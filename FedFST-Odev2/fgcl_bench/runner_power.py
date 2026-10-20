"""POWER koşucusu: resmi kod (Zenodo 10.5281/zenodo.15487654, zyl24-FCGL_POWER-78e30b6) + adaptör.

Algoritma kodu (flcore/Ours.py: OursClient, OursServer, GEModel; model.py; utils.py) olduğu gibi
import edilir. Adaptör yalnızca veri arayüzünü değiştirir:
  * POWER'ın veri katmanı (data.py, OpenFGL'in FGLDataset'i ve class_incremental) kullanılmaz;
    istemci grafikleri, görevler ve maskeler ortak cache'ten gelir.
  * OursClient.task_data, görev grafiğini ortak protokoldeki görev alt-grafiği olarak döndürür
    (FedFST'nin görev grafiğiyle aynı kenarlar). Orijinal task_data, edge_masking ile kenarları
    maskeleyip diske yazıyordu (bkz. notebook: edge_masking'in davranışı).
  * flcore/Ours.py'nin OpenFGL'den kullandığı tek fonksiyon (idx_to_mask_tensor) shims/ altında.
Eğitim döngüsü, POWER utils.start()'ın döngüsüdür; değerlendirme ortak Evaluator'la yapılır.
Karşılaştırma için POWER'ın kendi global değerlendirmesi (evaluate(use_global=True)) de
çalıştırılıp "acc_matrix_native" olarak kaydedilir.
"""
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Data

from .common import NUM_CLIENTS, set_seed, pick_device, task_view_hash


def load_power_args(power_dir):
    saved = sys.argv
    sys.argv = [saved[0]]          # config.py içe aktarılırken argparse çalıştırıyor -> varsayılanlar
    try:
        from config import args
    finally:
        sys.argv = saved
    return args


def run(cfg, cache, evaluator):
    power_dir = Path(cfg.power_dir)
    sys.path.insert(0, str(power_dir))
    sys.path.insert(1, str(Path(__file__).resolve().parent / "shims"))
    args = load_power_args(power_dir)
    from flcore.Ours import OursClient, OursServer
    import flcore.Ours as power_ours
    assert Path(power_ours.__file__).resolve().is_relative_to(power_dir.resolve()), power_ours.__file__

    meta = cache["meta"]
    args.num_clients = NUM_CLIENTS
    args.num_classes_per_task = meta["per_task"]
    args.dataset = cfg.dataset
    args.seed = cfg.seed
    args.num_rounds_per_task = cfg.rounds
    args.num_epochs = cfg.local_epochs
    args.input_dim = meta["in_dim"]
    args.output_dim = meta["out_dim"]
    args.task_dir = str(cfg.tmp_dir)
    args.global_eval = True   # bilgi amaçlı: ortak değerlendirme global modeli kullanıyor
    if cfg.smoke:
        args.num_it_recon = 3
        args.num_epoch_g = 2
    print("POWER args:", {k: getattr(args, k) for k in sorted(vars(args))})

    set_seed(cfg.seed)
    device = pick_device()
    print(f"Using device: {device}")

    class HarnessPowerClient(OursClient):
        """OursClient'ın yalnızca görev grafiğini kuran task_data yöntemi değiştirildi."""

        def task_data(self, task_id, whole_data, task):
            return Data(x=whole_data.x, edge_index=self.harness_task_edges[task_id].to(self.device), y=whole_data.y)

    fcgl_dataset = {}
    for cid, cl in enumerate(cache["clients"]):
        fcgl_dataset[cid] = {
            "data": Data(x=cl["x"].clone(), edge_index=cl["edge_index"].clone(), y=cl["y"].clone()),
            "task": [{"train_mask": t["train_mask"].clone(), "val_mask": t["valid_mask"].clone(),
                      "test_mask": t["test_mask"].clone()} for t in cl["tasks"]],
        }

    # POWER utils.load_clients_server ile aynı sıra: önce istemciler, sonra sunucu
    message_pool = {}
    clients = [HarnessPowerClient(args, client_id, fcgl_dataset[client_id], message_pool, device)
               for client_id in range(args.num_clients)]
    for cid, client in enumerate(clients):
        client.harness_task_edges = [t["edge_index"].clone() for t in cache["clients"][cid]["tasks"]]
    server = OursServer(args, message_pool, device)

    views = []
    for cid, client in enumerate(clients):
        tv = []
        for t, task in enumerate(client.data["task"]):
            nodes = torch.sort(cache["clients"][cid]["tasks"][t]["nodes"])[0]
            tv.append(dict(train=task["train_mask"].nonzero().view(-1), valid=task["val_mask"].nonzero().view(-1),
                           test=task["test_mask"].nonzero().view(-1), nodes=nodes,
                           edges=client.harness_task_edges[t].contiguous(), y=client.data["data"].y[nodes]))
        views.append(tv)
    used_hash = task_view_hash(views)
    assert used_hash == cache["hash"], "POWER'a verilen görevler ortak cache ile aynı değil!"
    print(f"[harness] görev maskeleri cache ile aynı (hash {used_hash[:12]})")

    def predict(client_id, eval_task_id):
        model = server.global_model
        was_training = model.training
        model.eval()
        whole_data = clients[client_id].data["data"].to(device)
        task = clients[client_id].data["task"][eval_task_id]
        task_data = clients[client_id].task_data(eval_task_id, whole_data, task)
        with torch.no_grad():
            logits, _, _ = model.forward(task_data)
        model.train(was_training)
        pred = logits.max(1)[1]
        return pred[evaluator.test_idx[client_id][eval_task_id].to(pred.device)]

    # ---- POWER utils.start() ----
    print("-" * 50)
    for client_id in range(args.num_clients):
        result_list = []
        for task_id in range(len(clients[client_id].data["task"])):
            label = clients[client_id].data["data"].y
            task = clients[client_id].data["task"][task_id]
            node_mask = task["train_mask"] | task["val_mask"] | task["test_mask"]
            result = torch.unique(label[node_mask]).cpu().tolist()
            result_list.append(str(result))
        print(f"client {client_id} tasks: {'=>'.join(result_list)}")
    print("-" * 50)
    num_tasks = len(fcgl_dataset[0]["task"])
    args.num_tasks = num_tasks
    native_acc = torch.zeros(size=(num_tasks, num_tasks)).to(device)
    task_seconds = []

    for task_id in range(num_tasks):
        t_task = time.time()
        native_acc[task_id, :] = 0
        message_pool["task_id"] = task_id
        for client_id in range(args.num_clients):
            clients[client_id].task_start(task_id)
        server.task_start(task_id)
        for round_id in range(args.num_rounds_per_task):
            message_pool["round_id"] = round_id
            server.send_message()
            for client_id in range(args.num_clients):
                clients[client_id].execute(task_id)
                clients[client_id].send_message(task_id)
            server.execute()

        # POWER'ın kendi global değerlendirmesi (karşılaştırma için)
        for eval_task_id in range(0, task_id + 1):
            total_nodes = 0
            for client_id in range(args.num_clients):
                num_nodes = clients[client_id].data["task"][eval_task_id]["test_mask"].sum()
                if int(num_nodes) == 0:
                    continue
                client_acc = clients[client_id].evaluate(task_id=eval_task_id, use_global=True)
                native_acc[task_id, eval_task_id] += client_acc * num_nodes
                total_nodes += num_nodes
            native_acc[task_id, eval_task_id] /= total_nodes
        # ortak değerlendirme
        evaluator.evaluate_after_task(task_id, predict)

        for client_id in range(args.num_clients):
            clients[client_id].task_done(task_id)
        server.task_done(task_id)
        task_seconds.append(round(time.time() - t_task, 2))
        print("-" * 50)

    native = native_acc.cpu()
    max_diff = float((native - evaluator.acc).abs().max())
    print(f"[harness] ortak değerlendirme ile POWER'ın kendi global değerlendirmesi arasındaki en büyük fark: {max_diff:.2e}")
    return dict(task_seconds=task_seconds,
                acc_matrix_native=[[round(float(v), 6) for v in row] for row in native],
                native_max_abs_diff=max_diff,
                hyperparameters={k: getattr(args, k) for k in (
                    "model", "hid_dim", "dropout", "lr", "optim", "weight_decay", "num_rounds_per_task", "num_epochs",
                    "alpha", "replay", "num_samples_per_class", "lr_g", "LBFGS_init_lr", "num_it_recon", "beta",
                    "num_recon_nodes", "decay", "num_epoch_g")},
                mask_hash=used_hash)
