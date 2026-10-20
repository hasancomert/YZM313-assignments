"""FedFST koşucusu: main.py'nin eğitim döngüsü, resmi (yamalı) Ours.py sınıflarıyla.

main.py'ye göre yalnızca üç fark var (eğitim mantığı aynı):
  1. Veri/bölümleme main.py'deki sırayla üretilir ve maskelerin ortak cache'le aynı olduğu
     hash ile doğrulanır.
  2. main.py istemcilerin jeneratör eğitimini ThreadPoolExecutor ile eşzamanlı çalıştırıyor.
     Üç iş parçacığı aynı küresel RNG'yi paylaştığı için sonuç aynı seed ile bile
     tekrarlanamıyor. Burada aynı adımlar aynı sırayla (istemci 0, 1, 2) ardışık çalışır.
  3. Değerlendirme ortak Evaluator ile yapılır. Karşılaştırma için FedFST'nin kendi
     değerlendirmesi de (client.evaluate, global_flag=True) aynı yerde çalıştırılıp
     sonuçları JSON'a "acc_matrix_native" olarak yazılır.
Kayıp grafikleri (plot_all_losses) algoritmanın parçası olmadığı için çizilmez.
"""
import sys
import time

import torch

from .common import DATASETS, task_view_hash


def fedfst_args(cfg):
    from args import parser  # FedFST'nin args.py'si
    info = DATASETS[cfg.dataset]
    argv = ["--dataset_name", info["fedfst_name"], "--seed", str(cfg.seed), "--device_id", "0",
            "--per_task_class_num", str(info["per_task"]), "--clients_num", "3",
            "--rounds", str(cfg.rounds), "--local_epochs", str(cfg.local_epochs),
            "--dataset_dir", str(cfg.data_root), "--save_dir", str(cfg.tmp_dir)]
    if cfg.smoke:
        argv += ["--gen_epochs", "2", "--kd_epochs", "3"]
    return parser.parse_args(argv)


def views_from_fedfst_clients(clients):
    views = []
    for client in clients:
        tv = []
        for task in client.tasks:
            nodes = torch.sort(torch.tensor([int(v) for v in task["local_data"].global_map], dtype=torch.long))[0]
            tv.append(dict(train=task["train_mask"].nonzero().view(-1), valid=task["valid_mask"].nonzero().view(-1),
                           test=task["test_mask"].nonzero().view(-1), nodes=nodes,
                           edges=task["local_data"].edge_index.detach().cpu().contiguous(),
                           y=client.data.y.detach().cpu()[nodes]))
        views.append(tv)
    return views


def run(cfg, cache, evaluator):
    sys.path.insert(0, str(cfg.fedfst_dir))
    from algorithm.Ours import load_server_clients
    from datasets.partition import louvain_partitioner
    from datasets.dataset_loader import load_dataset
    from utils import set_seed

    args = fedfst_args(cfg)
    # ---- main.py, satır 23-58 (aynı sıra) ----
    print(f"Seed: {args.seed}, Dataset: {args.dataset_name}, Clients: {args.clients_num}, Model: {args.model}")
    print(f"gen_lr={args.gen_lr}, kd_lr={args.kd_lr}, kd_khop={args.kd_khop}, gen_epochs={args.gen_epochs}, kd_epochs={args.kd_epochs}, lr={args.lr}")
    print(f"gen_rounds={args.gen_rounds}, tolerance={args.tolerance}, max_iterations={args.max_iterations}, gen_num_nodes={args.gen_num_nodes}, per_task_class_num={args.per_task_class_num}")
    print(f"kd_ce_weight={args.kd_ce_weight}, kd_low_weight={args.kd_low_weight}, temperature={args.kd_temperature}")
    print(f"gen_ce_weight={args.gen_ce_weight}, gen_kl_weight={args.gen_kl_weight}")
    print(f"S_high_x_weight={args.S_high_x_weight}, S_high_A_weight={args.S_high_A_weight}")

    set_seed(args.seed)

    if args.use_gpu:
        device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    data, input_dim, out_dim = load_dataset(args.dataset_dir, args.dataset_name, args.per_task_class_num, args)
    args.input_dim = input_dim
    args.output_dim = out_dim
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {tuple(data.x.shape)}, Classes: {data.y.max().item() + 1}")

    clients_data = louvain_partitioner(data, args.clients_num)
    args.input_dim = input_dim
    args.out_dim = out_dim
    classes_num = data.y.max().item() + 1
    server, clients, message_pool = load_server_clients(args, clients_data, device, classes_num)

    tasks_num = (data.y.max().item() + 1) // args.per_task_class_num
    print(f"Total number of tasks: {tasks_num}")

    used_hash = task_view_hash(views_from_fedfst_clients(clients))
    assert used_hash == cache["hash"], "FedFST'nin ürettiği görev maskeleri ortak cache ile aynı değil!"
    print(f"[harness] görev maskeleri cache ile aynı (hash {used_hash[:12]})")
    assert tasks_num == cache["meta"]["num_tasks"]

    native_acc = torch.zeros(size=(tasks_num, tasks_num)).to(device)
    task_seconds = []

    def predict(client_id, eval_task_id):
        model = server.global_model
        was_training = model.training
        model.eval()
        data_t = clients[client_id].tasks[eval_task_id]["local_data"].to(device)
        with torch.no_grad():
            _, out = model(data_t)
        model.train(was_training)
        _, pred = out.max(dim=1)
        return pred[evaluator.test_idx[client_id][eval_task_id].to(pred.device)]

    for task_id in range(tasks_num):
        t_task = time.time()
        print(f"\n========== Task {task_id}/{tasks_num} ==========")

        for round in range(args.rounds):
            print(f"GNN Round {round}")
            print(f"Training clients for task {task_id}...")
            client_losses = []
            for client_id, client in enumerate(clients):
                loss = client.train(task_id)
                client_losses.append(loss)
                print(f"Client {client_id} loss: {loss:.4f}")

            avg_client_loss = sum(client_losses) / len(client_losses)
            print(f"Average client loss: {avg_client_loss:.4f}")

            for client_id, client in enumerate(clients):
                client.send_message(task_id)

            server.clients_nodes_num = [server.message_pool[f"client_{client_id}"]["nodes_num"]
                                        for client_id in range(args.clients_num)]
            server.clients_learned_nodes_num = [server.message_pool[f"client_{client_id}"]["learned_nodes_num"]
                                                for client_id in range(args.clients_num)]

            print("Server aggregating client models...")
            server.aggregate()
            server.send_message()

        if task_id != 0:
            server.feature_gen_init(task_id = task_id)
            for client_id, client in enumerate(clients):
                client.feature_gen_init(task_id = task_id)

            for round in range(args.gen_rounds):
                print(f"Generator Round {round}")
                # main.py: ThreadPoolExecutor ile eşzamanlı. Burada aynı çağrılar aynı sırayla ardışık.
                for client_id, client in enumerate(clients):
                    client.feature_gen_train(task_id, round != 0)

                for client_id, client in enumerate(clients):
                    client.send_feature_gen(task_id)

                server.feature_gen_aggregate()
                server.send_feature_gen()

            server.KD_train(task_id)

        server.update_last_global_model()
        for client_id, client in enumerate(clients):
            client.last_global_model.load_state_dict(server.last_global_model.state_dict())

        for client_id, client in enumerate(clients):
            client.save_validation_acc(task_id)

        # --- FedFST'nin kendi değerlendirmesi (main.py satır 122-132), karşılaştırma için ---
        for eval_task_id in range(0, task_id + 1):
            total_nodes_num = 0
            for client_id in range(args.clients_num):
                nodes_num = clients[client_id].tasks[eval_task_id]["test_mask"].sum()
                if int(nodes_num) == 0:
                    continue
                evaluation = clients[client_id].evaluate(task_id = eval_task_id, global_flag=True)
                native_acc[task_id, eval_task_id] += evaluation["acc"] * nodes_num
                total_nodes_num += nodes_num
            native_acc[task_id, eval_task_id] /= total_nodes_num

        # --- ortak değerlendirme ---
        evaluator.evaluate_after_task(task_id, predict)
        task_seconds.append(_r2(time.time() - t_task))

    native = native_acc.cpu()
    max_diff = float((native - evaluator.acc).abs().max())
    print(f"[harness] ortak değerlendirme ile FedFST'nin kendi değerlendirmesi arasındaki en büyük fark: {max_diff:.2e}")
    return dict(task_seconds=task_seconds,
                acc_matrix_native=[[_r6(v) for v in row] for row in native],
                native_max_abs_diff=max_diff,
                hyperparameters={k: getattr(args, k) for k in (
                    "lr", "weight_decay", "hidden_dim", "dropout", "model", "rounds", "local_epochs",
                    "gen_rounds", "gen_epochs", "gen_lr", "kd_epochs", "kd_lr", "kd_khop", "max_iterations",
                    "tolerance", "gen_num_nodes", "num_samples_per_class", "per_task_class_num")},
                mask_hash=used_hash)


def _r2(x):
    return round(x, 2)


def _r6(v):
    return round(float(v), 6)
