"""FedSPA koşucusu: SFPD + HBDA (fedspa_port.py) görev akışında ardışık eğitilir.

FedSPA bir continual learning yöntemi değildir (standart FGL'de homofili heterojenliği için
tasarlandı). Burada, POWER makalesinin FedAvg/FedGTA gibi FGL yöntemlerini kullandığı biçimde,
görevler sırayla geldikçe aynı global model üzerinde eğitilen (fine-tuning) bir FGL baseline'ı
olarak çalıştırılır; hiçbir unutma önleyici mekanizma eklenmez.

Görev t için her istemcide:
  * görev alt-grafiği: cache'teki görev düğümleri (görevin sınıflarındaki tüm düğümler) ve
    aralarındaki kenarlar (FedFST'nin görev grafiğiyle aynı), düğümler yeniden numaralanır;
  * label_unk = görevin eğitim düğümlerinin etiketleri (ACM.preprocess, data.train_mask'ten);
  * adj_low_un / adj_low / adj_high (orijinalde logger'da, eğitimin başında bir kez) ve
    radio_diff = calculate_shigh(...) (orijinalde global_test'te, ilk round'da) görevin
    alt-grafiği için yeniden hesaplanır; HBDA'nın homofili ağırlıkları görev başında yenilenir.
Ortak FGCL bütçesi: görev başına `rounds` round, round başına `local_epochs` tam-grafik adım,
Adam(lr=1e-2, weight_decay=5e-4) (orijinal: SGD lr 0.1, 200 round, num_steps 4).
Her round'da 3 istemcinin hepsi katılır (orijinal betik flgo'nun varsayılan 'md' örneklemesini
kullanıyor; o örnekleme yerine koymalı olduğu için aynı istemci bir round'da birden çok kez seçilebiliyor).
Değerlendirmede global model eval modundadır (orijinal flgo kodu model.eval() çağırmıyor).
"""
import copy
import time

import torch
from torch_geometric.data import Data

from . import fedspa_port as port
from .common import set_seed, pick_device, task_view_hash

LR = 1e-2
WEIGHT_DECAY = 5e-4


def build_task_graph(client_cache, task):
    nodes = torch.sort(task["nodes"])[0]
    n_client = client_cache["x"].size(0)
    remap = torch.full((n_client,), -1, dtype=torch.long)
    remap[nodes] = torch.arange(nodes.numel())
    edge_index = remap[task["edge_index"]]
    assert bool((edge_index >= 0).all()), "görev kenarı görev düğümü dışına çıkıyor"
    data = Data(x=client_cache["x"][nodes].clone(), y=client_cache["y"][nodes].clone(), edge_index=edge_index,
                train_mask=task["train_mask"][nodes].clone(), val_mask=task["valid_mask"][nodes].clone(),
                test_mask=task["test_mask"][nodes].clone())
    return data, nodes


def run(cfg, cache, evaluator):
    meta = cache["meta"]
    set_seed(cfg.seed)
    device = pick_device()
    print(f"Using device: {device}")
    num_tasks = meta["num_tasks"]
    n_clients = len(cache["clients"])

    graphs = [[build_task_graph(cl, t) for t in cl["tasks"]] for cl in cache["clients"]]
    views = []
    for cid in range(n_clients):
        tv = []
        for t in range(num_tasks):
            g, nodes = graphs[cid][t]
            tv.append(dict(train=nodes[g.train_mask], valid=nodes[g.val_mask], test=nodes[g.test_mask], nodes=nodes,
                           edges=nodes[g.edge_index].contiguous(), y=g.y))
        views.append(tv)
    used_hash = task_view_hash(views)
    assert used_hash == cache["hash"], "FedSPA'ya verilen görevler ortak cache ile aynı değil!"
    print(f"[harness] görev maskeleri cache ile aynı (hash {used_hash[:12]})")

    graphs = [[(g.to(device), nodes) for g, nodes in row] for row in graphs]
    adj_cache = {}

    def adjs(cid, t):
        if (cid, t) not in adj_cache:
            adj_cache[(cid, t)] = port.compute_adjacencies(graphs[cid][t][0], device)
        return adj_cache[(cid, t)]

    model = port.Model(meta["in_dim"], meta["out_dim"], device=device).to(device)
    clients = [port.FedSPAClient(cid, LR, WEIGHT_DECAY, cfg.local_epochs) for cid in range(n_clients)]
    server = port.FedSPAServer(model, clients)
    print(f"FedSPA: hidden=128, dropout=0.1, Adam lr={LR} wd={WEIGHT_DECAY}, rounds/task={cfg.rounds}, "
          f"local steps={cfg.local_epochs}, istemci={n_clients}")

    # test sırası: evaluator.test_idx[c][t] istemci-içi numaralar; alt-grafik numarasına çevir
    test_pos = []
    for cid in range(n_clients):
        row = []
        for t in range(num_tasks):
            g, nodes = graphs[cid][t]
            pos = torch.searchsorted(nodes.to(device), evaluator.test_idx[cid][t].to(device))
            row.append(pos)
        test_pos.append(row)

    def predict(client_id, eval_task_id):
        g, _ = graphs[client_id][eval_task_id]
        adj_low_un, adj_low, adj_high = adjs(client_id, eval_task_id)
        c = clients[client_id]
        m = server.model
        was_training = m.training
        m.eval()
        with torch.no_grad():
            # Orijinal Client.test: istemcinin son eğitimdeki fr listeleriyle (denge dalı) çağrı
            out = m(g, adj_low_un, adj_low, adj_high, c.fr_high_g_list, c.fr_low_g_list, c.mlp_list)
        m.train(was_training)
        pred = out.max(1)[1]
        return pred[test_pos[client_id][eval_task_id]]

    task_seconds, hbda_weights, radio = [], [], []
    for task_id in range(num_tasks):
        t_task = time.time()
        print(f"\n========== Task {task_id}/{num_tasks} ==========")
        for cid, c in enumerate(clients):
            c.set_task(graphs[cid][task_id][0], adjs(cid, task_id))
            c.radio_diff = port.calculate_shigh(c.train_data.data)
        radio.append([float(c.radio_diff) for c in clients])
        print("radio_diff (S_high):", radio[-1])
        server.new_task()
        for round_id in range(cfg.rounds):
            models, losses = [], []
            for c in clients:
                local = copy.deepcopy(server.model)       # Server.pack: copy.deepcopy(self.model)
                c.model_g = copy.deepcopy(local)          # Client.reply
                losses.append(c.train(local))
                models.append(local)
            server.received_clients = list(range(n_clients))
            server.aggregate(models)
            print(f"round {round_id}: kayıplar {[round(l, 4) for l in losses]} | HBDA ağırlıkları {server.last_weights}")
        hbda_weights.append(server.last_weights)
        evaluator.evaluate_after_task(task_id, predict)
        task_seconds.append(round(time.time() - t_task, 2))

    return dict(task_seconds=task_seconds, mask_hash=used_hash, hbda_weights=hbda_weights, radio_diff=radio,
                hyperparameters=dict(model="ACM-SPA (2 katman)", hidden=128, dropout=0.1, optimizer="Adam", lr=LR,
                                     weight_decay=WEIGHT_DECAY, rounds=cfg.rounds, local_steps=cfg.local_epochs))
