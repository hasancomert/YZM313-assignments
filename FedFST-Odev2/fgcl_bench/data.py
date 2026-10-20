"""Veri katmanı: ham dosyalar, FedFST fonksiyonlarıyla bölümleme/görevler ve diske cache.

Bölümleme ve görevler, FedFST'nin main.py'sindeki sırayla ve FedFST'nin kendi
fonksiyonlarıyla üretilir:
    set_seed(seed) -> load_dataset -> louvain_partitioner (python-louvain, numpy RNG)
    -> sınıf sırası: random.seed(seed); random.shuffle (load_server_clients'taki gibi)
    -> her istemci için class_to_task (sınıf başına 0.6/0.2/0.2, numpy RNG)
main.py'de bu adımların arasında yalnızca model ağırlıkları başlatılır (torch RNG);
numpy/random akışını etkilemedikleri için maskeler main.py'dekiyle aynıdır. FedFST
koşucusu (runner_fedfst.py) bu sırayı modellerle birlikte yeniden çalıştırır ve
maskelerin cache'tekiyle aynı olduğunu hash ile doğrular.

Komut satırı:  python -m fgcl_bench.data --fedfst <dir> --work_dir <dir> --dataset cora --seed 24
"""
import argparse
import random
import sys
import time
import urllib.request
from pathlib import Path

import torch

from .common import DATASETS, NUM_CLIENTS, task_view_hash

GH = "https://raw.githubusercontent.com"
GEOM = f"{GH}/graphdml-uiuc-jlu/geom-gcn"
GEOM_SQUIRREL = f"{GEOM}/f1fc0d14b3b019c562737240d06ec83b07d16a8f"  # PyG'nin WikipediaNetwork'te sabitlediği commit


def raw_files(dataset):
    """(kök altındaki göreli yol, URL) çiftleri; PyG'nin beklediği klasör düzeninde."""
    if dataset in ("cora", "citeseer"):
        return [(f"{dataset}/raw/ind.{dataset}.{s}", f"{GH}/kimiyoung/planetoid/master/data/ind.{dataset}.{s}")
                for s in ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")]
    if dataset == "actor":
        files = [(f"raw/{f}", f"{GEOM}/master/new_data/film/{f}")
                 for f in ("out1_node_feature_label.txt", "out1_graph_edges.txt")]
        return files + [(f"raw/film_split_0.6_0.2_{i}.npz", f"{GEOM}/master/splits/film_split_0.6_0.2_{i}.npz")
                        for i in range(10)]
    if dataset == "squirrel":
        files = [(f"squirrel/geom_gcn/raw/{f}", f"{GEOM_SQUIRREL}/new_data/squirrel/{f}")
                 for f in ("out1_node_feature_label.txt", "out1_graph_edges.txt")]
        return files + [(f"squirrel/geom_gcn/raw/squirrel_split_0.6_0.2_{i}.npz",
                         f"{GEOM_SQUIRREL}/splits/squirrel_split_0.6_0.2_{i}.npz") for i in range(10)]
    if dataset == "minesweeper":
        return [("minesweeper/raw/minesweeper.npz",
                 f"{GH}/yandex-research/heterophilous-graphs/main/data/minesweeper.npz")]
    raise KeyError(dataset)


def prefetch_raw(data_root, dataset, retries=4):
    """Eksik ham dosyaları indirir. PyG de aynı dosyaları indirebilir; burada önceden indirip
    PyG'nin indirme adımını atlamasını sağlıyoruz (github.com/.../raw yönlendirmesi yerine
    doğrudan raw.githubusercontent.com). İndirme başarısız olursa PyG kendi indirmesini dener."""
    data_root = Path(data_root)
    for rel, url in raw_files(dataset):
        dest = data_root / rel
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries):
            try:
                tmp = dest.with_suffix(dest.suffix + ".part")
                urllib.request.urlretrieve(url, tmp)
                tmp.replace(dest)
                break
            except Exception as exc:  # ağ hatası: kısa bekleyip tekrar dene
                if attempt == retries - 1:
                    print(f"[uyarı] {url} indirilemedi ({exc}); PyG kendi indirmesini deneyecek.")
                else:
                    time.sleep(2 ** (attempt + 1))


def cache_path(work_dir, dataset, seed):
    return Path(work_dir) / "cache" / f"{dataset}_seed{seed}.pt"


def views_from_cache(cache):
    views = []
    for cl in cache["clients"]:
        tv = []
        for task in cl["tasks"]:
            nodes = torch.sort(task["nodes"])[0]
            tv.append(dict(train=task["train_mask"].nonzero().view(-1),
                           valid=task["valid_mask"].nonzero().view(-1),
                           test=task["test_mask"].nonzero().view(-1),
                           nodes=nodes, edges=task["edge_index"], y=cl["y"][nodes]))
        views.append(tv)
    return views


def build_cache(fedfst_dir, data_root, dataset, seed):
    sys.path.insert(0, str(fedfst_dir))
    from utils import set_seed                                     # FedFST
    from datasets.dataset_loader import load_dataset, class_to_task  # FedFST (yamalı)
    from datasets.partition import louvain_partitioner             # FedFST (yamalı)
    from torch_geometric.utils import homophily

    info = DATASETS[dataset]
    k = info["per_task"]
    t0 = time.time()
    set_seed(seed)
    data, in_dim, out_dim = load_dataset(str(data_root), info["fedfst_name"], k, None)
    y_original = data.y.clone()
    stats = dict(nodes=int(data.num_nodes), edges=int(data.edge_index.size(1)), features=int(data.x.size(1)),
                 classes=int(data.y.max()) + 1,
                 edge_homophily=float(homophily(data.edge_index, data.y, method="edge")))

    clients_data = louvain_partitioner(data, NUM_CLIENTS)
    classes_num = data.y.max().item() + 1
    classes_order = list(range(classes_num))
    random.seed(seed)                      # algorithm/Ours.py: load_server_clients (shuffle_flag=True)
    random.shuffle(classes_order)

    clients = []
    for cid, cdata in enumerate(clients_data):
        tasks = class_to_task(data=cdata, per_task_class_num=k, train_prop=0.6, valid_prop=0.2,
                              test_prop=0.2, shuffle_flag=True, classes_order=classes_order)
        ctasks = []
        for task in tasks:
            ctasks.append(dict(train_mask=task["train_mask"].clone(), valid_mask=task["valid_mask"].clone(),
                               test_mask=task["test_mask"].clone(),
                               nodes=torch.tensor([int(v) for v in task["local_data"].global_map], dtype=torch.long),
                               edge_index=task["local_data"].edge_index.contiguous().clone()))
        clients.append(dict(x=cdata.x.clone(), y=cdata.y.clone(), edge_index=cdata.edge_index.contiguous().clone(),
                            tasks=ctasks))

    num_tasks = classes_num // k
    task_classes = [[classes_order[t * k + i] for i in range(k)] for t in range(num_tasks)]
    dropped = [classes_order[j] for j in range(num_tasks * k, classes_num)]
    stats["clients"] = [dict(nodes=int(cl["x"].size(0)), edges=int(cl["edge_index"].size(1)),
                             tasks=[dict(nodes=int(t["nodes"].numel()), train=int(t["train_mask"].sum()),
                                         valid=int(t["valid_mask"].sum()), test=int(t["test_mask"].sum()),
                                         edges=int(t["edge_index"].size(1))) for t in cl["tasks"]])
                        for cl in clients]
    cache = dict(meta=dict(dataset=dataset, seed=seed, num_clients=NUM_CLIENTS, per_task=k, num_tasks=num_tasks,
                           num_classes=classes_num, in_dim=int(in_dim), out_dim=int(out_dim),
                           classes_order=classes_order, task_classes=task_classes, dropped_classes=dropped,
                           stats=stats, build_seconds=round(time.time() - t0, 2)),
                 clients=clients, y_original=y_original)
    cache["hash"] = task_view_hash(views_from_cache(cache))
    return cache


def load_cache(work_dir, dataset, seed):
    return torch.load(cache_path(work_dir, dataset, seed), weights_only=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fedfst", required=True)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--seed", type=int, default=24)
    args = ap.parse_args(argv)
    data_root = Path(args.work_dir) / "data"
    out = cache_path(args.work_dir, args.dataset, args.seed)
    if out.exists():
        cache = torch.load(out, weights_only=False)
        print(f"cache mevcut: {out} (hash {cache['hash'][:12]})")
        return 0
    prefetch_raw(data_root, args.dataset)
    cache = build_cache(args.fedfst, data_root, args.dataset, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".pt.tmp")
    torch.save(cache, tmp)
    tmp.replace(out)
    m = cache["meta"]
    print(f"cache yazıldı: {out} | görev={m['num_tasks']} atılan sınıf={m['dropped_classes']} "
          f"| istemci düğümleri={[c['nodes'] for c in m['stats']['clients']]} | hash {cache['hash'][:12]} "
          f"| {m['build_seconds']} sn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
