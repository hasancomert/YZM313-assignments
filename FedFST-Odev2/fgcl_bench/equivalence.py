"""Vektörizasyon eşdeğerlik testi (FedFST yamaları P5a–P5f).

Orijinal fonksiyonlar yamalı repodan değil, git geçmişinden (`git show
d83fac0:<dosya>`) okunur; böylece "orijinal" taraf gerçekten orijinal koddur.
Her karşılaştırmada iki taraf aynı seed'le başlatılır ve şunlar istenir:
  * çıktılar birebir aynı (torch.equal; dtype, şekil ve gerekirse stride dahil),
  * gradyanlar birebir aynı (KL kaybı için),
  * çağrıdan sonra RNG durumu aynı (aynı generator'dan aynı sayıda sayı çekildi).

Kullanım:  python -m fgcl_bench.equivalence --fedfst <yamalı FedFST klasörü>
"""
import argparse
import ast
import random
import subprocess
import sys
import textwrap
import time
import types

import numpy as np
import torch
from torch_geometric.data import Data

COMMIT = "d83fac0"


def git_show(repo, path):
    return subprocess.run(["git", "-C", str(repo), "show", f"{COMMIT}:{path}"],
                          check=True, capture_output=True, text=True).stdout


def module_from_source(name, source):
    mod = types.ModuleType(name)
    exec(compile(source, f"<{COMMIT}:{name}>", "exec"), mod.__dict__)
    return mod


def method_from_class(source, class_name, method_name, namespace):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    code = textwrap.dedent(ast.get_source_segment(source, item))
                    exec(compile(code, f"<{COMMIT}:{class_name}.{method_name}>", "exec"), namespace)
                    return namespace[method_name]
    raise KeyError(method_name)


def original_kl_block(ours_source):
    """feature_gen_train içindeki orijinal KL döngüsünü fonksiyona sarar."""
    lines = ours_source.split("\n")
    start = next(i for i, l in enumerate(lines) if l.strip() == "kl_losses = []")
    end = next(i for i in range(start, len(lines))
               if l_strip(lines[i]) == "loss_kl = torch.tensor(0.0, device=self.device)")
    block = textwrap.dedent("\n".join(lines[start:end + 1]))
    body = textwrap.indent(block, "    ")
    code = ("def original_kl(all_generated_features, all_generated_labels, subgraph, self):\n"
            f"{body}\n    return loss_kl\n")
    ns = {"torch": torch, "F": torch.nn.functional}
    exec(compile(code, f"<{COMMIT}:feature_gen_train KL block>", "exec"), ns)
    return ns["original_kl"]


def l_strip(s):
    return s.strip()


def rng_snapshot(device):
    snap = {"cpu": torch.get_rng_state().clone()}
    if device.type == "cuda":
        snap["cuda"] = torch.cuda.get_rng_state(device).clone()
    return snap


def same_rng(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def same_tensor(a, b, check_stride=True):
    if a.dtype != b.dtype or a.shape != b.shape or a.device != b.device:
        return False
    if check_stride and a.numel() > 1 and a.stride() != b.stride():
        return False
    return torch.equal(a, b)


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, n_cases, ok, detail=""):
        self.rows.append((name, n_cases, ok, detail))
        print(f"  {'OK ' if ok else 'FARK'}  {name:<46s} {n_cases:>4d} durum  {detail}")

    @property
    def all_ok(self):
        return all(r[2] for r in self.rows)


def random_graph(gen, n_nodes, n_classes, n_edges, device, n_feat=8):
    y = torch.randint(0, n_classes, (n_nodes,), generator=gen)
    x = torch.rand(n_nodes, n_feat, generator=gen)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), generator=gen)
    return Data(x=x.to(device), y=y.to(device), edge_index=edge_index.to(device))


def test_reduce_edges(orig_utils, new_utils, device, report, n_cases=60):
    # GPU'da orijinal döngü her kenarda .item() ile senkronize olur ve yavaştır: GPU'da daha az ve
    # daha küçük durum denenir (CPU'da tam kapsam).
    gpu = device.type == "cuda"
    if gpu:
        n_cases = 9
    gen = torch.Generator().manual_seed(1234)
    for fname in ["reduce_homophilic_edges", "reduce_heterophilic_edges"]:
        ok = True
        for case in range(n_cases):
            n = int(torch.randint(2, 300 if gpu else 900, (1,), generator=gen))
            c = int(torch.randint(1, 5, (1,), generator=gen))
            e = int(torch.randint(0, 2 * n + 1, (1,), generator=gen))
            ratio = [0.05, 0.1, 0.5][case % 3]
            data = random_graph(gen, n, c, e, device)
            outs = []
            for impl in (orig_utils, new_utils):
                seed_all(100 + case)
                ei = data.edge_index
                for _ in range(3):  # ardışık çağrılar (jeneratör eğitimindeki döngü gibi)
                    ei = getattr(impl, fname)(data, ei, reduction_ratio=ratio)
                outs.append((ei, rng_snapshot(device)))
            ok &= same_tensor(outs[0][0], outs[1][0], check_stride=False) and same_rng(outs[0][1], outs[1][1])
        report.add(f"utils.{fname}", n_cases, ok)

    # S_high / homofili ayar döngüsünü taklit eden karışık dizi
    ok = True
    n_seq = 2 if gpu else 20
    for case in range(n_seq):
        data = random_graph(gen, 200 if gpu else 800, 4, 400 if gpu else 1600, device)
        outs = []
        for impl in (orig_utils, new_utils):
            seed_all(500 + case)
            ei = data.edge_index
            for it in range(60):
                fn = impl.reduce_homophilic_edges if (it + case) % 3 else impl.reduce_heterophilic_edges
                ei = fn(data, ei, reduction_ratio=0.05)
            outs.append((ei, rng_snapshot(device)))
        ok &= same_tensor(outs[0][0], outs[1][0], check_stride=False) and same_rng(outs[0][1], outs[1][1])
    report.add("60 iterasyonluk karışık azaltma dizisi", n_seq, ok)


def test_homophily_level(orig_method, new_method, device, report, n_cases=40):
    if device.type == "cuda":
        n_cases = 10
    gen = torch.Generator().manual_seed(77)
    ok = True
    for case in range(n_cases):
        tasks = []
        n_tasks = int(torch.randint(1, 4, (1,), generator=gen))
        for _ in range(n_tasks):
            n = int(torch.randint(5, 400, (1,), generator=gen))
            g = random_graph(gen, n, 3, int(torch.randint(0, 3 * n, (1,), generator=gen)), device)
            train_mask = torch.rand(n, generator=gen) < 0.6
            tasks.append({"local_data": g, "train_mask": train_mask})
        fake = types.SimpleNamespace(tasks=tasks)
        a = orig_method(fake, n_tasks)
        b = new_method(fake, n_tasks)
        ok &= (type(a) is type(b)) and (a == b)
    report.add("OursClient.get_learned_Homophily_Level", n_cases, ok)


def test_kl(orig_kl, new_kl, device, report):
    gen = torch.Generator().manual_seed(4242)
    ok = True
    n_cases = 0
    max_abs = 0.0
    for n_classes in (1, 2, 4):
        for per_class in (3, 200):
            for n_feat in (7, 64, 1433):
                for order in ("block", "shuffled", "missing_class"):
                    n_cases += 1
                    labels = torch.arange(n_classes).repeat_interleave(per_class)
                    if order == "shuffled":
                        labels = labels[torch.randperm(labels.numel(), generator=gen)]
                    feats0 = torch.randn(labels.numel(), n_feat, generator=gen)
                    sub_n = 300
                    sub_y = torch.randint(0, n_classes, (sub_n,), generator=gen)
                    if order == "missing_class" and n_classes > 1:
                        sub_y[sub_y == n_classes - 1] = 0  # bu sınıfın gerçek düğümü yok -> atlanmalı
                    sub = Data(x=torch.rand(sub_n, n_feat, generator=gen).to(device), y=sub_y.to(device))
                    results = []
                    for fn in (orig_kl, new_kl):
                        seed_all(9 + n_cases)
                        feats = feats0.clone().to(device).requires_grad_(True)
                        out = fn(feats, labels.to(device), sub, types.SimpleNamespace(device=device)) \
                            if fn is orig_kl else fn(feats, labels.to(device), sub, device)
                        out.backward()
                        results.append((out.detach(), feats.grad.detach().clone(), rng_snapshot(device)))
                    (l1, g1, r1), (l2, g2, r2) = results
                    if device.type == "cpu":
                        ok &= torch.equal(l1, l2) and torch.equal(g1, g2) and same_rng(r1, r2)
                    else:  # GPU indirgemeleri farklı sıralarla toplayabilir: değerler çok yakın, RNG aynı
                        max_abs = max(max_abs, (l1 - l2).abs().item(), (g1 - g2).abs().max().item())
                        ok &= same_rng(r1, r2) and torch.allclose(l1, l2, rtol=1e-5, atol=1e-6) \
                            and torch.allclose(g1, g2, rtol=1e-4, atol=1e-7)
    detail = "(karşılaştırılan: kayıp, gradyan, RNG durumu)" if device.type == "cpu" \
        else f"(RNG birebir istenir; GPU'da max|fark|={max_abs:.2e})"
    report.add("feature_gen_train KL kaybı (+gradyan)", n_cases, ok, detail)


def test_subgraph(orig_part, new_part, report, n_cases=60):
    gen = torch.Generator().manual_seed(99)
    ok = True
    for case in range(n_cases):
        n = int(torch.randint(1, 500, (1,), generator=gen))
        e = 0 if case % 10 == 0 else int(torch.randint(1, 4 * n, (1,), generator=gen))
        data = random_graph(gen, n, 5, e, torch.device("cpu"))
        k = int(torch.randint(0, n + 1, (1,), generator=gen))
        nodes = torch.randperm(n, generator=gen)[:k]
        if case % 3 == 0:
            node_list = [np.int64(v) for v in nodes.numpy()]       # class_to_task'taki gibi numpy int
        else:
            node_list = sorted(int(v) for v in nodes)              # get_learned_graph'taki gibi
        if case % 15 == 7 and k > 1:
            node_list = node_list + node_list[:1]                  # tekrar eden düğüm -> döngü yolu
        for flag in (True, False):
            try:
                a = orig_part.get_subgraph_by_node(data, list(node_list), flag)
            except Exception as exc:  # orijinal kodun bozuk kenar-durumları
                a = exc
            b = new_part.get_subgraph_by_node(data, list(node_list), flag)
            if isinstance(a, Exception):
                continue
            ok &= same_tensor(a.x, b.x) and same_tensor(a.y, b.y)
            ok &= same_tensor(a.edge_index, b.edge_index, check_stride=True)
            ok &= list(a.global_map) == list(b.global_map)
    report.add("partition.get_subgraph_by_node (flag=True/False)", n_cases, ok)


def test_louvain(orig_part, new_part, data, name, report):
    seed_all(24)
    t0 = time.time()
    a = orig_part.louvain_partitioner(data, 3)
    t_orig = time.time() - t0
    state_a = np.random.get_state()[1].copy()
    seed_all(24)
    t0 = time.time()
    b = new_part.louvain_partitioner(data, 3)
    t_new = time.time() - t0
    state_b = np.random.get_state()[1].copy()
    ok = len(a) == len(b) and np.array_equal(state_a, state_b)
    for ca, cb in zip(a, b):
        ok &= same_tensor(ca.x, cb.x) and same_tensor(ca.y, cb.y) and same_tensor(ca.edge_index, cb.edge_index)
    report.add(f"partition.louvain_partitioner ({name})", 1, ok, f"orijinal {t_orig:.1f} sn, yeni {t_new:.1f} sn")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fedfst", required=True, help="yamalı FedFST klasörü (git deposu)")
    ap.add_argument("--data_root", default=None, help="Cora'nın indirildiği kök (louvain testi için)")
    ap.add_argument("--cuda", action="store_true", help="GPU varsa ayrıca GPU'da da test et")
    args = ap.parse_args(argv)

    torch.set_num_threads(1)  # FedFST'nin set_seed'i de tek iş parçacığı kullanıyor
    sys.path.insert(0, args.fedfst)
    import utils as new_utils                       # yamalı
    from datasets import partition as new_part      # yamalı
    from algorithm import Ours as new_ours          # yamalı

    orig_utils = module_from_source("orig_utils", git_show(args.fedfst, "utils.py"))
    orig_part = module_from_source("orig_partition", git_show(args.fedfst, "datasets/partition.py"))
    ours_src = git_show(args.fedfst, "algorithm/Ours.py")
    orig_homo = method_from_class(ours_src, "OursClient", "get_learned_Homophily_Level", {"torch": torch})
    orig_kl = original_kl_block(ours_src)

    report = Report()
    devices = [torch.device("cpu")]
    if args.cuda and torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    for device in devices:
        print(f"[{device}]")
        test_reduce_edges(orig_utils, new_utils, device, report)
        test_homophily_level(orig_homo, new_ours.OursClient.get_learned_Homophily_Level, device, report)
        test_kl(orig_kl, new_ours._gen_kl_loss_vectorized, device, report)
    print("[cpu]")
    test_subgraph(orig_part, new_part, report)
    if args.data_root:
        from torch_geometric.datasets import Planetoid
        cora = Planetoid(args.data_root, "cora")[0]
        test_louvain(orig_part, new_part, cora, "Cora", report)
    print("SONUÇ:", "tüm eşdeğerlik testleri geçti" if report.all_ok else "FARK BULUNDU")
    return 0 if report.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
