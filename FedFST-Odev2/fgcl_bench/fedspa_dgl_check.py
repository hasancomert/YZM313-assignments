"""FedSPA port'unun (fedspa_port.py) orijinal DGL'li koda eşdeğerlik testi.

DGL gerektirir (Colab'ın Python/torch sürümü için DGL wheel'i yok); yerelde ayrı bir ortamda
(Python 3.11, torch 2.2.1, DGL 2.1.0) çalıştırıldı. Orijinal kod FedSPA reposundan okunur:
  benchmark001/config.py  (GraphConvolution, ACM, mp_func_spa, agg_func_spa, DGL blok kurulumu)
  benchmark001/base.py    (MLP_spa)
  flgo/experiment/logger/simple_logger.py (adj matrisleri), flgo/algorithm/fedspa.py (fish_cal,
  Server.aggregate ve yardımcıları), flgo/algorithm/fedbase.py (calculate_shigh)
Aynı ağırlıklar (state_dict, strict=True) ve aynı seed'le iki model karşılaştırılır.

    python -m fgcl_bench.fedspa_dgl_check --fedspa <FedSPA klasörü> [--cache <cache.pt>]
"""
import argparse
import ast
import collections
import copy
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

import dgl  # noqa: F401  (yoksa test çalışmaz)

from . import fedspa_port as port


def extract(source, names, namespace, class_name=None):
    tree = ast.parse(source)
    body = tree.body
    if class_name:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(textwrap.dedent(ast.get_source_segment(source, node)), "<orig>", "exec"), namespace)
    return namespace


def load_original(fedspa_dir):
    code = Path(fedspa_dir) / "Code"
    base_ns = {}
    exec(compile((code / "benchmark001/base.py").read_text(), "<orig base.py>", "exec"), base_ns)
    cfg_src = (code / "benchmark001/config.py").read_text()
    kept = []
    for line in cfg_src.split("\n"):
        if line.startswith("from .base import") or line.startswith("from load_data.load_data import") \
                or line.startswith("train_data = load_dataset("):
            kept.append("# [test] " + line)
        else:
            kept.append(line)
    cfg = types.ModuleType("orig_config")
    cfg.__dict__["MLP_spa"] = base_ns["MLP_spa"]
    exec(compile("\n".join(kept), "<orig config.py>", "exec"), cfg.__dict__)

    import dgl as _dgl
    log_ns = {"torch": torch, "dgl": _dgl, "np": np}
    extract((code / "flgo/experiment/logger/simple_logger.py").read_text(),
            {"dgl_to_scipy", "compute_low_freq_matrix", "get_unnormalized_low_freq_matrix",
             "normalize_tensor", "convert_to_dglgraph"}, log_ns)
    fed_ns = {"torch": torch, "np": np, "F": F, "collections": collections}
    fedspa_src = (code / "flgo/algorithm/fedspa.py").read_text()
    extract(fedspa_src, {"sum_of_layer_fisher_differences", "normalize_list", "calculate_fisher_average",
                         "normalize", "calculate_weights", "fish_cal"}, fed_ns)
    extract(fedspa_src, {"aggregate"}, fed_ns, class_name="Server")
    extract((code / "flgo/algorithm/fedbase.py").read_text(), {"calculate_shigh"}, fed_ns)
    return cfg, log_ns, fed_ns


def orig_adjacencies(log_ns, data, device):
    # simple_logger.py: SimpleLogger.log_once (current_round == -1 dalı), device = cpu
    g = log_ns["convert_to_dglgraph"](data)
    adj_low_unnormalized = log_ns["get_unnormalized_low_freq_matrix"](g).to(device)
    eye_matrix = torch.eye(g.number_of_nodes()).to(device)
    adj_low = (log_ns["normalize_tensor"](eye_matrix + adj_low_unnormalized.to_dense())).to(device)
    adj_high = (torch.eye(g.number_of_nodes()).to(device) - adj_low).to(device).to_sparse()
    return adj_low_unnormalized, adj_low, adj_high


class OrigWrapper(torch.nn.Module):
    """default_model.Model ile aynı isimlendirme (model.xxx) için.

    Orijinal att_struc_low ilklenmemiş bellek (torch.FloatTensor); yerel denemelerde zaman zaman
    NaN/1e38 gibi değerler içerdi ve has_nan() kontrolü modeli birleştirmeden çıkardı (port'taki
    değişiklik 4). Karşılaştırmanın hesaba odaklanması için burada da sıfırlanır."""

    def __init__(self, acm):
        super().__init__()
        self.model = acm
        with torch.no_grad():
            for g in self.model.gcns:
                g.att_struc_low.zero_()

    def forward(self, *a, **k):
        return self.model(*a, **k)

    def has_nan(self):
        return any(torch.isnan(p).any().item() for p in self.parameters())


def sequential_sum_limit():
    """torch.sum(dim=1)'in sıralı toplamla birebir aynı kaldığı derece sınırı (çeşitli iç boyutlar
    için en küçüğü). DGL'in agg_func_spa'sı posta kutusunu sum(1) ile toplar; PyTorch bu toplamı
    tensör şekline göre farklı sıralarla yapar, index_add_ ise sıralı toplar."""
    gen = torch.Generator().manual_seed(0)
    limits = []
    for d in (1, 2, 3, 4, 5, 8, 16, 64, 128):
        lim = 200
        for deg in range(1, 200):
            t = torch.randn(7, deg, d, generator=gen)
            seq = torch.zeros(7, d)
            for k in range(deg):
                seq = seq + t[:, k]
            if not torch.equal(t.sum(1), seq):
                lim = deg
                break
        limits.append(lim)
    return min(limits)


SEQ_SUM_LIMIT = None


def seed_all(s):
    np.random.seed(s)
    torch.manual_seed(s)


def graphs_for_test(cache_path):
    gen = torch.Generator().manual_seed(5)
    out = []
    for n, e, f, c in [(40, 50, 6, 3), (12, 30, 6, 3), (40, 160, 16, 4), (200, 1500, 32, 2)]:
        edge_index = torch.randint(0, n, (2, e), generator=gen)
        edge_index = torch.unique(edge_index, dim=1)  # tekrar eden kenar yok (gerçek veride de yok)
        x = torch.rand(n, f, generator=gen)
        y = torch.randint(0, c, (n,), generator=gen)
        train_mask = torch.rand(n, generator=gen) < 0.6
        out.append((f"rastgele N={n} E={edge_index.size(1)}", Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask), c))
    if cache_path:
        cache = torch.load(cache_path, weights_only=False)
        cl = cache["clients"][0]
        task = cl["tasks"][0]
        nodes = torch.sort(task["nodes"])[0]
        remap = torch.full((cl["x"].size(0),), -1, dtype=torch.long)
        remap[nodes] = torch.arange(nodes.numel())
        ei = remap[task["edge_index"]]
        d = Data(x=cl["x"][nodes], y=cl["y"][nodes], edge_index=ei, train_mask=task["train_mask"][nodes])
        out.append((f"{cache['meta']['dataset']} istemci0/görev0 N={nodes.numel()} E={ei.size(1)}",
                    d, cache["meta"]["out_dim"]))
    return out


def compare(name, a, b, rows):
    """rows'a (ad, birebir mi, göreli fark) ekler; göreli fark = max|a-b| / max|a|."""
    if isinstance(a, torch.Tensor):
        exact = a.shape == b.shape and torch.equal(a, b)
        if exact:
            diff = 0.0
        elif a.shape != b.shape:
            diff = float("inf")
        else:
            diff = float((a - b).abs().max()) / max(float(a.abs().max()), 1e-30)
    else:
        exact = a == b
        diff = 0.0 if exact else float("inf")
    rows.append((name, exact, diff))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fedspa", required=True)
    ap.add_argument("--cache", default=None, help="gerçek bir görev alt-grafiği için cache dosyası")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    global SEQ_SUM_LIMIT
    SEQ_SUM_LIMIT = sequential_sum_limit()
    cfg, log_ns, fed_ns = load_original(args.fedspa)
    device = torch.device("cpu")
    print(f"torch {torch.__version__}, dgl {dgl.__version__}; torch.sum(dim=1) derece < {SEQ_SUM_LIMIT} iken sıralı toplamla birebir")
    all_ok, worst = True, 0.0
    for gname, data, nclass in graphs_for_test(args.cache):
        rows = []
        nfeat = data.x.size(1)
        seed_all(0)
        orig = OrigWrapper(cfg.get_spa(nfeat, 128, nclass, 2)).to(device)
        prt = port.Model(nfeat, nclass, device=device).to(device)
        prt.load_state_dict(orig.state_dict(), strict=True)   # aynı parametre adları ve şekilleri
        compare("parametre adları", [n for n, _ in orig.named_parameters()], [n for n, _ in prt.named_parameters()], rows)

        a_un, a_low, a_high = orig_adjacencies(log_ns, data, device)
        p_un, p_low, p_high = port.compute_adjacencies(data, device)
        compare("adj_low_un (yoğun)", a_un.to_dense(), p_un.to_dense(), rows)
        compare("adj_low", a_low, p_low, rows)
        compare("adj_high (yoğun)", a_high.to_dense(), p_high.to_dense(), rows)

        for branch in ("dengesiz (fr listesi yok)", "denge dalı (fr listeleri)"):
            outs = []
            for model, (au, al, ah) in ((orig, (a_un, a_low, a_high)), (prt, (p_un, p_low, p_high))):
                m = copy.deepcopy(model)
                m.train()
                m.zero_grad()
                lists = None
                if branch.startswith("denge"):
                    lists = ([g.att_vec_high_unk for g in m.model.gcns], [g.att_vec_low_unk for g in m.model.gcns],
                             [g.att_vec_mlp for g in m.model.gcns])
                seed_all(11)
                out = m(data, au, al, ah, *lists) if lists else m(data, au, al, ah)
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                loss.backward()
                grads = {n: (p.grad.clone() if p.grad is not None else None) for n, p in m.named_parameters()}
                flags = {n: p.requires_grad for n, p in m.named_parameters()}
                m.eval()
                with torch.no_grad():
                    out_eval = m(data, au, al, ah, *lists) if lists else m(data, au, al, ah)
                outs.append((out.detach(), grads, flags, out_eval))
            compare(f"{branch}: çıktı (train)", outs[0][0], outs[1][0], rows)
            compare(f"{branch}: çıktı (eval)", outs[0][3], outs[1][3], rows)
            compare(f"{branch}: requires_grad bayrakları", outs[0][2], outs[1][2], rows)
            g0, g1 = outs[0][1], outs[1][1]
            same_none = all((g0[k] is None) == (g1[k] is None) for k in g0)
            rows.append((f"{branch}: gradyanı olan parametreler", same_none, 0.0 if same_none else float("inf")))
            for k in g0:
                if g0[k] is not None and g1[k] is not None:
                    compare(f"{branch}: grad {k}", g0[k], g1[k], rows)

        # Fisher (fish_cal) ve HBDA birleştirmesi: 3 istemci, farklı ağırlıklar
        clients_o, clients_p, models_o, models_p = [], [], [], []
        for cid in range(3):
            seed_all(100 + cid)
            mo = OrigWrapper(cfg.get_spa(nfeat, 128, nclass, 2)).to(device)
            mp = port.Model(nfeat, nclass, device=device).to(device)
            mp.load_state_dict(mo.state_dict(), strict=True)
            lists_o = ([g.att_vec_low_unk for g in mo.model.gcns], [g.att_vec_high_unk for g in mo.model.gcns],
                       [g.att_vec_mlp for g in mo.model.gcns])
            lists_p = ([g.att_vec_low_unk for g in mp.model.gcns], [g.att_vec_high_unk for g in mp.model.gcns],
                       [g.att_vec_mlp for g in mp.model.gcns])
            seed_all(200 + cid)
            radio = fed_ns["calculate_shigh"](data)
            seed_all(200 + cid)
            radio_p = port.calculate_shigh(data)
            compare(f"calculate_shigh (istemci {cid})", float(radio), float(radio_p), rows)
            clients_o.append(types.SimpleNamespace(train_data=types.SimpleNamespace(data=data), adj_low_un=a_un, adj_low=a_low,
                                                   adj_high=a_high, fr_low_g_list=lists_o[0], fr_high_g_list=lists_o[1],
                                                   mlp_list=lists_o[2], radio_diff=radio, datavol=100 + cid))
            clients_p.append(types.SimpleNamespace(train_data=types.SimpleNamespace(data=data), adj_low_un=p_un, adj_low=p_low,
                                                   adj_high=p_high, fr_low_g_list=lists_p[0], fr_high_g_list=lists_p[1],
                                                   mlp_list=lists_p[2], radio_diff=radio_p, datavol=100 + cid))
            models_o.append(mo)
            models_p.append(mp)
        seed_all(300)
        fish_o = fed_ns["fish_cal"](models_o[0], data, a_un, a_low, a_high, clients_o[0].fr_high_g_list,
                                    clients_o[0].fr_low_g_list, clients_o[0].mlp_list)
        seed_all(300)
        fish_p = port.fish_cal(models_p[0], data, p_un, p_low, p_high, clients_p[0].fr_high_g_list,
                               clients_p[0].fr_low_g_list, clients_p[0].mlp_list)
        compare("fish_cal: anahtarlar", sorted(fish_o), sorted(fish_p), rows)
        for k in fish_o:
            compare(f"fish_cal: {k}", fish_o[k], fish_p[k], rows)

        seed_all(0)
        server_model_o = OrigWrapper(cfg.get_spa(nfeat, 128, nclass, 2)).to(device)
        server_o = types.SimpleNamespace(model=server_model_o, clients=clients_o, received_clients=[0, 1, 2],
                                         gv=types.SimpleNamespace(logger=types.SimpleNamespace(info=print)))
        server_p = port.FedSPAServer(port.Model(nfeat, nclass, device=device).to(device), clients_p)
        server_p.model.load_state_dict(server_model_o.state_dict(), strict=True)
        seed_all(400)
        agg_o = fed_ns["aggregate"](server_o, models_o)
        seed_all(400)
        agg_p = server_p.aggregate(models_p)
        for (n1, p1), (n2, p2) in zip(agg_o.named_parameters(), agg_p.named_parameters()):
            compare(f"HBDA birleştirme: {n1}", p1.detach(), p2.detach(), rows)

        n_exact = sum(r[1] for r in rows)
        max_diff = max(r[2] for r in rows)
        worst = max(worst, max_diff)
        max_in_deg = int(torch.bincount(data.edge_index[1], minlength=data.num_nodes).max())
        # PyTorch'un sum(1)'i (DGL'in posta kutusu toplamı) derece >= 18'de kademeli toplamaya geçer;
        # index_add_ sıralı toplar. Bu yüzden bit düzeyinde eşitlik yalnızca derece < 18 iken beklenir.
        ok = (n_exact == len(rows)) if max_in_deg < SEQ_SUM_LIMIT else (max_diff < 1e-4)
        all_ok &= ok
        print(f"[{gname}, maks. giriş derecesi {max_in_deg}] {n_exact}/{len(rows)} karşılaştırma birebir aynı; "
              f"en büyük göreli fark {max_diff:.2e} -> {'GEÇTİ' if ok else 'KALDI'}")
        for r in rows:
            if not r[1]:
                print(f"    birebir değil: {r[0]}  (göreli fark {r[2]:.2e})")
    print("SONUÇ:", "port, orijinal DGL'li kodla eşdeğer" if all_ok else "FARK BULUNDU", f"(en büyük göreli fark {worst:.2e})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
