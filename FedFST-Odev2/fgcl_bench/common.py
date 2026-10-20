"""Harness'in ortak parçaları: veri seti tanımları, yollar, seed, hash, JSON."""
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Veri seti -> FedFST'deki adı ve görev başına sınıf sayısı (Minesweeper: 1, bkz. notebook).
DATASETS = {
    "cora":        dict(title="Cora",        fedfst_name="cora",        per_task=2),
    "citeseer":    dict(title="CiteSeer",    fedfst_name="citeseer",    per_task=2),
    "actor":       dict(title="Actor",       fedfst_name="actor",       per_task=2),
    "squirrel":    dict(title="Squirrel",    fedfst_name="squirrel",    per_task=2),
    "minesweeper": dict(title="Minesweeper", fedfst_name="minesweeper", per_task=1),
}
METHODS = {"fedfst": "FedFST", "power": "POWER", "fedspa": "FedSPA"}
NUM_CLIENTS = 3


def run_name(dataset, method, seed):
    return f"{dataset}__{method}__seed{seed}"


def set_seed(seed):
    """FedFST'nin utils.set_seed'i ile aynı (POWER'ın seed_everything'i de aynı içerikte)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'


def pick_device(prefer="auto"):
    if prefer == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda:0")


def device_label(device):
    if device.type == "cuda":
        return f"cuda:0 ({torch.cuda.get_device_name(0)})"
    return "cpu"


def load_module_from_file(alias, path):
    """Bir .py dosyasını farklı bir modül adıyla yükler (ör. FedFST utils.py -> fedfst_utils),
    böylece POWER'ın kendi `utils` modülüyle çakışmaz."""
    spec = importlib.util.spec_from_file_location(alias, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _tensor_bytes(t):
    t = t.detach().cpu().contiguous()
    return str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.numpy().tobytes()


def task_view_hash(views):
    """views[c][t] = dict(train, valid, test, nodes, edges, y): düğümler istemci-içi numaralarla,
    train/valid/test/nodes sıralı LongTensor, edges [2, E] (sıra korunur), y = nodes'un etiketleri.
    Üç yöntem de beslediği veriden bu yapıyı kurup hash'i cache'teki ile karşılaştırır."""
    h = hashlib.sha256()
    for c, tasks in enumerate(views):
        for t, v in enumerate(tasks):
            h.update(f"client{c}/task{t}".encode())
            for key in ("train", "valid", "test", "nodes", "edges", "y"):
                h.update(key.encode())
                h.update(_tensor_bytes(v[key]))
    return h.hexdigest()


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def env_info(device):
    import torch_geometric
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "numpy": np.__version__,
        "device": device_label(device),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
