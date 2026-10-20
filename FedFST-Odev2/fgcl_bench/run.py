"""Tek koşu (bir yöntem, bir veri seti, bir seed) -> results/<veri>__<yöntem>__seed<s>.json

    python -m fgcl_bench.run --method {fedfst,power,fedspa} --dataset cora --seed 24 \
        --src_dir <kaynak kodlar> --work_dir <Drive klasörü> [--rounds 10 --local_epochs 3] [--smoke]

--smoke: 1 round, 1 yerel epoch; FedFST'de jeneratör/KD epoch'ları 2/3, POWER'da L-BFGS 3 iterasyon.
Sonuçlar results_smoke/ altına yazılır (gerçek sonuçlarla karışmaz).
"""
import argparse
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import torch

from .common import DATASETS, METHODS, run_name, write_json_atomic, env_info, pick_device
from .data import cache_path, load_cache
from .evaluate import Evaluator


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--seed", type=int, default=24)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--local_epochs", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--work_dir", required=True)
    args = ap.parse_args(argv)
    if args.rounds > 15:
        raise SystemExit("Görev başına en fazla 15 communication round (bütçe kuralı).")
    if args.smoke:
        args.rounds, args.local_epochs = 1, 1

    src, work = Path(args.src_dir).resolve(), Path(args.work_dir).resolve()
    tmp_dir = Path(tempfile.mkdtemp(prefix="fgcl_"))
    cfg = types.SimpleNamespace(method=args.method, dataset=args.dataset, seed=args.seed, rounds=args.rounds,
                                local_epochs=args.local_epochs, smoke=args.smoke, src_dir=src, work_dir=work,
                                fedfst_dir=src / "FedFST", power_dir=src / "POWER", fedspa_dir=src / "FedSPA",
                                data_root=work / "data", tmp_dir=tmp_dir)
    device = pick_device()
    print(f"=== {METHODS[args.method]} | {DATASETS[args.dataset]['title']} | seed {args.seed} | "
          f"rounds/görev {args.rounds} | yerel epoch {args.local_epochs} | smoke={args.smoke} ===")
    print("ortam:", env_info(device))

    if not cache_path(work, args.dataset, args.seed).exists():
        # Cache ayrı bir süreçte üretilir: FedFST'nin `utils` modülü POWER'ınkiyle çakışmasın.
        subprocess.run([sys.executable, "-m", "fgcl_bench.data", "--fedfst", str(cfg.fedfst_dir),
                        "--work_dir", str(work), "--dataset", args.dataset, "--seed", str(args.seed)], check=True)
    cache = load_cache(work, args.dataset, args.seed)
    evaluator = Evaluator(cache, cfg.fedfst_dir)

    t0 = time.time()
    if args.method == "fedfst":
        from .runner_fedfst import run
    elif args.method == "power":
        from .runner_power import run
    else:
        from .runner_fedspa import run
    extra = run(cfg, cache, evaluator)
    elapsed = time.time() - t0

    assert extra["mask_hash"] == cache["hash"]
    aa, af = evaluator.final()
    meta = cache["meta"]
    result = dict(method=args.method, method_title=METHODS[args.method], dataset=args.dataset,
                  dataset_title=DATASETS[args.dataset]["title"], seed=args.seed, smoke=args.smoke,
                  rounds_per_task=args.rounds, local_epochs=args.local_epochs, num_clients=meta["num_clients"],
                  per_task_class_num=meta["per_task"], num_tasks=meta["num_tasks"],
                  task_classes=meta["task_classes"], dropped_classes=meta["dropped_classes"],
                  acc_matrix=evaluator.matrix(), AA=round(aa, 4), AF=round(af, 4),
                  time_sec=round(elapsed, 1), env=env_info(device), cache_hash=cache["hash"], **extra)
    out_dir = work / ("results_smoke" if args.smoke else "results")
    out = out_dir / f"{run_name(args.dataset, args.method, args.seed)}.json"
    write_json_atomic(out, result)
    print(f"SONUÇ | {METHODS[args.method]} | {DATASETS[args.dataset]['title']} | AA={aa:.2f} | AF={af:.2f} | "
          f"süre={elapsed:.0f} sn | {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
