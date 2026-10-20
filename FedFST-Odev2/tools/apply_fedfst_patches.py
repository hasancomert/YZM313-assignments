"""Yerel test için: tools/patches.py'deki tüm yamaları bir FedFST klasörüne uygular.

    python tools/apply_fedfst_patches.py <FedFST klasörü> [--only P0,P1,...]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from fgcl_bench.patching import apply_patch, create_files  # noqa: E402
from patches import PATCHES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    for p in PATCHES:
        if only and p["id"] not in only:
            continue
        if p["kind"] == "create":
            create_files(args.root, p["id"], p["files"], p["content"])
        else:
            apply_patch(args.root, p["file"], p["id"], p["old"], p["new"], show_diff=False)


if __name__ == "__main__":
    main()
