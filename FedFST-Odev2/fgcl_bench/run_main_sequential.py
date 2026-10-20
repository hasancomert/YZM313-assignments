"""Sağlama: FedFST'nin (yamalı) orijinal main.py'sini, jeneratör eğitimini yapan
ThreadPoolExecutor tek iş parçacığına (max_workers=1) indirilmiş olarak çalıştırır.
İstemciler 0, 1, 2 sırasıyla, harness'teki sırayla aynı biçimde çalışır; main.py'nin
kodu değiştirilmez.

    python -m fgcl_bench.run_main_sequential <FedFST klasörü> --dataset_name cora --seed 24 --device_id 0 ...
"""
import concurrent.futures
import os
import runpy
import sys


class _SequentialExecutor(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(max_workers=1)


def main():
    fedfst_dir = os.path.abspath(sys.argv[1])
    concurrent.futures.ThreadPoolExecutor = _SequentialExecutor
    sys.argv = ["main.py"] + sys.argv[2:]
    sys.path.insert(0, fedfst_dir)
    os.chdir(fedfst_dir)
    runpy.run_path(os.path.join(fedfst_dir, "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
