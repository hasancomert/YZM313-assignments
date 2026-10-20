"""Ortak değerlendirme: her görevden sonra global modelin o ana kadarki tüm görevlerdeki test
doğruluğu -> doğruluk matrisi -> AA ve AF.

Aritmetik, FedFST main.py'deki değerlendirme bloğunun aynısıdır:
    ACC[t, e] += client_acc * n_test   (client_acc = doğru / n_test * 100, Python float)
    ACC[t, e] /= toplam n_test
AA ve AF, FedFST'nin utils.AA / utils.AF fonksiyonlarıyla hesaplanır (dosya `fedfst_utils`
adıyla yüklenir; POWER'ın kendi `utils` modülüyle çakışmasın diye).
"""
from pathlib import Path

import torch

from .common import load_module_from_file


class Evaluator:
    def __init__(self, cache, fedfst_dir):
        self.fedfst_utils = load_module_from_file("fedfst_utils", Path(fedfst_dir) / "utils.py")
        self.num_tasks = cache["meta"]["num_tasks"]
        self.acc = torch.zeros(size=(self.num_tasks, self.num_tasks))
        # test düğümleri (istemci-içi numara, sıralı) ve etiketleri
        self.test_idx = [[t["test_mask"].nonzero().view(-1) for t in cl["tasks"]] for cl in cache["clients"]]
        self.test_y = [[cl["y"][idx] for idx in tests] for cl, tests in zip(cache["clients"], self.test_idx)]
        self.test_count = [[t["test_mask"].sum() for t in cl["tasks"]] for cl in cache["clients"]]

    def evaluate_after_task(self, task_id, predict_fn):
        """predict_fn(client_id, eval_task_id) -> tahmin edilen etiketler; sırası self.test_idx[c][e] ile aynı."""
        for eval_task_id in range(task_id + 1):
            total_nodes_num = 0
            for client_id in range(len(self.test_idx)):
                nodes_num = self.test_count[client_id][eval_task_id]
                if int(nodes_num) == 0:
                    continue
                pred = predict_fn(client_id, eval_task_id).detach().cpu()
                correct = pred.eq(self.test_y[client_id][eval_task_id]).sum().item()
                client_acc = (correct / nodes_num.item()) * 100
                self.acc[task_id, eval_task_id] += client_acc * nodes_num
                total_nodes_num += nodes_num
            self.acc[task_id, eval_task_id] /= total_nodes_num
            print(f"Task {task_id} trained, Task {eval_task_id} global acc: {self.acc[task_id, eval_task_id]:.2f}")
        aa = self.fedfst_utils.AA(self.acc, T=task_id + 1)
        af = self.fedfst_utils.AF(self.acc, T=task_id + 1)
        aa, af = float(aa), float(af)
        print(f"Task {task_id} - AA: {aa:.4f}, AF: {af:.4f}")
        return aa, af

    def final(self):
        aa = float(self.fedfst_utils.AA(self.acc, T=self.num_tasks))
        af = float(self.fedfst_utils.AF(self.acc, T=self.num_tasks))
        return aa, af

    def matrix(self):
        return [[round(float(v), 6) for v in row] for row in self.acc]
