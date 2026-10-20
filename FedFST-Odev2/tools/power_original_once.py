"""Sağlama: POWER'ı kendi orijinal betik yoluyla Cora'da bir kez çalıştırır (yerel test).

optuna_search.py'nin __main__ bloğu ve objective() fonksiyonu, optuna olmadan ve config.py
varsayılanlarıyla (tek deneme) çalıştırılır: seed_everything -> load_fcgl_dataset (OpenFGL
FGLDataset, Louvain, 3 istemci, 20/40/40 bölme) -> load_clients_server -> start.
optuna_search.py'deki gibi CPU kullanılır (args.disable_cuda = True).

    POWER_ROOT=<veri kökü> OPENFGL=<OpenFGL klasörü> python power_original_once.py <POWER klasörü> [--global_eval]
"""
import os
import sys

power_dir = sys.argv[1]
global_eval = "--global_eval" in sys.argv
sys.argv = [sys.argv[0]]              # config.py ve openfgl.config argv'yi okuyor -> varsayılanlar
sys.path.insert(0, power_dir)
sys.path.insert(1, os.environ["OPENFGL"])
os.chdir(power_dir)

from config import args                # noqa: E402  (POWER)
import torch                           # noqa: E402
from data import load_fcgl_dataset     # noqa: E402  (POWER, OpenFGL'i kullanır)
from utils import load_clients_server, start, seed_everything  # noqa: E402  (POWER)

args.root = os.environ["POWER_ROOT"]
args.disable_cuda = True
args.global_eval = global_eval
seed_everything(args.seed)
device = torch.device("cpu")
fcgl_dataset, input_dim, output_dim, task_dir = load_fcgl_dataset(
    root=args.root, dataset=args.dataset, num_clients=args.num_clients,
    classes_per_task=args.num_classes_per_task, shuffle_task=args.shuffle_task)
args.input_dim = input_dim
args.output_dim = output_dim
args.task_dir = task_dir
clients, server, message_pool = load_clients_server(args, fcgl_dataset, device)
aa, af = start(args, fcgl_dataset, clients, server, message_pool, device)
print(f"POWER orijinal betik | {args.dataset} | global_eval={args.global_eval} | AA={float(aa):.2f} AF={float(af):.2f}")
