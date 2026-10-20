"""Sağlama (yerel): POWER'ın orijinal betik yolu, ama görev sırası tüm istemcilerde aynı.

power_original_once.py ile aynı; tek fark, data.py'deki class_incremental fonksiyonunun sonundaki
`np.random.shuffle(tasks)` satırının (her istemcinin görev sırasını ayrı ayrı karıştırır) bu koşu
için devre dışı bırakılması. Böylece ortak protokoldeki gibi tüm istemciler [0,1] => [2,3] => [4,5]
sırasını izler. Kaynak dosyalar değiştirilmez; fonksiyon bellekte yeniden tanımlanır.

    POWER_ROOT=<veri kökü> OPENFGL=<OpenFGL> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        python power_original_sync_order.py <POWER klasörü>
"""
import inspect
import os
import sys

power_dir = sys.argv[1]
sys.argv = [sys.argv[0]]
sys.path.insert(0, power_dir)
sys.path.insert(1, os.environ["OPENFGL"])
os.chdir(power_dir)

from config import args            # noqa: E402
import torch                       # noqa: E402
import data as pdata               # noqa: E402

src = inspect.getsource(pdata.class_incremental)
src = src.replace("    np.random.shuffle(tasks)\n", "    pass  # [yerel test] görev sırası karıştırılmadı\n")
assert "[yerel test]" in src
exec(src, pdata.__dict__)

from utils import load_clients_server, start, seed_everything  # noqa: E402

args.root = os.environ["POWER_ROOT"]
args.disable_cuda = True
seed_everything(args.seed)
fcgl_dataset, input_dim, output_dim, task_dir = pdata.load_fcgl_dataset(
    root=args.root, dataset=args.dataset, num_clients=args.num_clients,
    classes_per_task=args.num_classes_per_task, shuffle_task=args.shuffle_task)
args.input_dim, args.output_dim, args.task_dir = input_dim, output_dim, task_dir
clients, server, message_pool = load_clients_server(args, fcgl_dataset, torch.device("cpu"))
aa, af = start(args, fcgl_dataset, clients, server, message_pool, torch.device("cpu"))
print(f"POWER orijinal betik, görev sırası eşit | {args.dataset} | AA={float(aa):.2f} AF={float(af):.2f}")
