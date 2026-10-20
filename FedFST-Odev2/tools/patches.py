"""FedFST (YunQi572/FedFST @ d83fac0) yamalarının tek kaynağı.

Notebook'taki yama hücreleri bu listeden üretilir (tools/build_notebook.py);
yerel testler de aynı listeyi fgcl_bench.patching.apply_patch ile uygular.
Her yeni metin bir "[Odev2-Px]" işareti taşır; işaret dosyada varsa yama
tekrar uygulanmaz (idempotent).
"""

PATCHES = []


def _add(pid, file, title, reason, old, new):
    PATCHES.append(dict(kind="replace", id=pid, file=file, title=title, reason=reason,
                        old=old.strip("\n"), new=new.strip("\n")))


# ---------------------------------------------------------------- (Colab)
PATCHES.append(dict(
    kind="create", id="P0",
    files=["datasets/__init__.py", "algorithm/__init__.py", "models/__init__.py"],
    content="# [Odev2-P0] Klasörü normal bir Python paketi yapar (bkz. notebook, yama P0).\n",
    title="datasets/, algorithm/, models/ klasörlerine boş __init__.py ekle",
    reason="FedFST bu klasörleri `from datasets.dataset_loader import ...` gibi import ediyor ama klasörlerde "
           "`__init__.py` yok (namespace package). Colab'da Hugging Face `datasets` paketi önceden kurulu; "
           "Python normal bir paketi namespace package'a tercih ettiği için `datasets` adı HF paketine gidiyor ve "
           "import `ModuleNotFoundError: No module named 'datasets.dataset_loader'` ile çöküyor (HF datasets "
           "kurulu bir ortamda yeniden ürettim). Boş `__init__.py` dosyaları kodun davranışını değiştirmez."))


# ---------------------------------------------------------------- (a)
_add("P1", "datasets/dataset_loader.py",
     "torch_sparse import'unu kaldır",
     "`from torch_sparse import SparseTensor` satırı ModuleNotFoundError veriyor. "
     "SparseTensor dosyada hiç kullanılmıyor; torch_sparse derlenmiş bir eklenti ve "
     "Colab'ın güncel torch sürümü için wheel'i yok.",
     old='''
from torch_sparse import SparseTensor
''',
     new='''
# [Odev2-P1] "from torch_sparse import SparseTensor" kaldırıldı: SparseTensor kullanılmıyor.
''')

# ---------------------------------------------------------------- (b)
_add("P2", "algorithm/Ours.py",
     "Var olmayan iki fonksiyonun import'unu kaldır",
     "Ours.py 12. satır utils'ten `edge_distribution_low1` ve `edge_distribution_low2` "
     "import ediyor; bu fonksiyonlar utils.py'de yok (ImportError) ve Ours.py'de hiç kullanılmıyor.",
     old='''
from utils import Generator, edge_distribution_low, get_khop_feature, edge_distribution_low1, edge_distribution_low2, get_Shigh, reduce_homophilic_edges, reduce_heterophilic_edges, get_Homophily_Level
''',
     new='''
# [Odev2-P2] edge_distribution_low1/2 import'tan çıkarıldı (utils.py'de yoklar, kullanılmıyorlar).
from utils import Generator, edge_distribution_low, get_khop_feature, get_Shigh, reduce_homophilic_edges, reduce_heterophilic_edges, get_Homophily_Level
''')

# ---------------------------------------------------------------- (c)
_add("P3a", "args.py",
     "Eksik iki argümanı ekle",
     "main.py 26–27. satırlar `args.kd_ce_weight` ve `args.gen_ce_weight` yazdırıyor ama "
     "args.py'de tanımlı değiller (AttributeError). Varsayılanlar Ours.py'deki getattr "
     "varsayılanlarıyla aynı (254. satır: 0.5, 799. satır: 1.0); davranış değişmez.",
     old='''
parser.add_argument('--gen_kl_weight', type=float, default=100)
''',
     new='''
parser.add_argument('--gen_kl_weight', type=float, default=100)
# [Odev2-P3a] main.py bu iki argümanı yazdırıyor ama tanımlı değillerdi. Varsayılanlar
# Ours.py'deki getattr varsayılanlarıyla aynı (satır 254 ve 799); davranış değişmez.
parser.add_argument('--kd_ce_weight', type=float, default=0.5)
parser.add_argument('--gen_ce_weight', type=float, default=1.0)
''')

_add("P3b", "main.py",
     "Tanımsız argümanı yazdıran print'i düzelt",
     "main.py 28. satır tanımsız `args.S_high_D_weight`'i yazdırıyor (AttributeError). "
     "Bu değer hiçbir yerde kullanılmıyor; print, gerçekte kullanılan `S_high_A_weight`'i yazdıracak şekilde düzeltildi.",
     old='''
    print(f"S_high_x_weight={args.S_high_x_weight}, S_high_D_weight={args.S_high_D_weight}")
''',
     new='''
    print(f"S_high_x_weight={args.S_high_x_weight}, S_high_A_weight={args.S_high_A_weight}")  # [Odev2-P3b]
''')

# ---------------------------------------------------------------- (f)
_add("P4a", "datasets/dataset_loader.py",
     "Minesweeper ve Squirrel'i kabul edilen veri setlerine ekle (assert)",
     "`load_dataset` yalnızca belirli isimleri kabul ediyor; minesweeper ve squirrel listede yok.",
     old='''
    assert dataset_name in ('cora', 'citeseer', 'pubmed', 'ogbn-arxiv',
                            'computers', 'physics', 'roman_empire', 'year', 'actor', 'cs'), 'Invalid dataset'
''',
     new='''
    assert dataset_name in ('cora', 'citeseer', 'pubmed', 'ogbn-arxiv',
                            'computers', 'physics', 'roman_empire', 'year', 'actor', 'cs',
                            'minesweeper', 'squirrel'), 'Invalid dataset'  # [Odev2-P4a]
''')

_add("P4b", "datasets/dataset_loader.py",
     "Minesweeper ve Squirrel yükleme dallarını ekle",
     "İki veri seti PyG ile yükleniyor: `HeterophilousGraphDataset(root, name='Minesweeper')` ve "
     "`WikipediaNetwork(root, name='squirrel')` (geom-gcn ön işlemeli sürüm, PyG varsayılanı). "
     "İki sınıf da dosyada zaten import edilmiş.",
     old='''
    elif dataset_name in ['actor']:
        dataset = Actor(root=root_dir)
        data = dataset[0]
''',
     new='''
    elif dataset_name in ['actor']:
        dataset = Actor(root=root_dir)
        data = dataset[0]

    # [Odev2-P4b] Minesweeper ve Squirrel (iki sınıf da bu dosyada zaten import edilmiş).
    elif dataset_name in ['minesweeper']:
        dataset = HeterophilousGraphDataset(root=root_dir, name='Minesweeper')
        data = dataset[0]

    elif dataset_name in ['squirrel']:
        dataset = WikipediaNetwork(root=root_dir, name='squirrel')
        data = dataset[0]
''')

# ---------------------------------------------------------------- (i) vektörizasyon
_add("P5a", "utils.py",
     "reduce_homophilic_edges / reduce_heterophilic_edges: kenar döngüsünü vektörize et",
     "İki fonksiyon kenar kenar `.item()` ile dönüyor; jeneratör eğitiminde her epoch'ta 60 kez çağrılıyor. "
     "Yeni sürüm aynı kenarları aynı sırada seçiyor ve `torch.randperm`'i aynı generator'dan (varsayılan CPU "
     "generator'ı), aynı n ile, aynı sırada çağırıyor; çıktı ve RNG durumu birebir aynı (eşdeğerlik testi aşağıda).",
     old='''
def reduce_homophilic_edges(data, edge_index, reduction_ratio=0.1):
    labels = data.y

    homophilic_edges = []
    heterophilic_edges = []

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if labels[src] == labels[dst]:
            homophilic_edges.append(i)
        else:
            heterophilic_edges.append(i)

    num_to_remove = int(len(homophilic_edges) * reduction_ratio)
    if num_to_remove > 0:
        remove_indices = torch.randperm(len(homophilic_edges))[:num_to_remove]
        edges_to_remove = [homophilic_edges[i] for i in remove_indices]

        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        keep_mask[edges_to_remove] = False
        new_edge_index = edge_index[:, keep_mask]
    else:
        new_edge_index = edge_index

    return new_edge_index

def reduce_heterophilic_edges(data, edge_index, reduction_ratio=0.1):
    labels = data.y

    heterophilic_edges = []

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if labels[src] != labels[dst]:
            heterophilic_edges.append(i)

    num_to_remove = int(len(heterophilic_edges) * reduction_ratio)
    if num_to_remove > 0:
        remove_indices = torch.randperm(len(heterophilic_edges))[:num_to_remove]
        edges_to_remove = [heterophilic_edges[i] for i in remove_indices]

        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        keep_mask[edges_to_remove] = False
        new_edge_index = edge_index[:, keep_mask]
    else:
        new_edge_index = edge_index

    return new_edge_index
''',
     new='''
# [Odev2-P5a] Kenar döngüsü vektörize edildi. Seçilen kenarlar ve sıraları aynı; torch.randperm
# aynı generator'dan (varsayılan CPU generator'ı), aynı n ile çağrılıyor -> çıktı ve RNG durumu aynı.
def _reduce_edges_by_label_relation(data, edge_index, reduction_ratio, homophilic):
    labels = data.y
    ei = edge_index.to(labels.device)
    same = labels[ei[0]] == labels[ei[1]]
    selected = same if homophilic else ~same
    candidate_edges = torch.nonzero(selected, as_tuple=False).view(-1).cpu()

    num_to_remove = int(candidate_edges.numel() * reduction_ratio)
    if num_to_remove > 0:
        remove_indices = torch.randperm(candidate_edges.numel())[:num_to_remove]
        edges_to_remove = candidate_edges[remove_indices]

        keep_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        keep_mask[edges_to_remove] = False
        new_edge_index = edge_index[:, keep_mask]
    else:
        new_edge_index = edge_index

    return new_edge_index

def reduce_homophilic_edges(data, edge_index, reduction_ratio=0.1):
    return _reduce_edges_by_label_relation(data, edge_index, reduction_ratio, homophilic=True)

def reduce_heterophilic_edges(data, edge_index, reduction_ratio=0.1):
    return _reduce_edges_by_label_relation(data, edge_index, reduction_ratio, homophilic=False)
''')

_add("P5b", "algorithm/Ours.py",
     "get_learned_Homophily_Level: kenar döngüsünü vektörize et",
     "Önceki görevlerin eğitim düğümleri arasındaki kenarlarda homofili oranını kenar kenar `.item()` ile sayıyor. "
     "Yeni sürüm aynı iki sayacı (toplam kenar, aynı etiketli kenar) tensör işlemleriyle sayıp aynı bölmeyi yapıyor. "
     "RNG kullanılmıyor.",
     old='''
    def get_learned_Homophily_Level(self, task_id):
        total_edge_nums = 0
        same_label_edges = 0

        for task_i in range(task_id):
            task = self.tasks[task_i]
            graph = task["local_data"]
            train_mask = task["train_mask"]

            edge_index = graph.edge_index
            node_labels = graph.y

            for edge_idx in range(edge_index.size(1)):
                src_node = edge_index[0, edge_idx].item()
                dst_node = edge_index[1, edge_idx].item()

                if train_mask[src_node] and train_mask[dst_node]:
                    total_edge_nums += 1

                    if node_labels[src_node] == node_labels[dst_node]:
                        same_label_edges += 1

        if total_edge_nums > 0:
            homophily_ratio = same_label_edges / total_edge_nums
        else:
            homophily_ratio = 0.0

        return homophily_ratio
''',
     new='''
    def get_learned_Homophily_Level(self, task_id):
        # [Odev2-P5b] Kenar döngüsü vektörize edildi; iki sayaç ve sonuç aynı, RNG kullanılmıyor.
        total_edge_nums = 0
        same_label_edges = 0

        for task_i in range(task_id):
            task = self.tasks[task_i]
            graph = task["local_data"]
            train_mask = task["train_mask"]

            edge_index = graph.edge_index
            node_labels = graph.y

            src_nodes, dst_nodes = edge_index[0], edge_index[1]
            train_mask = train_mask.to(edge_index.device)
            both_train = train_mask[src_nodes] & train_mask[dst_nodes]
            same_label = node_labels[src_nodes] == node_labels[dst_nodes]
            total_edge_nums += int(both_train.sum().item())
            same_label_edges += int((both_train & same_label).sum().item())

        if total_edge_nums > 0:
            homophily_ratio = same_label_edges / total_edge_nums
        else:
            homophily_ratio = 0.0

        return homophily_ratio
''')

_add("P5c", "algorithm/Ours.py",
     "feature_gen_train: düğüm düğüm KL döngüsü için vektörize yardımcı fonksiyonu ekle",
     "Yardımcı fonksiyon, P5d'de döngünün yerine çağrılıyor. Örnekler aynı sırada işleniyor; aynı etiketli "
     "ardışık örnekler için tek bir `torch.randint(0, n_c, (k,))` çağrısı, varsayılan CPU generator'ından "
     "k ayrı `torch.randint(0, n_c, (1,))` çağrısıyla aynı sayıları çekiyor (testte doğrulandı).",
     old='''
class OursServer(BaseServer):
''',
     new='''
def _gen_kl_loss_vectorized(all_generated_features, all_generated_labels, subgraph, device):
    # [Odev2-P5c] OursClient.feature_gen_train'deki örnek örnek KL döngüsünün vektörize hali.
    # Örnekler aynı sırada işlenir. Aynı etiketli ardışık örnekler için tek bir torch.randint
    # çağrısı, varsayılan CPU generator'ından tek tek çağrılarla aynı sayıları aynı sırada çeker.
    labels_cpu = all_generated_labels.detach().cpu()
    num_generated = labels_cpu.numel()
    rows_list, real_list = [], []
    if num_generated > 0:
        change = torch.nonzero(labels_cpu[1:] != labels_cpu[:-1], as_tuple=False).view(-1) + 1
        bounds = [0] + change.tolist() + [num_generated]
        for start, end in zip(bounds[:-1], bounds[1:]):
            gen_label = all_generated_labels[start]
            same_class_mask = (subgraph.y == gen_label)
            if same_class_mask.sum() > 0:
                same_class_indices = torch.where(same_class_mask)[0]
                random_idx = torch.randint(0, len(same_class_indices), (end - start,))
                real_list.append(same_class_indices[random_idx.to(same_class_indices.device)])
                rows_list.append(torch.arange(start, end, device=all_generated_features.device))
    if rows_list:
        rows = torch.cat(rows_list)
        real_idx = torch.cat(real_list)
        kl_per_sample = F.kl_div(
            F.log_softmax(all_generated_features[rows], dim = 1),
            F.softmax(subgraph.x[real_idx], dim = 1),
            reduction = 'none'
        ).sum(dim = 1)
        return kl_per_sample.mean()
    return torch.tensor(0.0, device=device)


class OursServer(BaseServer):
''')

_add("P5d", "algorithm/Ours.py",
     "feature_gen_train: KL döngüsünü vektörize yardımcıyla değiştir",
     "Her epoch'ta üretilen her düğüm için (Cora'da 400–800 düğüm) ayrı ayrı rastgele gerçek düğüm seçip KL "
     "hesaplayan döngü, P5c'deki fonksiyona çağrıyla değiştirildi. Kayıp formülü aynı: "
     "KL(softmax(gerçek) || softmax(üretilen)) özniteliklerde toplanıp örneklerde ortalanıyor.",
     old='''
            kl_losses = []
            for i, gen_label in enumerate(all_generated_labels):
                same_class_mask = (subgraph.y == gen_label)
                if same_class_mask.sum() > 0:
                    same_class_indices = torch.where(same_class_mask)[0]
                    random_idx = torch.randint(0, len(same_class_indices), (1,)).item()
                    real_feature = subgraph.x[same_class_indices[random_idx]]

                    kl_loss = F.kl_div(
                        F.log_softmax(all_generated_features[i], dim = 0),
                        F.softmax(real_feature, dim = 0),
                        reduction = 'sum'
                    )

                    kl_losses.append(kl_loss)

            if kl_losses:
                loss_kl = torch.stack(kl_losses).mean()
            else:
                loss_kl = torch.tensor(0.0, device=self.device)
''',
     new='''
            # [Odev2-P5d] Örnek örnek KL döngüsü yerine eşdeğer vektörize hesap (bkz. _gen_kl_loss_vectorized).
            loss_kl = _gen_kl_loss_vectorized(all_generated_features, all_generated_labels, subgraph, self.device)
''')

_add("P5e", "datasets/partition.py",
     "get_subgraph_by_node: kenar döngüsünü vektörize et",
     "Görev alt-grafikleri (class_to_task), öğrenilmiş alt-graf (get_learned_graph) ve her round'daki "
     "get_subgraph bu fonksiyonu çağırıyor; kenar kenar `.item()` kullanıyor. Yeni sürüm aynı kenarları aynı "
     "sırada, aynı dtype ve bellek düzeniyle (transpoze görünüm) döndürüyor. Düğüm listesinde tekrar varsa "
     "orijinal döngü aynen çalışıyor. RNG kullanılmıyor.",
     old='''
def get_subgraph_by_node(dataset, node_list, flag):
    node_id_set = set(node_list)
''',
     new='''
def get_subgraph_by_node(dataset, node_list, flag):
    # [Odev2-P5e] Vektörize sürüm: aynı kenarlar, aynı sıra, aynı dtype ve bellek düzeni.
    node_ids = [int(v) for v in node_list]
    if len(set(node_ids)) != len(node_ids):
        return _get_subgraph_by_node_loop(dataset, node_list, flag)
    local_id_to_global_id = list(node_list)
    edge_index = dataset.edge_index.cpu()
    num_slots = int(dataset.x.shape[0])
    if node_ids:
        num_slots = max(num_slots, max(node_ids) + 1)
    if edge_index.numel() > 0:
        num_slots = max(num_slots, int(edge_index.max()) + 1)
    global_id_to_local_id = torch.full((num_slots,), -1, dtype=torch.long)
    if node_ids:
        global_id_to_local_id[torch.tensor(node_ids, dtype=torch.long)] = torch.arange(len(node_ids), dtype=torch.long)
    keep = (global_id_to_local_id[edge_index[0]] >= 0) & (global_id_to_local_id[edge_index[1]] >= 0) if edge_index.numel() > 0 else torch.zeros(0, dtype=torch.bool)
    if bool(keep.any()):
        # Orijinal kod torch.tensor(list_of_pairs).t() döndürüyordu: (2, E) şeklinde transpoze görünüm.
        global_edge_list = edge_index[:, keep].t().contiguous().t()
        local_edge_index = global_id_to_local_id[global_edge_list.t()].contiguous().t()
    else:
        local_edge_index = torch.empty((2, 0), dtype=torch.int64)
        global_edge_list = torch.tensor([]).t()
    if flag:
        local_subgraph = Data(x=dataset.x[node_list], edge_index=local_edge_index, y=dataset.y[node_list])
    else:
        local_subgraph = Data(x=dataset.x, edge_index=global_edge_list, y=dataset.y)
    local_subgraph.global_map = local_id_to_global_id

    return local_subgraph


def _get_subgraph_by_node_loop(dataset, node_list, flag):
    # Orijinal (döngülü) sürüm; yalnızca düğüm listesinde tekrar varsa kullanılır.
    node_id_set = set(node_list)
''')

_add("P5f", "datasets/partition.py",
     "louvain_partitioner: istemci alt-grafiği çıkarımını vektörize et",
     "Louvain bölümlemesi (python-louvain, numpy RNG) aynen korunuyor. Yalnızca her istemcinin kenarlarını "
     "kenar kenar `.item()` ile seçen döngü vektörize edildi; aynı kenarlar aynı sırada ve aynı yeni "
     "numaralarla seçiliyor. Düğüm listesinde tekrar varsa orijinal döngü çalışıyor.",
     old='''
    clients_data = []
    for nodes in clients_nodes:
        node_map = {node: i for i, node in enumerate(nodes)}
        sub_edge_index = []
        for i in range(data.edge_index.size(1)):
            if data.edge_index[0, i].item() in node_map and data.edge_index[1, i].item() in node_map:
                sub_edge_index.append([node_map[data.edge_index[0, i].item()], node_map[data.edge_index[1, i].item()]])
        sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long).t().contiguous()
''',
     new='''
    clients_data = []
    edge_src_all = data.edge_index[0]
    edge_dst_all = data.edge_index[1]
    for nodes in clients_nodes:
        # [Odev2-P5f] Kenar seçimi vektörize edildi: aynı kenarlar, aynı sıra, aynı yeni numaralar.
        node_ids = [int(v) for v in nodes]
        if len(set(node_ids)) == len(node_ids):
            node_map_t = torch.full((data.num_nodes,), -1, dtype=torch.long)
            node_map_t[torch.tensor(node_ids, dtype=torch.long)] = torch.arange(len(node_ids), dtype=torch.long)
            keep = (node_map_t[edge_src_all] >= 0) & (node_map_t[edge_dst_all] >= 0)
            if bool(keep.any()):
                sub_edge_index = torch.stack([node_map_t[edge_src_all[keep]], node_map_t[edge_dst_all[keep]]], dim=0).contiguous()
            else:
                sub_edge_index = torch.tensor([], dtype=torch.long).t().contiguous()
        else:
            node_map = {node: i for i, node in enumerate(nodes)}
            sub_edge_index = []
            for i in range(data.edge_index.size(1)):
                if data.edge_index[0, i].item() in node_map and data.edge_index[1, i].item() in node_map:
                    sub_edge_index.append([node_map[data.edge_index[0, i].item()], node_map[data.edge_index[1, i].item()]])
            sub_edge_index = torch.tensor(sub_edge_index, dtype=torch.long).t().contiguous()
''')
