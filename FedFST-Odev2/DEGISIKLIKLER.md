# Değişiklikler ve protokol sapmaları

Bu dosya, üç yöntemin resmi koduna göre yapılan tüm değişiklikleri ve ödevdeki ortak protokolden sapmaları listeler. Kural şuydu: hiçbir yöntemin algoritması "iyileştirilmedi" ve hiçbir yönteme unutma önleyici mekanizma eklenmedi. Yalnızca kodu çalıştırmak ve ortak veri/değerlendirme arayüzüne bağlamak için gereken en küçük değişiklikler yapıldı.

## 1. FedFST (`YunQi572/FedFST` @ `d83fac0`) – repoya uygulanan yamalar

Yamaların metni `tools/patches.py`'de; notebook'ta her biri ayrı bir hücre. Hepsi idempotenttir (yeni metin `[Odev2-Px]` işareti taşır).

| Yama | Dosya | Ne değişti | Neden |
|---|---|---|---|
| P0 | `datasets/`, `algorithm/`, `models/` | boş `__init__.py` eklendi | Klasörler namespace package. Colab'da önceden kurulu Hugging Face `datasets` paketi FedFST'nin `datasets` klasörünü gölgeliyor ve `from datasets.dataset_loader import ...` `ModuleNotFoundError` veriyor (HF `datasets` kurulu bir ortamda yeniden ürettim). |
| P1 | `datasets/dataset_loader.py` | `from torch_sparse import SparseTensor` kaldırıldı | ModuleNotFoundError; SparseTensor kullanılmıyor, torch_sparse derlenmiş bir eklenti. |
| P2 | `algorithm/Ours.py` | import'tan `edge_distribution_low1/2` çıkarıldı | ImportError; fonksiyonlar `utils.py`'de yok ve kullanılmıyor. |
| P3a | `args.py` | `--kd_ce_weight` (0.5) ve `--gen_ce_weight` (1.0) eklendi | `main.py` bunları yazdırıyor (AttributeError). Varsayılanlar Ours.py'deki getattr varsayılanlarıyla aynı (satır 254, 799); davranış değişmiyor. |
| P3b | `main.py` | `S_high_D_weight` print'i `S_high_A_weight` olarak düzeltildi | Tanımsız argüman (AttributeError); değer hiçbir yerde kullanılmıyor. |
| P4a, P4b | `datasets/dataset_loader.py` | `minesweeper` ve `squirrel` eklendi: `HeterophilousGraphDataset(root, name='Minesweeper')`, `WikipediaNetwork(root, name='squirrel')` | Yükleyici bu iki veri setini kabul etmiyordu. |
| P5a | `utils.py` | `reduce_homophilic_edges` / `reduce_heterophilic_edges` vektörize | Hız. Aynı kenarlar aynı sırada seçiliyor; `torch.randperm` aynı generator'dan, aynı n ile çağrılıyor. |
| P5b | `algorithm/Ours.py` | `get_learned_Homophily_Level` vektörize | Hız; RNG yok. |
| P5c, P5d | `algorithm/Ours.py` | `feature_gen_train`'deki örnek örnek KL döngüsü, vektörize bir yardımcı fonksiyonla değiştirildi | Hız. Aynı etiketli ardışık örnekler için tek `torch.randint(0, n, (k,))` çağrısı, k ayrı `(1,)` çağrısıyla aynı sayıları çekiyor (testte doğrulandı). |
| P5e | `datasets/partition.py` | `get_subgraph_by_node` vektörize (düğüm listesinde tekrar varsa orijinal döngü çalışıyor) | Hız; aynı kenarlar, aynı sıra, aynı dtype ve bellek düzeni (transpoze görünüm). |
| P5f | `datasets/partition.py` | `louvain_partitioner`'ın istemci alt-çizgisi çıkarımı vektörize | Hız; Louvain'in kendisi (python-louvain, numpy RNG) değişmedi. |

**Eşdeğerlik:** `fgcl_bench/equivalence.py` orijinal fonksiyonları git geçmişinden okuyup yamalı sürümlerle aynı seed'le karşılaştırır. Çıktı, gradyan ve RNG durumu CPU'da birebir aynı (Bölüm 6.1). Ayrıca harness'teki FedFST, yamalı orijinal `main.py` ile Cora'da bit düzeyinde aynı AA/AF'yi verdi (Bölüm 6.3).

**Komut satırı:** `--device_id 0` veriliyor (varsayılan 2, Colab'da `cuda:2` yok). `type=bool` tanımlı `--use_gpu` ve `--shuffle_flag` komut satırından verilmiyor. CUDA yoksa `main.py`'nin mantığıyla CPU seçiliyor.

## 2. FedFST – koşucu (`fgcl_bench/runner_fedfst.py`) ile `main.py` arasındaki farklar

| Fark | Neden |
|---|---|
| İstemcilerin jeneratör eğitimi `ThreadPoolExecutor` yerine aynı sırayla (istemci 0, 1, 2) ardışık çalışıyor | Üç iş parçacığı aynı küresel RNG'yi paylaştığı için orijinal `main.py` aynı seed'le tekrarlanamıyor (yerelde iki koşu: AA 38.97 ve 80.63). Algoritma aynı. |
| Değerlendirme ortak `Evaluator` ile; FedFST'nin kendi `evaluate`'i de aynı yerde çalıştırılıp `acc_matrix_native` olarak kaydediliyor | Ortak değerlendirme; iki değerlendirme yerelde birebir aynı çıktı (fark 0). |
| Görev maskelerinin ortak cache ile aynı olduğu hash ile doğrulanıyor | Adil karşılaştırma. |
| `plot_all_losses` çağrılmıyor | Algoritmanın parçası değil; yalnızca PNG üretiyor. |

## 3. POWER (Zenodo `zyl24-FCGL_POWER-78e30b6`) – adaptör (`fgcl_bench/runner_power.py`)

Algoritma dosyalarına (`flcore/Ours.py`, `model.py`, `utils.py`, `config.py`) dokunulmadı.

| Değişiklik | Neden |
|---|---|
| `flcore/__init__.py` (boş) eklendi | FedFST P0 ile aynı önlem (namespace package). |
| OpenFGL kurulmuyor; `flcore/Ours.py`'nin kullandığı tek fonksiyon `idx_to_mask_tensor`, OpenFGL'den (commit `5483768`) aynen kopyalanarak `fgcl_bench/shims/openfgl/` altına kondu | POWER'ın veri katmanı (`data.py`) kullanılmıyor; algoritma kodu yalnızca bu fonksiyonu import ediyor. |
| Veri katmanı (`data.py`: OpenFGL `FGLDataset`, scikit-network Louvain, `class_incremental`) yerine ortak cache | Ortak protokol: aynı istemciler, görevler ve maskeler. |
| `OursClient.task_data`, bir alt sınıfta görev alt-çizgisini döndürecek şekilde değiştirildi (orijinal: `edge_masking` + diske kayıt) | Ortak protokol: tüm yöntemler aynı görev çizgesini görüyor. Not: orijinal `edge_masking` boolean maske üzerinde döndüğü için görevler arası kenarları silmiyor (Bölüm 6.4). |
| `config.py` argümanları varsayılanlarıyla okunup veri setine göre dolduruldu (`num_clients=3`, `input_dim`, `output_dim`, `num_rounds_per_task=10`, `num_epochs=3`) | `config.py` import sırasında `sys.argv`'yi okuyor; argüman listesi boşaltılarak varsayılanlar alındı. Yöntem içi hiperparametreler (α=0.5, b=1, β=0.01, decay=0.3, num_epoch_g=3, L-BFGS 300 iterasyon/lr 1.0, lr 0.01, wd 5e-4, GAT 64, dropout 0.5) değişmedi. |
| Değerlendirme: yerel modeller (`global_eval=False`, POWER varsayılanı) yerine sunucunun global modeli; POWER'ın kendi `evaluate(use_global=True)`'si de karşılaştırma için kaydediliyor | Ortak protokol. |
| Görev sırası tüm istemcilerde aynı (orijinal veri katmanı her istemcide ayrı ayrı karıştırıyor) | Ortak protokol (FedFST'nin senaryosu). Bu fark, varsayılan β=0.01 ile birlikte POWER'ın yeni görevi öğrenememesine yol açıyor; POWER'ın kendi betiğiyle doğrulandı (Bölüm 6.4, `tools/power_original_sync_order.py`). |
| Seed: POWER'ın `seed_everything`'i yerine aynı içerikteki ortak `set_seed` | Ortak protokol (iki fonksiyonun içeriği aynı). |

## 4. FedSPA (`OakleyTan/FedSPA` @ `d2b3ddb`) – port (`fgcl_bench/fedspa_port.py`, `fgcl_bench/runner_fedspa.py`)

Repoya yama yapılmadı; yöntem harness'e taşındı. Orijinal sınıflar ve fonksiyonlar aynen kopyalandı; değişen yerler kodda `[port]` ile işaretli.

| Değişiklik | Neden |
|---|---|
| DGL `graph.update_all(mp_func_spa, agg_func_spa)` → `spa_update_all` (`index_select` + kenar türü maskesi + `index_add_`) | Colab'ın Python/torch sürümü için DGL wheel'i yok. DGL 2.1 + torch 2.2.1 ortamında orijinalle karşılaştırıldı: düşük dereceli çizgelerde bit düzeyinde aynı; yüksek dereceli düğümlerde yalnızca toplama sırası farkı (göreli ≤ 1.7·10⁻⁵). |
| DGL `g.adjacency_matrix()` → yoğun $A[src, dst]=1$; `adj_low`, `adj_high` logger'daki formülle | DGL'siz; DGL 2.1'de satır = kaynak (doğrulandı). |
| Sabit `torch.device('cuda')` / `'cuda:0'` → cihaz parametresi; sabit boyutlar (`in_size=1703`, `out_size=5`) → veri setinden; sabit Cornell yükleyici ve önceden üretilmiş 10 istemcili bölümleme kullanılmıyor | Kodun Cornell dışında ve GPU'suz çalışması. |
| `GraphConvolution.att_struc_low`: ilklenmemiş bellek (`torch.FloatTensor`) → sıfır | Parametre ileri geçişte kullanılmıyor. Ama rastgele bellek içeriği yerelde 150 denemenin 15'inde NaN içerdi; `has_nan()` modeli birleştirmeden çıkardı ve 8 denemede `ZeroDivisionError` oluştu. RNG tüketimi değişmiyor. |
| Kenarı olmayan çizgede düğüm sayısı doğrudan x'ten alınıyor | Orijinal `convert_to_dglgraph` boş tensörün max'ında çöküyor. |
| Görev akışında ardışık eğitim (FGL ince ayar baseline'ı) | FedSPA bir continual learning yöntemi değil; POWER makalesinin FedAvg/FedGTA'yı uyarladığı biçim. Unutma önleyici bir şey eklenmedi. |
| Her görevde `label_unk` = o görevin eğitim etiketleri; `adj_low_un/adj_low/adj_high` ve `radio_diff = calculate_shigh(...)` görev alt-çizgisinden yeniden; HBDA'nın homofili ağırlıkları (`diffs`/`weights2`) görev başında yeniden | Orijinalde bunlar eğitimin başında bir kez, istemcinin tüm alt-çizgisi için hesaplanıyor. |
| Optimizer: SGD lr 0.1 → Adam lr 0.01, weight decay 5e-4 (flgo'nun `get_optimizer`'ındaki Adam dalı); görev başına 10 round × 3 tam-çizge adımı (orijinal: 200 round, `num_steps=4`) | Ortak FGCL bütçesi (POWER makalesindeki ortak ayar). |
| Her round'da 3 istemcinin hepsi katılıyor | Orijinal betik flgo'nun varsayılan `md` örneklemesini kullanıyor (yerine koymalı, veri hacmiyle orantılı). Bu örneklemede aynı istemci bir round'da birden çok kez seçilebiliyor ve `weights2[i]` model sırasıyla hizalanmayabiliyor. FedFST ve POWER'da da tüm istemciler katılıyor. |
| Değerlendirmede global model eval modunda (dropout kapalı) | Ortak değerlendirme. Orijinal flgo kodu `model.eval()` hiç çağırmıyor. |
| Değerlendirilen görevin `label_unk`'i o görevin eğitim etiketleri; istemcinin son eğitimdeki fr listeleri bayrak olarak veriliyor (denge dalı) | Orijinal `Client.test` ile aynı çağrı biçimi. |

Kodda olduğu gibi korunan ayrıntılar: denge dalında altıncı terimin `neigh_hete_high` kullanması; HBDA'nın ham $|S_{high}|$ ile ağırlıklandırması; Fisher'ın tüm düğümlerin etiketleriyle ve train modunda (dropout açık) hesaplanması; `freeze_parameters`'ın yalnızca bir listeyi dondurması.

## 5. Ortak protokol kararları

| Karar | Ayrıntı |
|---|---|
| Bölümleme ve görevler | FedFST'nin `louvain_partitioner` (3 istemci) ve `class_to_task` (sınıf başına 0.6/0.2/0.2) fonksiyonları, `main.py`'deki sırayla; sınıf sırası seed ile karıştırılıyor; görevler arası kenar yok; artan sınıf atılıyor. Her (veri seti, seed) için bir kez üretilip `cache/` altına yazılıyor; üç yöntem de buradan okuyor ve maske hash'ini doğruluyor. |
| Görev başına sınıf | 2; Minesweeper'da 1 (2 sınıflı veri setinden tek görev çıkmasın ve unutma ölçülebilsin diye). |
| Bütçe | Görev başına 10 communication round, round başına 3 yerel epoch (tam-çizge), tek seed (24). |
| Omurga | FedFST ve POWER: 2 katmanlı GAT, 64 gizli birim, dropout 0.5. FedSPA: kendi ACM-SPA omurgası (128 gizli birim, dropout 0.1). |
| Değerlendirme | Her görevden sonra global modelin o ana kadarki tüm görevlerdeki test doğruluğu, istemciler arasında test düğümü sayısıyla ağırlıklı; aritmetik `main.py`'nin aynısı; AA/AF FedFST'nin `utils.AA`/`utils.AF` fonksiyonlarıyla. |
| Veri | PyG yükleyicileri; ham dosyalar `raw.githubusercontent.com`'dan önceden indiriliyor (Squirrel: PyG'nin sabitlediği geom-gcn commit'i `f1fc0d1`). |

## 6. Ödev yönergesinden sapmalar ve notlar

- **POWER'ın sonuçları, varsayılan β=0.01 ile yeni görevi öğrenemediğini gösteriyor.** Bu, ortak protokoldeki eşit görev sırasından kaynaklanıyor ve POWER'ın kendi betiğiyle doğrulandı. Yönerge gereği hiperparametre araması yapılmadı.
- **POWER makalesindeki değerler** (Cora 65.74/28.10, CiteSeer 54.47/28.94, Squirrel 40.27/35.24) ve "Squirrel'de 10 istemci" bilgisi, bu çalışma ortamından makalenin tam metnine (arXiv) erişilemediği için kaynağından doğrulanamadı. 20/40/40 ve 48/32/20 bölmeleri `data.py`'den, 10 koşu ortalaması teknik raporun Ek A.3'ünden doğrulandı.
- Colab'daki GPU koşuları atomik toplamalar nedeniyle bit düzeyinde tekrarlanamaz. Yerel CPU sonuçları ile Colab sonuçları arasında küçük farklar olabilir; FedFST'nin RNG'ye duyarlılığı nedeniyle fark büyük de olabilir.
