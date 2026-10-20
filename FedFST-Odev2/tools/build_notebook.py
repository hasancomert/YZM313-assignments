"""FedFST_WarmUp_Odev2.ipynb üreticisi.

Notebook'taki %%writefile hücreleri fgcl_bench/ altındaki dosyalardan, yama hücreleri
tools/patches.py'den üretilir; böylece notebook'taki kod yerelde test edilen kodla aynıdır.

    python tools/build_notebook.py   # -> FedFST_WarmUp_Odev2.ipynb (çıktısız)
"""
import sys
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from patches import PATCHES  # noqa: E402

cells = []


def md(text):
    cells.append(new_markdown_cell(textwrap.dedent(text).strip("\n")))


def code(text, tags=None):
    cell = new_code_cell(textwrap.dedent(text).strip("\n"))
    if tags:
        cell.metadata["tags"] = tags
    cells.append(cell)


def writefile(rel):
    content = (ROOT / rel).read_text(encoding="utf-8")
    cells.append(new_code_cell(f"%%writefile {rel}\n{content.rstrip()}\n"))


HARNESS_FILES = [
    ("fgcl_bench/__init__.py", "paket işareti"),
    ("fgcl_bench/common.py", "veri seti tanımları, seed (FedFST'nin set_seed'iyle aynı), görev hash'i, atomik JSON yazımı"),
    ("fgcl_bench/data.py", "ham dosyaların indirilmesi; FedFST fonksiyonlarıyla bölümleme ve görevlerin üretilip diske (cache) yazılması"),
    ("fgcl_bench/evaluate.py", "ortak değerlendirme: doğruluk matrisi, AA ve AF (FedFST'nin utils.AA / utils.AF fonksiyonlarıyla)"),
    ("fgcl_bench/runner_fedfst.py", "FedFST: main.py'nin eğitim döngüsü, resmi (yamalı) Ours.py sınıflarıyla"),
    ("fgcl_bench/runner_power.py", "POWER: resmi flcore/Ours.py + veri arayüzü adaptörü"),
    ("fgcl_bench/fedspa_port.py", "FedSPA: SFPD + HBDA'nın orijinal koddan port'u (DGL'siz)"),
    ("fgcl_bench/runner_fedspa.py", "FedSPA: görev akışında ardışık eğitim (FGL ince ayar baseline'ı)"),
    ("fgcl_bench/run.py", "tek komutla koşu: python -m fgcl_bench.run --method ... --dataset ... --seed ..."),
    ("fgcl_bench/equivalence.py", "vektörizasyon eşdeğerlik testi (orijinal kod git geçmişinden okunur)"),
    ("fgcl_bench/run_main_sequential.py", "sağlama: orijinal main.py'yi tek iş parçacıklı jeneratör eğitimiyle çalıştırır"),
    ("fgcl_bench/fedspa_dgl_check.py", "FedSPA port'u ile orijinal DGL'li kodun karşılaştırılması (DGL gerektirir; Colab'da çalıştırılmaz)"),
    ("fgcl_bench/shims/openfgl/__init__.py", "POWER için OpenFGL'den tek fonksiyon"),
    ("fgcl_bench/shims/openfgl/utils/__init__.py", ""),
    ("fgcl_bench/shims/openfgl/utils/basic_utils.py", "idx_to_mask_tensor (OpenFGL'den aynen)"),
]

# =====================================================================================================
md(r"""
# İleri Yapay Zeka (YZ313) – WarmUp Ödev 2
## FedFST'nin Google Colab'da çalıştırılması ve POWER (KDD'25) / FedSPA (CVPR'25) ile karşılaştırılması

| | |
|---|---|
| **Ad Soyad** | *(doldurunuz)* |
| **Öğrenci No** | *(doldurunuz)* |
| **Ders** | İleri Yapay Zeka (YZ313) |

**Özet.** Bu notebook, federe çizge sürekli öğrenmesi (Federated Graph Continual Learning, FGCL) için önerilen FedFST'nin resmi kodunu ([YunQi572/FedFST](https://github.com/YunQi572/FedFST), commit `d83fac0`) Colab'da çalıştırır ve sonuçlarını POWER ([Zhu vd., KDD'25](https://arxiv.org/abs/2411.18919)) ve FedSPA ([Tan vd., CVPR'25](https://github.com/OakleyTan/FedSPA)) ile karşılaştırır. Üç yöntem de beş veri setinde (Cora, CiteSeer, Actor, Squirrel, Minesweeper) aynı Louvain bölümlemesini (3 istemci), aynı sınıf-artımlı görevleri ve aynı eğitim/doğrulama/test maskelerini kullanır. Değerlendirme tek ve ortak bir fonksiyonla yapılır: ortalama doğruluk (AA ↑) ve ortalama unutma (AF ↓). Repoların olduğu gibi çalışmayan kısımları en küçük yamalarla düzeltildi. FedFST'nin yavaş Python döngüleri, sonuçları bit düzeyinde değiştirmeden vektörize edildi. FedSPA'nın DGL'e bağlı mesaj geçirme kısmı saf PyTorch'a taşındı.

> **Çalıştırma:** `Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU`, ardından `Çalışma zamanı → Tümünü çalıştır`. Drive erişim izni istenir. Sonuçlar, log'lar ve veri cache'i `MyDrive/FedFST_Odev2/` altına yazılır. Bağlantı koparsa notebook'u yeniden çalıştırmak yeterli: biten koşular atlanır ve deney kaldığı yerden devam eder. GPU kotası yoksa her şey otomatik olarak CPU'da çalışır. Hızlı bir deneme için Bölüm 8'deki CONFIG hücresinde `SMOKE = True` yapın.

**İçindekiler**
1. FedFST'nin çalışma mantığı
2. Karşılaştırılan yöntemler: POWER ve FedSPA
3. Kurulum
4. Kaynak kodların indirilmesi
5. Yamalar ve harness dosyaları
6. Eşdeğerlik testleri ve sağlamalar
7. Veri setleri
8. Deneyler
9. Sonuçlar
10. Tartışma
11. Kaynakça
""")

md(r"""
## 1. FedFST'nin çalışma mantığı

### 1.1 Problem: federe çizge sürekli öğrenmesi (FGCL) ve felaket unutma
Bir çizge, Louvain topluluk bölümlemesiyle **K = 3** istemciye dağıtılır. Sınıflar görevlere ayrılır (burada görev başına 2 sınıf, Minesweeper'da 1) ve görevler sırayla gelir. Buna *sınıf-artımlı* senaryo denir. Her görevde istemciler yalnızca o görevin düğümleriyle yerel eğitim yapar, sunucu da modelleri birleştirir. Her görevin sonunda global modelin o ana kadar görülen **tüm** sınıfları ayırt etmesi beklenir. Model yeni sınıflara uyum sağlarken eski sınıflardaki doğruluğunu kaybeder; buna **felaket unutma** (catastrophic forgetting) denir.

Değerlendirme bir doğruluk matrisi $A$ ile yapılır: $A_{t,i}$, görev $t$'nin eğitimi bittiğinde global modelin görev $i$'nin test düğümlerindeki doğruluğudur (istemciler arasında test düğümü sayısıyla ağırlıklı). $T$ görev için FedFST'nin `utils.AA` ve `utils.AF` fonksiyonları şunları hesaplar:

$$\mathrm{AA}=\frac{1}{T}\sum_{i=1}^{T}A_{T,i}\ (\uparrow),\qquad \mathrm{AF}=\frac{1}{T-1}\sum_{i=1}^{T-1}\left(A_{i,i}-A_{T,i}\right)\ (\downarrow)$$

POWER'ın kodundaki `utils.AA` / `utils.AF` aynı tanımları kullanır; POWER'ın teknik raporundaki grafiklerde bunlar **AM** ve **FM** olarak geçer. Tek görevli bir akışta unutma ölçülemez (`AF` bu durumda −1 döndürür).

### 1.2 Spektral bakış: yüksek ve düşük frekans
Kombinatoryal Laplasyen $L = D - A$ ile bir düğüm sinyali $x$ için $x^\top L x=\sum_{(i,j)\in E}(x_i-x_j)^2$ (yönsüz çizgede) komşular arasındaki farkları toplar. Rayleigh oranı $x^\top L x / x^\top x$ küçükse sinyal komşular arasında **pürüzsüzdür**. Bu düşük frekanstır: komşular benzerdir (homofili) ve sınıflar arası ortak, genel anlam bu bileşende taşınır. Oran büyükse sinyal kenarlar boyunca **hızla değişir**. Bu yüksek frekanstır: komşular arasında tutarsızlık (heterofili) ve ayırt edici ayrıntı vardır. FedFST'nin README'si unutmayı bu iki bileşen üzerinden tanımlar ([README](https://github.com/YunQi572/FedFST)):
* **Yüksek frekanslı tutarsızlık unutması:** yeni görevlerle ortaya çıkan düğüm tutarsızlıkları mesaj geçirmeyi bozar ve ayırt edici bilgiyi siler.
* **Düşük frekanslı tutarlılık unutması:** yeni görevlere aşırı uyum, global anlamsal tutarlılığı sulandırır.

README'ye göre FedFST iki bileşenden oluşur: **HHKR** (Historical High-Frequency Knowledge Restoration) geçmişteki yüksek frekanslı tutarsızlık bilgisini yeniden kurup korur; **HLST** (Historical Low-Frequency Semantic Transfer) düşük frekanslı tutarlılığı spektral damıtmayla korur.

Kodda yüksek frekans ölçüsü `utils.get_Shigh` fonksiyonunda hesaplanır. Önce rastgele seçilen özniteliklerin %20'si için $x^\top L x / x^\top x$ oranlarının ortalaması alınır ($S^{x}_{high}$). Ardından komşuluk matrisinin her sütunu bir sinyal gibi ele alınıp aynı oranın ortalaması alınır ($S^{A}_{high}$). Sonuç $S_{high}=0.5\,S^{x}_{high}+0.5\,S^{A}_{high}$ olur.
""")

md(r"""
### 1.3 Bileşenlerin koda karşılığı
Kodda "HHKR" ve "HLST" adları geçmiyor. Aşağıdaki eşleştirme, README'deki tanımlarla fonksiyonların yaptığı işi karşılaştırarak yaptığım bir okuma. Bu okumayı destekleyen bir ipucu var: KD kaybı kodda `Low Freq Loss` olarak adlandırılıyor (`algorithm/Ours.py`, `KD_train`). Hiperparametreler `args.py`'deki varsayılanlardır.

| Adım | Kod | Kodda gördüğüm kadarıyla ne yapıyor |
|---|---|---|
| Yerel eğitim | `OursClient.train` | Global ağırlıklarla başlar; yalnızca mevcut görevin eğitim düğümlerinde çapraz entropiyle 3 epoch eğitir (her round yeni Adam, lr 0.005, wd 5e-4). Yerel tarafta unutmaya karşı ek bir kayıp terimi yok. |
| Birleştirme | `OursServer.aggregate` | FedAvg; istemci ağırlıkları görevdeki düğüm sayısıyla orantılı. |
| HHKR-1: koşullu jeneratör | `OursClient.feature_gen_train`, `utils.Generator` | Gürültü ve sınıf gömmesinden düğüm özniteliği üreten bir MLP. Kaybın iki terimi var: (i) bir önceki görevin global modeli (`last_global_model`) üretilen düğümleri koşul sınıfına atamalı (CE); (ii) üretilen özniteliğin softmax dağılımı, aynı sınıftan rastgele seçilen **gerçek bir eski-görev düğümünün** dağılımına KL ile yaklaşmalı (ağırlık 100). İstemci başına 2 jeneratör round'u × 200 epoch çalışır; sunucu jeneratörleri öğrenilmiş düğüm sayısıyla ağırlıklı ortalar (`feature_gen_aggregate`). |
| HHKR-2: homofili eşleme | `get_learned_Homophily_Level`, `utils.reduce_homophilic_edges` / `reduce_heterophilic_edges` | Üretilen düğümler rastgele bir çizgeyle bağlanır (düğüm başına 2 kenar). Çizgenin homofilisi, istemcinin eski görevlerdeki homofili oranına (eğitim düğümleri arası kenarlarda) ±0.05 yaklaşana kadar her adımda homofilik ya da heterofilik kenarların %5'i silinir (en çok 60 adım). |
| HHKR-3: sunucuda $S_{high}$ eşleme | `send_feature_gen` (`get_Shigh`), `OursServer.synthesis_data` | İstemciler eski-görev alt-çizgilerinin $S_{high}$ değerini gönderir; sunucu bunların ağırlıklı ortalamasını alır. Global jeneratörle eski sınıfların her birinden 300 düğüm üretir ve rastgele çizgeden kenar silerek sentetik çizgenin $S_{high}$'ını bu değere ±0.05 yaklaştırır. Böylece sentetik tekrar çizgesi geçmişin yüksek frekans (tutarsızlık) düzeyini taşır. |
| HLST: düşük frekanslı damıtma | `OursServer.KD_train`, `utils.get_khop_feature`, `utils.edge_distribution_low` | Öğrenci global model, öğretmen önceki görevin global modeli. Kayıp = 0.5·CE (sentetik etiketler) + 1·KL. KL, iki modelin softmax çıktılarının **2 adımlı komşu ortalamasıyla yumuşatılmış** hâlleri (alçak geçiren filtre) arasında, kenar uçları üzerinden hesaplanır. 200 epoch, Adam lr 0.002. |
| Model seçimi | `evaluate_on_clients`, `evaluate_for_kd` | Her KD epoch'unda istemcilerin **doğrulama** maskeleriyle AA−AF hesaplanır. Eski görevlerin doğruluğu mevcut görevi geçince eğitim erken durur ve son iki kayıttan iyisi seçilir; erken durmazsa en iyi kayıt kullanılır. Test maskesi seçimde kullanılmıyor. |
""")

md(r"""
### 1.4 Akış şeması
```
Görev t = 0, 1, …, T−1
│
├─ 10 round ─┬─ istemci k: GAT'ı global ağırlıklarla başlat, yalnızca görev t verisiyle 3 epoch eğit
│            └─ sunucu: FedAvg (görevdeki düğüm sayısıyla ağırlıklı)
│
├─ t > 0 ise ─┬─ 2 jeneratör round'u
│             │    istemci k: koşullu jeneratör, CE(önceki global model) + 100·KL(gerçek eski düğüm)
│             │               sentetik çizgenin homofilisi ← eski görevlerin homofilisi       [HHKR]
│             │    istemci k → sunucu: jeneratör ağırlıkları + eski alt-çizginin S_high değeri
│             │    sunucu: jeneratörleri ağırlıklı ortala
│             ├─ synthesis_data: eski sınıflardan 300'er düğüm üret,
│             │                  sentetik çizgenin S_high'ı ← global S_high                    [HHKR]
│             └─ KD_train: global model ← 0.5·CE + KL(2-adım yumuşatılmış öğrenci ‖ öğretmen),
│                          doğrulama AA−AF ile model seçimi / erken durma                   [HLST]
│
└─ önceki global model ← global model;  değerlendirme: A[t, 0..t]
```

### 1.5 Kodu okurken dikkat ettiğim noktalar
* **Geçmiş veriye erişim.** Görev $t$'de istemciler önceki görevlerin gerçek verisini kullanıyor: jeneratörün KL terimi gerçek düğüm özniteliklerini, homofili ve $S_{high}$ hesapları da etiketleri ve kenarları kullanıyor (`get_learned_graph`, `get_subgraph`, `get_learned_Homophily_Level`). KD'deki model seçimi de önceki görevlerin doğrulama etiketlerine bakıyor. Yani istemciler geçmiş görevlerin yerel verisini saklıyor. POWER ise sınıf başına yalnızca 1 düğüm saklıyor (`num_samples_per_class = 1`). Karşılaştırmayı yorumlarken bu fark akılda tutulmalı.
* **KD kaybının biçimi.** `KD_train`'de değişken adı `student_log_probs` olsa da değer `F.softmax(...)`, yani olasılık. `edge_distribution_low` bu yumuşatılmış olasılıkları `nn.KLDivLoss(log_target=True)`'ya log-olasılık gibi veriyor.
* **Tekrarlanabilirlik.** `main.py`, istemcilerin jeneratör eğitimini `ThreadPoolExecutor` ile eşzamanlı çalıştırıyor. Üç iş parçacığı aynı küresel RNG'yi paylaştığı için aynı seed'le bile sonuç değişiyor (Bölüm 6.3'te iki ayrı koşu). Harness bu adımı aynı sırayla ardışık çalıştırıyor.
""")

md(r"""
## 2. Karşılaştırılan yöntemler

### 2.1 POWER (Zhu, Hu ve Wu, KDD'25)
POWER, FGCL problemini ilk tanımlayan çalışma. Makale iki zorluğu adlandırıyor ([arXiv:2411.18919](https://arxiv.org/abs/2411.18919)):
* **Yerel çizge unutması (LGF):** istemci modeli yeni göreve uyum sağlarken eski bilgiyi kaybeder.
* **Global uzmanlık çatışması (GEC):** istemcilerin tutarsız uzmanlıkları sunucuda birleştirilirken çatışır ve global model hem eski hem yeni görevlerde kötüleşir.

LGF için yerel-global kapsamı en geniş deneyim düğümleri saklanıp tekrar oynatılır. GEC için sunucuda **sözde prototip yeniden oluşturma** ve **yörünge-farkında bilgi aktarımı** kullanılır. Resmi kodda ([Zenodo](https://zenodo.org/records/15487655): `flcore/Ours.py`, `utils.py`) gördüğüm karşılıklar şunlar:

| Bileşen | Kod | Ne yapıyor |
|---|---|---|
| Deneyim düğümleri (LGF) | `OursClient.task_done` → `utils.update_converage_max` | Görev bitince her sınıftan 1 düğüm seçer (kapsama maksimizasyonu). Uzaklıklar, yerel ve global modelin gömmelerinin α = 0.5 karışımında hesaplanır; yarıçap, ortalama uzaklığın 0.1 katıdır. |
| Yerel eğitim | `OursClient.execute` | Kayıp = β·CE(mevcut görev) + (1−β)·CE(tampondaki düğümler, kenarsız). β'nın varsayılanı 0.01. |
| Prototip gradyanı | `OursClient.encode_gradient` | Her görevin ilk round'unda her sınıfın öznitelik ortalaması (prototip), tüm istemcilerde ortak ve hiç eğitilmeyen 4 katmanlı bir MLP'den (`GEModel`) geçirilir. CE kaybının bu MLP'nin parametrelerine göre gradyanı sunucuya gönderilir. |
| Sözde prototip (GEC) | `OursServer.execute`, `gradient2label` | Sunucu, sınıf etiketini son katman bias gradyanının en küçük elemanından çıkarır. Rastgele bir girdiyi, gradyanı gelen gradyanla eşleşene kadar L-BFGS ile optimize eder (300 iterasyon, lr 1.0). Bu, DLG'ye benzer bir gradyan ters çevirmedir. Sonuç, sınıfın prototip tamponuna eklenir. |
| Yörünge-farkında aktarım (GEC) | `OursServer.execute` (son kısım) | FedAvg'den sonra global model, tüm sınıfların sözde prototipleri üzerinde her istemcinin yerel modelinin çıktılarına (L1) yaklaştırılır (3 epoch, Adam lr 0.01). İstemci-sınıf ağırlıkları sınıf örnek sayılarından hesaplanır; geçmiş görevlerin katkısı `decay = 0.3` ile azaltılır. |

Kodda gördüğüm ve karşılaştırmayı etkileyen ayrıntılar:
* **β = 0.01.** `config.py`'de "explored" başlığı altında. Resmi eğitim betiği (`optuna_search.py`) β, `decay`, `num_recon_nodes` ve `num_epoch_g` için 300 denemelik bir Optuna araması yapıyor. Teknik raporda da β "keşif" hiperparametreleri arasında (Ek A.3). Ödev gereği arama yapmadım; kodun varsayılanlarını kullandım.
* **Görev sırası.** POWER'ın veri katmanı (`data.py`, `class_incremental`) görevlerin sırasını **her istemci için ayrı ayrı karıştırıyor** (`np.random.shuffle(tasks)`). Yani orijinal senaryoda istemciler aynı anda farklı sınıfları öğreniyor. Ortak protokolde ise, FedFST'deki gibi, tüm istemciler aynı sırayı izliyor.
* **Değerlendirme.** `utils.start()` varsayılan olarak (`global_eval = False`) her istemcinin **yerel** modelini kendi verisinde değerlendiriyor. Ortak protokolde global model değerlendiriliyor.
* **KNN çizgesi.** Teknik rapor KNN parametresini 1 olarak veriyor (Ek A.3). Kodda `construct_knn_graph` tanımlı ama kullanılmıyor; sunucu sözde prototipler için kenarsız bir çizge (`construct_self_loop_graph`) kuruyor.
* **Veri setleri ve bağımlılıklar.** POWER'ın veri katmanı (`data.py`) bölme oranlarını Cora, CiteSeer, Computers, Physics, Actor, ogbn-arxiv, Flickr, Squirrel ve Roman-empire için tanımlıyor; Minesweeper için tanım yok, bu yüzden orijinal betik Minesweeper'da çalışmaz. Ortak cache kullanıldığı için bu kısıt harness'te geçerli değil. Kod derlenmiş bir eklenti (torch_scatter, torch_sparse, DGL) kullanmıyor; PyTorch Geometric'e, veri katmanı için de OpenFGL'e bağlı. `config.py`'de `--gpuid 0` ve yer tutucu bir `--root` var; `optuna_search.py` ise cihazı CPU'ya zorluyor (`args.disable_cuda = True`). Harness'te cihaz otomatik seçiliyor (GPU varsa GPU).
* **`edge_masking`.** `task_data`, görev çizgesini `utils.edge_masking` ile kurmaya çalışıyor. Fonksiyon boolean bir maske üzerinde `for node in handled` ile döndüğü için görevler arası kenarları silmiyor ve self-loop'ları yalnızca 0 ve 1 numaralı düğümlere ekliyor (Bölüm 6.4'te küçük bir örnekle gösteriliyor). Harness bu fonksiyonu kullanmıyor; tüm yöntemlere aynı görev alt-çizgisini veriyor.

### 2.2 FedSPA (Tan vd., CVPR'25)
FedSPA, standart federe çizge öğrenmesinde (FGL) **homofili heterojenliği** sorununa yöneliyor. İstemcilerin yerel alt-çizgilerinin homofili düzeyleri farklı olduğunda global GNN'in genellemesi bozuluyor ([README](https://github.com/OakleyTan/FedSPA)). Yöntemin iki bileşeni var: **SFPD** (Subgraph Feature Propagation Decoupling) istemcilerin ortak bir homofili düzeyinde iş birliği yapmasını sağlıyor; **HBDA** (Homophily Bias-Driven Aggregation) birleştirmede homofili yanlılığı düşük istemcileri öne çıkarıyor.

**FedSPA bir sürekli öğrenme (continual learning) yöntemi değil; unutmaya karşı bir mekanizması yok.** Bu çalışmada, POWER makalesinin FedAvg ve FedGTA gibi FGL yöntemlerini FGCL'ye uyarladığı biçimde kullanıldı: görevler sırayla geliyor ve aynı global model her görevde FedSPA ile eğitiliyor ("FL/FGL fine-tuning"; [POWER teknik raporu](https://zenodo.org/records/15487655), Ek A.1).

| Bileşen | Kod (`Code/` altında) | Ne yapıyor |
|---|---|---|
| SFPD (omurga) | `benchmark001/config.py`: `ACM`, `GraphConvolution`, `mp_func_spa`, `agg_func_spa` | 2 katmanlı bir model; her katmanda üç kanal var: alçak geçiren ($\tilde D^{-1}(A+I)$), yüksek geçiren ($I-\tilde D^{-1}(A+I)$) ve MLP. Sınıf adı `ACM`; yapı, ACM-GCN'in (Luan vd., 2022) kanallarına benziyor. Komşu mesajları eğitim etiketlerine göre üçe ayrılır: iki uç da etiketli ve **aynı** sınıftan (homofilik), etiketli ve **farklı** sınıftan (heterofilik), en az bir ucu etiketsiz (bilinmeyen). Her grup için ayrı dikkat vektörleri öğrenilir. Bilinmeyen gruptaki mesajlar, öğrenilen bir kapıyla (`fc_balance`, sigmoid) heterofilik ("fr") ve homofilik ("be") dikkat vektörlerinin karışımıyla puanlanır. |
| HBDA | `flgo/algorithm/fedspa.py`: `Server.aggregate`, `fish_cal`, `calculate_weights`; `fedbase.py`: `calculate_shigh` | "fr" ve "be" dikkat parametreleri, Fisher bilgisi (log-olabilirlik gradyanının karesi) istemci ortalamasına yakın olan istemcilere daha büyük ağırlık verilerek birleştirilir: $w_k \propto e^{-\tilde d_k}$, burada $\tilde d_k$ min-max ölçeklenmiş farktır. Diğer tüm parametreler aynı formülle, istemcinin öznitelik yüksek frekans oranına ($S_{high}$, `calculate_shigh`) göre ağırlıklandırılır. |
| Komşuluk matrisleri | `flgo/experiment/logger/simple_logger.py` (`log_once`) | `adj_low` ve `adj_high`, eğitimin başında logger'ın içinde bir kez hesaplanıyor. |

Kodda gördüğüm ayrıntılar:
1. HBDA'nın homofili ağırlıkları, sunucunun hesapladığı "istemci $S_{high}$'ının ağırlıklı ortalamadan sapması" (`server.radio_diff`) ile değil, istemcinin **ham** $|S_{high}|$ değeriyle hesaplanıyor (`fedspa.py`: `self.diffs.append(abs(client.radio_diff))`).
2. Değerlendirmede `model.eval()` hiç çağrılmıyor; yani orijinal değerlendirme dropout açıkken yapılıyor.
3. `GraphConvolution.att_struc_low` ilklenmemiş bellekle (`torch.FloatTensor`) oluşturuluyor. Yerel denemelerde bu bellek zaman zaman NaN içerdi; birleştirmedeki `has_nan()` kontrolü o modeli dışarıda bıraktı ve `ZeroDivisionError` / `ValueError` hatalarına yol açtı (Bölüm 6.2).
4. Denge dalındaki altıncı dikkat terimi `neigh_unk_high` yerine `neigh_hete_high`'ı kullanıyor (`attention7`). Port'ta kodda olduğu gibi korundu.
""")

# =====================================================================================================
md(r"""
## 3. Kurulum
Colab'daki hazır torch'a dokunulmuyor (yeniden kurma ya da sürüm düşürme yok). Yalnızca eksik olan saf-Python paketleri pip ile kuruluyor: `torch_geometric`, `python-louvain` (FedFST `from community import community_louvain` kullanıyor; PyPI'daki `community` adlı paket başka bir şey), `ogb` ve `gdown` (FedFST'nin veri yükleyicisi bunları import ediyor). Önceden eğitilmiş bir model gerekmiyor; tüm GNN'ler sıfırdan eğitiliyor.

### 3.1 Ortam bilgisi ve Google Drive
""")

code(r"""
import os, sys, re, json, time, shutil, platform, subprocess, importlib, warnings
from pathlib import Path
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")   # ogb'nin bağımlılığından
warnings.filterwarnings("ignore", message=".*IProgress not found.*")

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/fedfst_odev2")                    # kodlar (oturum boyunca)
    WORK = Path("/content/drive/MyDrive/FedFST_Odev2")      # sonuçlar, log'lar, cache (kalıcı)
else:  # yerel test: Colab'a özgü adımlar atlanır
    BASE = Path(os.environ.get("FGCL_BASE", "fedfst_odev2_local")).resolve()
    WORK = Path(os.environ.get("FGCL_WORK", str(BASE / "work"))).resolve()
SRC = BASE / "src"
for p in (BASE, SRC, WORK, WORK / "data", WORK / "cache"):
    p.mkdir(parents=True, exist_ok=True)
os.chdir(BASE)
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
print("Colab:", IN_COLAB, "| kod klasörü:", BASE, "| kalıcı klasör:", WORK)

import torch
print("Python", platform.python_version(), "| torch", torch.__version__, "| torch'un CUDA sürümü:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    if shutil.which("nvidia-smi"):
        print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv"], capture_output=True, text=True).stdout)
else:
    print("GPU yok: tüm koşular otomatik olarak CPU'da çalışacak.")
""")

md(r"""
### 3.2 Bağımlılıklar
""")

code(r"""
def importable(module):
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False

missing = [pkg for mod, pkg in [("torch_geometric", "torch_geometric"),
                                ("community.community_louvain", "python-louvain"),
                                ("ogb", "ogb"), ("gdown", "gdown")] if not importable(mod)]
if missing:
    print("kuruluyor:", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    importlib.invalidate_caches()
else:
    print("gerekli paketler zaten kurulu")

from importlib.metadata import version, PackageNotFoundError
for pkg in ["torch", "torch_geometric", "python-louvain", "ogb", "gdown", "networkx", "numpy", "scipy",
            "scikit-learn", "pandas", "matplotlib"]:
    try:
        print(f"{pkg:16s} {version(pkg)}")
    except PackageNotFoundError:
        print(f"{pkg:16s} -")
""")

# =====================================================================================================
md(r"""
## 4. Kaynak kodların indirilmesi
### 4.1 FedFST ve FedSPA (sabit commit'ler)
""")

code(r"""
def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

REPOS = {"FedFST": ("https://github.com/YunQi572/FedFST", "d83fac0"),
         "FedSPA": ("https://github.com/OakleyTan/FedSPA", "d2b3ddb")}
for name, (url, commit) in REPOS.items():
    repo = SRC / name
    if not (repo / ".git").exists():
        git("clone", "-q", url, str(repo))
    if not git("rev-parse", "HEAD", cwd=repo).startswith(commit):
        git("checkout", "-q", commit, cwd=repo)
    print(f"{name:7s} {git('log', '-1', '--format=%h  %ad  %s', '--date=short', cwd=repo)}")
FEDFST = SRC / "FedFST"
""")

md(r"""
### 4.2 POWER (Zenodo)
POWER'ın GitHub reposu açılmıyor; resmi kod Zenodo'da ([10.5281/zenodo.15487654](https://doi.org/10.5281/zenodo.15487654), `zyl24/FCGL_POWER-official.zip`, md5 `567faa826ad62a3d8610682ee2e8474f`). Zip, Drive'daki çalışma klasörüne indirilir. Zenodo'ya erişilemezse zip'i elle indirip `MyDrive/FedFST_Odev2/FCGL_POWER-official.zip` olarak koymak yeterli; hücre önce oraya bakar.
""")

code(r"""
import hashlib, zipfile, urllib.request

POWER_URL = "https://zenodo.org/records/15487655/files/zyl24/FCGL_POWER-official.zip?download=1"
POWER_MD5 = "567faa826ad62a3d8610682ee2e8474f"
POWER_ZIP = WORK / "FCGL_POWER-official.zip"
POWER_DIR = SRC / "POWER"

def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

if not (POWER_DIR / "flcore" / "Ours.py").exists():
    if not (POWER_ZIP.exists() and md5sum(POWER_ZIP) == POWER_MD5):
        print("Zenodo'dan indiriliyor ...")
        part = POWER_ZIP.with_suffix(".part")
        if shutil.which("wget"):
            subprocess.run(["wget", "-q", "-O", str(part), POWER_URL])
        else:
            try:
                urllib.request.urlretrieve(POWER_URL, part)
            except Exception as exc:
                print("indirme hatası:", exc)
        if part.exists() and part.stat().st_size > 0 and md5sum(part) == POWER_MD5:
            part.replace(POWER_ZIP)
    assert POWER_ZIP.exists() and md5sum(POWER_ZIP) == POWER_MD5, (
        f"POWER zip'i indirilemedi ya da md5 tutmuyor. Zenodo'dan elle indirip {POWER_ZIP} olarak koyun.")
    with zipfile.ZipFile(POWER_ZIP) as z:
        z.extractall(SRC)
    (SRC / "zyl24-FCGL_POWER-78e30b6").rename(POWER_DIR)
if POWER_ZIP.exists():
    print("POWER zip md5:", md5sum(POWER_ZIP))
print("POWER dosyaları:", sorted(p.name for p in POWER_DIR.iterdir()))
""")

# =====================================================================================================
md(r"""
## 5. Yamalar ve harness dosyaları
Değişiklikler en küçük tutuldu ve her biri ayrı bir hücrede. Yamalar **idempotenttir**: yeni metin bir `[Odev2-Px]` işareti taşır; işaret dosyada zaten varsa yama atlanır, yani hücreler iki kez çalıştırılınca dosyalar bozulmaz. Yama motoru (`fgcl_bench/patching.py`) orijinal metni satır satır arar. Orijinal dosyalarda boş satırlarda girinti boşlukları olduğu için satır sonu boşluklarını yok sayar. Tam olarak bir eşleşme bulamazsa hata verir.
""")

code(r"""
for d in ["fgcl_bench", "fgcl_bench/shims/openfgl/utils"]:
    os.makedirs(d, exist_ok=True)
""")
writefile("fgcl_bench/patching.py")
code(r"""
from fgcl_bench.patching import apply_patch, create_files
""")

md(r"""
### 5.1 FedFST yamaları
**P0–P4**, repoyu çalışır hâle getirir; **P5a–P5f**, ölçülen darboğazlardaki Python döngülerini aynı matematik ve aynı RNG kullanımıyla vektörize eder. `--device_id` ve diğer argümanlar için yama gerekmiyor: koşucu `--device_id 0` veriyor ve `type=bool` tanımlı `--use_gpu` / `--shuffle_flag` argümanlarını komut satırından vermiyor (argparse `"False"` dizesini de True okur).
""")

for p in PATCHES:
    if p["kind"] == "create":
        md(f"**{p['id']} – {p['title']}**\n\n*Neden:* {p['reason']}")
        files = ", ".join(repr(f) for f in p["files"])
        code(f'create_files(FEDFST, "{p["id"]}", [{files}],\n'
             f'             {p["content"]!r})')
    else:
        md(f"**{p['id']} – `{p['file']}`: {p['title']}**\n\n*Neden:* {p['reason']}")
        code(f'apply_patch(FEDFST, "{p["file"]}", "{p["id"]}",\n'
             f"old=r'''\n{p['old']}\n''',\n"
             f"new=r'''\n{p['new']}\n''')")

md(r"""
### 5.2 POWER ve FedSPA
**POWER:** ödevde tercih edilen yol seçildi: resmi istemci ve sunucu sınıfları (`flcore/Ours.py`) olduğu gibi import edilip harness'e bir adaptörle bağlandı. Bu yol mümkündü, çünkü POWER'ın algoritma kodu veri katmanından ayrık; veriyi yalnızca `self.data["data"]`, `self.data["task"]` ve `task_data()` üzerinden görüyor. Bu yüzden veri sözlüğü ortak cache'ten kuruldu ve bir alt sınıfta yalnızca `task_data` değiştirildi. Eğitim döngüsü POWER'ın `utils.start()` döngüsünün aynısı. Algoritma dosyalarına (`flcore/Ours.py`, `model.py`, `utils.py`) dokunulmuyor. `flcore/` klasörü de FedFST'nin klasörleri gibi `__init__.py`'siz bir namespace package olduğu için aynı önlem (P0) burada da alınıyor. POWER'ın OpenFGL'den kullandığı tek fonksiyon (`idx_to_mask_tensor`) OpenFGL'den aynen kopyalanarak `fgcl_bench/shims/` altına konuyor. Veri arayüzü `fgcl_bench/runner_power.py`'deki adaptörde (Bölüm 5.4).

**FedSPA:** repoya yama yapılmıyor. Yöntem, gömülü flgo kopyasından ve DGL'den bağımsız olarak `fgcl_bench/fedspa_port.py`'ye taşındı (Bölüm 5.4 ve 6.2).
""")

code(r"""
create_files(POWER_DIR, "POWER-P0", ["flcore/__init__.py"],
             "# [Odev2-POWER-P0] flcore'u normal bir Python paketi yapar (bkz. FedFST P0).\n")
""")

md(r"""
### 5.3 Harness dosyaları
Ortak deney düzeneği `fgcl_bench/` paketinde. Her koşu ayrı bir süreçte çalışır (`python -m fgcl_bench.run --method {fedfst,power,fedspa} --dataset ... --seed ...`) ve sonucunu bir JSON dosyasına yazar.

| Dosya | Görevi |
|---|---|
""" + "\n".join(f"| `{f}` | {desc} |" for f, desc in HARNESS_FILES if desc))

for rel, _ in HARNESS_FILES:
    writefile(rel)

md(r"""
### 5.4 Değişikliklerin ve protokol kararlarının listesi

| Dosya / yer | Ne değişti | Neden |
|---|---|---|
| FedFST `datasets/`, `algorithm/`, `models/` (P0) | boş `__init__.py` eklendi | Colab'daki Hugging Face `datasets` paketi, FedFST'nin namespace package olan `datasets` klasörünü gölgeliyor |
| FedFST `datasets/dataset_loader.py` (P1) | `torch_sparse` import'u kaldırıldı | ModuleNotFoundError; SparseTensor kullanılmıyor |
| FedFST `algorithm/Ours.py` (P2) | var olmayan iki fonksiyonun import'u kaldırıldı | ImportError; fonksiyonlar kullanılmıyor |
| FedFST `args.py`, `main.py` (P3a/P3b) | `--kd_ce_weight 0.5`, `--gen_ce_weight 1.0` eklendi; tanımsız `S_high_D_weight` print'i düzeltildi | AttributeError; varsayılanlar Ours.py'deki getattr varsayılanlarıyla aynı |
| FedFST `datasets/dataset_loader.py` (P4a/P4b) | Minesweeper ve Squirrel eklendi | yükleyici bu iki veri setini kabul etmiyordu |
| FedFST `utils.py`, `algorithm/Ours.py`, `datasets/partition.py` (P5a–P5f) | Python döngüleri vektörize edildi | hız; çıktı, gradyan ve RNG durumu birebir aynı (Bölüm 6.1) |
| FedFST eğitim döngüsü (`runner_fedfst.py`) | `main.py`'deki döngü aynen; jeneratör eğitimi thread'ler yerine aynı sırayla ardışık; değerlendirme ortak fonksiyonla; kayıp grafikleri çizilmiyor | tekrarlanabilirlik (thread'ler aynı RNG'yi paylaşıyor); ortak değerlendirme |
| POWER `flcore/__init__.py` | boş dosya eklendi | P0 ile aynı önlem |
| POWER veri katmanı (`runner_power.py`) | `data.py` / OpenFGL yerine ortak cache; `task_data` yalnızca görev alt-çizgisini döndürecek şekilde alt sınıfta değiştirildi; `config.py` argümanları varsayılanlarla okunup veri setine göre dolduruldu | ortak protokol: aynı istemciler, görevler, maskeler, aynı görev çizgesi; OpenFGL kurulmuyor |
| POWER değerlendirme | yerel model yerine global model (ortak fonksiyon) | ortak protokol |
| FedSPA (`fedspa_port.py`) | DGL `update_all` → `index_add_`; DGL `adjacency_matrix` → yoğun A[src, dst]; sabit `cuda` ve sabit boyutlar kaldırıldı; `att_struc_low` sıfırla ilklendi | Colab'da DGL kurulamıyor; ilklenmemiş bellek birleştirmeyi rastgele bozuyordu |
| FedSPA eğitim (`runner_fedspa.py`) | görev akışında ardışık ince ayar; her görevde `label_unk`, adj matrisleri, $S_{high}$ ve HBDA homofili ağırlıkları görev alt-çizgisinden yeniden; Adam lr 0.01 / wd 5e-4, görev başına 10 round × 3 tam-çizge adımı (orijinal: SGD lr 0.1, 200 round, 4 adım); her round'da 3 istemci de katılıyor (orijinal flgo 'md' örneklemesi yerine koymalı); değerlendirmede eval modu | FedSPA bir FGL yöntemi; ortak FGCL bütçesi (POWER makalesindeki ortak ayar) |
| Protokol | tüm yöntemler aynı cache'ten okuyor; maske hash'i her koşuda doğrulanıp JSON'a yazılıyor | adil karşılaştırma |
""")

# =====================================================================================================
md(r"""
## 6. Eşdeğerlik testleri ve sağlamalar
### 6.1 FedFST vektörizasyonu
Test, orijinal fonksiyonları yamalı repodan değil **git geçmişinden** (`git show d83fac0:<dosya>`) okur ve yamalı sürümlerle aynı seed'le karşılaştırır. Çıktıların (dtype, şekil ve gerekirse bellek düzeni dahil), gradyanların ve çağrıdan sonraki RNG durumunun birebir aynı olması istenir. Böylece rastgele seçimlerin aynı generator'dan aynı sırayla çekildiği doğrulanır. GPU varsa indirgeme sırasından kaynaklanabilecek farklar için KL testi GPU'da ayrıca toleransla çalışır (RNG durumu yine birebir istenir). Testin gerçekten fark yakalayabildiğini yerelde kasıtlı hatalar enjekte ederek de denedim: yanlış RNG kullanımı ve değiştirilmiş oran yakalandı.
""")

code(r"""
from fgcl_bench.data import prefetch_raw
prefetch_raw(WORK / "data", "cora")            # Louvain testi Cora'yı kullanıyor
cmd = [sys.executable, "-m", "fgcl_bench.equivalence", "--fedfst", str(FEDFST), "--data_root", str(WORK / "data")]
if torch.cuda.is_available():
    cmd.append("--cuda")
res = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print(res.stderr[-3000:])
    raise RuntimeError("Vektörizasyon eşdeğerlik testi geçmedi.")
""")

md(r"""
### 6.2 FedSPA: DGL → saf PyTorch
Colab'daki Python/torch sürümü için DGL wheel'i yok (PyPI'da Python 3.13 için yalnızca çok eski 0.1.0–0.1.3 sürümleri var). Bu yüzden FedSPA'nın DGL kullanan iki parçası saf PyTorch ile yeniden yazıldı:
* `graph.update_all(mp_func_spa, agg_func_spa)` → `spa_update_all`: mesajlar DGL'deki gibi tüm kenarlar için `index_select` ile toplanır, kenar türü maskesiyle çarpılır ve `index_add_` ile hedef düğümlerde toplanır;
* `g.adjacency_matrix()` → yoğun $A[\mathrm{src}, \mathrm{dst}] = 1$ (DGL 2.1'de satır = kaynak; yerelde doğrulandı).

Port, yerelde ayrı bir ortamda (Python 3.11, torch 2.2.1, DGL 2.1.0) orijinal DGL'li koda karşı test edildi (`fgcl_bench/fedspa_dgl_check.py`). Orijinal sınıflar FedSPA reposundan okundu, aynı ağırlıklar `load_state_dict(strict=True)` ile yüklendi ve aynı seed kullanıldı. Her çizgede 152 karşılaştırma yapıldı: komşuluk matrisleri; iki dalın (fr listesi var/yok) train ve eval çıktıları; tüm gradyanlar; `requires_grad` bayrakları; `fish_cal` Fisher değerleri; 3 istemcili HBDA birleştirmesinin sonucu. Yerel çıktı:

```
torch 2.2.1+cu121, dgl 2.1.0; torch.sum(dim=1) derece < 5 iken sıralı toplamla birebir
[rastgele N=40 E=47, maks. giriş derecesi 3] 152/152 karşılaştırma birebir aynı; en büyük göreli fark 0.00e+00 -> GEÇTİ
[rastgele N=12 E=29, maks. giriş derecesi 4] 152/152 karşılaştırma birebir aynı; en büyük göreli fark 0.00e+00 -> GEÇTİ
[rastgele N=40 E=156, maks. giriş derecesi 8] 152/152 karşılaştırma birebir aynı; en büyük göreli fark 0.00e+00 -> GEÇTİ
[rastgele N=200 E=1475, maks. giriş derecesi 17] 86/152 karşılaştırma birebir aynı; en büyük göreli fark 1.31e-06 -> GEÇTİ
[squirrel istemci0/görev0 N=864 E=25474, maks. giriş derecesi 370] 74/152 karşılaştırma birebir aynı; en büyük göreli fark 1.68e-05 -> GEÇTİ
SONUÇ: port, orijinal DGL'li kodla eşdeğer (en büyük göreli fark 1.68e-05)
```

Düşük dereceli çizgelerde tüm çıktılar bit düzeyinde aynı. Yüksek dereceli düğümlerde kalan fark yalnızca toplama sırasından geliyor. DGL'in `agg_func_spa`'sı her düğümün posta kutusunu `torch.sum(dim=1)` ile topluyor; PyTorch bu toplamı tensör şekline göre farklı sırayla yapıyor: ölçtüğüm iç boyutlarda toplam, derece 5 ya da 18'den itibaren sıralı toplamdan ayrılıyor (ör. iç boyut 2 iken 5'ten, 128 iken 18'den itibaren). `index_add_` ise sıralı topluyor. Test sırasında orijinal kodun ayrıca bir hatası ortaya çıktı: ilklenmemiş `att_struc_low` belleği 150 denemenin 15'inde bir modele NaN soktu ve bunların 8'inde orijinal birleştirme `ZeroDivisionError` ile çöktü. Port bu parametreyi sıfırla ilkliyor; parametre ileri geçişte kullanılmadığı için hesap değişmiyor.
""")

md(r"""
### 6.3 Sağlama: harness'teki FedFST ile orijinal `main.py`
Yerelde (CPU, tek iş parçacığı) yamalı orijinal `main.py`, jeneratör eğitimini yapan `ThreadPoolExecutor` tek iş parçacığına indirilerek (`fgcl_bench/run_main_sequential.py`; `main.py`'nin kodu değişmiyor) Cora'da seed 24 ile çalıştırıldı:

| Koşu | AA | AF | Son doğruluk matrisi |
|---|---|---|---|
| Harness (`fgcl_bench.run --method fedfst`) | 61.7726 | 32.7114 | [[100, 0, 0], [99.15, 94.64, 0], [85.47, 43.75, 56.10]] |
| Orijinal `main.py`, sıralı jeneratör | 61.7726 | 32.7114 | aynı |
| Orijinal `main.py`, thread'li (1. koşu) | 38.9687 | 55.5823 | [[100, 0, 0], [92.31, 86.61, 0], [70.09, 5.36, 41.46]] |
| Orijinal `main.py`, thread'li (2. koşu) | 80.6263 | 12.9273 | [[100, 0, 0], [94.87, 93.75, 0], [99.15, 68.75, 73.98]] |

Harness, orijinal döngüyle bit düzeyinde aynı sonucu veriyor. Thread'li orijinal ise aynı seed'le iki koşuda çok farklı sonuçlar veriyor. Bu, hem tekrarlanamazlığın kaynağını hem de FedFST'nin tek seed'li sonuçlarının RNG akışına ne kadar duyarlı olduğunu gösteriyor. GPU'da GAT ve `index_add` atomik toplamalar kullandığı için Colab'daki koşular bit düzeyinde tekrarlanamaz; aynı karşılaştırma Colab'da isteğe bağlı olarak Bölüm 9.6'da çalıştırılabilir.
""")

md(r"""
### 6.4 Sağlama: POWER
**`edge_masking`** (Bölüm 2.1): aşağıdaki hücre POWER'ın kendi fonksiyonunu küçük bir örnekte çalıştırıyor. İki "görev" var, {0, 1, 2} ve {3, 4, 5}; görevler arası kenarlar 0–5 ve 2–3.
""")

code(r"""
demo = '''
import sys, torch
sys.path.insert(0, sys.argv[1])
from utils import edge_masking            # POWER/utils.py
edge_index = torch.tensor([[0,1,1,2,2,3,3,4,4,5,0,5],[1,0,2,1,3,2,4,3,5,4,5,0]])
handled = torch.tensor([True, True, True, False, False, False])   # görev 1'in düğümleri (task_data'daki gibi bool maske)
out = edge_masking(edge_index, handled=handled, device=torch.device("cpu"))
inter = sorted({(int(a), int(b)) for a, b in out.t() if (int(a) < 3) != (int(b) < 3)})
print("girdi kenarı:", edge_index.size(1), "| çıktı kenarı:", out.size(1))
print("çıktıda kalan görevler arası kenarlar:", inter)
print("self-loop eklenen düğümler:", sorted({int(a) for a, b in out.t() if a == b}))
'''
print(subprocess.run([sys.executable, "-c", demo, str(POWER_DIR)], capture_output=True, text=True).stdout)
""")

md(r"""
**Orijinal betikle Cora.** POWER yerelde kendi orijinal betik yoluyla bir kez çalıştırıldı (`optuna_search.py`'nin `__main__` bloğu ve `objective()`'i, Optuna olmadan, `config.py` varsayılanlarıyla, tek deneme; OpenFGL veri katmanı, CPU; betik: `tools/power_original_once.py`):

| Koşu (POWER'ın kendi veri hattı, seed 2025) | Görev sıraları | AA | AF |
|---|---|---|---|
| Varsayılan değerlendirme (yerel modeller) | istemciye göre karışık: [4,5]⇒[2,3]⇒[0,1], [4,5]⇒[2,3]⇒[0,1], [2,3]⇒[0,1]⇒[4,5] | 64.07 | 8.14 |
| Global model değerlendirmesi (`global_eval`) | aynı | 57.49 | 2.62 |
| Varsayılan değerlendirme, görev sırası tüm istemcilerde aynı (`tools/power_original_sync_order.py`) | [0,1]⇒[2,3]⇒[4,5] | 30.75 | 1.90 |

Üçüncü koşuda görev 1 ve görev 2'nin doğruluğu, kendi eğitimlerinin hemen ardından %0 (doğruluk matrisi: [[96.05, 0, 0], [91.67, 0, 0], [18.86, 73.39, 0]]). Yani β = 0.01 iken tüm istemciler aynı anda yeni sınıflara geçince yeni görev 10 round içinde öğrenilemiyor. Ortak protokolde de aynı durum görülüyor (Bölüm 9 ve 10). Bu, adaptörden değil yöntemin varsayılan ayarı ile görev senaryosunun etkileşiminden kaynaklanıyor.
""")

# =====================================================================================================
md(r"""
## 7. Veri setleri
Veri setleri PyTorch Geometric ile yükleniyor. Ham dosyalar, PyG'nin kullandığı GitHub depolarından (`raw.githubusercontent.com`) Drive'daki `data/` klasörüne indiriliyor; Squirrel için PyG'nin sabitlediği geom-gcn commit'i kullanılıyor. Bölümleme ve görevler, **FedFST'nin kendi fonksiyonlarıyla** ve `main.py`'deki sırayla üretiliyor:

`set_seed` → `load_dataset` → `louvain_partitioner` (python-louvain, 3 istemci) → sınıf sırasının seed'le karıştırılması (`load_server_clients`) → her istemcide `class_to_task` (sınıf başına 0.6/0.2/0.2 eğitim/doğrulama/test).

Sonuç her (veri seti, seed) için bir kez `cache/<veri>_seed<s>.pt` dosyasına yazılıyor ve üç yöntem de buradan okuyor. Görev çizgesi, FedFST'deki gibi yalnızca görevin sınıflarındaki düğümler arasındaki kenarları içeriyor (görevler arası kenar yok). Artan sınıf atılıyor: `out_dim = (C // k)·k`. Minesweeper 2 sınıflı olduğu için görev başına 1 sınıf kullanılıyor (2 görev, her görevde tek sınıf). Bu yapay ve zor bir senaryo: her görevde model "her düğüm bu sınıftandır" demeyi öğrenebiliyor ve sınıf-artımlı değerlendirmede AA kolayca 50 civarına kilitleniyor (Bölüm 10.1).
""")

code(r"""
ALL_DATASETS = ["cora", "citeseer", "actor", "squirrel", "minesweeper"]
DATA_SEED = 24      # FedFST varsayılanı; CONFIG'de başka seed'ler seçilirse cache'leri koşucu ayrıca üretir
for ds in ALL_DATASETS:
    r = subprocess.run([sys.executable, "-m", "fgcl_bench.data", "--fedfst", str(FEDFST), "--work_dir", str(WORK),
                        "--dataset", ds, "--seed", str(DATA_SEED)], cwd=BASE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise RuntimeError(f"{ds} hazırlanamadı")
    print(r.stdout.strip().splitlines()[-1])
""")

code(r"""
import pandas as pd
from IPython.display import display, Markdown
from fgcl_bench.data import load_cache
from fgcl_bench.common import DATASETS as DS_INFO

rows, task_rows = [], []
for ds in ALL_DATASETS:
    m = load_cache(WORK, ds, DATA_SEED)["meta"]
    st = m["stats"]
    rows.append({"Veri seti": DS_INFO[ds]["title"], "Düğüm": st["nodes"], "Kenar": st["edges"],
                 "Öznitelik": st["features"], "Sınıf": st["classes"], "Kenar homofilisi": round(st["edge_homophily"], 3),
                 "Sınıf/görev": m["per_task"], "Görev": m["num_tasks"],
                 "Görev sınıfları (orijinal id)": " ⇒ ".join(str(c) for c in m["task_classes"]),
                 "Atılan sınıf": ", ".join(map(str, m["dropped_classes"])) or "—",
                 "İstemci düğümleri": " / ".join(str(c["nodes"]) for c in st["clients"])})
    for cid, c in enumerate(st["clients"]):
        for t, task in enumerate(c["tasks"]):
            task_rows.append({"Veri seti": DS_INFO[ds]["title"], "İstemci": cid, "Görev": t + 1,
                              "Görev düğümü": task["nodes"], "Eğitim": task["train"], "Doğrulama": task["valid"],
                              "Test": task["test"], "Görev içi kenar": task["edges"]})
display(Markdown(f"**Veri setleri ve görevler (seed {DATA_SEED})**"))
display(pd.DataFrame(rows).set_index("Veri seti"))
display(Markdown("**İstemci × görev düğüm sayıları**"))
task_df = pd.DataFrame(task_rows)
task_df["Veri seti"] = pd.Categorical(task_df["Veri seti"], [DS_INFO[d]["title"] for d in ALL_DATASETS], ordered=True)
display(task_df.set_index(["Veri seti", "İstemci", "Görev"]).unstack("Görev").astype("Int64"))
""")

# =====================================================================================================
md(r"""
## 8. Deneyler
### 8.1 CONFIG
Koşu sırası: önce Cora'da üç yöntem (elde hemen tam bir karşılaştırma olsun diye), sonra CiteSeer, Actor, Squirrel ve Minesweeper. Ortak bütçe görev başına 10 communication round ve round başına 3 yerel epoch (tam-çizge). Bir round, her istemcinin yerel epoch'ları ile sunucudaki birleştirmeden oluşur. Görev başına 15 round'un üstü kabul edilmiyor.
""")

code(r"""
# ============================== CONFIG ==============================
DATASETS = ["cora", "citeseer", "actor", "squirrel", "minesweeper"]   # çalıştırma sırası
METHODS = ["fedfst", "power", "fedspa"]
ROUNDS = 10            # görev başına communication round (en fazla 15)
LOCAL_EPOCHS = 3       # round başına yerel epoch
SEEDS = [24]           # FedFST varsayılanı
SMOKE = False          # True: 1 round, 1 epoch, FedFST jeneratör/KD 2/3 epoch, POWER L-BFGS 3 iterasyon, yalnızca Cora
RUN_TIMEOUT_MIN = 90   # tek bir koşu için üst sınır (dakika)
# ====================================================================
""", tags=["parameters"])

code(r"""
assert ROUNDS <= 15, "Görev başına en fazla 15 communication round."
if SMOKE:
    DATASETS = ["cora"]
RESULTS_DIR = WORK / ("results_smoke" if SMOKE else "results")
LOG_DIR = WORK / ("logs_smoke" if SMOKE else "logs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RUNS = [(ds, m, s) for s in SEEDS for ds in DATASETS for m in METHODS]
print(f"{len(RUNS)} koşu | SMOKE={SMOKE} | sonuçlar: {RESULTS_DIR} | log'lar: {LOG_DIR}")
""")

md(r"""
### 8.2 Koşucu
Her koşu ayrı bir süreç. Çıktısının tamamı Drive'daki log dosyasına gidiyor; burada yalnızca bir özet satırı ve log'un son satırları görünüyor. Biten koşunun JSON'u Drive'a yazıldığı için hücre yeniden çalıştırılınca o koşu atlanıyor. Her koşudan sonra, o ana kadarki sürelerden kalan süre tahmin ediliyor.
""")

code(r"""
def run_name(ds, m, s):
    return f"{ds}__{m}__seed{s}"

def estimate_remaining(done_times, remaining):
    if not done_times:
        return None
    overall = sum(done_times.values()) / len(done_times)
    total = 0.0
    for ds, m, s in remaining:
        same = [t for (d2, m2, _), t in done_times.items() if m2 == m]
        total += sum(same) / len(same) if same else overall
    return total

done_times, failed = {}, []
for i, (ds, m, s) in enumerate(RUNS, 1):
    name = run_name(ds, m, s)
    out = RESULTS_DIR / f"{name}.json"
    if out.exists():
        r = json.loads(out.read_text())
        done_times[(ds, m, s)] = r["time_sec"]
        print(f"[{i}/{len(RUNS)}] {name}: daha önce bitmiş (AA={r['AA']:.2f}, AF={r['AF']:.2f}) -> atlandı")
        continue
    cmd = [sys.executable, "-u", "-m", "fgcl_bench.run", "--method", m, "--dataset", ds, "--seed", str(s),
           "--rounds", str(ROUNDS), "--local_epochs", str(LOCAL_EPOCHS), "--src_dir", str(SRC), "--work_dir", str(WORK)]
    if SMOKE:
        cmd.append("--smoke")
    log = LOG_DIR / f"{name}.log"
    t0 = time.time()
    with open(log, "w") as f:
        try:
            rc = subprocess.run(cmd, cwd=BASE, stdout=f, stderr=subprocess.STDOUT, timeout=RUN_TIMEOUT_MIN * 60).returncode
        except subprocess.TimeoutExpired:
            rc = "zaman aşımı"
    elapsed = time.time() - t0
    tail = log.read_text(errors="replace").strip().splitlines()[-3:]
    if rc == 0 and out.exists():
        r = json.loads(out.read_text())
        done_times[(ds, m, s)] = elapsed
        print(f"[{i}/{len(RUNS)}] {name}: AA={r['AA']:.2f} AF={r['AF']:.2f} | {elapsed/60:.1f} dk | {r['env']['device']}")
    else:
        failed.append(name)
        print(f"[{i}/{len(RUNS)}] {name}: HATA ({rc}) | log: {log}")
    for line in tail:
        print("    │ " + line[:150])
    rest = estimate_remaining(done_times, [x for x in RUNS[i:] if not (RESULTS_DIR / f"{run_name(*x)}.json").exists()])
    if rest:
        print(f"    tahmini kalan süre: ~{rest/60:.0f} dk")
print("\nBiten:", len(done_times), "| hatalı:", failed or "yok")
""")

code(r"""
# İlerleme tablosu (✓: sonuç var)
status = pd.DataFrame([{"Veri seti": DS_INFO[ds]["title"], "Seed": s, "Yöntem": m,
                        "Durum": "✓" if (RESULTS_DIR / f"{run_name(ds, m, s)}.json").exists() else "—"}
                       for ds, m, s in RUNS]).pivot_table(index=["Veri seti", "Seed"], columns="Yöntem",
                                                           values="Durum", aggfunc="first")
display(status)
""")

# =====================================================================================================
md(r"""
## 9. Sonuçlar
### 9.1 Sonuçların yüklenmesi ve maske kontrolü
Her JSON, koşunun beslediği görev maskelerinin hash'ini taşıyor. Aşağıdaki hücre, aynı (veri seti, seed) için üç yöntemin hash'lerinin aynı olduğunu ve cache'teki hash'le eşleştiğini doğruluyor.
""")

code(r"""
from fgcl_bench.common import METHODS as METHOD_TITLES

results = {}
for ds, m, s in RUNS:
    p = RESULTS_DIR / f"{run_name(ds, m, s)}.json"
    if p.exists():
        results[(ds, m, s)] = json.loads(p.read_text())
print(f"{len(results)}/{len(RUNS)} koşunun sonucu bulundu.")

for ds in DATASETS:
    for s in SEEDS:
        hashes = {m: results[(ds, m, s)]["mask_hash"] for m in METHODS if (ds, m, s) in results}
        if not hashes:
            continue
        cache_hash = load_cache(WORK, ds, s)["hash"]
        assert all(h == cache_hash for h in hashes.values()), f"{ds}/seed{s}: maske hash'leri farklı! {hashes}"
        print(f"{DS_INFO[ds]['title']:12s} seed {s}: {len(hashes)} yöntem aynı maskeleri kullandı (hash {cache_hash[:12]})")
""")

md(r"""
### 9.2 AA / AF tablosu
Her veri setinde en iyi değer kalın: AA için en yüksek, AF için en düşük. Birden fazla seed varsa ortalama ± standart sapma gösteriliyor. AF tek başına okunmamalı. Negatif AF, bir görevin doğruluğunun sonradan arttığını gösterir. Düşük AF ile birlikte görev sonu doğruluğu (köşegen) da düşükse bu "az unutma" değil, yeni görevin hiç öğrenilmediği anlamına gelir. Bölüm 9.5'teki tablo bu ayrımı gösteriyor.
""")

code(r"""
import numpy as np

def agg(ds, m, key):
    vals = [results[(ds, m, s)][key] for s in SEEDS if (ds, m, s) in results]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None

def cell(v, best):
    if v is None:
        return "—"
    txt = f"{v[0]:.2f}" + (f" ± {v[1]:.2f}" if v[2] > 1 else "")
    return f"**{txt}**" if best else txt

header = "| Veri seti | " + " | ".join(f"AA ↑ {METHOD_TITLES[m]}" for m in METHODS) + " | " + \
         " | ".join(f"AF ↓ {METHOD_TITLES[m]}" for m in METHODS) + " |"
lines = [header, "|" + "---|" * (1 + 2 * len(METHODS))]
for ds in DATASETS:
    aa = {m: agg(ds, m, "AA") for m in METHODS}
    af = {m: agg(ds, m, "AF") for m in METHODS}
    best_aa = max((v[0] for v in aa.values() if v), default=None)
    best_af = min((v[0] for v in af.values() if v), default=None)
    lines.append(f"| {DS_INFO[ds]['title']} | " +
                 " | ".join(cell(aa[m], aa[m] is not None and aa[m][0] == best_aa) for m in METHODS) + " | " +
                 " | ".join(cell(af[m], af[m] is not None and af[m][0] == best_af) for m in METHODS) + " |")
display(Markdown("\n".join(lines)))
""")

md(r"""
### 9.3 Doğruluk matrisleri
Satır: eğitimi biten görev; sütun: değerlendirilen görev. Hücreler test doğruluğunu (%) gösteriyor. Köşegen, bir görevin hemen ardından ölçülen doğruluk; son satır, akışın sonundaki durum.
""")

code(r"""
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
METHOD_COLORS = {"fedfst": "#2a78d6", "power": "#eb6834", "fedspa": "#1baf7a"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                     "#256abf", "#184f95", "#0d366b"])
SEQ.set_bad("#f0efec")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

seed0 = SEEDS[0]
ds_present = [ds for ds in DATASETS if any((ds, m, seed0) in results for m in METHODS)]
if ds_present:
    fig, axes = plt.subplots(len(ds_present), len(METHODS), figsize=(3.1 * len(METHODS), 2.7 * len(ds_present)),
                             squeeze=False, constrained_layout=True)
    im = None
    for r_i, ds in enumerate(ds_present):
        for c_i, m in enumerate(METHODS):
            ax = axes[r_i][c_i]
            res_ = results.get((ds, m, seed0))
            if res_ is None:
                ax.set_axis_off()
                continue
            A = np.array(res_["acc_matrix"], dtype=float)
            T = A.shape[0]
            A_masked = np.where(np.tril(np.ones_like(A)) > 0, A, np.nan)
            im = ax.imshow(A_masked, cmap=SEQ, vmin=0, vmax=100)
            for i in range(T):
                for j in range(i + 1):
                    ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if A[i, j] >= 55 else INK)
            ax.set_xticks(range(T), [f"G{j + 1}" for j in range(T)])
            ax.set_yticks(range(T), [f"G{j + 1}" for j in range(T)])
            ax.set_title(f"{DS_INFO[ds]['title']} · {METHOD_TITLES[m]}\nAA {res_['AA']:.1f} · AF {res_['AF']:.1f}",
                         fontsize=9)
            if r_i == len(ds_present) - 1:
                ax.set_xlabel("değerlendirilen görev")
            if c_i == 0:
                ax.set_ylabel("eğitimi biten görev")
            for side in ax.spines.values():
                side.set_visible(False)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="test doğruluğu (%)")
    fig.savefig(WORK / ("acc_matrices_smoke.png" if SMOKE else "acc_matrices.png"), dpi=150)
    plt.show()
""")

md(r"""
### 9.4 AA ve AF karşılaştırması
İki ölçü farklı yönlerde iyidir (AA ↑, AF ↓), bu yüzden ayrı panellerde gösteriliyor.
""")

code(r"""
if ds_present:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    x = np.arange(len(ds_present))
    width = 0.26
    for ax, key, label in [(axes[0], "AA", "AA (%) ↑"), (axes[1], "AF", "AF (%) ↓")]:
        for k, m in enumerate(METHODS):
            vals = [agg(ds, m, key)[0] if agg(ds, m, key) else np.nan for ds in ds_present]
            ax.bar(x + (k - 1) * width, vals, width, color=METHOD_COLORS[m], edgecolor=SURFACE, linewidth=1.5,
                   label=METHOD_TITLES[m], zorder=3)
        ax.axhline(0, color=AXIS, linewidth=1, zorder=4)
        ax.set_xticks(x, [DS_INFO[ds]["title"] for ds in ds_present])
        ax.set_ylabel(label)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.15))
    fig.savefig(WORK / ("aa_af_smoke.png" if SMOKE else "aa_af.png"), dpi=150)
    plt.show()
""")

md(r"""
### 9.5 Ayrıntılı göstergeler ve süreler
* **Görev sonu doğruluğu (ort. köşegen):** her görevin, kendi eğitimi biter bitmez ölçülen doğruluğu. Yeni bilgiyi öğrenebilme (plastisite) göstergesi.
* **Eski görevler (son):** akış bitince önceki görevlerin ortalama doğruluğu. Kalıcılık (stabilite) göstergesi.
* **Son görev:** son görevin doğruluğu.
""")

code(r"""
detail = []
for (ds, m, s), r in sorted(results.items(), key=lambda kv: (DATASETS.index(kv[0][0]), METHODS.index(kv[0][1]), kv[0][2])):
    A = np.array(r["acc_matrix"], dtype=float)
    T = A.shape[0]
    detail.append({"Veri seti": DS_INFO[ds]["title"], "Yöntem": METHOD_TITLES[m], "Seed": s,
                   "AA": round(r["AA"], 2), "AF": round(r["AF"], 2),
                   "Görev sonu doğruluğu (ort. köşegen)": round(float(np.mean(np.diag(A))), 2),
                   "Eski görevler (son)": round(float(np.mean(A[T - 1, :T - 1])), 2) if T > 1 else np.nan,
                   "Son görev": round(float(A[T - 1, T - 1]), 2),
                   "Süre (dk)": round(r["time_sec"] / 60, 2),
                   "Görev süreleri (sn)": " / ".join(f"{t:.0f}" for t in r.get("task_seconds", [])),
                   "Cihaz": r["env"]["device"]})
detail_df = pd.DataFrame(detail).set_index(["Veri seti", "Yöntem", "Seed"])
display(detail_df)
if detail:
    total_min = sum(r["time_sec"] for r in results.values()) / 60
    display(Markdown(f"**Toplam deney süresi:** {total_min:.1f} dk ({len(results)} koşu)"))
    print("\nOtomatik özet (Bölüm 10'daki yorumlar bu sayılarla birlikte okunmalı):")
    for ds in ds_present:
        aa = {m: agg(ds, m, "AA")[0] for m in METHODS if agg(ds, m, "AA")}
        af = {m: agg(ds, m, "AF")[0] for m in METHODS if agg(ds, m, "AF")}
        best_aa = max(aa, key=aa.get)
        best_af = min(af, key=af.get)
        print(f"  {DS_INFO[ds]['title']:12s} en yüksek AA: {METHOD_TITLES[best_aa]} ({aa[best_aa]:.2f}) | "
              f"en düşük AF: {METHOD_TITLES[best_af]} ({af[best_af]:.2f})")
""")

md(r"""
### 9.6 İsteğe bağlı sağlama: Colab'da harness ile `main.py`
`RUN_MAIN_PY_CHECK = True` yapılırsa yamalı orijinal `main.py`, jeneratör eğitimi tek iş parçacığına indirilerek Cora'da (seed 24, varsayılan ayarlar) çalıştırılır ve sonucu harness'in sonucuyla karşılaştırılır. Süre, FedFST'nin Cora koşusu kadardır. Bölüm 6.3'teki yerel karşılaştırma CPU'da birebir aynı sonucu verdi. GPU'da atomik toplamalar nedeniyle küçük farklar beklenebilir.
""")

code(r"""
RUN_MAIN_PY_CHECK = False
if RUN_MAIN_PY_CHECK and not SMOKE:
    log = WORK / "logs" / "sanity_main_py_cora.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        subprocess.run([sys.executable, "-m", "fgcl_bench.run_main_sequential", str(FEDFST), "--dataset_name", "cora",
                        "--seed", "24", "--device_id", "0", "--dataset_dir", str(WORK / "data"),
                        "--save_dir", str(BASE / "main_py_plots")], cwd=BASE, stdout=f, stderr=subprocess.STDOUT, check=True)
    txt = log.read_text()
    main_aa = float(re.search(r"Final AA: ([-\d.]+)", txt).group(1))
    main_af = float(re.search(r"Final AF: ([-\d.]+)", txt).group(1))
    print(f"main.py (sıralı jeneratör): AA={main_aa:.4f} AF={main_af:.4f}")
    if ("cora", "fedfst", 24) in results:
        r = results[("cora", "fedfst", 24)]
        print(f"harness:                    AA={r['AA']:.4f} AF={r['AF']:.4f}")
else:
    print("Atlandı (RUN_MAIN_PY_CHECK = False). Yerel CPU karşılaştırması Bölüm 6.3'te.")
""")

# =====================================================================================================
md(r"""
## 10. Tartışma
Aşağıdaki yorumlar Bölüm 9'daki tablolara ve kodu okurken gördüklerime dayanıyor. Bölüm 9.5'teki otomatik özet bu koşunun sayılarını veriyor; metindeki bir genelleme bu sayılarla çelişirse sayılar esas alınmalı. Yerel CPU testlerinde (aynı kod, aynı seed) gördüğüm değerleri "yerel testte" diye belirttim. Colab'daki GPU koşuları, atomik toplamalar ve FedFST'nin RNG duyarlılığı nedeniyle bunlardan farklı çıkabilir.

### 10.1 Homofilik ve heterofilik veri setleri
* **Cora ve CiteSeer** yüksek homofililidir: kenar homofilisi sırasıyla ≈0.81 ve ≈0.74 (Bölüm 7). GAT'ın komşu toplaması burada işe yarıyor ve görev sonu doğrulukları yüksek. Yerel testte FedFST bu iki veri setinde en yüksek AA'yı verdi. FedSPA'ya karşı farkı eski görevleri korumasından (son satırdaki eski görev doğrulukları), POWER'a karşı farkı ise yeni görevleri öğrenebilmesinden (köşegen) geliyordu. Bölüm 9.5'teki "Görev sonu doğruluğu" ve "Eski görevler (son)" sütunları bu ayrımı Colab sonuçları için gösteriyor.
* **Actor ve Squirrel** heterofiliktir (≈0.22): komşular çoğunlukla farklı sınıftan. Mesaj geçirme ayırt edici bilgiyi taşımakta zorlanıyor ve görev sonu doğrulukları da düşük. FedFST'nin $S_{high}$ eşleştirmesi tam bu yüksek frekans bilgisini sentetik tekrar çizgesinde korumaya yönelik. FedSPA'nın homofilik/heterofilik/bilinmeyen kanal ayrımı da heterofiliye yönelik, ama unutmayı önlemiyor.
* **Minesweeper**'ın kenar homofilisi ≈0.68 görünüyor. Ancak sınıflar %80 / %20 dengesiz olduğu için rastgele kenarlarda bile beklenen kenar homofilisi $0.8^2+0.2^2=0.68$'dir. Ölçülen değer bu taban çizgisine çok yakın; yani etiketler komşulukla neredeyse ilişkisiz ve veri seti pratikte heterofilik sayılmalı. Görev başına tek sınıf olduğu için sınıf-artımlı değerlendirmede iki uç durum var. Model yalnızca son sınıfı tahmin ederse AA = 50 ve AF = 100 olur (tam unutma). Yalnızca ilk sınıfı tahmin etmeye devam ederse yine AA = 50, ama AF = 0 olur (yeni görev hiç öğrenilmemiş). Bu nedenle Minesweeper'da AA tek başına yanıltıcıdır; doğruluk matrisiyle birlikte okunmalı.

### 10.2 Unutma analizi
* **FedSPA** unutmaya karşı bir mekanizma içermiyor. Her görev yalnızca yeni sınıfların etiketleriyle eğitildiği için çıkış katmanı yeni sınıflara kayıyor ve eski görevlerin doğruluğu çöküyor. Yerel testte Cora'da köşegen %89–99 iken son satırdaki eski görev doğrulukları 0'dı (AF ≈ 96). HBDA, istemciler arasındaki homofili farkına göre ağırlık veriyor; zamandaki (görevler arası) kaymayı ele almıyor. FedSPA'nın yüksek AF'si, POWER makalesindeki "FL/FGL fine-tuning" yöntemlerinden beklenen davranış.
* **POWER**, varsayılan β = 0.01 ile mevcut görevin kaybına yalnızca 0.01 ağırlık veriyor. Ortak protokolde tüm istemciler aynı anda yeni sınıflara geçiyor ve yerel testte yeni görevler 10 round içinde neredeyse hiç öğrenilemedi (görev sonu doğrulukları ≈ 0). Bu durumda düşük AF "az unutma" anlamına gelmiyor; yeni bilgi hiç gelmemiş oluyor. Cora'da AF negatif bile çıktı: görev 2'nin doğruluğu görev 3 sırasında 0'dan yükseldi. O sırada görev 2'nin sınıfları yalnızca tampondaki tekrar düğümleri ve sunucudaki sözde prototipler üzerinden eğitiliyordu. Bunun adaptörden değil, β ile görev senaryosunun etkileşiminden kaynaklandığını POWER'ın kendi betiğiyle doğruladım (Bölüm 6.4). Orijinal veri hattında istemcilerin görev sıraları karışıkken Cora'da AA 64.07 (yerel modeller) / 57.49 (global model) çıktı; sıra tüm istemcilerde aynı yapılınca yeni görev doğrulukları 0'a düştü.
* **FedFST**, sentetik tekrar ve düşük frekanslı damıtmayla eski görevleri koruyor. KD aşamasındaki doğrulama tabanlı model seçimi, eski ve yeni görev doğrulukları arasında bir denge noktası seçiyor. Bununla birlikte iki noktaya dikkat etmek gerek. Birincisi, istemciler geçmiş görevlerin gerçek verisini kullanıyor (Bölüm 1.5). İkincisi, sonuçlar RNG'ye çok duyarlı: yerel testte thread'li orijinal `main.py` aynı seed'le iki kez çalıştırıldığında Cora'da AA 38.97 ve 80.63 verdi (Bölüm 6.3). Tek seed'li karşılaştırmada FedFST'nin AA'sı bu belirsizlikle birlikte okunmalı.

### 10.3 POWER makalesindeki sonuçlarla kıyas
POWER makalesinde ([arXiv:2411.18919](https://arxiv.org/abs/2411.18919)) raporlanan AM/FM değerleri: Cora 65.74 / 28.10, CiteSeer 54.47 / 28.94, Squirrel 40.27 / 35.24. Bu değerler bizim protokolümüzle birebir karşılaştırılamaz:
* **Bölme:** POWER'ın veri katmanı Cora ve CiteSeer'de 0.2/0.4/0.4, Squirrel'de 0.48/0.32/0.2 eğitim/doğrulama/test oranı kullanıyor (`data.py`). Biz FedFST'nin 0.6/0.2/0.2 bölmesini kullandık.
* **İstemciler ve bölümleme:** makalede Squirrel deneyleri 10 istemciyle yapılmış ve Louvain bölümlemesi OpenFGL'in (scikit-network) uygulamasıyla yapılıyor. Biz her veri setinde FedFST'nin `louvain_partitioner`'ı (python-louvain) ile 3 istemci kullandık.
* **Senaryo ve değerlendirme:** POWER'da görev sırası istemciden istemciye karışık ve değerlendirme varsayılan olarak yerel modelle yapılıyor. Makaledeki değerler 10 koşunun ortalaması (Ek A.3) ve keşif hiperparametreleri Optuna ile aranmış.

Yerel sağlama koşusunda POWER'ın orijinal betiği (tek koşu, `config.py` varsayılanları, CPU) Cora'da AA 64.07 / AF 8.14 verdi. AA, makaledeki 65.74'e yakın; AF ise farklı.

### 10.4 Kısıtlar
* **Tek seed ve görev başına 10 round.** FedFST'nin RNG duyarlılığı düşünüldüğünde, yöntemler arasındaki farklar istatistiksel olarak sınanmadı. CONFIG'deki `SEEDS` listesi genişletilerek deney tekrarlanabilir; süre seed sayısıyla doğrusal artar.
* **Minesweeper'da görev başına tek sınıf.** Yapay ve zor bir senaryo (10.1).
* **FedSPA'nın uyarlanması.** FedSPA bir FGL yöntemi; FGCL'ye ince ayarla uyarlandı. Ortak bütçe için optimizer değiştirildi: SGD lr 0.1 ve 200 round yerine Adam lr 0.01, wd 5e-4, görev başına 10 round × 3 adım. Her görevde komşuluk matrisleri, $S_{high}$ ve HBDA'nın homofili ağırlıkları görev alt-çizgisinden yeniden hesaplandı. FedSPA kendi omurgasını (ACM-SPA, 128 gizli birim, dropout 0.1) kullanıyor; FedFST ve POWER ise 2 katmanlı GAT (64 gizli birim, dropout 0.5) kullanıyor.
* **DGL → PyTorch.** Düşük dereceli çizgelerde bit düzeyinde aynı sonuç; yüksek dereceli düğümlerde toplama sırasından kaynaklanan, göreli olarak en çok ≈2·10⁻⁵'lik fark (Bölüm 6.2).
* **Senaryo farkları.** POWER ve FedFST'nin kendi senaryoları (görev sırası, bölme, değerlendirilen model) ortak protokol için değiştirildi. Yöntemler kendi en iyi koşullarında değil, aynı koşullarda karşılaştırıldı.
* **Tekrarlanabilirlik.** GPU'da GAT ve `index_add` atomik toplamalar kullandığı için Colab koşuları bit düzeyinde tekrarlanamaz; CPU'da tekrarlanabilir.
""")

md(r"""
## 11. Kaynakça
1. H. Guo, Z. Tan, W. Huang, B. Yang, M. Ye. *FedFST: Mitigating Spectral Catastrophic Forgetting in Federated Graph Continual Learning.* SIGKDD 2026. Kod: https://github.com/YunQi572/FedFST (commit `d83fac0`).
2. Y. Zhu, M. Hu, D. Wu. *Federated Continual Graph Learning.* KDD 2025. https://arxiv.org/abs/2411.18919 ; DOI 10.1145/3711896.3736956. Kod ve teknik rapor: https://zenodo.org/records/15487655 (DOI 10.5281/zenodo.15487654).
3. Z. Tan, G. Wan, W. Huang, H. Li, G. Zhang, C. Yang, M. Ye. *FedSPA: Generalizable Federated Graph Learning under Homophily Heterogeneity.* CVPR 2025, s. 15464–15475. Kod: https://github.com/OakleyTan/FedSPA (commit `d2b3ddb`).
4. P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio. *Graph Attention Networks.* ICLR 2018.
5. M. Fey, J. E. Lenssen. *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.
6. V. D. Blondel, J.-L. Guillaume, R. Lambiotte, E. Lefebvre. *Fast unfolding of communities in large networks.* J. Stat. Mech., 2008. Uygulama: python-louvain, https://github.com/taynaud/python-louvain.
7. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017 (FedAvg).
8. P. Sen, G. Namata, M. Bilgic, L. Getoor, B. Gallagher, T. Eliassi-Rad. *Collective Classification in Network Data.* AI Magazine, 2008 (Cora, CiteSeer).
9. Z. Yang, W. Cohen, R. Salakhutdinov. *Revisiting Semi-Supervised Learning with Graph Embeddings.* ICML 2016 (Planetoid sürümü).
10. H. Pei, B. Wei, K. C.-C. Chang, Y. Lei, B. Yang. *Geom-GCN: Geometric Graph Convolutional Networks.* ICLR 2020 (Actor ve Squirrel'in kullanılan ön işlenmiş sürümleri).
11. B. Rozemberczki, C. Allen, R. Sarkar. *Multi-scale Attributed Node Embedding.* Journal of Complex Networks, 2021 (Squirrel).
12. O. Platonov, D. Kuznedelev, M. Diskin, A. Babenko, L. Prokhorenkova. *A critical look at the evaluation of GNNs under heterophily: Are we really making progress?* ICLR 2023 (Minesweeper).
13. S. Luan, C. Hua, Q. Lu, J. Zhu, M. Zhao, S. Zhang, X.-W. Chang, D. Precup. *Revisiting Heterophily For Graph Neural Networks.* NeurIPS 2022 (ACM-GCN).
14. L. Zhu, Z. Liu, S. Han. *Deep Leakage from Gradients.* NeurIPS 2019 (gradyan ters çevirme).
15. F. Zhou, C. Cao. *Overcoming Catastrophic Forgetting in Graph Neural Networks with Experience Replay.* AAAI 2021 (ER-GNN, kapsama maksimizasyonu).
16. X. Li vd. *OpenFGL: A Comprehensive Benchmark for Federated Graph Learning.* arXiv:2408.16288. Kod: https://github.com/xkLi-Allen/OpenFGL (POWER'ın veri katmanı).
17. M. Wang vd. *Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks.* arXiv:1909.01315 (DGL).
""")

for i, cell in enumerate(cells):
    cell.id = f"cell-{i:03d}"   # deterministik kimlikler: yeniden üretimde diff yalnızca içerik değişikliklerini gösterir
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nb.metadata["accelerator"] = "GPU"
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
nbformat.validate(nb)
out = ROOT / "FedFST_WarmUp_Odev2.ipynb"
nbformat.write(nb, out)
print(f"{out} yazıldı: {len(cells)} hücre")
