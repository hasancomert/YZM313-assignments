# FedFST – Google Colab Ödevi (WarmUp Ödev 2)

Bu klasörde, [YunQi572/FedFST](https://github.com/YunQi572/FedFST) reposunu (SIGKDD 2026) Google Colab'da çalıştıran ve FedFST'yi aynı veri setlerinde POWER ([KDD'25](https://arxiv.org/abs/2411.18919)) ve FedSPA ([CVPR'25](https://github.com/OakleyTan/FedSPA)) ile karşılaştıran notebook bulunur.

| Dosya / klasör | Açıklama |
|---|---|
| `FedFST_WarmUp_Odev2.ipynb` | **Teslim edilecek notebook.** Kendi başına çalışır: kodları indirir, yamaları uygular, harness dosyalarını `%%writefile` ile yazar, deneyleri çalıştırır, tabloları ve grafikleri üretir. Çıktıları temizlenmiş hâldedir. |
| `DEGISIKLIKLER.md` | Yapılan tüm değişikliklerin ve ortak protokolden sapmaların listesi |
| `fgcl_bench/` | Notebook'un `%%writefile` hücrelerindeki harness dosyalarının aynısı (incelemek için) |
| `tools/` | Notebook üreticisi (`build_notebook.py`), FedFST yamalarının tek kaynağı (`patches.py`) ve yerel sağlama betikleri |
| `yerel_test_kayitlari/` | Yerel testlerin çıktıları (Colab sonucu **değildir**) |

## Colab'da nasıl çalıştırılır?

1. Notebook'u Colab'da açın:
   - Colab'da `Dosya → Not defteri yükle` ile `FedFST_WarmUp_Odev2.ipynb` dosyasını yükleyin, **ya da**
   - doğrudan GitHub'dan açın: [Colab'da aç](https://colab.research.google.com/github/hasancomert/YZM313Odevleri/blob/ccr-797905e4-4sm9t1/FedFST-Odev2/FedFST_WarmUp_Odev2.ipynb). Repo gizliyse Colab GitHub yetkisi ister; dal birleştirildikten sonra bağlantıdaki dal adını `main` yapın.
   - Teslim için notebook'un **kendi Google Drive'ınızda** bir kopyası olmalı (`Dosya → Drive'a kopya kaydet`). Paylaşım bu kopya üzerinden yapılır.
2. `Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU`.
3. `Çalışma zamanı → Tümünü çalıştır`. İlk hücrelerden biri Google Drive erişim izni ister; izin verin. Sonuçlar, log'lar ve veri cache'i `MyDrive/FedFST_Odev2/` altına yazılır.
4. Hızlı bir deneme yapmak isterseniz önce Bölüm 8'deki CONFIG hücresinde `SMOKE = True` yapın (yalnızca Cora, 1 round, 1 epoch; birkaç dakika). Gerçek çalıştırma için `SMOKE = False` olmalı. SMOKE sonuçları ayrı bir klasöre (`results_smoke/`) yazılır ve gerçek sonuçlarla karışmaz.

### Tahmini süre

- **Kurulum, indirmeler, eşdeğerlik testi, veri hazırlığı:** ~3–5 dk.
- **Deneyler (15 koşu: 5 veri seti × 3 yöntem):** yerelde CPU'da (tek iş parçacığı) toplam eğitim süresi 15.9 dk ölçüldü (`yerel_test_kayitlari/cpu_tam_ayar_kosulari.txt`). En uzun koşular FedFST'nin CiteSeer (5.8 dk) ve Cora (2.7 dk) koşuları. T4'te süre tahminen 10–30 dk arasında olacaktır. Modeller küçük olduğu için GPU'nun kazancı sınırlı; çekirdek başlatma ve senkronizasyon maliyeti baskın. Gerçek süre, ilk koşudan sonra notebook'ta "tahmini kalan süre" olarak yazdırılır. GPU kotası yoksa her şey CPU'da çalışır; Colab'ın CPU'su daha yavaş olduğu için bu durumda ~30–45 dk beklenebilir.

### Koşu yarıda kalırsa

- Her koşu bitince sonucu Drive'a JSON olarak yazılır (`MyDrive/FedFST_Odev2/results/`). Notebook'u yeniden `Tümünü çalıştır` ile başlatmak yeterlidir: biten koşular atlanır, yarıda kalan koşu baştan başlar. Kodlar oturuma özel klasöre (`/content/fedfst_odev2`) yeniden indirilir; veri ve cache Drive'dan okunur.
- Hata veren bir koşu sonraki koşuları durdurmaz. Hücrenin sonunda hatalı koşular listelenir; ayrıntı `MyDrive/FedFST_Odev2/logs/<veri>__<yöntem>__seed24.log` dosyasındadır.
- POWER'ın kodu Zenodo'dan indirilir (md5 kontrolüyle). Zenodo'ya erişilemezse zip'i ([10.5281/zenodo.15487654](https://doi.org/10.5281/zenodo.15487654), dosya `zyl24/FCGL_POWER-official.zip`) elle indirip `MyDrive/FedFST_Odev2/FCGL_POWER-official.zip` olarak koyun ve hücreyi yeniden çalıştırın.

## Teslim öncesi kontrol listesi

- [ ] Notebook'un en üstüne ad, soyad ve öğrenci numarası yazıldı.
- [ ] Tüm hücreler Colab'da `SMOKE = False` ile hatasız çalıştı ve çıktılar notebook'ta kayıtlı (`Dosya → Kaydet`).
- [ ] Bölüm 9'daki tablo ve grafikler 5 veri seti × 3 yöntemin tamamını gösteriyor; 9.1'deki maske kontrolü geçti.
- [ ] Bölüm 9.5'teki otomatik özet okundu; Bölüm 10'daki yorumlar Colab'daki sayılarla çelişmiyor (gerekirse bir-iki cümle güncellendi).
- [ ] Bölüm 10.3'teki POWER makalesi değerleri (Cora 65.74/28.10, CiteSeer 54.47/28.94, Squirrel 40.27/35.24) ve "Squirrel'de 10 istemci" bilgisi makalenin sonuç tablosuyla bir kez daha karşılaştırıldı.
- [ ] `Paylaş` düğmesiyle **mustafacoskun566@gmail.com** (ders sorumlusu) ve **betulerkantarci@gmail.com** (ders asistanı) adreslerine erişim verildi.
- [ ] Son teslim: **21 Ekim 2026, saat 12:00 (gece)**.

## Yerel test notu

Notebook, Colab'ın güncel ortamına benzeyen bir ortamda (Python 3.13, torch 2.11, PyG 2.8.1; Hugging Face `datasets` kurulu, GPU yok) `papermill` ile baştan sona çalıştırıldı. `SMOKE = True` ile tüm hücreler hatasız çalıştı. Ardından `SMOKE = False` ile Cora'da üç yöntem gerçek ayarlarla çalıştırıldı. Beş veri setinin tamamı da üç yöntemle, tam ayarlarla harness üzerinden çalıştırıldı (15/15 koşu başarılı). Eşdeğerlik testleri ve sağlamalar geçti:

- FedFST vektörizasyonu: çıktı, gradyan ve RNG durumu orijinal kodla bit düzeyinde aynı.
- Harness'teki FedFST, yamalı orijinal `main.py` (jeneratör eğitimi sıralı) ile Cora'da birebir aynı AA/AF'yi veriyor (61.7726 / 32.7114).
- FedSPA'nın PyTorch port'u, DGL 2.1 + torch 2.2.1 ortamında orijinal DGL'li koda karşı test edildi.
- POWER, kendi orijinal betik yoluyla Cora'da bir kez çalıştırıldı.

Ayrıntılar notebook'un 6. bölümünde ve `yerel_test_kayitlari/` klasöründe. **Gerçek sonuçlar notebook Colab'da çalıştırıldığında üretilir;** GPU'daki atomik toplamalar nedeniyle yerel CPU sayılarından biraz farklı olabilir.
