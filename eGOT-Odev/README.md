# eGOT – Google Colab Ödevi

Bu klasördeki notebook, [NNeuralDynamics/eGOT](https://github.com/NNeuralDynamics/eGOT) reposunu (commit `3d37906`) Google Colab'da uçtan uca çalıştırır. eGOT, Graph-of-Thoughts tabanlı bir GraphRAG yöntemidir ([Sanda vd., *Bioinformatics* 2026, doi:10.1093/bioinformatics/btag216](https://doi.org/10.1093/bioinformatics/btag216)).

| Dosya | Açıklama |
|---|---|
| `eGOT_Odev.ipynb` | Teslim edilecek Colab notebook'u: Türkçe açıklamalar, kod ve değerlendirme. Çıktısız gelir; çıktılar Colab'da çalıştırınca oluşur. |

## Notebook neler içeriyor?

0. Tek ayar hücresi (`LLM_BASE_URL`, `LLM_MODEL`, anahtarın Secrets'taki adı, Neo4j modu, deney boyutu) ve Colab Secrets ile anahtar okuma.
1. eGOT'un çalışma mantığı: KG kurulumu (builder) → Neo4j → hibrit, grafla genişletilen retrieval → Graph-of-Thoughts döngüsü → sentez. Hangi dosyanın ne yaptığı ve kodun kökeni (ai-knowledge-graph, Neo4j LLM Graph Builder).
2. Kurulum: commit'e sabitlenmiş klon, yalnızca kodun import ettiği paketler, Colab'a Neo4j Community 5.26 LTS + APOC (yedek: AuraDB Free).
3. Dil modeli: **`gpt-4o-mini`** ve seçimin gerekçesi. Bağlantı testi; anahtar, kota (`insufficient_quota`) ya da model adı sorunlarında Türkçe ve anahtarı göstermeyen bir açıklamayla durur.
4. Repo olduğu gibi çalışmıyor. 17 hata (H1–H17) tek tek **tetiklenip** görünür yamalarla düzeltilir: `ImportError`, sabit `cuda:2`, `print(exc_info=...)`, sabit Jetstream adresi, `src.` import'ları, LangChain 1.x, ragas, eksik `FORMAT_ANSWER_PROMPT`, sentezdeki tanımsız `client`, değerlendirme betikleri, veri biçimi, CSV üreticisinin sabit yolları.
5. MultiHop-RAG alt kümesi: 8 soru, `question_type`'lara dengeli, sabit tohum.
6. KG kurulumu, istatistikler, örnek üçlüler ve pyvis görselleştirmesi.
7. Repoda olmayan Neo4j yükleyicisi (batched `UNWIND`, yüklem → ilişki tipi), indeksler, APOC testi, retrieval sorgusunun doğrudan testi.
8. FastAPI sunucusu, retrieval sağlık testi, tek soruluk GoT demosu (thought → ara cevap → kaynak). Demonun cevabı kaydedilir ve Bölüm 9'da yeniden kullanılır.
9. Toplu çalıştırma: EM/F1 (repodaki `eval_multihopqa.py`), içerme, kanıt bulma, süre, ölçülen LLM çağrısı/token sayısı, maliyet. GoT'suz modla soru soru karşılaştırma ve özet grafik. Geçici hatada soru başına bir otomatik yeniden deneme; tüm cevaplar düşünce süreçleriyle birlikte dosyaya yazılır.
10. Hata tablosu (dosya:satır) ve Colab'da karşılaşılacak ek hatalar için boş satırlar.
11. eGOT ile Microsoft GraphRAG'in teknik karşılaştırması (indeksleme, sorgu, maliyet, sınırlamalar). Bu çalıştırmanın ölçülen değerlerinden otomatik üretilen bir özet paragrafı içerir.
12. Kaynaklar.

## Nasıl çalıştırılır?

1. Notebook'u Colab'da açın:
   - `Dosya → Not defteri yükle` ile `eGOT_Odev.ipynb` dosyasını yükleyin, **ya da**
   - doğrudan GitHub'dan açın: [Colab'da aç](https://colab.research.google.com/github/hasancomert/YZM313-assignments/blob/ccr-8e9383b3-eupmed/eGOT-Odev/eGOT_Odev.ipynb). Repo gizliyse Colab GitHub yetkisi ister. Dal birleştirildikten sonra bağlantıdaki dal adını `main` yapın.
   - Teslim için notebook'un **kendi Google Drive'ınızda** bir kopyası olmalı (`Dosya → Drive'a kopya kaydet`).
2. `Çalışma zamanı → Çalışma zamanı türünü değiştir → CPU` (GPU gerekmez).
3. Sol menüdeki 🔑 **Secrets** bölümüne `OPENAI_API_KEY` ekleyin ve "Notebook access" iznini açın.
4. `Çalışma zamanı → Tümünü çalıştır`. Tahmini süre 20–40 dakikadır; gerçek süreler hücre çıktılarında yazar. Varsayılan ayarlarla gpt-4o-mini maliyetinin 1 doların altında kalması beklenir; gerçek maliyeti Bölüm 9.3 ölçülen token sayılarından hesaplar.
5. Neo4j'nin Colab'a kurulumu başarısız olursa Bölüm 0.1'de `NEO4J_MODE = "aura"` yapın. Ardından AuraDB Free bağlantı bilgilerini `NEO4J_URI`, `NEO4J_USERNAME` ve `NEO4J_PASSWORD` secret'ları olarak ekleyin.

## Teslim öncesi kontrol listesi

- [ ] Notebook'un en üstüne ad, soyad ve öğrenci numarası yazıldı.
- [ ] Tüm hücreler Colab'da hatasız çalıştı ve çıktılar kaydedildi (`Dosya → Kaydet`).
- [ ] Bölüm 3.1'deki model testi, Bölüm 8.2'deki retrieval sağlık testi ve Bölüm 8.3'teki GoT demosu beklenen çıktıyı verdi.
- [ ] Bölüm 9–11'deki tablolar okundu ve yorumlar sayılarla çelişmiyor (gerekirse bir-iki cümle eklendi).
- [ ] Colab'da yeni bir hata çıktıysa Bölüm 10.1'deki boş tabloya (C1–C3) eklendi.
- [ ] Notebook, ödev PDF'inde yazan ders sorumlusu ve asistanın e-posta adresleriyle paylaşıldı.
- [ ] Son teslim: **4 Aralık 2026, saat 12:00 (gece)**.

## Test notu

Notebook bu repoya konmadan önce, Colab'ın yayımladığı paket listesine ([googlecolab/backend-info](https://github.com/googlecolab/backend-info)) göre kurulmuş bir ortamda baştan sona çalıştırıldı. Ortam şunları içeriyordu: Python 3.13.18, torch 2.11, transformers 5.22, sentence-transformers 5.7, langchain 1.4.7, openai 2.54.

- **Neo4j:** Test ortamından `dist.neo4j.org`'a erişilemedi. Bu yüzden Neo4j Community **5.26.32** ve APOC 5.26.32, Maven Central'daki resmî jar'lardan tarball düzeninde kuruldu. `local` kurulum hücresi bu pakete karşı çalıştırıldı: indirme, APOC'u `labs/`'tan `plugins/`'e kopyalama, `neo4j.conf`, `neo4j-admin dbms set-initial-password`, `neo4j start`. Colab'da gerçek tarball `dist.neo4j.org`'dan indirilecek; bu adres ve tarball'daki `bin/` betikleri test edilemedi. `aura` modu, çalışan bir Neo4j'ye bağlanarak test edildi; yükleyicinin başka etiketlerdeki verilere dokunmadığı doğrulandı.
- **LLM:** Test ortamında OpenAI anahtarı yoktu ve `api.openai.com`'a erişilemiyordu, bu yüzden **gpt-4o-mini ile çalıştırılamadı**. Hattın tamamı iki yolla sınandı: (1) eGOT'un prompt'larına biçimce doğru cevaplar veren OpenAI-uyumlu bir sahte sunucuyla, (2) gerçek bir model olan `Qwen/Qwen2.5-14B-Instruct` ile, OpenAI-uyumlu bir API (Featherless) üzerinden kısa uçtan uca çalıştırmalarla (1–3 soru, GoT derinliği 1–2). İkinci test yalnızca ayar hücresindeki üç değer değiştirilerek yapıldı. Bu testlerde ölçülen LLM çağrısı sayısı teorik değerle (2 + 9·d + 1) aynı çıktı. Sahte sunucuyla elde edilen sayılar anlamsızdır; bunlar yalnızca kodun çalıştığını gösterir.
- Hücrelerin ikinci kez çalıştırılması da test edildi: Yamalar tekrar uygulanmıyor; builder, demo ve bütün tahminler dosyadan okunuyor (ikinci çalıştırmada yalnızca 3 küçük doğrulama çağrısı yapılıyor).
- Hata yolları da denendi: geçersiz API anahtarı, ulaşılamayan LLM adresi, yanlış Neo4j şifresi ve kapalı Neo4j için anlaşılır mesaj veriliyor; anahtar ve şifre çıktıda görünmüyor.
