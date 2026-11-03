# eGOT – Google Colab Ödevi

Bu klasördeki notebook, [NNeuralDynamics/eGOT](https://github.com/NNeuralDynamics/eGOT) reposunu (commit `3d37906`) Google Colab'da uçtan uca çalıştırır. eGOT, Graph-of-Thoughts tabanlı bir GraphRAG yöntemidir ([Sanda vd., *Bioinformatics* 2026, doi:10.1093/bioinformatics/btag216](https://doi.org/10.1093/bioinformatics/btag216)).

| Dosya | Açıklama |
|---|---|
| `eGOT_Odev.ipynb` | Teslim edilecek Colab notebook'u: Türkçe açıklamalar, kod ve değerlendirme. Çıktısız gelir; çıktılar Colab'da çalıştırınca oluşur. |

## Notebook neler içeriyor?

0. Tek ayar hücresi (`LLM_BASE_URL`, `LLM_MODEL`, anahtarın Secrets'taki adı, Neo4j modu, deney boyutu).
1. eGOT'un çalışma mantığı: KG kurulumu (builder) → Neo4j → hibrit, grafla genişletilen retrieval → Graph-of-Thoughts döngüsü → sentez. Hangi dosyanın ne yaptığı ve kodun kökeni (ai-knowledge-graph, Neo4j LLM Graph Builder).
2. Kurulum: commit'e sabitlenmiş klon, yalnızca kodun import ettiği paketler, Colab'a Neo4j Community 5.26 LTS + APOC (yedek: AuraDB Free) ve yerel dil modeli için Ollama 0.40.14.
3. Dil modeli: **Qwen2.5-7B-Instruct**, Colab'ın T4 GPU'sunda Ollama ile yerel ve **ücretsiz** çalışır; API anahtarı ya da kredi gerekmez. Seçimin gerekçesi ve ücretli alternatif (OpenAI `gpt-4o-mini`, ayar hücresinde üç değer değiştirilerek). Bağlantı ve JSON testi; sorun çıkarsa Türkçe bir açıklamayla durur.
4. Repo olduğu gibi çalışmıyor. 17 hata (H1–H17) tek tek **tetiklenip** görünür yamalarla düzeltilir: `ImportError`, sabit `cuda:2`, `print(exc_info=...)`, sabit Jetstream adresi, `src.` import'ları, LangChain 1.x, ragas, eksik `FORMAT_ANSWER_PROMPT`, sentezdeki tanımsız `client`, değerlendirme betikleri, veri biçimi, CSV üreticisinin sabit yolları.
5. MultiHop-RAG alt kümesi: 8 soru, `question_type`'lara dengeli, sabit tohum.
6. KG kurulumu, istatistikler, örnek üçlüler ve pyvis görselleştirmesi.
7. Repoda olmayan Neo4j yükleyicisi (batched `UNWIND`, yüklem → ilişki tipi), indeksler, APOC testi, retrieval sorgusunun doğrudan testi.
8. FastAPI sunucusu, retrieval sağlık testi, tek soruluk GoT demosu (thought → ara cevap → kaynak). Demonun cevabı kaydedilir ve Bölüm 9'da yeniden kullanılır.
9. Toplu çalıştırma: EM/F1 (repodaki `eval_multihopqa.py`), içerme, kanıt bulma, süre, ölçülen LLM çağrısı/token sayısı. GoT'suz modla soru soru karşılaştırma ve özet grafik. Geçici hatada soru başına bir otomatik yeniden deneme; tüm cevaplar düşünce süreçleriyle birlikte dosyaya yazılır.
10. Hata tablosu (dosya:satır) ve Colab'da karşılaşılacak ek hatalar için boş satırlar.
11. eGOT ile Microsoft GraphRAG'in teknik karşılaştırması (indeksleme, sorgu, maliyet, sınırlamalar). Bu çalıştırmanın ölçülen değerlerinden otomatik üretilen bir özet paragrafı içerir.
12. Kaynaklar.

## Nasıl çalıştırılır?

1. Notebook'u Colab'da açın:
   - doğrudan GitHub'dan: [Colab'da aç](https://colab.research.google.com/github/hasancomert/YZM313-assignments/blob/ccr-8e9383b3-eupmed/eGOT-Odev/eGOT_Odev.ipynb) (dal birleştirildikten sonra bağlantıdaki dal adını `main` yapın), **ya da**
   - `Dosya → Not defteri yükle` ile `eGOT_Odev.ipynb` dosyasını yükleyin.
   - Sonra `Dosya → Drive'a kopya kaydet` deyin ve kopyada çalışın. GitHub'dan açılan sürüm kaydedilmez; teslim edilecek olan Drive'daki kopyadır.
2. `Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU → Kaydet`. GPU, dil modelini yerel çalıştırmak için gerekir. GPU'suz bir çalışma zamanında notebook Bölüm 2.6'da durur ve ne yapılacağını yazar.
3. Varsayılan ayarlarla **hiçbir secret gerekmez**.
4. `Çalışma zamanı → Tümünü çalıştır`. Tahmini süre 30–60 dakikadır; gerçek süreler hücre çıktılarında yazar. Sekmeyi kapatmayın.
5. Neo4j'nin Colab'a kurulumu başarısız olursa Bölüm 0.1'de `NEO4J_MODE = "aura"` yapın. Ardından AuraDB Free bağlantı bilgilerini `NEO4J_URI`, `NEO4J_USERNAME` ve `NEO4J_PASSWORD` secret'ları olarak ekleyin.
6. Colab o an GPU vermezse (ücretsiz kota ya da yoğunluk), bir süre sonra yeniden deneyin. Ücretli bir API kullanmak isterseniz Bölüm 0.1'deki üç değeri değiştirin (ör. `"https://api.openai.com/v1"`, `"gpt-4o-mini"`, `"OPENAI_API_KEY"`) ve anahtarı Secrets'a ekleyin; o zaman CPU yeterlidir.

## Teslim öncesi kontrol listesi

- [ ] Notebook'un en üstüne ad, soyad ve öğrenci numarası yazıldı.
- [ ] Tüm hücreler Colab'da hatasız çalıştı ve çıktılar kaydedildi (`Dosya → Kaydet`).
- [ ] Bölüm 2.6'da model GPU'ya yüklendi, Bölüm 3.1'deki model testi, Bölüm 8.2'deki retrieval sağlık testi ve Bölüm 8.3'teki GoT demosu beklenen çıktıyı verdi.
- [ ] Bölüm 9–11'deki tablolar okundu ve yorumlar sayılarla çelişmiyor (gerekirse bir-iki cümle eklendi).
- [ ] Colab'da yeni bir hata çıktıysa Bölüm 10.1'deki boş tabloya (C1–C3) eklendi.
- [ ] Notebook, ödev PDF'inde yazan ders sorumlusu ve asistanın e-posta adresleriyle paylaşıldı.
- [ ] Son teslim: **4 Aralık 2026, saat 12:00 (gece)**.

## Test notu

Notebook bu repoya konmadan önce, Colab'ın yayımladığı paket listesine ([googlecolab/backend-info](https://github.com/googlecolab/backend-info)) göre kurulmuş bir ortamda baştan sona çalıştırıldı. Ortam şunları içeriyordu: Python 3.13.18, torch 2.11, transformers 5.22, sentence-transformers 5.7, langchain 1.4.7, openai 2.54. Test ortamında GPU yoktu.

- **Neo4j:** Test ortamından `dist.neo4j.org`'a erişilemedi. Bu yüzden Neo4j Community **5.26.32** ve APOC 5.26.32, Maven Central'daki resmî jar'lardan tarball düzeninde kuruldu. `local` kurulum hücresi bu pakete karşı çalıştırıldı: indirme, APOC'u `labs/`'tan `plugins/`'e kopyalama, `neo4j.conf`, `neo4j-admin dbms set-initial-password`, `neo4j start`. Colab'da gerçek tarball `dist.neo4j.org`'dan indirilecek; bu adres ve tarball'daki `bin/` betikleri test edilemedi. `aura` modu, çalışan bir Neo4j'ye bağlanarak test edildi; yükleyicinin başka etiketlerdeki verilere dokunmadığı doğrulandı.
- **Ollama (Bölüm 2.6):** Test ortamında GPU yoktu ve Ollama'nın model deposuna (`registry.ollama.ai`) erişilemiyordu. Bu yüzden notebook, Ollama 0.40.14 ile CPU'da, Hugging Face'teki Qwen2.5-3B-Instruct GGUF dosyasından oluşturulan bir modelle baştan sona çalıştırıldı (1 soru, GoT derinliği 1). Şunlar hatasız çalıştı: Ollama'nın GitHub'dan indirilip `/usr/local` altına kurulması, sunucunun başlatılması, modelin belleğe yüklenmesi, KG kurulumu, GoT ve GoT'suz mod, Bölüm 11.0'daki yerel model satırları. Notebook ikinci kez çalıştırıldığında Ollama yeniden kurulmadı ve cevaplar dosyadan okundu. Bu Ollama sürümünde bağlam penceresini aşan bir istek kesilmiyor, hatayla reddediliyor; pencere bu yüzden 16.384 token yapıldı. T4 GPU'da çalışma ve `qwen2.5:7b`'nin Ollama deposundan indirilmesi test edilemedi.
- **Qwen2.5-7B-Instruct ile tam deney:** Varsayılan deney (8 soru, GoT derinliği 2, GoT'suz karşılaştırma dahil), aynı model (`Qwen/Qwen2.5-7B-Instruct`) ile OpenAI-uyumlu bir API (Featherless) üzerinden baştan sona çalıştırıldı; model yalnızca ayar hücresindeki üç değer değiştirilerek seçildi. Bütün hücreler hatasız tamamlandı (toplam ~19 dakika). Her GoT sorusu teorik değer kadar (21) LLM çağrısı yaptı. Model, thought puanlama adımlarının birkaçında geçersiz JSON üretti; eGOT bu adımlarda varsayılan puanla devam etti. Bu sayı artık soru başına raporlanıyor (`json_errors`). Bu testte "içerme" ölçüsünün alt dize eşleştirdiği fark edildi ("no", "economy" içinde bulunuyordu); ölçü kelime düzeyine çevrildi.
- **Diğer LLM testleri:** Hat ayrıca eGOT'un prompt'larına biçimce doğru cevaplar veren OpenAI-uyumlu bir sahte sunucuyla ve `Qwen/Qwen2.5-14B-Instruct` ile (Featherless; 1–3 soru, derinlik 1–2) çalıştırıldı. Sahte sunucuyla elde edilen sayılar anlamsızdır; bunlar yalnızca kodun çalıştığını gösterir. OpenAI `gpt-4o-mini` ile çalıştırılamadı (test ortamında anahtar yoktu).
- Hücrelerin ikinci kez çalıştırılması da test edildi: Yamalar tekrar uygulanmıyor; builder, demo ve bütün tahminler dosyadan okunuyor (ikinci çalıştırmada yalnızca 3 küçük doğrulama çağrısı yapılıyor).
- Hata yolları da denendi: geçersiz API anahtarı, ulaşılamayan LLM adresi, yanlış Neo4j şifresi ve kapalı Neo4j için anlaşılır mesaj veriliyor; anahtar ve şifre çıktıda görünmüyor.
