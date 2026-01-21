import os
from dotenv import load_dotenv
load_dotenv()

# DİNAMİK DOSYA YOLU AYARLARI
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Veri Klasörleri
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "mevzuat")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

# Dosya Yolları (Mutlak Yollar)
DB_PATH = os.path.join(VECTOR_STORE_DIR, "chroma_db")
JSON_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_laws_final.json")
STATS_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_laws_final_stats.json")

# Data Seti Yolu
DATASET_PATH = os.path.join(DATA_DIR, "test_datasets", "bronze_dataset_final.json")

# Koleksiyon Adı
COLLECTION_NAME = "turkish_law_collection"

# MODEL AYARLARI
MODEL_NAME = "gemini-2.0-flash"
EVALUATOR_MODEL_NAME = "gpt-3.5-turbo"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# TEXT SPLITTING PARAMETRELERİ
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# RETRIEVER AYARLARI
RETRIEVER_K = 7 #5 - 9 - 13 test edilcek
TEMPERATURE = 0 # hukuk asistanı için 0 olmalı

# DATA INGESTION AYARLARI
SOURCE_MAPPING = {
    "1.5.1475.docx": "1475 Sayılı İş Kanunu",
    "1.5.2709.docx": "2709 Sayılı T.C. Anayasası",
    "1.5.3308.docx": "3308 Sayılı Mesleki Eğitim Kanunu",
    "1.5.4447.docx": "4447 Sayılı İşsizlik Sigortası Kanunu",
    "1.5.4857.docx": "4857 Sayılı İş Kanunu",
    "1.5.6098.docx": "6098 Sayılı Türk Borçlar Kanunu",
    "1.5.6331.docx": "6331 Sayılı İş Sağlığı ve Güvenliği Kanunu",
    "1.5.6356.docx": "6356 Sayılı Sendikalar ve Toplu İş Sözleşmesi Kanunu",
    "1.5.7036.docx": "7036 Sayılı İş Mahkemeleri Kanunu"
}

SOURCE_FORMAT_MAPPING = {
    "1.5.1475.docx": {
        "display_name": "1475 Sayılı İş Kanunu",
        "short_name": "İş Kanunu",
        "law_number": "1475"
    },
    "1.5.2709.docx": {
        "display_name": "T.C. Anayasası",
        "short_name": "Anayasa",
        "law_number": "2709"
    },
    "1.5.3308.docx": {
        "display_name": "3308 Sayılı Mesleki Eğitim Kanunu",
        "short_name": "Mesleki Eğitim Kanunu",
        "law_number": "3308"
    },
    "1.5.4447.docx": {
        "display_name": "4447 Sayılı İşsizlik Sigortası Kanunu",
        "short_name": "İşsizlik Sigortası Kanunu",
        "law_number": "4447"
    },
    "1.5.4857.docx": {
        "display_name": "4857 Sayılı İş Kanunu",
        "short_name": "İş Kanunu",
        "law_number": "4857"
    },
    "1.5.6098.docx": {
        "display_name": "6098 Sayılı Türk Borçlar Kanunu",
        "short_name": "Borçlar Kanunu",
        "law_number": "6098"
    },
    "1.5.6331.docx": {
        "display_name": "6331 Sayılı İş Sağlığı ve Güvenliği Kanunu",
        "short_name": "İSG Kanunu",
        "law_number": "6331"
    },
    "1.5.6356.docx": {
        "display_name": "6356 Sayılı Sendikalar ve Toplu İş Sözleşmesi Kanunu",
        "short_name": "Sendikalar Kanunu",
        "law_number": "6356"
    },
    "1.5.7036.docx": {
        "display_name": "7036 Sayılı İş Mahkemeleri Kanunu",
        "short_name": "İş Mahkemeleri Kanunu",
        "law_number": "7036"
    }
}

LEGAL_TERM_WEIGHTS = {
    "kıdem tazminatı": 5, "ihbar tazminatı": 5, "iş sözleşmesi": 5, "hizmet sözleşmesi": 5,
    "fazla çalışma": 5, "yıllık izin": 5, "sendika": 5, "toplu iş sözleşmesi": 5,
    "grev": 5, "lokavt": 5, "arabuluculuk": 5, "işe iade": 5, "haklı fesih": 5,
    "geçerli fesih": 5, "iş güvencesi": 5, "sendikal tazminat": 5, "mobbing": 5,
    "psikolojik taciz": 5, "işveren": 3, "işçi": 3, "ücret": 3, "çalışma süresi": 3,
    "tazminat": 3, "zamanaşımı": 3, "iş mahkemesi": 3, "eşitlik": 3, "ayrımcılık": 3,
    "fesih": 3, "bildirim süresi": 3, "deneme süresi": 3, "belirli süreli": 3,
    "belirsiz süreli": 3, "kısmi süreli": 3, "tam süreli": 3, "işyeri": 2,
    "işletme": 2, "sözleşme": 2, "hak": 2, "yükümlülük": 2,
}

STOP_WORDS = {
    "acaba", "ama", "ancak", "arada", "aslında", "ayrıca", "bana", "bazı", "belki",
    "ben", "benden", "beni", "benim", "beri", "bile", "bin", "bir", "biri", "birkaç",
    "birkez", "birçok", "birşey", "birşeyi", "biz", "bize", "bizden", "bizi", "bizim",
    "böyle", "böylece", "bu", "buna", "bunda", "bundan", "bunlar", "bunları", "bunların",
    "bunu", "bunun", "burada", "bütün", "çoğu", "çünkü", "da", "daha", "dahi", "de",
    "defa", "değil", "diğer", "diye", "doksan", "dokuz", "dolayı", "dolayısıyla", "dört",
    "edecek", "eden", "ederek", "edilecek", "ediliyor", "edilmesi", "ediyor", "eğer",
    "elli", "en", "etmesi", "etti", "ettiği", "ettiğini", "gibi", "göre", "halen",
    "hangi", "hata", "hepsi", "her", "herkes", "herkese", "herkesi", "herkesin", "hiç",
    "hiçbir", "için", "iki", "ile", "ilgili", "ise", "işte", "itibaren", "itibariyle",
    "kadar", "karşın", "katrilyon", "kendi", "kendilerine", "kendini", "kendisi",
    "kendisine", "kendisini", "kez", "ki", "kim", "kimden", "kime", "kimi", "kimse",
    "kırk", "milyar", "milyon", "mu", "mü", "mı", "nasıl", "ne", "neden", "nedenle",
    "nerde", "nerede", "nereye", "niçin", "niye", "o", "olan", "olarak", "oldu",
    "olduğu", "olduğunu", "olduklarını", "olmadı", "olmadığı", "olmak", "olması",
    "olmayan", "olmaz", "olsa", "olsun", "olup", "olur", "olursa", "oluyor", "on",
    "ona", "onlar", "onlardan", "onları", "onların", "onu", "onun", "orada", "oysa",
    "oyysa", "pek", "rağmen", "sadece", "sanki", "sayı", "sayılı", "sekiz", "seksen",
    "sen", "senden", "seni", "senin", "siz", "sizden", "sizi", "sizin", "sonra",
    "tarafından", "trilyon", "tüm", "üç", "üzere", "var", "vardı", "ve", "veya",
    "ya", "yani", "yapacak", "yapılan", "yapılması", "yapıyor", "yapmak", "yaptı",
    "yaptığı", "yaptığını", "yaptıkları", "yedi", "yerine", "yetmiş", "yine", "yirmi",
    "yok", "yoksa", "yüz", "zaten", "madde", "kanun", "fıkra", "bent", "hüküm", "tarihli"
}

LEGAL_TERM_MAPPINGS = {
    "sataşma": ["mobbing", "psikolojik taciz", "hakaret", "zorbalık", "kötü muamele"],
    "şeref ve namus": ["hakaret", "onur kırma", "küfür", "mobbing", "iftira"],
    "cinsel taciz": ["taciz", "sarkıntılık", "haklı fesih", "rahatsız etme", "cinsel istismar"],
    "ahlak ve iyi niyet": ["haklı fesih", "24/2", "tazminatlı çıkış", "yüz kızartıcı", "hırsızlık", "kavga", "güven sarsıcı"],
    "sağlık sebepleri": ["iş kazası riski", "meslek hastalığı", "haklı fesih", "raporlu", "sağlık raporu"],
    "fesih": ["işten çıkarma", "kovulma", "istifa", "ayrılma", "kapının önüne koyma", "iş akdinin sona ermesi"],
    "kıdem tazminatı": ["çıkış parası", "tazminat hesaplama", "yıllık tazminat", "kıdem hesabı"],
    "ihbar tazminatı": ["haber verme süresi", "bildirim süresi", "hemen çıkış", "ihbar öneli"],
    "ücret": ["maaş", "aylık", "yevmiye", "fazla mesai parası", "prim", "ikramiye", "ödeme"],
    "fazla çalışma": ["mesai", "fazla mesai", "overtime", "ek çalışma", "hafta sonu çalışma"],
    "yıllık izin": ["yıllık ücretli izin", "tatil", "izin hakkı", "dinlenme hakkı"],
    "belirli süreli": ["sözleşmeli personel", "geçici iş", "proje bazlı", "süreli sözleşme"],
    "belirsiz süreli": ["kadrolu", "daimi", "süresiz sözleşme"],
    "sendikal": ["sendika üyeliği", "sendikal tazminat", "sendikalı olma", "sendika hakkı"],
    "toplu iş sözleşmesi": ["TİS", "toplu sözleşme", "sendika sözleşmesi"],
    "grev": ["iş bırakma", "toplu eylem", "grev hakkı"],
    "verim": ["performans", "düşük performans", "verimsizlik", "yetersizlik", "hedef tutmama", "satış hedefi"],
    "savunma": ["ifade", "savunması alınmadan", "sözlü savunma", "savunma yazısı", "savunma hakkı"],
    "davranış": ["tutum", "hal ve hareketler", "uyumsuzluk", "kavga", "disiplin"],
    "işe iade": ["işe geri dönüş", "haksız fesih davası", "geçersiz fesih", "işe iade davası"],
    "arabuluculuk": ["zorunlu arabuluculuk", "dava şartı", "uzlaşma", "arabulucu"],
    "iş mahkemesi": ["iş davası", "işçi davası", "işveren davası"],
}

# RAG GENERATOR PROMPT (Asistanın kullanacağı prompt)
PROMPT_TEMPLATE = """
Sen Türk hukuku konusunda uzman bir Yapay Zeka Hukuk Asistanısın.

KURALLAR:
1. SADECE verilen BAĞLAM'daki bilgileri kullan
2. Bağlamda yoksa "Bu konuda elimdeki yasal metinlerde yeterli bilgi bulunmamaktadır" de
3. KISA ve ÖZ cevap ver (maksimum 3-4 cümle)
4. TÜRKÇE cevap ver

CEVAP FORMATI:
BÖLÜM 1 - AÇIKLAMA
[Soruyu 2-4 cümleyle açıkla. Madde numarası KULLANMA.]

BÖLÜM 2 - HUKUKİ DAYANAKLAR
- [Kanun Adı Madde X]

BAĞLAM: 
{context}

SORU: {question}

CEVAP (TÜRKÇE, KISA VE ÖZ):
"""

# LLM JUDGE PROMPT (Test aşamasında puanlama yapacak prompt)
ENHANCED_JUDGE_PROMPT = """
SİSTEM ROLÜ:
Sen, Türk Hukuku alanında uzmanlaşmış, son derece titiz ve objektif bir RAG (Retrieval-Augmented Generation) Denetçisisin. Görevin, sistemin ürettiği cevabı aşağıda belirtilen katı kriterlere göre puanlamaktır. Asla "kibar" olmaya çalışma, sadece teknik doğruluğu ve kanıta dayalılığı değerlendir.

GİRDİ VERİLERİ:
----------------
[SORU]: {question}
[REFERANS CEVAP]: {expected_answer}
[ZORUNLU UNSURLAR]: {must_include}
[ÖNERİLEN UNSURLAR]: {should_include}
[KULLANILAN CONTEXT]: {context}
[MODELİN CEVABI]: {actual_answer}
[KAYNAK]: {source}

DEĞERLENDİRME ALGORİTMASI (Adım Adım Uygula):

ADIM 1 - RETRIEVAL ANALİZİ:
- [KULLANILAN CONTEXT] soruyla alakalı bilgi içeriyor mu?
- Eğer context boşsa/alakasızsa VE model "Bu konuda yeterli bilgi bulunmamaktadır" gibi bir cevap verdiyse → Bu DOĞRU davranıştır, Faithfulness tam puan almalı.

ADIM 2 - HALLUCINATION KONTROLÜ (Faithfulness):
- [MODELİN CEVABI] içindeki HER İDDİAYI tek tek kontrol et
- Her iddianın [KULLANILAN CONTEXT] içinde karşılığı var mı?
- Context'te OLMAYAN bilgi varsa → Faithfulness < 0.5
- Tamamen uydurma bilgi varsa → Faithfulness < 0.3

ADIM 3 - HUKUKİ DOĞRULUK (Correctness):
- [MODELİN CEVABI] ile [REFERANS CEVAP] anlamsal olarak örtüşüyor mu?
- Kanun maddesi numaraları DOĞRU mu? (Md. 25 ≠ Md. 17)
- Hukuki terimler doğru kullanılmış mı? (fesih ≠ istifa)
- Süreler/rakamlar doğru mu? (14 gün ≠ 30 gün)

ADIM 4 - KAPSAM KONTROLÜ (Completeness):
- [ZORUNLU UNSURLAR] listesindeki kavramlar cevapta var mı?
- [ÖNERİLEN UNSURLAR] listesinden kaçı cevapta geçiyor?
- Kritik bilgi eksikliği var mı?

PUANLAMA REHBERİ (0.0 - 1.0 Arası):

| Skor | Anlam |
|------|-------|
| 0.9-1.0 | Mükemmel - Hata yok, tam ve doğru |
| 0.7-0.9 | İyi - Küçük eksiklikler var ama doğru |
| 0.5-0.7 | Orta - Önemli eksiklikler veya küçük hatalar |
| 0.3-0.5 | Zayıf - Ciddi hatalar veya eksiklikler |
| 0.0-0.3 | Başarısız - Yanlış, alakasız veya uydurma |

METRIK TANIMLAMALARI:
- relevance_score: Cevap soruya odaklı mı? Konu dağılmış mı?
- faithfulness_score: Cevap SADECE context'e mi dayanıyor? Dış bilgi var mı?
- correctness_score: Hukuki sonuç referans cevapla aynı mı?
- completeness_score: Kritik bilgiler tam mı?
- overall_score: Tüm faktörlerin ağırlıklı ortalaması (faith×0.3 + rel×0.25 + corr×0.25 + comp×0.2)

STRENGTHS/WEAKNESSES:
- strengths: Cevabın 2-3 güçlü yönünü kısa maddeler halinde yaz
- weaknesses: Cevabın 2-3 zayıf yönünü veya eksikliğini kısa maddeler halinde yaz

ÖNEMLİ KURALLAR:
1. Pozitiflik Yanlılığı (Positivity Bias) YAPMA. Hata varsa acımasızca puan kır.
2. "Kısmen doğru" ifadeleri hukukta TEHLİKELİDİR. Şüphe durumunda düşük puan ver.
3. Madde numarası yanlışsa → Correctness < 0.5
4. Context dışı bilgi varsa → Faithfulness < 0.5
5. Model "bilgi yok" deyip context gerçekten boşsa → Bu BAŞARI, cezalandırma

ÇIKTI:
Bu analize dayanarak JSON formatındaki skorları, güçlü/zayıf yönleri ve gerekçeni oluştur.
"""

TEST_CONFIG_DEFAULTS = {
    "delay_between_questions": 2.0,  
    "output_dir": "reports",
    "save_contexts": True,
    "verbose": True,
}
LATENCY_THRESHOLD_MS = 8000


TEMPLATES_DIR = os.path.join(SRC_DIR, "templates")


# LLM Judge vs Heuristic
SCORING_WEIGHTS = {
    "judge_weight": 0.70,      # LLM Judge kararı (%70)
    "heuristic_weight": 0.30,  # Matematiksel/Semantic hesaplamalar (%30)
}


JUDGE_SUBWEIGHTS = {
    "faithfulness": 0.30,  # En kritik: Hallucination kontrolü
    "relevance": 0.25,     # Soruyla ilgililik
    "correctness": 0.25,   # Hukuki doğruluk
    "completeness": 0.20,  # Cevap tamlığı
}


HEURISTIC_SUBWEIGHTS = {
    "semantic_correctness": 0.25,  # Anlamsal benzerlik (E5 model) - alternatif cevaplar dahil
    "quote_presence": 0.15,        # YENİ: Kaynak alıntı bulunma (retrieval kalitesi)
    "citation_accuracy": 0.20,     # Doğru madde/kaynak atıfı
    "answer_consistency": 0.18,    # Sayısal değer tutarlılığı
    "keyword_coverage": 0.12,      # Hibrit keyword eşleşme
    "response_quality": 0.10,      # Cevap kalitesi (uzunluk/yapı)
}

HYBRID_KEYWORD_WEIGHTS = {
    "exact_match": 0.40,   # Kritik terimler için tam eşleşme
    "semantic": 0.60,      # Genel anlam için semantic similarity
}

MUST_SHOULD_WEIGHTS = {
    "must_weight": 0.80,   # Zorunlu terimler daha önemli
    "should_weight": 0.20, # Önerilen terimler
}

TEST_THRESHOLDS = {
    # Genel geçer notu
    "pass_threshold": 0.70,
    
    # LLM Judge eşikleri
    "relevance_threshold": 0.60,      # İlgisiz cevaba tolerans yok
    "faithfulness_threshold": 0.75,   # Hallucination'a düşük tolerans
    
    # Heuristic eşikleri  
    "citation_threshold": 0.50,       # Atıf doğruluğu
    "consistency_threshold": 0.50,    # Sayısal tutarlılık
    "semantic_threshold": 0.40,       # Minimum anlamsal benzerlik
}

METRIC_WEIGHTS = {
    "citation_weight": 2.0,          # Yanlış atıf cezası ağır
    "keyword_weight": 1.2,           # Semantic search güvenilir
    "response_quality_weight": 0.5,  # Format daha az önemli
    "latency_weight": 0.1,           # Hız en az önemli
}