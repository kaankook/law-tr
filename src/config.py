from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# PROJE DİZİN YAPISI
# ─────────────────────────────────────────────
SRC_DIR    = Path(__file__).parent.resolve()
ROOT_DIR   = SRC_DIR.parent

DATA_DIR         = ROOT_DIR / "data"
RAW_DATA_DIR     = DATA_DIR / "raw" / "mevzuat"
PROCESSED_DIR    = DATA_DIR / "processed"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
REPORTS_DIR      = ROOT_DIR / "reports"

DB_PATH           = str(VECTOR_STORE_DIR / "qdrant_db")
JSON_FILE_PATH    = str(PROCESSED_DIR / "chunks_final.json")
STATS_FILE_PATH   = str(PROCESSED_DIR / "chunks_stats.json")
MAPPING_FILE_PATH = str(DATA_DIR / "raw" / "mülga" / "mapping_registry.json")

DATASET_PATH = str(DATA_DIR / "test_datasets" / "bronze_dataset_final.json")

TEMPLATES_DIR = SRC_DIR / "templates"


# ─────────────────────────────────────────────
# MODEL AYARLARI
# ─────────────────────────────────────────────
COLLECTION_NAME      = "turkish_law_collection"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
SPARSE_MODEL_NAME    = "Qdrant/bm25"

MODEL_NAME           = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EVALUATOR_MODEL_NAME = os.getenv("EVALUATOR_MODEL", "gpt-4o-mini")

LLM_ENRICHMENT_MODEL       = "gpt-4o-mini"
LLM_ENRICHMENT_CONCURRENCY = 10
LLM_ENRICHMENT_RETRIES     = 3


# ─────────────────────────────────────────────
# CHUNKING PARAMETRELERİ
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class ChunkConfig:
    target_size: int      = 2000
    overlap_chars: int    = 0
    min_chunk_chars: int  = 100
    header_repeat: bool   = True

CHUNK_CFG = ChunkConfig()

CHUNK_SIZE    = CHUNK_CFG.target_size
CHUNK_OVERLAP = CHUNK_CFG.overlap_chars


# ─────────────────────────────────────────────
# RAG PİPELİNE & RETRIEVAL AYARLARI
# ─────────────────────────────────────────────
TEMPERATURE             = 0.1
MIN_CHUNK_CHARS         = 1
ENABLE_NEIGHBOR_CHUNKS  = False
HYBRID_ALPHA = 0.35  # 1.0'a yaklaştıkça anlamsal, 0.0'a yaklaştıkça kelime odaklı olur.

# Qdrant Native Hybrid Search için genişletilmiş retrieval konfigürasyonu.
RETRIEVAL_CFG: Dict[str, Any] = {
    "k":               15,
    "search_kwargs":   {},      
    "use_mmr":         False,    
    "mmr_fetch_k":     40,      
    "lambda_mult":     0.85,
    "score_threshold": 0.40,
}

RERANKER_CFG: Dict[str, Any] = {
    "model_name":      "BAAI/bge-reranker-v2-m3",
    "top_n":           7,       
    "batch_size":      16,
    "threshold_score": 0.40,
}

# Sorgudan kanun numarası çıkarıp Qdrant'ta Metadata Filtresi uygular.
QUERY_ANALYSIS_CFG: Dict[str, Any] = {
    "extract_law_number": True,         
    "llm_model":          "gpt-4o-mini",
}

# ─────────────────────────────────────────────
# API VE STREAMING ENDPOINTLERİ
# ─────────────────────────────────────────────
SAMBANOVA_API_BASE_URL  = "https://api.sambanova.ai/v1"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"

STREAM_ENABLED    = os.getenv("STREAM_ENABLED", "false").lower() == "true"
STREAM_CHUNK_SIZE = 50
STREAM_TIMEOUT    = 300


# ─────────────────────────────────────────────
# KAYNAK EŞLEŞTİRME (METADATA)
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class LawSourceInfo:
    display_name: str
    short_name: str
    law_number: str

SOURCE_FORMAT_MAPPING: Dict[str, LawSourceInfo] = {
    "1.5.1475.docx": LawSourceInfo("1475 Sayılı İş Kanunu", "1475 İş Kanunu", "1475"),
    "1.5.2709.docx": LawSourceInfo("T.C. Anayasası", "Anayasa", "2709"),
    "1.5.3308.docx": LawSourceInfo("3308 Sayılı Mesleki Eğitim Kanunu", "Mesleki Eğitim Kanunu", "3308"),
    "1.5.4447.docx": LawSourceInfo("4447 Sayılı İşsizlik Sigortası Kanunu", "İşsizlik Sigortası Kanunu", "4447"),
    "1.5.4857.docx": LawSourceInfo("4857 Sayılı İş Kanunu", "İş Kanunu", "4857"),
    "1.5.5510.docx": LawSourceInfo("5510 Sayılı SSGSSK", "SGK Kanunu", "5510"),
    "1.5.6098.docx": LawSourceInfo("6098 Sayılı Türk Borçlar Kanunu", "Borçlar Kanunu", "6098"),
    "1.5.6331.docx": LawSourceInfo("6331 Sayılı İş Sağlığı ve Güvenliği Kanunu", "İSG Kanunu", "6331"),
    "1.5.6356.docx": LawSourceInfo("6356 Sayılı Sendikalar ve Toplu İş Sözleşmesi Kanunu", "Sendikalar Kanunu", "6356"),
    "1.5.6100.docx": LawSourceInfo("6100 Sayılı Hukuk Muhakemeleri Kanunu", "Hukuk Muhakemeleri Kanunu", "6100"),
    "1.5.7036.docx": LawSourceInfo("7036 Sayılı İş Mahkemeleri Kanunu", "İş Mahkemeleri Kanunu", "7036"),
    "1.3.5953.docx": LawSourceInfo("5953 Sayılı Basın Mesleği Kanunu", "Basın Mesleği Kanunu", "5953"),
    "1.4.193.docx": LawSourceInfo("193 Sayılı Gelir Vergisi Kanunu", "Gelir Vergisi Kanunu", "193"),
    "1.5.4688.docx": LawSourceInfo("4688 Sayılı Kamu Görevlileri Sendikaları ve Toplu Sözleşme Kanunu", "Kamu Görevlileri Sendikaları Kanunu", "4688"),
    "1.5.657.docx":  LawSourceInfo("657 Sayılı Devlet Memurları Kanunu", "Devlet Memurları Kanunu", "657"),
    "1.5.6458.docx": LawSourceInfo("6458 Sayılı Yabancılar ve Uluslararası Koruma Kanunu", "Yabancılar Kanunu", "6458"),
    "1.5.6735.docx": LawSourceInfo("6735 Sayılı Uluslararası İşgücü Kanunu", "Uluslararası İşgücü Kanunu", "6735"),
    "1.5.854.docx":  LawSourceInfo("854 Sayılı Deniz İş Kanunu", "Deniz İş Kanunu", "854"),
}

SOURCE_MAPPING: Dict[str, str] = {fname: info.display_name for fname, info in SOURCE_FORMAT_MAPPING.items()}


# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────
PROMPT_TEMPLATE = """\
Sen, Türk iş hukuku konusunda uzman, samimi ve güven verici bir yapay zeka asistanısın. Kullanıcıların sorularını sıcak, anlaşılır ve doğal bir dille yanıtlarsın.

TEMEL KURAL:
Yalnızca sağlanan BAĞLAM'ı kullan. Bağlamda cevap yoksa asla uydurma; nazikçe bilmediğini belirt.

---

ÖZEL DURUMLAR:
1. [MÜLGA]: Bağlamda mülga uyarısı varsa, yanıtın başında bunu kısaca belirt: "Küçük bir not: İlgili hüküm artık yürürlükte değil, ancak güncel durumu şöyle özetleyebilirim:"
2. [BİLGİ YOK]: Bağlamda cevap yoksa şöyle yanıtla:
   "Bu konuda elimdeki yasal kaynaklarda net bir bilgi bulunmuyor. Seni yanlış yönlendirmemek adına detay veremiyorum."

---

YANIT KURALLARI:
- Sayısal değerleri bağlamdan olduğu gibi al; tüm sayıları rakamla yaz.
- Hukuki jargonu herkesin anlayacağı günlük dile çevir.
- "Harika bir soru!", "Gelin birlikte inceleyelim" gibi yapay, aşırı şatafatlı ifadeler kullanma. Doğal ve samimi ol.
- Bölüm başlıklarını (AÇIKLAMA, HUKUKİ DAYANAKLAR vb.) yanıtta yazma; sadece içeriği yaz.
- Hukuki dayanakları yanıt metnine dahil etme; bunlar zaten ayrıca gösterilmektedir.

---

ÇIKTI FORMATI:

[Kullanıcıya samimi bir dille, durumu anlatan kısa ve öz bir açıklama. Tarafların haklarını vurgula. Bölüm başlığı olmadan doğrudan yaz.]

UYARI *(yalnızca gerçekten gerekirse — istisnalar veya dikkat edilmesi gereken kritik bir nokta varsa)*
- [İstisna veya uyarı]

---

BAĞLAM:
{context}

SORU: {question}

CEVAP:
"""

JUDGE_SYSTEM_PROMPT = """
Sen katı, tarafsız ve sıfır toleranslı bir Hukuk Profesörü ve Değerlendiricisin (LLM Judge). 
Görevin, bir yapay zeka asistanının verdiği cevabı, 'Beklenen Cevap' ve 'Kanun Bağlamı' ile karşılaştırarak 0.0 ile 1.0 arasında puanlamaktır.

HUKUKİ KATI KURALLAR (BUNLARA KESİNLİKLE UY):
1. Rakamlar ve Süreler: Beklenen cevapta "16 hafta" yazıyorsa ve asistan "4 ay" diyorsa puan kır. Kanun lafzı esastır.
2. Eksik Şartlar: Beklenen cevapta "toplu iş sözleşmesiyle 4 aya uzatılabilir" istisnası varsa ve asistan bunu yazmamışsa "Completeness" (Tamlık) puanını ciddi şekilde düşür.
3. Uydurma (Hallucination): Asistanın cevabında bağlamda (context) OLMAYAN tek bir ekstra kelime veya yorum varsa "Faithfulness" (Sadakat) puanını 0.0'a çek.

Şu 4 metriği tam olarak 0.0, 0.25, 0.50, 0.75 veya 1.0 şeklinde puanla:
- relevance: Cevap, doğrudan sorulan soruya odaklanıyor mu? (Gereksiz laf kalabalığı var mı?)
- faithfulness: Cevap SADECE verilen bağlama mı dayanıyor?
- correctness: Asistanın cevabı, beklenen cevap ile hukuken ve anlamsal olarak %100 örtüşüyor mu?
- completeness: Beklenen cevaptaki tüm istisnalar ve süreler asistanın cevabında eksiksiz var mı?

Dönüş formatı SADECE aşağıdaki gibi geçerli bir JSON olmalıdır. Başka hiçbir açıklama yazma:
{
  "relevance": 1.0,
  "faithfulness": 1.0,
  "correctness": 0.5,
  "completeness": 0.75
}
"""

LLM_ENRICHMENT_PROMPT = """\
Sen bir Türk iş hukuku uzmanısın. Sana verilen kanun maddesini analiz ederek \
aşağıdaki JSON formatında yanıt üret.

Görevin:
1. "ozet": Maddenin pratikte ne anlama geldiğini günlük dilde, hukuki jargon \
kullanmadan 1-2 cümleyle açıkla. İşçi veya işveren bakış açısından yaz; \
"bu maddeye göre..." gibi kalıplardan kaçın, doğrudan konuya gir.

2. "sorular": Bu maddenin yanıtladığı 3-5 adet gerçekçi kullanıcı sorusu yaz. \
Sorular bir işçinin, işverenin veya İK uzmanının gerçekten sorabileceği türden \
olsun. "nasıl", "ne zaman", "kim", "ne kadar", "kaç gün", "hangi durumlarda", \
"zorunlu mu" gibi somut, pratik ifadeler kullan. Her soru farklı bir açıdan \
yaklaşsın; aynı şeyi farklı kelimelerle tekrarlama.

Kurallar:
- YALNIZCA JSON döndür. Markdown kod bloğu, açıklama, giriş veya kapanış \
cümlesi kesinlikle ekleme.
- Mülga (yürürlükten kalkmış) maddeler için de aynı formatı uygula; özette \
mülga olduğunu belirt.
- Madde metni Türkçe karakter içerebilir, bunu göz önünde bulundur.

Beklenen çıktı (başka hiçbir şey yok):
{"ozet": "...", "sorular": ["...", "...", "...", "..."]}
"""

# ─────────────────────────────────────────────
# TEST VE PUANLAMA AĞIRLIKLARI
# ─────────────────────────────────────────────
# RETRIEVAL METRİKLERİ AĞIRLIKLARI
RETRIEVAL_WEIGHTS = {
    "precision": 0.50,
    "recall":    0.50,
}

# GENERATION METRİKLERİ AĞIRLIKLARI
GENERATION_WEIGHTS = {
    "faithfulness":        0.35,
    "answer_relevance":    0.35,
    "answer_correctness":  0.30,
}

# GENEL PUANLAMA AĞIRLIKLARI
SCORING_WEIGHTS = {
    "retrieval_weight":  0.40,
    "generation_weight": 0.60,
}

# METRIK EŞİKLERİ
TEST_THRESHOLDS = {
    "pass_threshold":              0.70,
    "precision_threshold":         0.60,
    "recall_threshold":            0.50,
    "faithfulness_threshold":      0.75,
    "answer_relevance_threshold":  0.65,
    "answer_correctness_threshold": 0.70,
}

METRIC_WEIGHTS = {
    "latency_weight": 0.1,
}

TEST_CONFIG_DEFAULTS = {
    "delay_between_questions": 1.5,                                 
    "output_dir":              str(REPORTS_DIR),
    "save_contexts":           True,                                         
    "verbose":                 os.getenv("EVAL_VERBOSE", "true").lower() == "true",  
}

LATENCY_THRESHOLD_MS = 8_000

RETRIEVER_K = 20
HYBRID_KEYWORD_WEIGHTS = {
    "exact_match": 0.40,
    "semantic":    0.60,
}