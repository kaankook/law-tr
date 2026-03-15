from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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

DB_PATH           = str(VECTOR_STORE_DIR / "chroma_db")
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

MODEL_NAME           = os.getenv("LLM_MODEL",       "gemini-2.0-flash")
EVALUATOR_MODEL_NAME = os.getenv("EVALUATOR_MODEL", "gpt-3.5-turbo")


# ─────────────────────────────────────────────
# CHUNKING PARAMETRELERİ
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for document chunking with semantic boundaries.
    
    Attributes:
        target_size: Target chunk size in characters (2000 for balanced context)
        overlap_chars: Character overlap between chunks (0 for non-overlapping)
        min_chunk_chars: Minimum chunk size to prevent fragmentation (100 chars = ~20-25 words)
        header_repeat: Whether to repeat headers in multi-part chunks
    """
    target_size: int      = 2000
    overlap_chars: int    = 0
    min_chunk_chars: int  = 100
    header_repeat: bool   = True

CHUNK_CFG = ChunkConfig()

CHUNK_SIZE    = CHUNK_CFG.target_size
CHUNK_OVERLAP = CHUNK_CFG.overlap_chars


# ─────────────────────────────────────────────
# RETRIEVER AYARLARI
# ─────────────────────────────────────────────
RETRIEVER_K  = 7
TEMPERATURE  = 0


# ─────────────────────────────────────────────
# API ENDPOINT SUBLAMASı
# ─────────────────────────────────────────────
# External API endpoints for model providers
SAMBANOVA_API_BASE_URL = "https://api.sambanova.ai/v1"
"""SambaNova API base URL for Qwen model access.

Used when SAMBANOVA_API_KEY environment variable is set.
"""

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434"
"""Default Ollama server base URL for local model access.

Overridable via OLLAMA_BASE_URL environment variable.
Use as fallback when no other Qwen provider available.
"""


# ─────────────────────────────────────────────
# STREAMING AYARLARI (Future)
# ─────────────────────────────────────────────
STREAM_ENABLED = os.getenv("STREAM_ENABLED", "false").lower() == "true"
"""Enable streaming responses from LLM.

Stream responses in chunks instead of waiting for full completion.
Useful for long answer generation and real-time UX.
"""

STREAM_CHUNK_SIZE = 50
"""Minimum chunk size for streamed responses (characters)."""

STREAM_TIMEOUT = 300
"""Timeout for streaming response in seconds (5 minutes)."""


# ─────────────────────────────────────────────
# PROMPT ŞABLONU
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class LawSourceInfo:
    display_name: str
    short_name: str
    law_number: str


SOURCE_FORMAT_MAPPING: Dict[str, LawSourceInfo] = {
    "1.5.1475.docx": LawSourceInfo(
        display_name="1475 Sayılı İş Kanunu",
        short_name="1475 İş Kanunu",
        law_number="1475",
    ),
    "1.5.2709.docx": LawSourceInfo(
        display_name="T.C. Anayasası",
        short_name="Anayasa",
        law_number="2709",
    ),
    "1.5.3308.docx": LawSourceInfo(
        display_name="3308 Sayılı Mesleki Eğitim Kanunu",
        short_name="Mesleki Eğitim Kanunu",
        law_number="3308",
    ),
    "1.5.4447.docx": LawSourceInfo(
        display_name="4447 Sayılı İşsizlik Sigortası Kanunu",
        short_name="İşsizlik Sigortası Kanunu",
        law_number="4447",
    ),
    "1.5.4857.docx": LawSourceInfo(
        display_name="4857 Sayılı İş Kanunu",
        short_name="İş Kanunu",
        law_number="4857",
    ),
    "1.5.5510.docx": LawSourceInfo(
        display_name="5510 Sayılı SSGSSK",
        short_name="SGK Kanunu",
        law_number="5510",
    ),
    "1.5.6098.docx": LawSourceInfo(
        display_name="6098 Sayılı Türk Borçlar Kanunu",
        short_name="Borçlar Kanunu",
        law_number="6098",
    ),
    "1.5.6331.docx": LawSourceInfo(
        display_name="6331 Sayılı İş Sağlığı ve Güvenliği Kanunu",
        short_name="İSG Kanunu",
        law_number="6331",
    ),
    "1.5.6356.docx": LawSourceInfo(
        display_name="6356 Sayılı Sendikalar ve Toplu İş Sözleşmesi Kanunu",
        short_name="Sendikalar Kanunu",
        law_number="6356",
    ),
    "1.5.6100.docx": LawSourceInfo(
        display_name="6100 Sayılı Hukuk Muhakemeleri Kanunu",
        short_name="Hukuk Muhakemeleri Kanunu",
        law_number="6100",
    ),
    "1.5.7036.docx": LawSourceInfo(
        display_name="7036 Sayılı İş Mahkemeleri Kanunu",
        short_name="İş Mahkemeleri Kanunu",
        law_number="7036",
    ),
    "1.3.5953.docx": LawSourceInfo(
        display_name="5953 Sayılı Basın Mesleği Kanunu",
        short_name="Basın Mesleği Kanunu",
        law_number="5953",
    ),
    "1.4.193.docx": LawSourceInfo(
        display_name="193 Sayılı Gelir Vergisi Kanunu",
        short_name="Gelir Vergisi Kanunu",
        law_number="193",
    ),
    "1.5.4688.docx": LawSourceInfo(
        display_name="4688 Sayılı Kamu Görevlileri Sendikaları ve Toplu Sözleşme Kanunu",
        short_name="Kamu Görevlileri Sendikaları Kanunu",
        law_number="4688",
    ),
    "1.5.657.docx": LawSourceInfo(
        display_name="657 Sayılı Devlet Memurları Kanunu",
        short_name="Devlet Memurları Kanunu",
        law_number="657",
    ),
    "1.5.6458.docx": LawSourceInfo(
        display_name="6458 Sayılı Yabancılar ve Uluslararası Koruma Kanunu",
        short_name="Yabancılar Kanunu",
        law_number="6458",
    ),
    "1.5.6735.docx": LawSourceInfo(
        display_name="6735 Sayılı Uluslararası İşgücü Kanunu",
        short_name="Uluslararası İşgücü Kanunu",
        law_number="6735",
    ),
    "1.5.854.docx": LawSourceInfo(
        display_name="854 Sayılı Deniz İş Kanunu",
        short_name="Deniz İş Kanunu",
        law_number="854",
    ),
}

# Geriye dönük uyumluluk
SOURCE_MAPPING: Dict[str, str] = {
    fname: info.display_name
    for fname, info in SOURCE_FORMAT_MAPPING.items()
}


# ─────────────────────────────────────────────
# ANAHTAR KELİME AĞIRLIKLARI
# ─────────────────────────────────────────────
LEGAL_TERM_WEIGHTS: Dict[str, int] = {
    # Çok yüksek öncelik (5)
    "kıdem tazminatı": 5,
    "ihbar tazminatı": 5,
    "iş sözleşmesi": 5,
    "hizmet sözleşmesi": 5,
    "işe iade": 5,
    "haksız fesih": 5,
    "haklı fesih": 5,
    "fazla çalışma": 5,
    "yıllık izin": 5,
    "sendika": 5,
    "toplu iş sözleşmesi": 5,
    "arabuluculuk": 5,
    "iş mahkemesi": 5,
    "mobbing": 5,
    "iş kazası": 5,
    "meslek hastalığı": 5,
    # Orta öncelik (3)
    "ücret": 3,
    "bordro": 3,
    "asgari ücret": 3,
    "işveren": 3,
    "işçi": 3,
    "fesih": 3,
    "deneme süresi": 3,
    "bildirim süresi": 3,
    "kıdem": 3,
    "ihbar": 3,
    "tazminat": 3,
    "iş güvencesi": 3,
    "grev": 3,
    "lokavt": 3,
    "performans": 3,
    "mücbir sebep": 3,
    "alt işveren": 3,
    "işyeri": 3,
    "çalışma süresi": 3,
    "fazla mesai ücreti": 3,
    "işe başlama": 3,
    # Düşük öncelik (2)
    "sözleşme": 2,
    "hukuk": 2,
    "delil": 2,
    "ispat": 2,
    "bilirkişi": 2,
    "kısa çalışma": 2,
    "işsizlik sigortası": 2,
    "iş sağlığı ve güvenliği": 2,
    "işveren vekili": 2,
}

LEGAL_TERM_MAPPINGS: Dict[str, List[str]] = {
    "sataşma": ["mobbing", "psikolojik taciz"],
    "taciz": ["cinsel taciz", "mobbing"],
    "cinsel taciz": ["cinsel taciz"],
    "şeref ve namus": ["hakaret", "onur kırma"],
    "ahlak ve iyi niyet": ["haklı fesih", "tazminatlı çıkış"],
    "sağlık sebepleri": ["iş kazası", "meslek hastalığı", "haklı fesih"],
    "fesih": ["fesih", "işten çıkarma", "iş akdinin sona ermesi"],
    "çıkış parası": ["kıdem tazminatı"],
    "tazminat hesaplama": ["kıdem tazminatı", "ihbar tazminatı"],
    "bildirim süresi": ["ihbar tazminatı", "ihbar"],
    "maaş": ["ücret", "bordro"],
    "aylık": ["ücret"],
    "yevmiye": ["ücret"],
    "fazla mesai parası": ["fazla mesai ücreti"],
    "prim": ["ücret"],
    "ikramiye": ["ücret"],
    "mesai": ["fazla çalışma"],
    "yıllık ücretli izin": ["yıllık izin"],
    "sözleşmeli personel": ["belirli süreli"],
    "kadrolu": ["belirsiz süreli"],
    "tis": ["toplu iş sözleşmesi"],
    "toplu sözleşme": ["toplu iş sözleşmesi"],
    "iş bırakma": ["grev"],
    "savunma yazısı": ["savunma"],
    "arabulucu": ["arabuluculuk"],
    "işe geri dönüş": ["işe iade"],
    "haksız fesih davası": ["haksız fesih", "işe iade"],
    "asgariücret": ["asgari ücret"],
}

STOP_WORDS: set = {
    "acaba","ama","ancak","arada","aslında","ayrıca","bana","bazı","belki",
    "ben","benden","beni","benim","beri","bile","bin","bir","biri","birkaç",
    "birkez","birçok","biz","bize","bizden","bizi","bizim","böyle","böylece",
    "bu","buna","bunda","bundan","bunlar","bunları","bunların","bunu","bunun",
    "burada","bütün","çoğu","çünkü","da","daha","dahi","de","defa","değil",
    "diğer","diye","dolayı","dolayısıyla","edecek","eden","ederek","edilecek",
    "ediliyor","edilmesi","ediyor","eğer","en","etmesi","etti","ettiği",
    "ettiğini","gibi","göre","halen","hangi","hepsi","her","herkes","herkesi",
    "hiç","hiçbir","için","iki","ile","ilgili","ise","itibaren","kadar",
    "karşın","kendi","kendini","kendisi","kez","ki","kim","kimden","kime",
    "kimi","kimse","mı","nasıl","ne","neden","nedenle","nerede","niçin",
    "o","olan","olarak","oldu","olduğu","olmak","olması","olmayan","olmaz",
    "olsa","olsun","olup","olur","on","ona","onlar","onları","onların","onu",
    "onun","oysa","pek","rağmen","sadece","sanki","sayı","sen","siz","sonra",
    "tarafından","tüm","üç","üzere","var","vardı","ve","veya","ya","yani",
    "yapacak","yapılan","yapılması","yapıyor","yapmak","yaptı","yaptığı",
    "yaptığını","yaptıkları","yerine","yine","yok","yoksa","zaten",
    "maddeki","tarih","tarihi","sayılı","kanun","madde",
}


# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────
PROMPT_TEMPLATE = """\
Sen Türk iş hukuku konusunda uzman bir Yapay Zeka Hukuk Asistanısın.

KURALLAR:
1. SADECE aşağıdaki BAĞLAM'daki bilgileri kullan.
2. Bağlamda yoksa: "Bu konuda elimdeki yasal metinlerde yeterli bilgi bulunmamaktadır" de.
3. KISA ve ÖZ cevap ver (maksimum 3-4 cümle).
4. TÜRKÇE cevap ver.
5. [MÜLGA UYARISI] etiketi gördüğünde bunu kullanıcıya açıkça belirt.

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

ENHANCED_JUDGE_PROMPT = """\
SİSTEM ROLÜ:
Sen, Türk Hukuku alanında uzmanlaşmış, titiz ve objektif bir RAG Denetçisisin.
Görevin: sistemin ürettiği cevabı aşağıdaki kriterlere göre puanlamak.

GİRDİ VERİLERİ:
[SORU]: {question}
[REFERANS CEVAP]: {expected_answer}
[ZORUNLU UNSURLAR]: {must_include}
[ÖNERİLEN UNSURLAR]: {should_include}
[KULLANILAN CONTEXT]: {context}
[MODELİN CEVABI]: {actual_answer}
[KAYNAK]: {source}

DEĞERLENDİRME ADIMLARI:
1. RETRIEVAL: Context soruyla alakalı mı? Boşsa ve model "bilgi yok" dediyse → başarı.
2. HALLUCINATION (Faithfulness): Her iddiayı context'te doğrula. Context dışı bilgi → < 0.5
3. HUKUKİ DOĞRULUK: Madde no, terimler, süreler, rakamlar doğru mu?
4. KAPSAM: Zorunlu unsurlar var mı?

PUANLAMA (0.0–1.0):
| 0.9–1.0 | Mükemmel | 0.7–0.9 | İyi | 0.5–0.7 | Orta | 0.3–0.5 | Zayıf | 0.0–0.3 | Başarısız |

METRİKLER:
- relevance_score    : Soruya odaklılık
- faithfulness_score : Context'e bağlılık (hallucination yok)
- correctness_score  : Hukuki sonuç doğruluğu
- completeness_score : Kritik bilgilerin tamlığı
- overall_score      : faith×0.30 + rel×0.25 + corr×0.25 + comp×0.20

KURALLAR:
- Madde numarası yanlışsa → correctness < 0.5
- Context dışı bilgi varsa → faithfulness < 0.5
- Pozitiflik yanlılığı YAPMA

ÇIKTI: Sadece JSON formatında döndür.
"""


# ─────────────────────────────────────────────
# PUANLAMA AĞIRLIKLARI
# ─────────────────────────────────────────────
SCORING_WEIGHTS = {
    "judge_weight":     0.70,
    "heuristic_weight": 0.30,
}

JUDGE_SUBWEIGHTS = {
    "faithfulness":  0.30,
    "relevance":     0.25,
    "correctness":   0.25,
    "completeness":  0.20,
}

HEURISTIC_SUBWEIGHTS = {
    "semantic_correctness": 0.25,
    "quote_presence":       0.15,
    "citation_accuracy":    0.20,
    "answer_consistency":   0.18,
    "keyword_coverage":     0.12,
    "response_quality":     0.10,
}

HYBRID_KEYWORD_WEIGHTS = {
    "exact_match": 0.40,
    "semantic":    0.60,
}

MUST_SHOULD_WEIGHTS = {
    "must_weight":   0.80,
    "should_weight": 0.20,
}

TEST_THRESHOLDS = {
    "pass_threshold":          0.70,
    "relevance_threshold":     0.60,
    "faithfulness_threshold":  0.75,
    "citation_threshold":      0.50,
    "consistency_threshold":   0.50,
    "semantic_threshold":      0.40,
}

METRIC_WEIGHTS = {
    "citation_weight":          2.0,
    "keyword_weight":           1.2,
    "response_quality_weight":  0.5,
    "latency_weight":           0.1,
}

TEST_CONFIG_DEFAULTS = {
    "delay_between_questions": 2.0,
    "output_dir":              str(REPORTS_DIR),
    "save_contexts":           True,
    "verbose":                 True,
}

LATENCY_THRESHOLD_MS = 8_000