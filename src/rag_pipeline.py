from __future__ import annotations
import os
import time
import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder
else:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        CrossEncoder = None
from src.config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    SPARSE_MODEL_NAME,
    MODEL_NAME,
    TEMPERATURE,
    PROMPT_TEMPLATE,
    MAPPING_FILE_PATH,
    SAMBANOVA_API_BASE_URL,
    OLLAMA_DEFAULT_BASE_URL,
    STREAM_ENABLED,
    STREAM_CHUNK_SIZE,
    RETRIEVAL_CFG,
    RERANKER_CFG,
    QUERY_ANALYSIS_CFG,
    ENABLE_NEIGHBOR_CHUNKS,
)
from src.law_mapping_resolver import LawMappingResolver

logger = logging.getLogger(__name__)

# Resolver başlatılamazsa mülga uyarıları devre dışı kalır.
try:
    _law_resolver = LawMappingResolver(MAPPING_FILE_PATH)
except Exception as e:
    logger.warning(
        f"LawMappingResolver başlatılamadı: {e}. "
        "Mülga kanun uyarıları devre dışı."
    )
    _law_resolver = None


# Model bazlı düşünme etiketi desenleri.
MODEL_THINKING_PATTERNS: Dict[str, str] = {
    "deepseek": r"<think>.*?</think>\s*",
    "qwen":     r"<think>.*?</think>\s*",
    "claude":   r"<claude_thinking>.*?</claude_thinking>\s*",
}


def _get_thinking_pattern(model_name: str) -> Optional[str]:
    # Model adına göre uygun düşünme etiketi regex'ini döndür.
    model_lower = model_name.lower()
    for model_key, pattern in MODEL_THINKING_PATTERNS.items():
        if model_key in model_lower:
            return pattern
    return None


def _cuda_available() -> bool:
    # Torch yüklüyse CUDA kullanılabilirliğini kontrol et.
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _validate_config() -> None:
    # Kritik config alanlarını başlangıçta doğrula.
    logger.info("Konfigürasyon validasyonu başlıyor...")

    required_retrieval = {
        "k", "search_kwargs", "use_mmr", "mmr_fetch_k", "lambda_mult", "score_threshold"
    }
    missing = required_retrieval - RETRIEVAL_CFG.keys()
    if missing:
        raise ValueError(f"RETRIEVAL_CFG eksik anahtarlar: {missing}")
    if not isinstance(RETRIEVAL_CFG["k"], int) or RETRIEVAL_CFG["k"] <= 0:
        raise ValueError(
            f"RETRIEVAL_CFG['k'] pozitif int olmalıdır: {RETRIEVAL_CFG['k']}"
        )
    if RETRIEVAL_CFG["mmr_fetch_k"] < RETRIEVAL_CFG["k"]:
        raise ValueError(
            f"RETRIEVAL_CFG['mmr_fetch_k'] ({RETRIEVAL_CFG['mmr_fetch_k']}) "
            f"k'dan ({RETRIEVAL_CFG['k']}) büyük olmalıdır"
        )
    logger.debug(
        f"  ✓ RETRIEVAL_CFG: k={RETRIEVAL_CFG['k']}, "
        f"mmr_fetch_k={RETRIEVAL_CFG['mmr_fetch_k']}"
    )

    required_reranker = {"model_name", "top_n", "batch_size", "threshold_score"}
    missing = required_reranker - RERANKER_CFG.keys()
    if missing:
        raise ValueError(f"RERANKER_CFG eksik anahtarlar: {missing}")
    if RERANKER_CFG["top_n"] > RETRIEVAL_CFG["k"]:
        raise ValueError(
            f"RERANKER_CFG['top_n'] ({RERANKER_CFG['top_n']}) "
            f"RETRIEVAL_CFG['k']'dan ({RETRIEVAL_CFG['k']}) büyük olamaz"
        )
    logger.debug(
        f"  ✓ RERANKER_CFG: model={RERANKER_CFG['model_name']}, "
        f"top_n={RERANKER_CFG['top_n']}"
    )

    required_qa = {"extract_law_number", "llm_model"}
    missing = required_qa - QUERY_ANALYSIS_CFG.keys()
    if missing:
        raise ValueError(f"QUERY_ANALYSIS_CFG eksik anahtarlar: {missing}")
    logger.debug(
        f"  ✓ QUERY_ANALYSIS_CFG: "
        f"extract_law_number={QUERY_ANALYSIS_CFG['extract_law_number']}"
    )

    # Temel string alanlarının tipini doğrula.
    for var_name, var_val in [
        ("DB_PATH",              DB_PATH),
        ("COLLECTION_NAME",      COLLECTION_NAME),
        ("EMBEDDING_MODEL_NAME", EMBEDDING_MODEL_NAME),
        ("SPARSE_MODEL_NAME",    SPARSE_MODEL_NAME),
        ("MODEL_NAME",           MODEL_NAME),
    ]:
        if not var_val or not isinstance(var_val, str):
            raise ValueError(f"Config hatası: {var_name} boş veya geçersiz tip")
        logger.debug(f"  ✓ {var_name}: {var_val}")

    if not isinstance(TEMPERATURE, (int, float)) or not 0 <= TEMPERATURE <= 2:
        raise ValueError(f"TEMPERATURE 0-2 aralığında olmalıdır: {TEMPERATURE}")
    logger.debug(f"  ✓ TEMPERATURE: {TEMPERATURE}")

    if not PROMPT_TEMPLATE or not isinstance(PROMPT_TEMPLATE, str):
        raise ValueError("Config hatası: PROMPT_TEMPLATE boş veya geçersiz tip")
    if "{context}" not in PROMPT_TEMPLATE or "{question}" not in PROMPT_TEMPLATE:
        raise ValueError(
            "Config hatası: PROMPT_TEMPLATE '{context}' ve '{question}' içermelidir"
        )
    logger.debug(f"  ✓ PROMPT_TEMPLATE: {len(PROMPT_TEMPLATE)} karakter")

    if not isinstance(ENABLE_NEIGHBOR_CHUNKS, bool):
        raise ValueError(
            f"Config hatası: ENABLE_NEIGHBOR_CHUNKS boolean olmalıdır, "
            f"alınan: {type(ENABLE_NEIGHBOR_CHUNKS).__name__}"
        )
    logger.debug(f"  ✓ ENABLE_NEIGHBOR_CHUNKS: {ENABLE_NEIGHBOR_CHUNKS}")

    logger.info("✓ Konfigürasyon validasyonu başarılı")


try:
    _validate_config()
except ValueError as e:
    logger.error(f"Konfigürasyon hatası: {e}")
    raise


def create_chat_model(
    model_name: str,
    temperature: float = TEMPERATURE,
) -> Union[ChatGoogleGenerativeAI, ChatOpenAI, ChatOllama]:
    # Model adına göre uygun sağlayıcıyı seç.
    model_lower = model_name.lower()

    if "gemini" in model_lower:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Google Gemini API key bulunamadı. "
                "GOOGLE_API_KEY veya GEMINI_API_KEY ortam değişkenini ayarlayın."
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            convert_system_message_to_human=True,
            api_key=api_key,
        )

    if "gpt" in model_lower or "openai" in model_lower:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key bulunamadı. "
                "OPENAI_API_KEY ortam değişkenini ayarlayın."
            )
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

    if "qwen" in model_lower:
        sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")
        qwen_api_key      = os.getenv("QWEN_API_KEY")
        qwen_base_url     = os.getenv("QWEN_BASE_URL")

        if sambanova_api_key:
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=sambanova_api_key,
                base_url=SAMBANOVA_API_BASE_URL,
            )
        if qwen_api_key and qwen_base_url:
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=qwen_api_key,
                base_url=qwen_base_url,
            )

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=ollama_base_url,
        )

    # Varsayılan fallback: Gemini.
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        convert_system_message_to_human=True,
    )


def get_article_reference(metadata: Dict[str, Any]) -> str:
    # Önce yeni alanı, yoksa geriye dönük uyumluluk alanını kullan.
    return metadata.get("article_reference") or metadata.get("article", "")


@dataclass
class RetrievalResult:
    # Retrieval sonuçlarını ve gecikme metriklerini taşır.
    documents:           List[Document]
    scores:              List[float] = field(default_factory=list)
    latency_ms:          float = 0.0   # Toplam retrieval süresi (qdrant + reranker)
    qdrant_latency_ms:   float = 0.0   # Qdrant native hybrid search süresi
    reranker_latency_ms: float = 0.0   # CrossEncoder reranking süresi

    @property
    def contexts(self) -> List[str]:
        return [doc.page_content for doc in self.documents]

    @property
    def metadata_list(self) -> List[Dict[str, Any]]:
        return [doc.metadata for doc in self.documents]


@dataclass
class GenerationResult:
    # LLM üretim çıktısı ve metrikleri.
    answer:      str
    latency_ms:  float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    model_name:  str = ""


@dataclass
class RAGResult:
    # Uçtan uca RAG sonucu.
    question:         str
    answer:           str
    retrieval:        RetrievalResult
    generation:       GenerationResult
    total_latency_ms: float = 0.0
    mulga_warnings:   List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question":              self.question,
            "answer":                self.answer,
            "retrieved_contexts":    self.retrieval.contexts,
            "retrieved_articles":    [get_article_reference(m) for m in self.retrieval.metadata_list],
            "retrieved_sources":     [m.get("source", "") for m in self.retrieval.metadata_list],
            # Latency metrikleri raporlama için ayrı tutulur.
            "retrieval_latency_ms":  self.retrieval.latency_ms,
            "qdrant_latency_ms":     self.retrieval.qdrant_latency_ms,
            "reranker_latency_ms":   self.retrieval.reranker_latency_ms,
            "generation_latency_ms": self.generation.latency_ms,
            "total_latency_ms":      self.total_latency_ms,
            "model_name":            self.generation.model_name,
            "token_usage":           self.generation.token_usage,
            "mulga_warnings":        self.mulga_warnings,
        }

class RAGPipeline:
    # Türk hukuk dokümanları için retrieval + generation pipeline'ı.

    def __init__(
        self,
        model_name:      Optional[str] = None,
        embedding_model: Optional[str] = None,
        db_path:         Optional[str] = None,
        temperature:     float         = TEMPERATURE,
    ):
        if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise ValueError(
                f"Geçersiz temperature: {temperature}. 0-2 aralığında olmalıdır."
            )

        self.model_name           = model_name or MODEL_NAME
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL_NAME
        self.db_path              = db_path or DB_PATH
        self.temperature          = temperature

        logger.info(
            f"RAGPipeline başlatıldı: model={self.model_name}, "
            f"embedding={self.embedding_model_name}, "
            f"k={RETRIEVAL_CFG['k']}, reranker_top_n={RERANKER_CFG['top_n']}"
        )

        # Lazy-load cache — tüm ağır bileşenler ilk kullanımda yüklenir
        self._embeddings:        Optional[HuggingFaceEmbeddings] = None
        self._sparse_embeddings: Optional[FastEmbedSparse]       = None
        self._qdrant_client:     Optional[QdrantClient]          = None
        self._vectorstore:       Optional[QdrantVectorStore]     = None
        self._llm                                                 = None
        self._prompt:            Optional[ChatPromptTemplate]    = None
        self._reranker:          Optional["CrossEncoder"]        = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            try:
                logger.info(
                    f"Dense embedding modeli yükleniyor: {self.embedding_model_name}"
                )
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={"device": "cuda" if _cuda_available() else "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info("Dense embedding modeli başarıyla yüklendi")
            except Exception as e:
                raise RuntimeError(
                    f"Dense embedding modeli '{self.embedding_model_name}' "
                    f"yüklenemedi: {e}"
                ) from e
        return self._embeddings

    @property
    def sparse_embeddings(self) -> FastEmbedSparse:
        if self._sparse_embeddings is None:
            try:
                logger.info(
                    f"Sparse embedding modeli yükleniyor: {SPARSE_MODEL_NAME}"
                )
                self._sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)
                logger.info("Sparse embedding modeli başarıyla yüklendi")
            except Exception as e:
                raise RuntimeError(
                    f"Sparse embedding modeli '{SPARSE_MODEL_NAME}' "
                    f"yüklenemedi: {e}"
                ) from e
        return self._sparse_embeddings

    @property
    def qdrant_client(self) -> QdrantClient:
        # Qdrant client, vectorstore ilklenirken hazırlanır.
        _ = self.vectorstore  # _qdrant_client'ı initialize eder
        return self._qdrant_client

    @property
    def vectorstore(self) -> QdrantVectorStore:
        if self._vectorstore is None:
            try:
                logger.info(f"Qdrant vektör veritabanı yükleniyor: {self.db_path}")
                self._qdrant_client = QdrantClient(path=self.db_path)
                self._vectorstore   = QdrantVectorStore(
                    client=self._qdrant_client,
                    collection_name=COLLECTION_NAME,
                    embedding=self.embeddings,
                    sparse_embedding=self.sparse_embeddings,
                    retrieval_mode="hybrid",
                )
                logger.info(
                    f"Qdrant native hybrid search aktif "
                    f"(Collection: {COLLECTION_NAME})"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Qdrant vektör veritabanı '{self.db_path}' yüklenemedi: {e}"
                ) from e
        return self._vectorstore

    @property
    def llm(self) -> Union[ChatGoogleGenerativeAI, ChatOpenAI, ChatOllama]:
        if self._llm is None:
            try:
                logger.info(f"LLM yükleniyor: {self.model_name}")
                self._llm = create_chat_model(
                    self.model_name, temperature=self.temperature
                )
                logger.info(f"LLM yüklendi: {self.model_name}")
            except ValueError as e:
                raise RuntimeError(f"LLM konfigürasyonu başarısız: {e}") from e
            except Exception as e:
                raise RuntimeError(
                    f"LLM '{self.model_name}' yüklenemedi: {e}"
                ) from e
        return self._llm

    @property
    def prompt(self) -> ChatPromptTemplate:
        if self._prompt is None:
            self._prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        return self._prompt

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            try:
                import torch
                from sentence_transformers import CrossEncoder
                
                model_name = RERANKER_CFG["model_name"]
                device     = "cuda" if _cuda_available() else "cpu"
                logger.info(f"Cross-encoder yükleniyor: {model_name} ({device})")
                
                # Reranker modeli yüklenir.
                self._reranker = CrossEncoder(
                    model_name,
                    device=device,
                    max_length=RERANKER_CFG.get("max_length", 1024)
                )
                
                # GPU'da FP16 ile hızlandır.
                if device == "cuda":
                    self._reranker.model.half()
                
                logger.info(f"Tokenizer max_length → {RERANKER_CFG.get('max_length', 1024)}")
                
                self._reranker.predict([("warm-up", "warm-up")], batch_size=RERANKER_CFG.get("batch_size", 32))
                logger.info("Cross-encoder başarıyla yüklendi ve FP16'ya optimize edildi")
            except Exception as e:
                raise RuntimeError(f"Cross-encoder yüklenemedi: {e}") from e
        return self._reranker   
    
    def _retrieve_and_rerank(
        self,
        question: str,
    ) -> Tuple[List[Document], float, float]:
        # Qdrant retrieval + reranking adımlarını çalıştır.
        law_filter = None
        if QUERY_ANALYSIS_CFG.get("extract_law_number"):
            # Sorgudan kanun numarasını yakalayıp metadata filtresi uygula.
            law_match = re.search(r"(\d{3,5})", question)
            if law_match:
                law_no = law_match.group(1)
                law_filter = Filter(
                    must=[
                        FieldCondition(
                            key="metadata.law_number",
                            match=MatchValue(value=law_no),
                        )
                    ]
                )
                logger.info(f"🔍 Metadata Filtresi Uygulanıyor: {law_no} Sayılı Kanun")

        qdrant_start = time.time()
        try:
            if RETRIEVAL_CFG.get("use_mmr"):
                logger.debug(f"MMR arama aktif: λ={RETRIEVAL_CFG.get('lambda_mult')}")
                docs = self.vectorstore.max_marginal_relevance_search(
                    question,
                    k=RETRIEVAL_CFG["k"],
                    fetch_k=RETRIEVAL_CFG.get("mmr_fetch_k", 40),
                    lambda_mult=RETRIEVAL_CFG.get("lambda_mult", 0.5),
                    filter=law_filter,
                )
            else:
                docs = self.vectorstore.similarity_search(
                    question,
                    k=RETRIEVAL_CFG["k"],
                    filter=law_filter,
                )
        except Exception as e:
            logger.error(f"Qdrant retrieval hatası: {e}")
            raise RuntimeError(f"Qdrant hybrid retrieval başarısız: {e}") from e

        qdrant_ms = (time.time() - qdrant_start) * 1000

        # Çok kısa chunk'ları ele.
        from src.config import MIN_CHUNK_CHARS

        filtered = [d for d in docs if len(d.page_content) >= MIN_CHUNK_CHARS]

        if not filtered and docs:
            logger.warning("Boyut filtresi her şeyi eledi, en büyük parça korunuyor.")
            filtered = [max(docs, key=lambda d: len(d.page_content))]
        elif not filtered:
            logger.warning(f"⚠️ Qdrant '{question}' için sonuç bulamadı.")
            return [], qdrant_ms, 0.0

        reranker_start = time.time()
        reranked = self.rerank(question, filtered)
        reranker_ms = (time.time() - reranker_start) * 1000

        logger.info(
            f"✅ Pipeline: Qdrant={qdrant_ms:.0f}ms ({len(docs)} doc) → "
            f"Reranker={reranker_ms:.0f}ms ({len(reranked)} doc)"
        )

        return reranked, qdrant_ms, reranker_ms
        
    def rerank(self, question: str, documents: List[Document]) -> List[Document]:
        if not documents:
            return documents

        top_x = RERANKER_CFG.get("top_n", 5)
        threshold = RERANKER_CFG.get("threshold_score", 0.60)

        try:
            pairs = []
            for doc in documents:
                # Small-to-Big: Reranker'a kısa özet metni ver.
                summary = doc.metadata.get("llm_summary", "")
                questions = doc.metadata.get("llm_questions", "")
                
                if summary or questions:
                    small_text = f"ÖZET: {summary} SORULAR: {questions}".strip()
                else:
                    # Güvenli fallback: metnin baş kısmını kullan.
                    small_text = doc.page_content[:400] 
                
                pairs.append((question, small_text))

            scores = self.reranker.predict(
                pairs,
                batch_size=RERANKER_CFG.get("batch_size", 16),
            )

            scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

            logger.info(f"── TOP-{top_x} Reranker Sonuçları (Threshold: {threshold} | Mod: Small-to-Big) ──")
            
            filtered_docs = []
            
            for i, (doc, score) in enumerate(scored[:top_x]):
                ref = doc.metadata.get('article_reference', 'Bilinmiyor')
                bar_len = int(max(0, score) * 20)
                bar = "█" * bar_len
                
                doc.metadata["rerank_score"] = float(score)
                
                if score >= threshold:
                    logger.info(f"   [{i+1}] {score:.4f} {bar.ljust(20)} | {ref} (✅ LLM'e Gidiyor)")
                    filtered_docs.append(doc)
                else:
                    logger.info(f"   [{i+1}] {score:.4f} {bar.ljust(20)} | {ref} (❌ Elendi)")

            if not filtered_docs and scored:
                logger.warning(f"⚠️ Hiçbir belge {threshold} eşiğini geçemedi! Sadece en yüksek skorlu 1 belge kurtarıldı.")
                best_doc = scored[0][0]
                filtered_docs.append(best_doc)

            return filtered_docs

        except Exception as e:
            logger.error(f"Reranking hatası: {e}")
            return documents[:top_x]
   
    def _fetch_chunk_by_id(self, chunk_id: str) -> Optional[Document]:
        # Qdrant'tan chunk_id'ye göre tek doküman çek.
        try:
            results, _ = self.qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.chunk_id",
                            match=MatchValue(value=chunk_id),
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if results:
                payload = results[0].payload or {}
                return Document(
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata", {}),
                )
        except Exception as e:
            logger.debug(f"Chunk fetch hatası ({chunk_id}): {e}")
        return None

    def retrieve_neighbor_chunks(
        self,
        main_chunks: List[Document],
        k_neighbors: int = 1,
    ) -> List[Document]:
        # Rerank sonrası komşu chunk'ları context'e ekle.
        if not ENABLE_NEIGHBOR_CHUNKS:
            return main_chunks

        neighbors = []
        seen_ids = {
            doc.metadata.get("chunk_id")
            for doc in main_chunks
            if doc.metadata.get("chunk_id")
        }

        for doc in main_chunks:
            meta = doc.metadata

            # Önceki komşular
            for i in range(1, k_neighbors + 1):
                prev_key = "prev_chunk_id" if i == 1 else f"prev_chunk_id_{i}"
                prev_meta_id = meta.get(prev_key)
                if prev_meta_id and prev_meta_id not in seen_ids:
                    neighbor = self._fetch_chunk_by_id(prev_meta_id)
                    if neighbor:
                        neighbors.append(neighbor)
                        seen_ids.add(prev_meta_id)
                        logger.debug(f"Komşu chunk: prev={prev_meta_id}")

            # Sonraki komşular
            for i in range(1, k_neighbors + 1):
                next_key = "next_chunk_id" if i == 1 else f"next_chunk_id_{i}"
                next_meta_id = meta.get(next_key)
                if next_meta_id and next_meta_id not in seen_ids:
                    neighbor = self._fetch_chunk_by_id(next_meta_id)
                    if neighbor:
                        neighbors.append(neighbor)
                        seen_ids.add(next_meta_id)
                        logger.debug(f"Komşu chunk: next={next_meta_id}")

        if neighbors:
            logger.info(f"Neighbor chunks eklendi: {len(neighbors)} komşu")
            return main_chunks + neighbors

        return main_chunks

    def generate(self, question: str, context: str) -> GenerationResult:
        # Prompt zinciri ile yanıt üret ve gerekiyorsa think etiketlerini temizle.
        start_time = time.time()

        chain = self.prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        thinking_pattern = _get_thinking_pattern(self.model_name)
        if thinking_pattern:
            answer = re.sub(
                thinking_pattern,
                "",
                answer,
                flags=re.DOTALL,
            ).strip()

        latency_ms = (time.time() - start_time) * 1000
        token_usage = getattr(self.llm, "last_token_usage", {}) or {}

        return GenerationResult(
            answer=answer,
            latency_ms=latency_ms,
            token_usage=token_usage,
            model_name=self.model_name,
        )

    def query(
        self,
        question: str,
        k: Optional[int] = None,  # Gelecekteki override için; şu an RETRIEVAL_CFG["k"] geçerli
    ) -> RAGResult:
        # Tam akış: retrieve -> rerank -> komşu zenginleştirme -> generate.
        if not question or not question.strip():
            raise ValueError("Soru metni boş olamaz.")

        total_start = time.time()

        reranked_docs, qdrant_ms, reranker_ms = self._retrieve_and_rerank(question)

        # Komşu chunk zenginleştirmesi.
        final_docs = self.retrieve_neighbor_chunks(reranked_docs)

        retrieval_result = RetrievalResult(
            documents=final_docs,
            scores=[],
            latency_ms=qdrant_ms + reranker_ms,
            qdrant_latency_ms=qdrant_ms,
            reranker_latency_ms=reranker_ms,
        )

        # Context birleştir ve mülga uyarılarını topla.
        context_parts = []
        mulga_warnings = []
        seen_warnings = set()

        for i, doc in enumerate(final_docs, 1):
            law_no = doc.metadata.get("law_number", "")
            article_no = doc.metadata.get("article_number", "")

            warning = None
            if _law_resolver:
                warning = _law_resolver.build_context_warning(law_no, article_no)

            chunk_text = doc.page_content
            if warning:
                chunk_text = warning + "\n\n" + chunk_text
                if warning not in seen_warnings:
                    mulga_warnings.append(warning)
                    seen_warnings.add(warning)

            logger.debug(
                f"  [{i}] {doc.metadata.get('source', 'N/A')[:25]} | "
                f"{doc.metadata.get('article_reference', '')} | "
                f"{len(chunk_text)} char"
            )
            context_parts.append(chunk_text)

        context = "\n\n---\n\n".join(context_parts)

        # Nihai cevabı üret.
        generation_result = self.generate(question, context)
        total_latency = (time.time() - total_start) * 1000

        logger.info(
            f"Query tamamlandı | "
            f"Qdrant={qdrant_ms:.0f}ms  "
            f"Reranker={reranker_ms:.0f}ms  "
            f"LLM={generation_result.latency_ms:.0f}ms  "
            f"Toplam={total_latency:.0f}ms | "
            f"Docs: {len(reranked_docs)} ranked → {len(final_docs)} (komşularla)"
        )

        return RAGResult(
            question=question,
            answer=generation_result.answer,
            retrieval=retrieval_result,
            generation=generation_result,
            total_latency_ms=total_latency,
            mulga_warnings=mulga_warnings,
        )

    def stream_query(self, question: str):
        # Yanıtı stream ederken düşünme etiketlerini stateful şekilde temizle.
        if not question or not question.strip():
            raise ValueError("Soru metni boş olamaz.")

        if not STREAM_ENABLED:
            logger.warning("Streaming devre dışı, normal sorgu moduna geçiliyor")
            result = self.query(question)
            yield result.answer
            return

        reranked_docs, qdrant_ms, reranker_ms = self._retrieve_and_rerank(question)
        final_docs = self.retrieve_neighbor_chunks(reranked_docs)

        logger.debug(
            f"Stream retrieval: Qdrant={qdrant_ms:.0f}ms, "
            f"Reranker={reranker_ms:.0f}ms, docs={len(final_docs)}"
        )

        context_parts = []
        for doc in final_docs:
            law_no = doc.metadata.get("law_number", "")
            article_no = doc.metadata.get("article_number", "")

            warning = None
            if _law_resolver:
                warning = _law_resolver.build_context_warning(law_no, article_no)

            chunk_text = doc.page_content
            if warning:
                chunk_text = warning + "\n\n" + chunk_text
            context_parts.append(chunk_text)

        context = "\n\n---\n\n".join(context_parts)

        model_lower = self.model_name.lower()
        if "claude" in model_lower:
            open_tag = "<claude_thinking>"
            close_tag = "</claude_thinking>"
        elif "deepseek" in model_lower or "qwen" in model_lower:
            open_tag = "<think>"
            close_tag = "</think>"
        else:
            open_tag = ""
            close_tag = ""

        try:
            chain = self.prompt | self.llm | StrOutputParser()
            output_buffer = ""
            pending = ""
            is_thinking = False

            for chunk in chain.stream({"context": context, "question": question}):
                if not chunk:
                    continue

                if not open_tag:
                    output_buffer += chunk
                else:
                    data = pending + chunk
                    pending = ""
                    cursor = 0

                    while cursor < len(data):
                        if is_thinking:
                            end_idx = data.find(close_tag, cursor)
                            if end_idx == -1:
                                tail_len = max(len(close_tag) - 1, 0)
                                keep_from = max(cursor, len(data) - tail_len)
                                pending = data[keep_from:]
                                cursor = len(data)
                            else:
                                cursor = end_idx + len(close_tag)
                                is_thinking = False
                        else:
                            start_idx = data.find(open_tag, cursor)
                            if start_idx == -1:
                                tail_len = max(len(open_tag) - 1, 0)
                                if tail_len:
                                    safe_end = max(cursor, len(data) - tail_len)
                                    if safe_end > cursor:
                                        output_buffer += data[cursor:safe_end]
                                    pending = data[safe_end:]
                                else:
                                    output_buffer += data[cursor:]
                                cursor = len(data)
                            else:
                                if start_idx > cursor:
                                    output_buffer += data[cursor:start_idx]
                                cursor = start_idx + len(open_tag)
                                is_thinking = True

                while len(output_buffer) >= STREAM_CHUNK_SIZE:
                    emit_chunk = output_buffer[:STREAM_CHUNK_SIZE]
                    output_buffer = output_buffer[STREAM_CHUNK_SIZE:]
                    yield emit_chunk

            if open_tag and (not is_thinking) and pending:
                output_buffer += pending

            if output_buffer.strip():
                yield output_buffer

        except Exception as e:
            logger.error(f"Stream generation hatası: {e}", exc_info=True)
            raise RuntimeError(f"Generation streaming başarısız: {e}") from e

    def batch_query(
        self,
        questions: List[str],
        delay_seconds: float = 1.0,
        show_progress: bool = True,
    ) -> List[RAGResult]:
        # Soruları sırayla çalıştır, hata olsa da batch'i sürdür.
        if not questions or not isinstance(questions, list):
            raise ValueError("Sorular listesi boş olamaz.")

        results = []
        total = len(questions)
        failed_count = 0

        for i, question in enumerate(questions):
            if show_progress:
                logger.info(f"Batch [{i + 1}/{total}]: {question[:60]}...")

            try:
                results.append(self.query(question))

            except ValueError as e:
                logger.warning(f"Soru #{i + 1} validation hatası: {e}")
                failed_count += 1
                results.append(
                    RAGResult(
                        question=question,
                        answer=f"VALIDATION_ERROR: {e}",
                        retrieval=RetrievalResult(documents=[]),
                        generation=GenerationResult(answer="", model_name=self.model_name),
                        total_latency_ms=0,
                    )
                )
            except RuntimeError as e:
                logger.error(f"Soru #{i + 1} runtime hatası: {e}")
                failed_count += 1
                results.append(
                    RAGResult(
                        question=question,
                        answer=f"RUNTIME_ERROR: {e}",
                        retrieval=RetrievalResult(documents=[]),
                        generation=GenerationResult(answer="", model_name=self.model_name),
                        total_latency_ms=0,
                    )
                )
            except Exception as e:
                logger.error(f"Soru #{i + 1} beklenmeyen hata: {e}", exc_info=True)
                failed_count += 1
                results.append(
                    RAGResult(
                        question=question,
                        answer=f"UNEXPECTED_ERROR: {e}",
                        retrieval=RetrievalResult(documents=[]),
                        generation=GenerationResult(answer="", model_name=self.model_name),
                        total_latency_ms=0,
                    )
                )

            if i < total - 1 and delay_seconds > 0:
                time.sleep(delay_seconds)

        if show_progress:
            success_rate = (total - failed_count) / total * 100
            logger.info(
                f"Batch tamamlandı: {total - failed_count}/{total} başarılı "
                f"({success_rate:.1f}%)"
            )

        return results

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    logger.info("RAG Pipeline Test başlıyor")
    pipeline = RAGPipeline()

    test_question = "Hırsızlıığın cezası nedir?"
    logger.info(f"Test sorusu: {test_question}")

    result = pipeline.query(test_question)

    logger.info(f"Qdrant   : {result.retrieval.qdrant_latency_ms:.0f}ms")
    logger.info(f"Reranker : {result.retrieval.reranker_latency_ms:.0f}ms")
    logger.info(f"LLM      : {result.generation.latency_ms:.0f}ms")
    logger.info(f"Toplam   : {result.total_latency_ms:.0f}ms")
    logger.info(f"Belgeler : {len(result.retrieval.metadata_list)}")

    for i, meta in enumerate(result.retrieval.metadata_list[:5], 1):
        source  = meta.get("source", "N/A")
        article = meta.get("article_reference", meta.get("article", "N/A"))
        part    = meta.get("chunk_part", "")
        logger.info(f"  {i}. {source} | {article} {part}")

    if result.mulga_warnings:
        logger.warning(f"Mülga uyarıları: {len(result.mulga_warnings)}")
        for w in result.mulga_warnings:
            logger.warning(f"  ⚠ {w[:80]}")

    if result.answer:
        logger.info(f"Cevap (ilk 500 karakter):\n{result.answer[:500]}")
    else:
        logger.warning("Cevap üretilmedi")