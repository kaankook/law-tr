import os
import time
import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    MODEL_NAME,
    RETRIEVER_K,
    TEMPERATURE,
    PROMPT_TEMPLATE,
    MAPPING_FILE_PATH,
    SAMBANOVA_API_BASE_URL,
    OLLAMA_DEFAULT_BASE_URL,
    STREAM_ENABLED,
    STREAM_CHUNK_SIZE,
    STREAM_TIMEOUT,
)
from law_mapping_resolver import LawMappingResolver

logger = logging.getLogger(__name__)
_law_resolver = LawMappingResolver(MAPPING_FILE_PATH)

# ─────────────────────────────────────────────────────────────
# MODEL-SPECIFIC THINKING TAG PATTERNS
# ─────────────────────────────────────────────────────────────
# Maps model names to regex patterns for removing thinking/reasoning tags
# Some models include internal reasoning in output that should be stripped
MODEL_THINKING_PATTERNS: Dict[str, str] = {
    "deepseek": r"<think>.*?</think>\s*",
    "qwen": r"<think>.*?</think>\s*",
    "claude": r"<claude_thinking>.*?</claude_thinking>\s*",
    # Most OpenAI models (GPT-4, GPT-3.5) don't include thinking tags in standard output
    # Gemini may include thinking in certain modes
}


def _get_thinking_pattern(model_name: str) -> Optional[str]:
    """Get thinking tag regex pattern for model.
    
    Args:
        model_name: Model identifier (e.g., 'gpt-4', 'deepseek-v3')
    
    Returns:
        Regex pattern string or None if model has no thinking tags
    """
    model_lower = model_name.lower()
    for model_key, pattern in MODEL_THINKING_PATTERNS.items():
        if model_key in model_lower:
            return pattern
    return None


def _validate_config() -> None:
    """Validate configuration values at startup.
    
    Checks critical config parameters and logs warnings/errors.
    
    Raises:
        ValueError: If critical configuration is missing or invalid
    """
    logger.info("Konfigürasyon validasyonu başlıyor...")
    
    # Validate RETRIEVER_K
    if not isinstance(RETRIEVER_K, int) or RETRIEVER_K <= 0 or RETRIEVER_K > 100:
        raise ValueError(
            f"Config hatası: RETRIEVER_K geçersiz ({RETRIEVER_K}). "
            "Değer 1-100 aralığında pozitif bir tam sayı olmalıdır."
        )
    logger.debug(f"  ✓ RETRIEVER_K: {RETRIEVER_K}")
    
    # Validate TEMPERATURE
    if not isinstance(TEMPERATURE, (int, float)) or TEMPERATURE < 0 or TEMPERATURE > 2:
        raise ValueError(
            f"Config hatası: TEMPERATURE geçersiz ({TEMPERATURE}). "
            "Değer 0-2 aralığında olmalıdır."
        )
    logger.debug(f"  ✓ TEMPERATURE: {TEMPERATURE}")
    
    # Validate PROMPT_TEMPLATE
    if not PROMPT_TEMPLATE or not isinstance(PROMPT_TEMPLATE, str):
        raise ValueError(
            "Config hatası: PROMPT_TEMPLATE boş veya geçersiz tip. "
            "Geçerli bir prompt şablonu gereklidir."
        )
    if "{context}" not in PROMPT_TEMPLATE or "{question}" not in PROMPT_TEMPLATE:
        raise ValueError(
            "Config hatası: PROMPT_TEMPLATE \'{context}\'  ve \'{question}\' yer tutucularını içermemelidir. "
            "Bu yer tutucuları ekleyin."
        )
    logger.debug(f"  ✓ PROMPT_TEMPLATE: {len(PROMPT_TEMPLATE)} karakter")
    
    # Validate DB_PATH
    if not DB_PATH or not isinstance(DB_PATH, str):
        raise ValueError(
            "Config hatası: DB_PATH boş veya geçersiz tip. "
            "Harika bir vektör veritabanı yolu gereklidir."
        )
    logger.debug(f"  ✓ DB_PATH: {DB_PATH}")
    
    # Validate COLLECTION_NAME
    if not COLLECTION_NAME or not isinstance(COLLECTION_NAME, str):
        raise ValueError(
            "Config hatası: COLLECTION_NAME boş veya geçersiz tip. "
            "Collection adı gereklidir."
        )
    logger.debug(f"  ✓ COLLECTION_NAME: {COLLECTION_NAME}")
    
    # Validate EMBEDDING_MODEL_NAME
    if not EMBEDDING_MODEL_NAME or not isinstance(EMBEDDING_MODEL_NAME, str):
        raise ValueError(
            "Config hatası: EMBEDDING_MODEL_NAME boş veya geçersiz tip. "
            "Embedding modeli adı gereklidir."
        )
    logger.debug(f"  ✓ EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")
    
    # Validate MODEL_NAME
    if not MODEL_NAME or not isinstance(MODEL_NAME, str):
        raise ValueError(
            "Config hatası: MODEL_NAME boş veya geçersiz tip. "
            "LLM modeli adı gereklidir."
        )
    logger.debug(f"  ✓ MODEL_NAME: {MODEL_NAME}")
    
    # Validate MAPPING_FILE_PATH
    if not MAPPING_FILE_PATH or not isinstance(MAPPING_FILE_PATH, str):
        logger.warning("Config uyarı: MAPPING_FILE_PATH belirtilmemiş, mülga eşlemesi devre dışı olabilir")
    else:
        logger.debug(f"  ✓ MAPPING_FILE_PATH: {MAPPING_FILE_PATH}")
    
    logger.info("✓ Konfigürasyon validasyonu başarılı")


# De-lazy validate config when module loads
try:
    _validate_config()
except ValueError as e:
    logger.error(f"Konfigürasyon hatası: {e}")
    raise


def create_chat_model(model_name: str, temperature: float = TEMPERATURE) -> Union[ChatGoogleGenerativeAI, ChatOpenAI, ChatOllama]:
    """Create and return appropriate chat model based on model name.
    
    Supports: Gemini, GPT/OpenAI, Qwen (via SambaNova, native, or Ollama fallback), Ollama.
    
    Args:
        model_name: Model identifier (e.g., 'gpt-4', 'gemini-2.0', 'qwen-32b')
        temperature: Temperature parameter (0-2 for GPT, 0-1 for others)
    
    Returns:
        Configured chat model instance
    
    Raises:
        ValueError: If required API keys not found for the model
    """
    model_lower = model_name.lower()

    if "gemini" in model_lower:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Google Gemini API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            convert_system_message_to_human=True,
            api_key=api_key
        )

    if "gpt" in model_lower or "openai" in model_lower:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            )
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

    if "qwen" in model_lower:
        sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")
        qwen_api_key = os.getenv("QWEN_API_KEY")
        qwen_base_url = os.getenv("QWEN_BASE_URL")

        if sambanova_api_key:
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=sambanova_api_key,
                base_url=SAMBANOVA_API_BASE_URL
            )
        if qwen_api_key and qwen_base_url:
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=qwen_api_key,
                base_url=qwen_base_url
            )

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=ollama_base_url
        )

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        convert_system_message_to_human=True
    )


def get_article_reference(metadata: Dict[str, Any]) -> str:
    """Extract standardized article reference from chunk metadata.
    
    Tries 'article_reference' field first, falls back to 'article' for backward compatibility.
    
    Args:
        metadata: Chunk metadata dictionary
    
    Returns:
        Article reference string (e.g., "Madde 5") or empty string if not found
    """
    return metadata.get("article_reference") or metadata.get("article", "")


@dataclass
class RetrievalResult:
    """Result object from semantic document retrieval.
    
    Attributes:
        documents: Retrieved LangChain documents
        scores: Cosine similarity scores (0-1, lower is better)
        latency_ms: Retrieval operation latency in milliseconds
    """
    documents: List[Document]
    scores: List[float] = field(default_factory=list)
    latency_ms: float = 0.0
    
    @property
    def contexts(self) -> List[str]:
        """Extract page contents from all retrieved documents.
        
        Returns:
            List of document texts
        """
        return [doc.page_content for doc in self.documents]
    
    @property
    def metadata_list(self) -> List[Dict[str, Any]]:
        """Extract metadata from all retrieved documents.
        
        Returns:
            List of metadata dictionaries
        """
        return [doc.metadata for doc in self.documents]


@dataclass
class GenerationResult:
    """Result object from LLM generation.
    
    Attributes:
        answer: Generated answer text (thinking tags stripped)
        latency_ms: LLM generation latency in milliseconds
        token_usage: Dictionary with input/output token counts (model-dependent)
        model_name: Name of model that generated this result
    """
    answer: str
    latency_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    model_name: str = ""


@dataclass
class RAGResult:
    """Complete result from RAG pipeline (retrieval + generation).
    
    Attributes:
        question: Original user question
        answer: Generated answer from LLM
        retrieval: Retrieval operation results (documents, scores, latency)
        generation: Generation operation results (answer, latency, tokens)
        total_latency_ms: Total end-to-end latency in milliseconds
    
    Methods:
        to_dict(): Serialize to dictionary for API responses or logging
    """
    question: str
    answer: str
    retrieval: RetrievalResult
    generation: GenerationResult
    total_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "retrieved_contexts": self.retrieval.contexts,
            "retrieved_articles": [get_article_reference(m) for m in self.retrieval.metadata_list],
            "retrieved_sources": [m.get('source', '') for m in self.retrieval.metadata_list],
            "retrieval_latency_ms": self.retrieval.latency_ms,
            "generation_latency_ms":  self.generation.latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "model_name": self.generation.model_name,
            "token_usage": self.generation.token_usage
        }


class RAGPipeline:
    """Turkish Legal Document RAG Pipeline.
    
    Orchestrates retrieval-augmented generation for Turkish legal documents:
    - Lazy-loads embeddings, vector store, and LLM with error handling
    - Supports multiple models (Gemini, OpenAI GPT, Qwen, Ollama)
    - Retrieves relevant documents via semantic search
    - Generates answers from context with model-specific post-processing
    - Batch processes multiple queries with rate limiting
    
    Example:
        >>> pipeline = RAGPipeline(model_name="gpt-4", retriever_k=5)
        >>> result = pipeline.query("SGK zaman aşımı kaç gündür?")
        >>> print(result.answer)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        db_path: Optional[str] = None,
        retriever_k: int = RETRIEVER_K,
        temperature: float = TEMPERATURE
    ):
        """Initialize RAG pipeline with optional custom configuration.
        
        Args:
            model_name: LLM identifier (default: from config)
            embedding_model: Embedding model name (default: from config)
            db_path: Vector store directory (default: from config)
            retriever_k: Number of documents to retrieve (default: from config)
            temperature: LLM temperature parameter (default: from config)
        
        Raises:
            ValueError: If retriever_k invalid or temperature out of range
        """
        # Validate retriever_k
        if not isinstance(retriever_k, int) or retriever_k <= 0 or retriever_k > 100:
            raise ValueError(
                f"Geçersiz retriever_k değeri: {retriever_k}. "
                "Değer 1-100 aralığında olmalıdır."
            )
        
        # Validate temperature (model-agnostic check: 0-2 covers all major models)
        if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
            raise ValueError(
                f"Geçersiz temperature değeri: {temperature}. "
                "Değer 0-2 aralığında olmalıdır (GPT için 0-2, diğer modeller için 0-1)."
            )
        
        self.model_name = model_name or MODEL_NAME
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL_NAME
        self.db_path = db_path or DB_PATH
        self.retriever_k = retriever_k
        self.temperature = temperature
        
        logger.info(
            f"RAGPipeline başlatıldı: model={self.model_name}, "
            f"embedding={self.embedding_model_name}, retriever_k={self.retriever_k}"
        )
        
        # Lazy loading için None olarak başlat
        self._embeddings = None
        self._vectorstore = None
        self._llm = None
        self._prompt = None
        
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load HuggingFace embeddings model with error handling.
        
        Returns:
            HuggingFaceEmbeddings instance
        
        Raises:
            RuntimeError: If embedding model download/initialization fails
        """
        if self._embeddings is None:
            try:
                logger.info(f"Embedding modeli yükleniyor: {self.embedding_model_name}")
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                logger.info("Embedding modeli başarıyla yüklendi")
            except Exception as e:
                logger.error(f"Embedding modeli yükleme hatası: {e}", exc_info=True)
                raise RuntimeError(
                    f"Embedding modeli '{self.embedding_model_name}' yüklenemedi: {e}. "
                    "HuggingFace bağlantısını, model adını ve internet bağlantısını kontrol edin."
                ) from e
        return self._embeddings
    
    @property
    def vectorstore(self) -> Chroma:
        """Lazy-load Chroma vector store with error handling.
        
        Returns:
            Chroma vector store instance
        
        Raises:
            RuntimeError: If vector store initialization or collection access fails
        """
        if self._vectorstore is None:
            try:
                logger.info(f"Vektör veritabanı yükleniyor: {self.db_path}")
                self._vectorstore = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=self.embeddings,
                    collection_name=COLLECTION_NAME
                )
                logger.info(f"Vektör veritabanı başarıyla yüklendi (Collection: {COLLECTION_NAME})")
            except Exception as e:
                logger.error(f"Vektör veritabanı yükleme hatası: {e}", exc_info=True)
                raise RuntimeError(
                    f"Vektör veritabanı '{self.db_path}' yüklenemedi: {e}. "
                    "Veri tabanı dosyasının var olduğunu, yazma izinlerini ve collection adını kontrol edin."
                ) from e
        return self._vectorstore
    
    @property
    def llm(self) -> Union[ChatGoogleGenerativeAI, ChatOpenAI, ChatOllama]:
        """Lazy-load language model with error handling.
        
        Returns:
            Configured chat model instance
        
        Raises:
            RuntimeError: If model initialization or API connection fails
        """
        if self._llm is None:
            try:
                logger.info(f"LLM yükleniyor: {self.model_name}")
                self._llm = self._create_llm(self.model_name)
                logger.info(f"LLM başarıyla yüklendi: {self.model_name}")
            except ValueError as e:
                logger.error(f"LLM konfigürasyon hatası: {e}")
                raise RuntimeError(f"LLM konfigürasyonu başarısız: {e}") from e
            except Exception as e:
                logger.error(f"LLM yükleme hatası: {e}", exc_info=True)
                raise RuntimeError(
                    f"LLM '{self.model_name}' yüklenemedi: {e}. "
                    "Model adını, API anahtarlarını ve ağ bağlantısını kontrol edin."
                ) from e
        return self._llm
    
    def _create_llm(self, model_name: str):
        return create_chat_model(model_name, temperature=self.temperature)
    
    @property
    def prompt(self) -> ChatPromptTemplate:
        """Lazy-load prompt template for LLM chain.
        
        Returns:
            ChatPromptTemplate with {context} and {question} placeholders
        """
        if self._prompt is None:
            self._prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        return self._prompt
    
    def retrieve(self, question: str, k: Optional[int] = None) -> RetrievalResult:
        """Retrieve similar documents from vector store using semantic search.
        
        Args:
            question: Query text to find similar documents
            k: Number of documents to retrieve (default: self.retriever_k)
        
        Returns:
            RetrievalResult containing documents, similarity scores, and latency
        
        Raises:
            ValueError: If k parameter is invalid (must be 0 < k <= 100)
        """
        k = self.retriever_k if k is None else k
        
        # Validate k parameter
        if not isinstance(k, int) or k <= 0 or k > 100:
            raise ValueError(
                f"Geçersiz k parametresi: {k}. k değeri 1-100 aralığında olmalıdır. "
                f"Boş sonuçlar (k=0) veya aşırı büyük sonuç kümeleri (k>100) istenmiyor."
            )
        
        if not question or not question.strip():
            raise ValueError(
                "Soru metni boş olamaz. Lütfen geçerli bir soru girin."
            )
        
        start_time = time.time()
        
        results_with_scores = self.vectorstore.similarity_search_with_score(
            question, 
            k=k
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        documents = [doc for doc, _ in results_with_scores]
        scores = [score for _, score in results_with_scores]
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            latency_ms=latency_ms
        )
    
    def generate(self, question: str, context: str) -> GenerationResult:
        """Generate answer from question and context using LLM chain.
        
        Strips model-specific thinking tags based on model type:
        - Deepseek/Qwen: <think>...</think>
        - Claude: <claude_thinking>...</claude_thinking>
        
        Args:
            question: User's original question
            context: Concatenated context from retrieved documents
        
        Returns:
            GenerationResult with answer, latency, and token usage
        
        Raises:
            Any LLM-specific exception (propagated from model or chain)
        """
        start_time = time.time()
        
        chain = self.prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context,
            "question": question
        })
        
        # Strip model-specific thinking tags if applicable
        thinking_pattern = _get_thinking_pattern(self.model_name)
        if thinking_pattern:
            logger.debug(f"Thinking tags temizleniyor ({self.model_name})")
            answer = re.sub(thinking_pattern, '', answer, flags=re.DOTALL)
        
        latency_ms = (time.time() - start_time) * 1000
        
        token_usage = {}
        if hasattr(self.llm, 'last_token_usage'):
            token_usage = self.llm.last_token_usage
        
        return GenerationResult(
            answer=answer,
            latency_ms=latency_ms,
            token_usage=token_usage,
            model_name=self.model_name
        )
    
    def query(self, question: str, k: Optional[int] = None) -> RAGResult:
        """Execute full RAG pipeline: retrieve documents and generate answer.
        
        Workflow:
        1. Retrieve most relevant documents via semantic search
        2. Build context with mülga (deprecated law) warnings if applicable
        3. Generate answer from context
        4. Return combined results with latencies
        
        Args:
            question: User's question
            k: Number of documents to retrieve (default: self.retriever_k)
        
        Returns:
            RAGResult with question, answer, retrieval, generation, and latencies
        
        Raises:
            ValueError: If question invalid or k out of range
            RuntimeError: If retrieval or generation fails
        """
        total_start = time.time()
        
        # 1. Retrieval
        retrieval_result = self.retrieve(question, k)
        
        # 2. Mulga uyarıları ile context oluştur
        context_parts = []
        for doc in retrieval_result.documents:
            law_no = doc.metadata.get("law_number", "")
            article_no = doc.metadata.get("article_number", "")
            warning = _law_resolver.build_context_warning(law_no, article_no)
            chunk_text = doc.page_content
            if warning:
                chunk_text = warning + "\n\n" + chunk_text
            context_parts.append(chunk_text)
        context = "\n\n---\n\n".join(context_parts)
        
        # 3. Generation
        generation_result = self.generate(question, context)
        
        total_latency = (time.time() - total_start) * 1000
        
        return RAGResult(
            question=question,
            answer=generation_result.answer,
            retrieval=retrieval_result,
            generation=generation_result,
            total_latency_ms=total_latency
        )
    
    def stream_query(self, question: str, k: Optional[int] = None):
        """Stream answer generation in chunks (generator-based).
        
        Yields answer text in chunks instead of waiting for full completion.
        Useful for long responses and real-time UX.
        
        Args:
            question: User's question
            k: Number of documents to retrieve (default: self.retriever_k)
        
        Yields:
            String chunks of generated answer
        
        Raises:
            ValueError: If question invalid or k out of range
            RuntimeError: If retrieval or generation fails
        
        Note:
            Requires STREAM_ENABLED config to be true.
            Falls back to regular query() if streaming not supported by model.
        """
        if not STREAM_ENABLED:
            logger.warning("Streaming devre dışı, normal sorgu moduna geçiliyor")
            result = self.query(question, k)
            yield result.answer
            return
        
        # Retrieve documents
        retrieval_result = self.retrieve(question, k)
        
        # Build context with mülga warnings
        context_parts = []
        for doc in retrieval_result.documents:
            law_no = doc.metadata.get("law_number", "")
            article_no = doc.metadata.get("article_number", "")
            warning = _law_resolver.build_context_warning(law_no, article_no)
            chunk_text = doc.page_content
            if warning:
                chunk_text = warning + "\n\n" + chunk_text
            context_parts.append(chunk_text)
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate and stream answer
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            
            # Attempt to stream from LLM (if supported)
            full_answer = ""
            if hasattr(chain.llm, 'stream'):
                logger.debug(f"Streaming {self.model_name} ile başlandı")
                for chunk in chain.stream({"context": context, "question": question}):
                    if chunk:
                        full_answer += chunk
                        # Yield when chunk threshold reached
                        if len(full_answer) >= STREAM_CHUNK_SIZE:
                            yield full_answer
                            full_answer = ""
            else:
                # Fallback: model doesn't support streaming
                logger.debug(f"{self.model_name} streaming desteklemiyor, tam sonuç dönüşü yapılıyor")
                full_answer = chain.invoke({"context": context, "question": question})
                yield full_answer
                return
            
            # Yield remaining content
            if full_answer:
                yield full_answer
                
        except Exception as e:
            logger.error(f"Stream generation hatası: {e}", exc_info=True)
            raise RuntimeError(f"Generation streaming başarısız: {e}") from e
    
    def batch_query(
        self, 
        questions: List[str], 
        k: Optional[int] = None,
        delay_seconds: float = 1.0,
        show_progress: bool = True
    ) -> List[RAGResult]:
        """Process multiple questions in batch with rate limiting and error handling.
        
        Args:
            questions: List of question texts to process
            k: Number of documents to retrieve per question (default: self.retriever_k)
            delay_seconds: Delay between queries for rate limiting (default: 1.0)
            show_progress: Whether to log progress indicators (default: True)
        
        Returns:
            List of RAGResult objects (one per question, even on error)
        
        Raises:
            ValueError: If questions list is empty or None
        """
        if not questions or not isinstance(questions, list) or len(questions) == 0:
            raise ValueError(
                "Sorular listesi boş olamaz. Lütfen en az bir soru girin."
            )
        
        results = []
        total = len(questions)
        failed_count = 0
        
        for i, question in enumerate(questions):
            if show_progress:
                logger.info(f"Batch işleniyor [{i+1}/{total}]: {question[:50]}...")
            
            try:
                result = self.query(question, k)
                results.append(result)
            except ValueError as e:
                # Validation errors - client's fault
                logger.warning(f"Soru #{i+1} validation hatası: {e}")
                failed_count += 1
                results.append(RAGResult(
                    question=question,
                    answer=f"VALIDATION_ERROR: {str(e)}",
                    retrieval=RetrievalResult(documents=[]),
                    generation=GenerationResult(answer="", model_name=self.model_name),
                    total_latency_ms=0
                ))
            except RuntimeError as e:
                # Resource errors - environment issue
                logger.error(f"Soru #{i+1} runtime hatası: {e}")
                failed_count += 1
                results.append(RAGResult(
                    question=question,
                    answer=f"RUNTIME_ERROR: {str(e)}",
                    retrieval=RetrievalResult(documents=[]),
                    generation=GenerationResult(answer="", model_name=self.model_name),
                    total_latency_ms=0
                ))
            except Exception as e:
                # Unexpected errors - log with full trace
                logger.error(f"Soru #{i+1} beklenmeyen hatası: {e}", exc_info=True)
                failed_count += 1
                results.append(RAGResult(
                    question=question,
                    answer=f"UNEXPECTED_ERROR: {str(e)}",
                    retrieval=RetrievalResult(documents=[]),
                    generation=GenerationResult(answer="", model_name=self.model_name),
                    total_latency_ms=0
                ))
            
            # Rate limiting
            if i < total - 1 and delay_seconds > 0:
                time.sleep(delay_seconds)
        
        if show_progress:
            success_rate = ((total - failed_count) / total) * 100
            logger.info(f"Batch tamamlandı: {total - failed_count}/{total} başarılı ({success_rate:.1f}%)")
        
        return results


# Test için
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    
    logger.info("RAG Pipeline Test başlıyor")
    
    pipeline = RAGPipeline()
    
    test_question = "SGK zaman aşımı kaç gündür?"
    logger.info(f"Test sorusu: {test_question}")
    
    result = pipeline.query(test_question)
    
    logger.info(f"Retrieval Süresi: {result.retrieval.latency_ms:.0f}ms")
    logger.info(f"Generation Süresi: {result.generation.latency_ms:.0f}ms")
    logger.info(f"Toplam Süre: {result.total_latency_ms:.0f}ms")
    logger.info(f"Bulunan Maddeler: {len(result.retrieval.metadata_list)}")
    
    for i, meta in enumerate(result.retrieval.metadata_list[:3], 1):
        source = meta.get('source', 'N/A')
        article = meta.get('article_reference', meta.get('article', 'N/A'))
        logger.info(f"  {i}. {source} | {article}")
    
    if result.answer:
        logger.info(f"Cevap açılıyor (ilk 500 karakter): {result.answer[:500]}...")
    else:
        logger.warning("Cevap üretilmedi (Empty response)")