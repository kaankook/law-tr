import os
import time
from typing import List, Dict, Any, Optional, Tuple
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
    PROMPT_TEMPLATE
)


@dataclass
class RetrievalResult:
    documents:  List[Document]
    scores: List[float] = field(default_factory=list)
    latency_ms: float = 0.0
    
    @property
    def contexts(self) -> List[str]:
        return [doc.page_content for doc in self.documents]
    
    @property
    def metadata_list(self) -> List[Dict]:
        return [doc.metadata for doc in self.documents]


@dataclass
class GenerationResult: 
    answer: str
    latency_ms: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    model_name: str = ""


@dataclass
class RAGResult:
    question: str
    answer: str
    retrieval:  RetrievalResult
    generation: GenerationResult
    total_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "retrieved_contexts": self.retrieval.contexts,
            "retrieved_articles": [m.get('article', '') for m in self.retrieval.metadata_list],
            "retrieved_sources": [m.get('source', '') for m in self.retrieval.metadata_list],
            "retrieval_latency_ms": self.retrieval.latency_ms,
            "generation_latency_ms":  self.generation.latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "model_name": self.generation.model_name,
            "token_usage": self.generation.token_usage
        }


class RAGPipeline:
    def __init__(
        self,
        model_name: Optional[str] = None,
        embedding_model:  Optional[str] = None,
        db_path: Optional[str] = None,
        retriever_k: int = RETRIEVER_K,
        temperature:  float = TEMPERATURE
    ):
        self.model_name = model_name or MODEL_NAME
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL_NAME
        self. db_path = db_path or DB_PATH
        self.retriever_k = retriever_k
        self.temperature = temperature
        
        # Lazy loading için None olarak başlat
        self._embeddings = None
        self._vectorstore = None
        self._llm = None
        self._prompt = None
        
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            print(f"📦 Embedding modeli yükleniyor: {self. embedding_model_name}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        return self._embeddings
    
    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            print(f"📚 Vektör veritabanı yükleniyor:  {self.db_path}")
            self._vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings,
                collection_name=COLLECTION_NAME
            )
        return self._vectorstore
    
    @property
    def llm(self):
        if self._llm is None:
            print(f"🤖 LLM yükleniyor: {self. model_name}")
            self._llm = self._create_llm(self.model_name)
        return self._llm
    
    def _create_llm(self, model_name: str):
        import os
        model_lower = model_name.lower()
        
        if "gemini" in model_lower:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "Google Gemini API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
                )
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature,
                convert_system_message_to_human=True,
                api_key=api_key
            )
        elif "gpt" in model_lower or "openai" in model_lower: 
            return ChatOpenAI(
                model=model_name,
                temperature=self.temperature
            )
        elif "qwen" in model_lower:
            # Qwen modelleri için - SambaNova (ücretsiz), Together AI, veya Ollama
            sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")
            qwen_api_key = os.getenv("QWEN_API_KEY")
            qwen_base_url = os.getenv("QWEN_BASE_URL")
            
            if sambanova_api_key:
                # SambaNova Cloud - ÜCRETSİZ Qwen 2.5 72B
                return ChatOpenAI(
                    model=model_name,
                    temperature=self.temperature,
                    api_key=sambanova_api_key,
                    base_url="https://api.sambanova.ai/v1"
                )
            elif qwen_api_key and qwen_base_url:
                # OpenAI uyumlu API (Together AI, OpenRouter vb.)
                return ChatOpenAI(
                    model=model_name,
                    temperature=self.temperature,
                    api_key=qwen_api_key,
                    base_url=qwen_base_url
                )
            else:
                ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                return ChatOllama(
                    model=model_name,
                    temperature=self.temperature,
                    base_url=ollama_base_url
                )
        else:
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=self.temperature,
                convert_system_message_to_human=True
            )
    
    @property
    def prompt(self) -> ChatPromptTemplate:
        """Prompt template'i oluştur"""
        if self._prompt is None:
            self._prompt = ChatPromptTemplate. from_template(PROMPT_TEMPLATE)
        return self._prompt
    
    def retrieve(self, question: str, k: Optional[int] = None) -> RetrievalResult:

        k = self.retriever_k
        
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
        import re
        start_time = time.time()
        
        chain = self.prompt | self.llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context,
            "question":  question
        })
        
        answer = re.sub(r'<think>.*?</think>\s*', '', answer, flags=re.DOTALL)
        
        latency_ms = (time. time() - start_time) * 1000
        
        token_usage = {}
        if hasattr(self.llm, 'last_token_usage'):
            token_usage = self.llm. last_token_usage
        
        return GenerationResult(
            answer=answer,
            latency_ms=latency_ms,
            token_usage=token_usage,
            model_name=self. model_name
        )
    
    def query(self, question: str, k: Optional[int] = None) -> RAGResult:
        total_start = time.time()
        
        # 1. Retrieval
        retrieval_result = self.retrieve(question, k)
        
        # 2. Context oluştur
        context = "\n\n---\n\n".join(retrieval_result.contexts)
        
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
    
    def batch_query(
        self, 
        questions:  List[str], 
        k: Optional[int] = None,
        delay_seconds: float = 1.0,
        show_progress: bool = True
    ) -> List[RAGResult]: 
        results = []
        total = len(questions)
        
        for i, question in enumerate(questions):
            if show_progress:
                print(f"  [{i+1}/{total}] {question[: 50]}...")
            
            try:
                result = self.query(question, k)
                results.append(result)
            except Exception as e:
                print(f"  ❌ Hata: {e}")
                # Hata durumunda boş sonuç ekle
                results.append(RAGResult(
                    question=question,
                    answer=f"HATA: {str(e)}",
                    retrieval=RetrievalResult(documents=[]),
                    generation=GenerationResult(answer="", model_name=self.model_name),
                    total_latency_ms=0
                ))
            
            # Rate limiting
            if i < total - 1 and delay_seconds > 0:
                time.sleep(delay_seconds)
        
        return results


# Test için
if __name__ == "__main__":
    print("RAG Pipeline Test")
    print("=" * 50)
    
    pipeline = RAGPipeline()
    
    test_question = "Türkiye Cumhuriyeti hangi niteliklere sahip bir devlettir?"
    print(f"\n📝 Soru: {test_question}\n")
    
    result = pipeline.query(test_question)
    
    print(f"📊 Retrieval Süresi: {result.retrieval.latency_ms:.0f}ms")
    print(f"📊 Generation Süresi: {result.generation.latency_ms:.0f}ms")
    print(f"📊 Toplam Süre: {result.total_latency_ms:.0f}ms")
    print(f"\n🔍 Bulunan Maddeler:")
    for meta in result.retrieval.metadata_list[: 3]:
        print(f"  - {meta. get('source', 'N/A')} | {meta.get('article', 'N/A')}")
    
    print(f"\n💬 Cevap:\n{result.answer[: 500]}...")