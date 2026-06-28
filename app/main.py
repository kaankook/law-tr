"""
Turkish Law RAG Backend - FastAPI Application

Bu modül Turkish Law RAG sisteminin FastAPI backend'ini sağlar.
"""

import sys
import logging
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import path ayarla
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.config import MODEL_NAME, RETRIEVER_K

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG pipeline
rag: Optional[RAGPipeline] = None


# Pydantic models
class QueryRequest(BaseModel):
    """Soru sorma isteği"""
    question: str


class SourceInfo(BaseModel):
    """Kaynak belge bilgisi"""
    law_number: str
    article_reference: str
    source: Optional[str] = None
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """Soru cevaplama yanıtı"""
    answer: str
    sources: list[SourceInfo]
    error: Optional[str] = None


class ConfigResponse(BaseModel):
    """Sistem konfigürasyonu"""
    model: str
    top_k: int


# Lifespan context manager - RAG pipeline başlatma ve kapatma
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Uygulamanın başlatılması ve kapatılması sırasında çalışacak kod.
    RAG pipeline'ı burada başlatıyoruz.
    """
    # Startup
    global rag
    try:
        logger.info(f"RAG Pipeline başlatılıyor... (Model: {MODEL_NAME})")
        rag = RAGPipeline()
        logger.info("✓ RAG Pipeline başarıyla başlatıldı")
    except Exception as e:
        logger.error(f"✗ RAG Pipeline başlatma hatası: {e}", exc_info=True)
        raise RuntimeError(f"RAG Pipeline başlatılamadı: {e}") from e

    yield

    # Shutdown
    logger.info("Uygulama kapatılıyor...")
    rag = None


# FastAPI App
app = FastAPI(
    title="Turkish Law RAG API",
    description="Türk Hukuku RAG Sistemi - Backend API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files - HTML/CSS/JS
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_class=FileResponse)
async def root():
    """Ana sayfa - index.html'i serve et"""
    index_path = Path(__file__).parent.parent / "static" / "index.html"
    if not index_path.exists():
        logger.error(f"index.html bulunamadı: {index_path}")
        raise HTTPException(status_code=404, detail="index.html not found")
    return str(index_path)


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Sistem konfigürasyonu döndür"""
    return ConfigResponse(
        model=MODEL_NAME,
        top_k=RETRIEVER_K
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    RAG pipeline ile soru cevapla
    
    Request body:
        { "question": "string" }
    
    Response:
        {
            "answer": "string",
            "stats": { "qdrant_ms", "reranker_ms", "llm_ms", "total_ms" },
            "sources": [ { "law_number", "article_reference", "source", "score" } ],
            "error": null or "string"
        }
    """
    
    # Soru validasyonu
    if not request.question or not request.question.strip():
        logger.warning("Boş soru geldi")
        return QueryResponse(
            answer="",
            sources=[],
            error="Soru metni boş olamaz."
        )
    
    # RAG pipeline kontrolü
    if rag is None:
        logger.error("RAG pipeline yüklü değil")
        return QueryResponse(
            answer="",
            sources=[],
            error="RAG pipeline yüklü değil. Lütfen sunucuyu yeniden başlatın."
        )
    
    try:
        # RAG sorgusu
        logger.info(f"Sorgu alındı: {request.question[:50]}...")
        result = rag.query(request.question)
        
        # Kaynakları format et
        sources = []
        for doc in result.retrieval.documents:
            # metadata içindeki score değerini güvenli şekilde al
            raw_score = doc.metadata.get("score")
            
            sources.append(SourceInfo(
                law_number=doc.metadata.get("law_number", "N/A"),
                article_reference=doc.metadata.get("article_reference", "N/A"),
                source=doc.metadata.get("source", "N/A"),
                score=raw_score if isinstance(raw_score, float) else None
            ))
        
        return QueryResponse(
            answer=result.answer,
            sources=sources,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Sorgu işleme hatası: {e}", exc_info=True)
        return QueryResponse(
            answer="",
            sources=[],
            error=f"Sorgu işleme hatası: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Sistem sağlık kontrolü"""
    return {
        "status": "ok",
        "rag_ready": rag is not None,
        "model": MODEL_NAME
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global hata işleyicisi"""
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Sunucu hatası oluştu. Lütfen daha sonra tekrar deneyin.",
            "error": str(exc)
        }
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 50)
    logger.info("Turkish Law RAG Backend Başlatılıyor...")
    logger.info("=" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
