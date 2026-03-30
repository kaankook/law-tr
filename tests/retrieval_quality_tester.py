import json
import logging
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

# python -m src.retrieval_quality_tester

from src.rag_pipeline import RAGPipeline
from src.config import (
    RETRIEVAL_CFG,
    RERANKER_CFG,
    MIN_CHUNK_CHARS,
    COLLECTION_NAME
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# DEBUG EDİLECEK SORGU
# ============================================================
TEST_QUERY = "Türk Borçlar Kanunu'nda genel zamanaşımı süresi ne kadardır ve hangi alacaklar için daha kısa süre öngörülmüştür?"
# ============================================================

def debug_retrieval_flow(query: str):
    pipeline = RAGPipeline()
    report_steps = []
    
    logger.info(f"🔍 DEBUG BAŞLADI: '{query}'")

    qdrant_start = time.time()
    
    raw_results = pipeline.vectorstore.similarity_search_with_score(
        query, 
        k=50 
    )
    qdrant_ms = (time.time() - qdrant_start) * 1000
    
    logger.info(f"📡 Qdrant 50 aday getirdi ({qdrant_ms:.0f}ms)")

    processed_candidates = []

    for i, (doc, score) in enumerate(raw_results):
        ref = doc.metadata.get('article_reference', 'Bilinmiyor')
        content_len = len(doc.page_content)
        
        passed_qdrant_threshold = score >= RETRIEVAL_CFG.get("score_threshold", 0.0)
        passed_size_filter = content_len >= MIN_CHUNK_CHARS
        
        status = "✅ GEÇTİ"
        reason = ""
        if not passed_qdrant_threshold:
            status = "❌ ELENDİ (Düşük Skor)"
            reason = f"Skor {score:.4f} < Eşik {RETRIEVAL_CFG.get('score_threshold')}"
        elif not passed_size_filter:
            status = "❌ ELENDİ (Kısa Metin)"
            reason = f"Karakter {content_len} < Min {MIN_CHUNK_CHARS}"

        candidate_info = {
            "rank": i + 1,
            "ref": ref,
            "score": float(score),
            "len": content_len,
            "status": status,
            "reason": reason,
            "content": doc.page_content[:200] + "...",
            "metadata": doc.metadata
        }
        processed_candidates.append(candidate_info)

    survivors = [res for res in processed_candidates if "GEÇTİ" in res["status"]]
    reranker_results = []
    
    if survivors:
        # survivor dökümanlarını tekrar Document nesnesine çeviriyoruz
        from langchain_core.documents import Document
        survivor_docs = [Document(page_content=c["content"], metadata=c["metadata"]) for c in survivors]
        
        reranker_start = time.time()
        # Reranker skorlarını al
        pairs = [(query, d.page_content) for d in survivor_docs]
        rerank_scores = pipeline.reranker.predict(pairs)
        reranker_ms = (time.time() - reranker_start) * 1000
        
        for cand, r_score in zip(survivors, rerank_scores):
            passed_reranker = r_score >= RERANKER_CFG["threshold_score"]
            cand["reranker_score"] = float(r_score)
            cand["reranker_status"] = "✅ OK" if passed_reranker else "❌ ELENDİ (Reranker)"
            if not passed_reranker:
                cand["reason"] += f" | Reranker: {r_score:.4f} < {RERANKER_CFG['threshold_score']}"

    return {
        "query": query,
        "candidates": processed_candidates,
        "qdrant_ms": qdrant_ms,
        "config": {
            "q_threshold": RETRIEVAL_CFG.get("score_threshold"),
            "r_threshold": RERANKER_CFG["threshold_score"],
            "min_chars": MIN_CHUNK_CHARS,
            "k": RETRIEVAL_CFG["k"]
        }
    }

def generate_debug_html(data):
    """Gelişmiş Debug Raporu"""
    rows = ""
    for c in data["candidates"]:
        row_class = "table-danger" if "❌" in c["status"] or "❌" in c.get("reranker_status", "") else ""
        reranker_val = f"{c['reranker_score']:.4f}" if "reranker_score" in c else "N/A"
        
        rows += f"""
        <tr class="{row_class}">
            <td>{c['rank']}</td>
            <td><strong>{c['ref']}</strong></td>
            <td>{c['score']:.4f}</td>
            <td>{reranker_val}</td>
            <td>{c['len']}</td>
            <td>{c['status']}<br><small>{c.get('reranker_status','')}</small></td>
            <td style="font-size: 0.8rem;">{c['reason']}</td>
            <td style="font-size: 0.7rem; max-width: 300px;">{c['content']}</td>
        </tr>
        """

    html = f"""
    <html>
    <head>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <title>RAG Forensic Debug</title>
    </head>
    <body class="p-4 bg-light">
        <div class="container-fluid">
            <h2>🔬 RAG Forensic Retrieval Debug</h2>
            <div class="alert alert-info">Sorgu: <strong>{data['query']}</strong></div>
            
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="card p-3">
                        <h6>Ayarlar</h6>
                        <small>Q-Threshold: {data['config']['q_threshold']}</small><br>
                        <small>R-Threshold: {data['config']['r_threshold']}</small><br>
                        <small>Min Chars: {data['config']['min_chars']}</small>
                    </div>
                </div>
            </div>

            <table class="table table-bordered table-hover bg-white">
                <thead class="table-dark">
                    <tr>
                        <th>Sıra</th>
                        <th>Madde Referans</th>
                        <th>Qdrant Skoru</th>
                        <th>Reranker Skoru</th>
                        <th>Karakter</th>
                        <th>Durum</th>
                        <th>Elenme Nedeni</th>
                        <th>İçerik Önizleme</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    results = debug_retrieval_flow(TEST_QUERY)
    html = generate_debug_html(results)
    output_file = Path("retrieval_forensic.html")
    output_file.write_text(html, encoding="utf-8")
    logger.info(f"📊 Forensic rapor oluşturuldu: {output_file.absolute()}")
    import os
    os.startfile(output_file)