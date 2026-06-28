#!/usr/bin/env python
"""
Turkish Law RAG Backend Runner

Projeyi başlatmak için:
    python run.py
    
Tarayıcıda açın:
    http://localhost:8000
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
