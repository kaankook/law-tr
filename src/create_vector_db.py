# python -m src.create_vector_db

from __future__ import annotations

import json
import importlib.util
import logging
import sys
from pathlib import Path
from typing import List

# Proje kökünü import yolu içine ekle.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.documents import Document as LangchainDocument

from src.config import (
    COLLECTION_NAME,
    DB_PATH,
    EMBEDDING_MODEL_NAME,
    JSON_FILE_PATH,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# Hibrit arama için kullanılacak sparse model.
SPARSE_MODEL_NAME = "Qdrant/bm25"


def _check_dependencies() -> None:
    required_modules = {
        "qdrant_client": "qdrant-client",
        "langchain_qdrant": "langchain-qdrant",
        "langchain_huggingface": "langchain-huggingface",
        "fastembed": "fastembed",
    }

    missing_packages = [
        package_name
        for module_name, package_name in required_modules.items()
        if importlib.util.find_spec(module_name) is None
    ]

    if missing_packages:
        install_command = f"pip install {' '.join(missing_packages)}"
        logger.error(
            "Eksik bağımlılıklar tespit edildi. Lütfen aşağıdaki komutu çalıştırın:\n"
            f"  {install_command}"
        )
        sys.exit(1)


def save_to_qdrant(
    documents: List[LangchainDocument],
    persist_directory: str = DB_PATH,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 50,
) -> bool:
    if not documents:
        logger.error("Kayıt yapılmadı: Boş belge listesi")
        raise ValueError("Documents list cannot be empty")

    # Modül import edildiğinde çökme olmaması için bağımlılıkları geç yükle.
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_qdrant import FastEmbedSparse, QdrantVectorStore
    except ImportError as e:
        raise RuntimeError(
            "Qdrant kayıt bağımlılıkları yüklenemedi. "
            "Lütfen _check_dependencies() çıktısındaki pip komutunu çalıştırın."
        ) from e

    try:
        logger.info("=" * 60)
        logger.info(f"Qdrant vektör veritabanına kayıt başlanıyor: {len(documents)} chunk")
        logger.info(f"Veritabanı yolu  : {persist_directory}")
        logger.info(f"Koleksiyon adı   : {collection_name}")
        logger.info(f"Batch boyutu     : {batch_size}")
        logger.info("=" * 60)

        # Yoğun (dense) embedding modeli.
        logger.info(f"Dense embedding modeli yükleniyor  : {EMBEDDING_MODEL_NAME}")
        try:
            dense_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"HuggingFaceEmbeddings başlatılamadı ({EMBEDDING_MODEL_NAME}): {e}"
            ) from e
        logger.info("✓ Dense embedding modeli yüklendi")

        # Anahtar kelime odaklı sparse embedding modeli.
        logger.info(f"Sparse (BM25) embedding modeli yükleniyor: {SPARSE_MODEL_NAME}")
        try:
            sparse_embeddings = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"FastEmbedSparse başlatılamadı ({SPARSE_MODEL_NAME}): {e}"
            ) from e
        logger.info("✓ Sparse (BM25) embedding modeli yüklendi")

        logger.info(f"Qdrant veritabanı başlatılıyor (yerel disk): {persist_directory}")
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Belgeleri sabit boyutlu parçalara ayır.
        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]
        total_batches = len(batches)
        first_batch = batches[0]
        
        logger.info(f"İlk batch işleniyor ve koleksiyon oluşturuluyor ({len(first_batch)} döküman)…")
        try:
            # İlk batch ile koleksiyon kurulur; sonraki batch'ler eklenir.
            qdrant_vector_store = QdrantVectorStore.from_documents(
                documents=first_batch,
                embedding=dense_embeddings,
                sparse_embedding=sparse_embeddings,
                path=persist_directory,
                collection_name=collection_name,
                retrieval_mode="hybrid",
            )
        except Exception as e:
            raise RuntimeError(f"QdrantVectorStore ilk batch hatası: {e}") from e
            
        logger.info(f"✓ Batch 1/{total_batches} kaydedildi ({len(first_batch)} döküman)")

        # Kalan batch'leri aynı koleksiyona ekle.
        for batch_num, batch in enumerate(batches[1:], start=2):

            try:
                qdrant_vector_store.add_documents(batch)
            except Exception as e:
                raise RuntimeError(
                    f"Batch {batch_num}/{total_batches} eklenirken hata: {e}"
                ) from e

            logger.info(
                f"✓ Batch {batch_num}/{total_batches} kaydedildi ({len(batch)} döküman)"
            )

        try:
            # Harici client açmadan mevcut store client'ı ile sayım doğrulaması yap.
            client = qdrant_vector_store.client
            count_result = client.count(collection_name=collection_name)
            logger.info(f"✓ Koleksiyon doğrulaması: {count_result.count} döküman Qdrant'ta")
        except Exception as e:
            logger.warning(f"Koleksiyon doğrulama yapılamadı (devam ediliyor): {e}")

        logger.info("✓ Qdrant vektör veritabanı başarıyla oluşturuldu ve kaydedildi")
        return True

    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        logger.error(f"Qdrant kayıt hatası: {e}", exc_info=True)
        raise RuntimeError(f"Vector store save failed: {e}") from e


def load_chunks_from_json(json_path: str) -> List[LangchainDocument]:
    try:
        logger.info(f"JSON dosyasından chunk'lar yükleniyor: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = [
            LangchainDocument(
                page_content=item["page_content"],
                metadata=item["metadata"],
            )
            for item in data
        ]
        logger.info(f"✓ {len(documents)} chunk yüklendi")
        return documents
    except Exception as e:
        logger.error(f"JSON yükleme hatası: {e}", exc_info=True)
        raise


def main() -> None:
    # CLI çalıştırmasında önce bağımlılıkları doğrula.
    _check_dependencies()

    logger.info("=" * 70)
    logger.info("### QDRANT HYBRID VECTOR DATABASE OLUŞTURMA ###")
    logger.info("=" * 70)

    try:
        # Hazırlanmış chunk JSON dosyasını yükle.
        documents = load_chunks_from_json(JSON_FILE_PATH)

        if not documents:
            logger.error(
                "Belge yüklenemedi — Önce data_ingestion.py çalıştırılması gerekmektedir."
            )
            sys.exit(1)

        # Vektör veritabanını oluştur ve dokümanları yaz.
        logger.info("=" * 70)
        save_to_qdrant(documents)
        logger.info("=" * 70)
        logger.info("✓ Qdrant Hybrid vector database oluşturuldu ve hazır!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"✗ Vector database oluşturma başarısız: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()