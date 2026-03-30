"""
Turkish Law RAG Assistant - Main Entry Point

Bu modül Turkish Law RAG sisteminin ana giriş noktasıdır.
Kullanıcılar hukuki sorular sorabilir ve ilgili kanun metinlerine
dayalı cevaplar alabilirler.
"""

import sys
import logging
from pathlib import Path

# Absolute imports düzeltmesi için src dizinini path'e ekle
SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR.parent))

from src.rag_pipeline import RAGPipeline
from src.config import MODEL_NAME, EVALUATOR_MODEL_NAME, RETRIEVER_K

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_rag():
    """RAG pipeline'ı başlat ve hazırla.
    
    Returns:
        RAGPipeline: Başlatılmış RAG pipeline nesnesi
        
    Raises:
        RuntimeError: Pipeline başlatılamadığında
    """
    try:
        logger.info(f"RAG Pipeline başlatılıyor... (Model: {MODEL_NAME})")
        rag = RAGPipeline()
        logger.info("RAG Pipeline başarıyla başlatıldı")
        return rag
    except Exception as e:
        logger.error(f"RAG Pipeline başlatma hatası: {e}", exc_info=True)
        raise RuntimeError(f"RAG Pipeline başlatılamadı: {e}") from e


def interactive_mode(rag: RAGPipeline) -> None:
    """RAG sistemi ile interaktif soru-cevap modu.
    
    Args:
        rag: Başlatılmış RAG pipeline nesnesi
    """
    logger.info("İnteraktif Mod başlatılıyor...")
    print("\n" + "="*70)
    print("Turkish Law RAG Assistant - Hukuki Bilgi Sorma Sistemi")
    print("="*70)
    print("\nSorunuzu Turkish dilinde girin (Çıkmak için 'quit' yazın):\n")
    
    while True:
        try:
            user_query = input("\n📝 Sorunuz: ").strip()
            
            if user_query.lower() in ['quit', 'çıkış', 'exit']:
                logger.info("Kullanıcı uygulamayı kapatıyor")
                print("\n✓ Görüşmek üzere!")
                break
            
            if not user_query:
                print("⚠️  Lütfen geçerli bir soru girin.")
                continue
            
            logger.info(f"Soru işleniyor: {user_query[:50]}...")
            print("\n⏳ Cevap hazırlanıyor...")
            
            result = rag.query(user_query)
            
            print("\n" + "-"*70)
            print("💬 CEVAP:")
            print("-"*70)
            print(result.answer)
            print("\n" + "-"*70)
            
            # Mülga kanun uyarıları varsa göster
            if result.mulga_warnings:
                print("\n⚠️  MÜLGA KANUN UYARILARI:")
                print("-"*70)
                for warning in result.mulga_warnings:
                    print(warning)
                print("-"*70)
            
            print(f"\n📚 Kaynak Sayısı: {len(result.retrieval.documents) if result.retrieval.documents else 0}")
            
            if result.retrieval.documents:
                print("\n📖 İlgili Kaynaklar:")
                for i, source in enumerate(result.retrieval.documents[:3], 1):
                    source_info = source.metadata.get('source', 'Bilinmeyen Kaynak')
                    print(f"  {i}. {source_info}")
                    
        except KeyboardInterrupt:
            logger.info("Kullanıcı Ctrl+C ile çıkış yaptı")
            print("\n\n✓ Program sonlandırılıyor...")
            break
        except Exception as e:
            logger.error(f"Sorgu işleme hatası: {e}", exc_info=True)
            print(f"\n❌ Hata oluştu: {str(e)}")
            print("Lütfen sorunuzu tekrar deneyin.\n")


def main():
    """Ana giriş noktası.
    
    RAG pipeline'ı başlatır ve interaktif modu çalıştırır.
    """
    try:
        # RAG sistemini başlat
        rag = initialize_rag()
        
        # İnteraktif mode geç
        interactive_mode(rag)
        
    except KeyboardInterrupt:
        logger.info("Program kullanıcı tarafından durduruldu")
        print("\n✓ Program kapatılıyor...")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Kritik hata: {e}", exc_info=True)
        print(f"\n❌ Kritik Hata: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
