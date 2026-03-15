"""Unit tests for RAG Pipeline module.

Tests RAGPipeline class functionality:
- Configuration validation
- Model initialization
- Lazy loading behavior
- Document retrieval
- Answer generation
- Batch processing
- Error handling
"""

import logging
import pytest
from typing import Optional

from src.config import (
    MODEL_NAME,
    RETRIEVER_K,
    TEMPERATURE,
    SAMBANOVA_API_BASE_URL,
    OLLAMA_DEFAULT_BASE_URL,
)
from src.rag_pipeline import (
    RAGPipeline,
    RetrievalResult,
    GenerationResult,
    RAGResult,
    create_chat_model,
    get_article_reference,
    _get_thinking_pattern,
    _validate_config,
)

logger = logging.getLogger(__name__)


class TestConfiguration:
    """Test configuration validation."""
    
    def test_config_validation_succeeds(self) -> None:
        """Test that config validation passes with valid config."""
        # Should not raise
        _validate_config()
    
    def test_thinking_patterns(self) -> None:
        """Test model thinking pattern detection."""
        # Deepseek should map to <think> pattern
        pattern = _get_thinking_pattern("deepseek-v3")
        assert pattern is not None
        assert "<think>" in pattern
        
        # Claude should map to <claude_thinking> pattern
        pattern = _get_thinking_pattern("claude-3.5-sonnet")
        assert pattern is not None
        assert "claude_thinking" in pattern
        
        # GPT should have no pattern
        pattern = _get_thinking_pattern("gpt-4")
        assert pattern is None


class TestDataClasses:
    """Test dataclass structures."""
    
    def test_retrieval_result(self) -> None:
        """Test RetrievalResult dataclass."""
        result = RetrievalResult(documents=[], scores=[], latency_ms=100.5)
        assert result.documents == []
        assert result.scores == []
        assert result.latency_ms == 100.5
        assert result.contexts == []
        assert result.metadata_list == []
    
    def test_generation_result(self) -> None:
        """Test GenerationResult dataclass."""
        result = GenerationResult(
            answer="Test answer",
            latency_ms=50.0,
            token_usage={"input": 10, "output": 20},
            model_name="gpt-4"
        )
        assert result.answer == "Test answer"
        assert result.latency_ms == 50.0
        assert result.token_usage == {"input": 10, "output": 20}
        assert result.model_name == "gpt-4"
    
    def test_rag_result_to_dict(self) -> None:
        """Test RAGResult serialization."""
        retrieval = RetrievalResult(documents=[], scores=[], latency_ms=100.0)
        generation = GenerationResult(answer="Test", latency_ms=50.0, model_name="test-model")
        result = RAGResult(
            question="Test?",
            answer="Test",
            retrieval=retrieval,
            generation=generation,
            total_latency_ms=150.0
        )
        
        result_dict = result.to_dict()
        assert result_dict["question"] == "Test?"
        assert result_dict["answer"] == "Test"
        assert result_dict["retrieval_latency_ms"] == 100.0
        assert result_dict["generation_latency_ms"] == 50.0
        assert result_dict["total_latency_ms"] == 150.0
        assert result_dict["model_name"] == "test-model"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_article_reference_primary_field(self) -> None:
        """Test article reference extraction with primary field."""
        metadata = {"article_reference": "Madde 5", "article": "Madde 3"}
        ref = get_article_reference(metadata)
        assert ref == "Madde 5"
    
    def test_get_article_reference_fallback_field(self) -> None:
        """Test article reference extraction with fallback."""
        metadata = {"article": "Madde 3"}
        ref = get_article_reference(metadata)
        assert ref == "Madde 3"
    
    def test_get_article_reference_empty(self) -> None:
        """Test article reference extraction when empty."""
        metadata = {}
        ref = get_article_reference(metadata)
        assert ref == ""


class TestRAGPipelineInitialization:
    """Test RAGPipeline initialization and validation."""
    
    def test_invalid_retriever_k_zero(self) -> None:
        """Test that k=0 raises ValueError."""
        with pytest.raises(ValueError, match="retriever_k"):
            RAGPipeline(retriever_k=0)
    
    def test_invalid_retriever_k_negative(self) -> None:
        """Test that negative k raises ValueError."""
        with pytest.raises(ValueError, match="retriever_k"):
            RAGPipeline(retriever_k=-5)
    
    def test_invalid_retriever_k_too_large(self) -> None:
        """Test that k > 100 raises ValueError."""
        with pytest.raises(ValueError, match="retriever_k"):
            RAGPipeline(retriever_k=200)
    
    def test_invalid_temperature_negative(self) -> None:
        """Test that negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            RAGPipeline(temperature=-0.5)
    
    def test_invalid_temperature_too_high(self) -> None:
        """Test that temperature > 2 raises ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            RAGPipeline(temperature=2.5)
    
    def test_valid_initialization(self) -> None:
        """Test successful pipeline initialization."""
        pipeline = RAGPipeline(retriever_k=5, temperature=0.7)
        assert pipeline.retriever_k == 5
        assert pipeline.temperature == 0.7
        assert pipeline.model_name is not None
        assert pipeline.embedding_model_name is not None


class TestRetrieveValidation:
    """Test parameter validation in retrieve method."""
    
    def test_retrieve_invalid_k_zero(self) -> None:
        """Test that retrieve with k=0 raises ValueError."""
        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="k"):
            pipeline.retrieve("Test question", k=0)
    
    def test_retrieve_invalid_k_negative(self) -> None:
        """Test that retrieve with negative k raises ValueError."""
        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="k"):
            pipeline.retrieve("Test question", k=-1)
    
    def test_retrieve_invalid_k_too_large(self) -> None:
        """Test that retrieve with k > 100 raises ValueError."""
        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="k"):
            pipeline.retrieve("Test question", k=150)
    
    def test_retrieve_empty_question(self) -> None:
        """Test that empty question raises ValueError."""
        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="Soru"):
            pipeline.retrieve("")
    
    def test_retrieve_whitespace_question(self) -> None:
        """Test that whitespace-only question raises ValueError."""
        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="Soru"):
            pipeline.retrieve("   ")


class TestBatchQueryValidation:
    """Test batch_query validation."""
    
    def test_batch_query_empty_list(self) -> None:
        """Test that empty questions list raises ValueError."""
        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="Sorular listesi"):
            pipeline.batch_query([])
    
    def test_batch_query_invalid_type(self) -> None:
        """Test that non-list input raises ValueError."""
        pipeline = RAGPipeline()
        with pytest.raises(ValueError, match="Sorular listesi"):
            pipeline.batch_query(None)  # type: ignore


class TestMagicStringConfiguration:
    """Test that magic strings are properly configured."""
    
    def test_sambanova_url_configured(self) -> None:
        """Test that SambaNova URL is in config."""
        assert SAMBANOVA_API_BASE_URL == "https://api.sambanova.ai/v1"
    
    def test_ollama_default_url_configured(self) -> None:
        """Test that Ollama default URL is in config."""
        assert OLLAMA_DEFAULT_BASE_URL == "http://localhost:11434"


# Integration tests (commented out as they require actual services)
# @pytest.mark.integration
# class TestIntegration:
#     """Integration tests with actual models."""
#     
#     def test_full_pipeline(self) -> None:
#         """Test full RAG pipeline with real models."""
#         pipeline = RAGPipeline()
#         result = pipeline.query("Test question")
#         
#         assert isinstance(result, RAGResult)
#         assert result.question == "Test question"
#         assert result.answer is not None
#         assert result.retrieval.latency_ms > 0
#         assert result.generation.latency_ms > 0


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_rag_pipeline.py -v
    pytest.main([__file__, "-v"])
