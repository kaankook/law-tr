import csv
import json
import logging
import re
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.config import (
    DATASET_PATH,
    GENERATION_WEIGHTS,
    MAPPING_FILE_PATH,
    METRIC_WEIGHTS,
    MODEL_NAME,
    EVALUATOR_MODEL_NAME,
    RERANKER_CFG,
    RETRIEVAL_CFG,
    RETRIEVAL_WEIGHTS,
    SCORING_WEIGHTS,
    TEST_CONFIG_DEFAULTS,
    TEST_THRESHOLDS,
)
from src.law_mapping_resolver import LawMappingResolver
from src.rag_pipeline import RAGPipeline, RAGResult, get_article_reference

# Genel log ayarlari
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def print_startup_banner() -> None:
    logger.info("━" * 70)
    logger.info("📦 Test altyapısı hazır")
    logger.info("  ✓ sentence_transformers")
    logger.info("  ✓ langchain_core")
    logger.info("  ✓ pydantic")
    logger.info("  ✓ rag_pipeline")
    logger.info("  ✓ law_mapping_resolver")
    logger.info("  ✓ config")
    logger.info("━" * 70)




@dataclass
class MetricResult:
    name: str
    score: float
    weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    def weighted_score(self) -> float:
        return self.score * self.weight


class HeuristicEvaluator:
    _embedding_model = None
    _model_name = "intfloat/multilingual-e5-large"

    @classmethod
    def get_embedding_model(cls) -> SentenceTransformer:
        if cls._embedding_model is None:
            logger.info(f"  🔄 Embedding modeli indiriliyor/yükleniyor: {cls._model_name}")
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._embedding_model = SentenceTransformer(cls._model_name, device=device)
            logger.info(f"  ✓ Embedding modeli hazır: {cls._model_name} ({device})")
        return cls._embedding_model

    @classmethod
    def compute_semantic_similarity(cls, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        model = cls.get_embedding_model()
        embeddings = model.encode(
            [f"query: {text1}", f"passage: {text2}"],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(max(0.0, min(1.0, similarity)))

    @classmethod
    def precision_score(cls, retrieved_articles: List[str], expected_article: str) -> MetricResult:
        """Retrieval Precision: Getirilen makalelerden kaçı ilgili?"""
        if not retrieved_articles or not expected_article:
            return MetricResult(
                name="precision",
                score=1.0 if not expected_article else 0.0,
                weight=RETRIEVAL_WEIGHTS["precision"],
                details={"note": "Eksik veri"}
            )
        
        expected_lower = expected_article.lower().strip()
        exact_matches = sum(1 for art in retrieved_articles if expected_lower in art.lower())
        
        precision = exact_matches / len(retrieved_articles) if retrieved_articles else 0.0
        
        return MetricResult(
            name="precision",
            score=min(1.0, precision),
            weight=RETRIEVAL_WEIGHTS["precision"],
            details={
                "exact_matches": exact_matches,
                "total_retrieved": len(retrieved_articles),
                "precision_value": round(precision, 3)
            }
        )

    @classmethod
    def recall_score(cls, retrieved_articles: List[str], expected_article: str) -> MetricResult:
        """Retrieval Recall: Beklenen makale retrieve edildi mi?"""
        if not expected_article:
            return MetricResult(
                name="recall",
                score=1.0,
                weight=RETRIEVAL_WEIGHTS["recall"],
                details={"note": "Beklenen makale yok"}
            )
        
        expected_lower = expected_article.lower().strip()
        found = any(expected_lower in art.lower() for art in retrieved_articles)
        
        # Semantic similarity de kontrol et
        if not found and retrieved_articles:
            similarities = [cls.compute_semantic_similarity(expected_lower, art.lower()) for art in retrieved_articles]
            max_sim = max(similarities) if similarities else 0.0
            found = max_sim >= 0.7
            score = max_sim if max_sim >= 0.7 else 0.0
        else:
            score = 1.0 if found else 0.0
        
        return MetricResult(
            name="recall",
            score=float(score),
            weight=RETRIEVAL_WEIGHTS["recall"],
            details={
                "expected_article_found": bool(found),
                "total_retrieved": len(retrieved_articles)
            }
        )

    @classmethod
    def faithfulness_score(cls, answer: str, context: List[str]) -> MetricResult:
        """Generation Faithfulness: Cevap context'e sadık mı?"""
        if not context or not answer:
            return MetricResult(
                name="faithfulness",
                score=1.0 if not answer else 0.5,
                weight=GENERATION_WEIGHTS["faithfulness"],
                details={"note": "Eksik veri"}
            )
        
        context_text = " ".join(str(c) for c in context if c)
        if not context_text.strip():
            return MetricResult(
                name="faithfulness",
                score=0.5,
                weight=GENERATION_WEIGHTS["faithfulness"],
                details={"note": "Context boş"}
            )
        
        # Cevabın her cümlesini context ile karşılaştır
        sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 10]
        if not sentences:
            return MetricResult(name="faithfulness", score=0.7, weight=GENERATION_WEIGHTS["faithfulness"])
        
        faithfulness_scores = [
            cls.compute_semantic_similarity(sent, context_text) for sent in sentences
        ]
        
        avg_score = float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0        
        
        return MetricResult(
            name="faithfulness",
            score=min(1.0, avg_score),
            weight=GENERATION_WEIGHTS["faithfulness"],
            details={
                "sentences_checked": len(sentences),
                "avg_similarity": round(avg_score, 3)
            }
        )

    @classmethod
    def answer_relevance_score(cls, answer: str, question: str) -> MetricResult:
        """Generation Answer Relevance: Cevap soruya ne kadar ilgili?"""
        if not answer or not question:
            return MetricResult(
                name="answer_relevance",
                score=0.0,
                weight=GENERATION_WEIGHTS["answer_relevance"],
                details={"note": "Eksik veri"}
            )
        
        relevance = cls.compute_semantic_similarity(question, answer)
        
        return MetricResult(
            name="answer_relevance",
            score=min(1.0, relevance),
            weight=GENERATION_WEIGHTS["answer_relevance"],
            details={
                "question_length": len(question.split()),
                "answer_length": len(answer.split()),
                "relevance_score": round(relevance, 3)
            }
        )

    @classmethod
    def answer_correctness_score(cls, answer: str, expected_answer: str, 
                                  alternative_answers: List[str] = None) -> MetricResult:
        """Generation Answer Correctness: Cevap doğru mu?"""
        if not expected_answer:
            return MetricResult(
                name="answer_correctness",
                score=1.0,
                weight=GENERATION_WEIGHTS["answer_correctness"],
                details={"note": "Beklenen cevap yok"}
            )
        
        if not answer:
            return MetricResult(
                name="answer_correctness",
                score=0.0,
                weight=GENERATION_WEIGHTS["answer_correctness"],
                details={"note": "Cevap boş"}
            )
        
        main_score = cls.compute_semantic_similarity(answer, expected_answer)
        best_score = main_score
        best_match = "main_answer"
        
        if alternative_answers:
            for i, alt in enumerate(alternative_answers):
                if alt and alt.strip():
                    alt_score = cls.compute_semantic_similarity(answer, alt)
                    if alt_score > best_score:
                        best_score = alt_score
                        best_match = f"alt_{i+1}"
        
        return MetricResult(
            name="answer_correctness",
            score=min(1.0, best_score),
            weight=GENERATION_WEIGHTS["answer_correctness"],
            details={
                "main_answer_score": round(main_score, 3),
                "best_match": best_match,
                "best_score": round(best_score, 3)
            }
        )




@dataclass
class EvaluationResult:
    question_id: int
    question: str
    category: str
    difficulty: str
    source: str
    generated_answer: str
    expected_answer: str
    retrieved_contexts: List[str] = field(default_factory=list)
    retrieved_articles: List[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    heuristic_metrics: List[MetricResult] = field(default_factory=list)
    passed: bool = False
    final_score: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)
    mulga_warnings: List[str] = field(default_factory=list)
    article_mapping_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id, "question": self.question,
            "category": self.category, "difficulty": self.difficulty, "source": self.source,
            "generated_answer": self.generated_answer, "expected_answer": self.expected_answer,
            "retrieved_articles": self.retrieved_articles, "total_latency_ms": float(self.total_latency_ms),
            "mulga_warnings": self.mulga_warnings, "article_mapping_detected": bool(self.article_mapping_detected),
            "heuristic_metrics": [asdict(m) for m in self.heuristic_metrics],
            "passed": bool(self.passed), "final_score": float(self.final_score), "failure_reasons": self.failure_reasons
        }


@dataclass
class TestSummary:
    total_questions: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    avg_final_score: float = 0.0
    avg_relevance: float = 0.0
    avg_faithfulness: float = 0.0
    avg_correctness: float = 0.0
    avg_completeness: float = 0.0
    avg_overall_judge: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_source: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model_name: str = ""
    evaluator_model: str = ""
    dataset_path: str = ""
    qdrant_k: int = 0         
    reranker_top_n: int = 0   
    timestamp: str = ""
    duration_seconds: float = 0.0
    best_questions: List[Dict] = field(default_factory=list)
    worst_questions: List[Dict] = field(default_factory=list)
    common_failure_reasons: Dict[str, int] = field(default_factory=dict)


@dataclass
class TestConfig:
    dataset_path: str = DATASET_PATH
    model_name: str = MODEL_NAME
    evaluator_model: str = EVALUATOR_MODEL_NAME
    delay_between_questions: float = TEST_CONFIG_DEFAULTS["delay_between_questions"]
    output_dir: str = TEST_CONFIG_DEFAULTS["output_dir"]
    run_name: Optional[str] = None
    pass_threshold: float = TEST_THRESHOLDS["pass_threshold"]
    save_contexts: bool = TEST_CONFIG_DEFAULTS["save_contexts"]
    verbose: bool = TEST_CONFIG_DEFAULTS["verbose"]

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


class RAGTestRunner:
    def __init__(self, config: TestConfig) -> None:
        self.config = config
        self.pipeline: Optional[RAGPipeline] = None
        self.heuristic = HeuristicEvaluator()
        self.law_resolver = LawMappingResolver(MAPPING_FILE_PATH)
        self.results: List[EvaluationResult] = []
        self.summary: Optional[TestSummary] = None
        self.start_time: Optional[datetime] = None
        self.questions_processed = 0
        self.total_questions = 0
        self.output_path = Path(config.output_dir) / config.run_name
        self.output_path.mkdir(parents=True, exist_ok=True)
        self._print_header()

    def _print_header(self) -> None:
        logger.info("━" * 70)
        logger.info("🧪  RAG TEST RUNNER — Gelişmiş Değerlendirme Sistemi")
        logger.info("━" * 70)
        logger.info(f"  📁 Dataset      : {self.config.dataset_path}")
        logger.info(f"  🤖 Model        : {self.config.model_name}")
        logger.info(f"  ⚖️  Evaluator    : {self.config.evaluator_model}")
        logger.info(f"  🔍 Qdrant K     : {RETRIEVAL_CFG.get('k', 'N/A')}")
        logger.info(f"  🎯 Reranker Top N: {RERANKER_CFG.get('top_n', 'N/A')}")
        logger.info(f"  📂 Output       : {self.output_path}")
        logger.info(f"  🎯 Pass Eşiği   : {self.config.pass_threshold}")
        logger.info("━" * 70)

    def load_dataset(self) -> List[Dict[str, Any]]:
        logger.info("📂 Dataset yükleniyor...")
        try:
            with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"  ❌ Dataset bulunamadı: {self.config.dataset_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"  ❌ Dataset JSON formatı geçersiz: {e}")
            raise ValueError(f"Geçersiz dataset formatı: {e}") from e

        questions = data.get('questions', data) if isinstance(data, dict) else data
        self.total_questions = len(questions)

        difficulties = defaultdict(int)
        categories = defaultdict(int)
        for q in questions:
            meta = q.get('metadata', {})
            difficulties[meta.get('difficulty', q.get('difficulty', 'unknown'))] += 1
            categories[meta.get('category', q.get('question_type', 'unknown'))] += 1

        logger.info(f"  ✓ {len(questions)} soru yüklendi")
        logger.info(f"  📊 Zorluk dağılımı  : {dict(difficulties)}")
        logger.info(f"  📊 Kategori dağılımı: {dict(categories)}")
        return questions

    def initialize(self) -> None:
        logger.info("━" * 70)
        logger.info("⚙️  Sistemler başlatılıyor...")

        logger.info("  🔄 RAG Pipeline yükleniyor...")
        self.pipeline = RAGPipeline(model_name=self.config.model_name)
        _ = self.pipeline.vectorstore
        _ = self.pipeline.llm
        logger.info("  ✓ RAG Pipeline hazır")

        logger.info("  ✓ Tüm sistemler hazır")
        logger.info("━" * 70)

    def _get_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {current}/{total} ({current/total*100:.1f}%)"

    def _get_eta(self) -> str:
        if self.questions_processed == 0:
            return "Hesaplanıyor..."
        elapsed = (datetime.now() - self.start_time).total_seconds()
        eta_seconds = (elapsed / self.questions_processed) * (self.total_questions - self.questions_processed)
        if eta_seconds < 60: return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600: return f"{eta_seconds/60:.1f}dk"
        else: return f"{eta_seconds/3600:.1f}sa"

    @staticmethod
    def _normalize_question_id(raw_id: Any, fallback_id: int) -> int:
        try:
            q_id = int(raw_id)
            return q_id if q_id > 0 else fallback_id
        except (TypeError, ValueError):
            return fallback_id

    def evaluate_single(
        self,
        question_data: Dict[str, Any],
        rag_result: RAGResult,
        question_id: Optional[int] = None,
    ) -> EvaluationResult:
        q_id = question_id if question_id is not None else self._normalize_question_id(question_data.get('id'), 1)
        question = question_data.get('question', '')

        # Dataset alanlarini normalize et
        meta = question_data.get('metadata', {})
        category = meta.get('category', question_data.get('question_type', 'unknown'))
        difficulty = meta.get('difficulty', question_data.get('difficulty', 'unknown'))
        source = question_data.get('source', '')

        raw_answer = question_data.get('expected_answer', question_data.get('answer', ''))
        if isinstance(raw_answer, dict):
            expected_answer = raw_answer.get('main_answer', '')
        else:
            expected_answer = raw_answer

        expected_article = (
            meta.get('article_reference')
            or question_data.get('source_details', {}).get('article', '')
        )

        alternative_answers = question_data.get('alternative_correct_answers', [])
        retrieved_articles = [get_article_reference(m) for m in rag_result.retrieval.metadata_list]

        logger.debug(f"    🔍 Metrikler hesaplanıyor...")
        
        # RETRIEVAL METRİKLERİ
        precision_metric = self.heuristic.precision_score(retrieved_articles, expected_article)
        recall_metric = self.heuristic.recall_score(retrieved_articles, expected_article)
        
        # GENERATION METRİKLERİ
        faithfulness_metric = self.heuristic.faithfulness_score(rag_result.answer, rag_result.retrieval.contexts)
        answer_relevance_metric = self.heuristic.answer_relevance_score(rag_result.answer, question)
        answer_correctness_metric = self.heuristic.answer_correctness_score(
            rag_result.answer, expected_answer, alternative_answers=alternative_answers
        )
        
        heuristic_metrics = [
            precision_metric,
            recall_metric,
            faithfulness_metric,
            answer_relevance_metric,
            answer_correctness_metric,
        ]

        logger.info(f"    ✓ Retrieval - Precision: {precision_metric.score:.2f} | Recall: {recall_metric.score:.2f}")
        logger.info(f"    ✓ Generation - Faithfulness: {faithfulness_metric.score:.2f} | Relevance: {answer_relevance_metric.score:.2f} | Correctness: {answer_correctness_metric.score:.2f}")

        final_score, passed, failure_reasons = self._calculate_final_result(heuristic_metrics, source)

        return EvaluationResult(
            question_id=q_id, question=question, category=category,
            difficulty=difficulty, source=source,
            generated_answer=rag_result.answer, expected_answer=expected_answer,
            retrieved_contexts=rag_result.retrieval.contexts if self.config.save_contexts else [],
            retrieved_articles=retrieved_articles,
            retrieval_latency_ms=rag_result.retrieval.latency_ms,
            generation_latency_ms=rag_result.generation.latency_ms,
            total_latency_ms=rag_result.total_latency_ms,
            passed=passed, final_score=final_score, failure_reasons=failure_reasons,
            mulga_warnings=rag_result.mulga_warnings, article_mapping_detected=False
        )

    def _calculate_final_result(self, heuristic_metrics: List[MetricResult], source: str) -> Tuple[float, bool, List[str]]:
        """
        Yeni metrik yapısı kullanarak final skoru hesapla.
        
        Retrieval: Precision + Recall
        Generation: Faithfulness + Answer Relevance + Answer Correctness
        """
        metric_map = {metric.name: metric for metric in heuristic_metrics}
        failure_reasons = []
        
        # Retrieval metrikleri
        precision = metric_map.get("precision")
        recall = metric_map.get("recall")
        
        # Generation metrikleri
        faithfulness = metric_map.get("faithfulness")
        answer_relevance = metric_map.get("answer_relevance")
        answer_correctness = metric_map.get("answer_correctness")
        
        # Eğer metrik hesaplanamadıysa
        if not all([precision, recall, faithfulness, answer_relevance, answer_correctness]):
            return 0.0, False, ["Metrik hesaplama hatası"]
        
        # Retrieval skoru
        retrieval_score = (
            precision.score * RETRIEVAL_WEIGHTS["precision"] +
            recall.score * RETRIEVAL_WEIGHTS["recall"]
        )
        
        # Generation skoru
        generation_score = (
            faithfulness.score * GENERATION_WEIGHTS["faithfulness"] +
            answer_relevance.score * GENERATION_WEIGHTS["answer_relevance"] +
            answer_correctness.score * GENERATION_WEIGHTS["answer_correctness"]
        )
        
        # Final skor
        final_score = float(
            retrieval_score * SCORING_WEIGHTS["retrieval_weight"] +
            generation_score * SCORING_WEIGHTS["generation_weight"]
        )
        
        # Eşik kontrolü
        passed = bool(final_score >= self.config.pass_threshold)        
        
        if precision.score < TEST_THRESHOLDS["precision_threshold"]:
            failure_reasons.append(f"Düşük Precision ({precision.score:.2f})")
        if recall.score < TEST_THRESHOLDS["recall_threshold"]:
            failure_reasons.append(f"Düşük Recall ({recall.score:.2f})")
        if faithfulness.score < TEST_THRESHOLDS["faithfulness_threshold"]:
            failure_reasons.append(f"Düşük Faithfulness ({faithfulness.score:.2f})")
        if answer_relevance.score < TEST_THRESHOLDS["answer_relevance_threshold"]:
            failure_reasons.append(f"Düşük Answer Relevance ({answer_relevance.score:.2f})")
        if answer_correctness.score < TEST_THRESHOLDS["answer_correctness_threshold"]:
            failure_reasons.append(f"Düşük Answer Correctness ({answer_correctness.score:.2f})")
        
        if not failure_reasons and final_score < self.config.pass_threshold:
            failure_reasons.append(f"Düşük final skor ({final_score:.2f})")
        
        return final_score, passed, failure_reasons

    def run(self) -> List[EvaluationResult]:
        questions = self.load_dataset()
        self.initialize()

        self.start_time = datetime.now()
        self.results = []

        logger.info("━" * 70)
        logger.info(f"🚀 TEST BAŞLIYOR — {len(questions)} soru")
        logger.info("━" * 70)

        passed_count = 0
        failed_count = 0

        for i, q_data in enumerate(questions):
            self.questions_processed = i + 1
            question = q_data.get('question', '').strip()
            q_id = self._normalize_question_id(q_data.get('id'), i + 1)
            
            if not question:
                logger.warning(f"⚠️  Soru {q_id}: Soru metni boş! Atlaniyor...")
                continue
            
            _meta = q_data.get('metadata', {})
            _category  = _meta.get('category',  q_data.get('question_type', '?'))
            _difficulty = _meta.get('difficulty', q_data.get('difficulty',   '?'))

            progress = self._get_progress_bar(i + 1, len(questions))
            eta = self._get_eta()

            logger.info("")
            logger.info(f"┌─ Soru {q_id}/{len(questions)} {progress} — ETA: {eta}")
            logger.info(f"│  📝 {question[:80]}{'...' if len(question) > 80 else ''}")
            logger.info(f"│  🏷️  Kategori: {_category} | Zorluk: {_difficulty}")

            try:
                logger.debug(f"│  🔍 RAG Pipeline sorgulanıyor...")
                rag_result = self.pipeline.query(question)
                logger.debug(f"│  ✓ Retrieval: {rag_result.retrieval.latency_ms:.0f}ms | Generation: {rag_result.generation.latency_ms:.0f}ms | Toplam: {rag_result.total_latency_ms:.0f}ms")
                logger.debug(f"│  💬 Cevap: {rag_result.answer[:100]}{'...' if len(rag_result.answer) > 100 else ''}")

                eval_result = self.evaluate_single(q_data, rag_result, question_id=q_id)
                self.results.append(eval_result)

                status = "✅ PASS" if eval_result.passed else "❌ FAIL"
                if eval_result.passed:
                    passed_count += 1
                else:
                    failed_count += 1

                metric_map = {m.name: m for m in eval_result.heuristic_metrics}
                rel = metric_map.get("answer_relevance").score if "answer_relevance" in metric_map else 0.0
                faith = metric_map.get("faithfulness").score if "faithfulness" in metric_map else 0.0
                corr = metric_map.get("answer_correctness").score if "answer_correctness" in metric_map else 0.0

                logger.info(f"└─ {status} | Final Skor: {eval_result.final_score:.3f} | "
                            f"Rel: {rel:.2f} | Faith: {faith:.2f} | Corr: {corr:.2f}")

                if not eval_result.passed and eval_result.failure_reasons:
                    logger.warning(f"   ⚠️  Başarısızlık: {', '.join(eval_result.failure_reasons)}")

                if (i + 1) % 5 == 0:
                    current_pass_rate = passed_count / (i + 1) * 100
                    logger.info(f"   📊 Anlık Durum: {passed_count} geçti / {failed_count} kaldı | Başarı: %{current_pass_rate:.1f}")

            except (ValueError, RuntimeError, TimeoutError) as e:
                logger.error(f"└─ ❌ Hata (Q{q_id}): {type(e).__name__}: {e}")
                failed_count += 1
                self.results.append(EvaluationResult(
                    question_id=q_id, question=question,
                    category=q_data.get('question_type', 'unknown'),
                    difficulty=q_data.get('difficulty', 'unknown'),
                    source=q_data.get('source', ''),
                    generated_answer=f"Hata: {str(e)}", expected_answer="",
                    passed=False, final_score=0.0,
                    failure_reasons=[f"Processing error: {str(e)}"]
                ))
            except Exception as e:
                logger.exception(f"└─ ❌ Beklenmeyen hata (Q{q_id}): {e}")

            if i < len(questions) - 1:
                time.sleep(self.config.delay_between_questions)

        self.questions_processed = len(questions)
        logger.info("")
        logger.info("━" * 70)
        logger.info(f"✅ Tüm sorular tamamlandı: {passed_count} geçti / {failed_count} kaldı")
        logger.info("📈 Özet hesaplanıyor...")
        self._generate_summary()
        return self.results

    def _generate_summary(self) -> None:
        if not self.results:
            return

        duration = (datetime.now() - self.start_time).total_seconds()

        passed = 0
        scores: List[float] = []
        latencies: List[float] = []
        
        # Yeni metrikler
        precision_scores: List[float] = []
        recall_scores: List[float] = []
        faithfulness_scores: List[float] = []
        answer_relevance_scores: List[float] = []
        answer_correctness_scores: List[float] = []

        by_difficulty = defaultdict(lambda: {"total": 0, "passed": 0, "avg_score": 0, "scores": []})
        by_category = defaultdict(lambda: {"total": 0, "passed": 0, "avg_score": 0, "scores": []})
        by_source = defaultdict(lambda: {"total": 0, "passed": 0, "avg_score": 0, "scores": []})

        failure_reasons = defaultdict(int)

        for r in self.results:
            if r.passed:
                passed += 1

            scores.append(r.final_score)
            if r.total_latency_ms > 0:
                latencies.append(r.total_latency_ms)

            # Metrikleri topla
            metric_map = {m.name: m for m in r.heuristic_metrics}
            
            if "precision" in metric_map:
                precision_scores.append(metric_map["precision"].score)
            if "recall" in metric_map:
                recall_scores.append(metric_map["recall"].score)
            if "faithfulness" in metric_map:
                faithfulness_scores.append(metric_map["faithfulness"].score)
            if "answer_relevance" in metric_map:
                answer_relevance_scores.append(metric_map["answer_relevance"].score)
            if "answer_correctness" in metric_map:
                answer_correctness_scores.append(metric_map["answer_correctness"].score)

            for key, d in [(r.difficulty, by_difficulty), (r.category, by_category), (r.source, by_source)]:
                d[key]["total"] += 1
                d[key]["passed"] += 1 if r.passed else 0
                d[key]["scores"].append(r.final_score)

            for reason in r.failure_reasons:
                reason_key = reason.split("(")[0].strip()
                failure_reasons[reason_key] += 1

        for d in [by_difficulty, by_category, by_source]:
            for key, val in d.items():
                val["avg_score"] = sum(val["scores"]) / len(val["scores"]) if val["scores"] else 0
                val["pass_rate"] = val["passed"] / val["total"] if val["total"] else 0
                del val["scores"]

        sorted_results = sorted(self.results, key=lambda x: x.final_score, reverse=True)
        best = [{"id": r.question_id, "score": r.final_score, "q": r.question[:50]} for r in sorted_results[:5]]
        worst = [{"id": r.question_id, "score": r.final_score, "q": r.question[:50], "reasons": r.failure_reasons}
                 for r in sorted_results[-5:] if not r.passed]

        self.summary = TestSummary(
            total_questions=len(self.results), passed=passed,
            failed=len(self.results) - passed,
            pass_rate=passed / len(self.results) if self.results else 0,
            avg_final_score=statistics.mean(scores) if scores else 0,
            avg_relevance=statistics.mean(precision_scores) if precision_scores else 0,
            avg_faithfulness=statistics.mean(faithfulness_scores) if faithfulness_scores else 0,
            avg_correctness=statistics.mean(answer_correctness_scores) if answer_correctness_scores else 0,
            avg_completeness=statistics.mean(answer_relevance_scores) if answer_relevance_scores else 0,
            avg_overall_judge=statistics.mean(recall_scores) if recall_scores else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0,
            by_difficulty=dict(by_difficulty), by_category=dict(by_category), by_source=dict(by_source),
            model_name=self.config.model_name, evaluator_model=self.config.evaluator_model,
            dataset_path=self.config.dataset_path, 
            qdrant_k=RETRIEVAL_CFG.get("k", 0),
            reranker_top_n=RERANKER_CFG.get("top_n", 0),
            timestamp=datetime.now().isoformat(), duration_seconds=duration,
            best_questions=best, worst_questions=worst, common_failure_reasons=dict(failure_reasons)
        )

    def save_results(self) -> None:
        if not self.results:
            logger.warning("Kaydedilecek sonuç yok")
            return

        logger.info("━" * 70)
        logger.info(f"💾 Sonuçlar kaydediliyor: {self.output_path}")

        try:
            with open(self.output_path / "summary.json", 'w', encoding='utf-8') as f:
                json.dump(asdict(self.summary), f, ensure_ascii=False, indent=2)
            logger.info("  ✓ summary.json")

            with open(self.output_path / "detailed_results.json", 'w', encoding='utf-8') as f:
                json.dump([r.to_dict() for r in self.results], f, ensure_ascii=False, indent=2)
            logger.info("  ✓ detailed_results.json")

            self._save_csv()
            logger.info("  ✓ results.csv")

            logger.info(f"  📂 Tüm dosyalar kaydedildi: {self.output_path}")
        except (IOError, OSError) as e:
            logger.error(f"  ❌ Kaydetme hatası: {e}")
            raise RuntimeError(f"Sonuçlar kaydedilemedi: {e}") from e

    def _save_csv(self) -> None:
        with open(self.output_path / "results.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Question', 'Category', 'Difficulty', 'Source',
                             'Final Score', 'Passed', 'Precision', 'Recall',
                             'Faithfulness', 'Answer Relevance', 'Answer Correctness',
                             'Latency (ms)', 'Failure Reasons'])
            for r in self.results:
                metric_map = {m.name: m for m in r.heuristic_metrics}
                precision_m = metric_map.get("precision")
                recall_m = metric_map.get("recall")
                faithfulness_m = metric_map.get("faithfulness")
                relevance_m = metric_map.get("answer_relevance")
                correctness_m = metric_map.get("answer_correctness")
                
                writer.writerow([
                    r.question_id, r.question[:100], r.category, r.difficulty, r.source,
                    f"{r.final_score:.3f}", "Yes" if r.passed else "No",
                    f"{precision_m.score:.2f}" if precision_m else "N/A",
                    f"{recall_m.score:.2f}" if recall_m else "N/A",
                    f"{faithfulness_m.score:.2f}" if faithfulness_m else "N/A",
                    f"{relevance_m.score:.2f}" if relevance_m else "N/A",
                    f"{correctness_m.score:.2f}" if correctness_m else "N/A",
                    f"{r.total_latency_ms:.0f}", "; ".join(r.failure_reasons)
                ])

    def print_summary(self) -> None:
        if not self.summary:
            logger.warning("Yazdırılacak özet yok")
            return

        s = self.summary
        status_emoji = "🎉" if s.pass_rate >= 0.8 else "⚠️" if s.pass_rate >= 0.6 else "❌"

        logger.info("━" * 70)
        logger.info("📊  TEST SONUÇ ÖZETİ")
        logger.info("━" * 70)
        logger.info(f"  {status_emoji} Başarı Oranı    : %{s.pass_rate*100:.1f} ({s.passed} geçti / {s.failed} kaldı / {s.total_questions} toplam)")
        logger.info(f"  📈 Final Skor Ort. : {s.avg_final_score:.3f}")
        logger.info("")
        logger.info("  📊 RETRİEVAL METRİKLERİ:")
        logger.info(f"    Precision (Kesinlik)     : {s.avg_relevance:.3f}")
        logger.info(f"    Recall (Hatırlama)       : {s.avg_overall_judge:.3f}")
        logger.info("")
        logger.info("  📊 GENERATION METRİKLERİ:")
        logger.info(f"    Faithfulness (Sadakat)   : {s.avg_faithfulness:.3f}")
        logger.info(f"    Answer Correctness       : {s.avg_correctness:.3f}")
        logger.info(f"    Answer Relevance         : {s.avg_completeness:.3f}")
        logger.info("")
        logger.info("  ⚡ PERFORMANS:")
        logger.info(f"    Ort. Yanıt Süresi  : {s.avg_latency_ms:.0f}ms")
        logger.info(f"    P95 Yanıt Süresi   : {s.p95_latency_ms:.0f}ms")
        logger.info(f"    Min / Max Süre     : {s.min_latency_ms:.0f}ms / {s.max_latency_ms:.0f}ms")
        logger.info(f"    Toplam Test Süresi : {s.duration_seconds:.1f}s")

        if s.by_difficulty:
            logger.info("")
            logger.info("  📊 ZORLUK BAZLI SONUÇLAR:")
            for diff, stats in sorted(s.by_difficulty.items()):
                logger.info(f"    {diff:12s}: %{stats['pass_rate']*100:.0f} başarı | Ort. Skor: {stats['avg_score']:.3f} ({stats['passed']}/{stats['total']})")

        if s.by_category:
            logger.info("")
            logger.info("  📂 KATEGORİ BAZLI SONUÇLAR:")
            for cat, stats in sorted(s.by_category.items()):
                logger.info(f"    {cat:15s}: %{stats['pass_rate']*100:.0f} başarı | Ort. Skor: {stats['avg_score']:.3f} ({stats['passed']}/{stats['total']})")

        if s.common_failure_reasons:
            logger.info("")
            logger.info("  ⚠️  EN YAYGIN HATALAR:")
            for reason, count in sorted(s.common_failure_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"    • {reason}: {count}x")

        if s.best_questions:
            logger.info("")
            logger.info("  ✅ EN YÜKSEK SKORLU SORULAR:")
            for q in s.best_questions[:3]:
                logger.info(f"    Q{q['id']} ({q['score']:.2f}): {q['q']}")

        if s.worst_questions:
            logger.info("")
            logger.info("  ❌ EN DÜŞÜK SKORLU SORULAR:")
            for q in s.worst_questions[:3]:
                reasons_str = " | ".join(q.get('reasons', [])[:2])
                logger.info(f"    Q{q['id']} ({q['score']:.2f}): {q['q']}")
                if reasons_str:
                    logger.info(f"      Sebepler: {reasons_str}")

        logger.info("━" * 70)


def run_test(
    dataset_path: str = DATASET_PATH,
    model_name: str = MODEL_NAME,
    output_dir: str = TEST_CONFIG_DEFAULTS["output_dir"],
    verbose: bool = TEST_CONFIG_DEFAULTS["verbose"]
) -> TestSummary:
    print_startup_banner()
    config = TestConfig(
        dataset_path=dataset_path, model_name=model_name,
        output_dir=output_dir, verbose=verbose
    )
    runner = RAGTestRunner(config)
    runner.run()
    runner.save_results()
    runner.print_summary()
    return runner.summary


if __name__ == "__main__":
    run_test()