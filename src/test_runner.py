import json
import csv
import time
import re
import statistics
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from rag_pipeline import RAGPipeline, RAGResult
from config import (
    MODEL_NAME,
    RETRIEVER_K,
    DATASET_PATH,
    EVALUATOR_MODEL_NAME,
    ENHANCED_JUDGE_PROMPT,
    TEST_CONFIG_DEFAULTS,
    TEST_THRESHOLDS,
    METRIC_WEIGHTS,
    LATENCY_THRESHOLD_MS,
    TEMPLATES_DIR,
    SCORING_WEIGHTS,
    JUDGE_SUBWEIGHTS,
    HEURISTIC_SUBWEIGHTS,
    HYBRID_KEYWORD_WEIGHTS,
    MUST_SHOULD_WEIGHTS,
)

def load_template(template_name: str) -> str:
    template_path = Path(TEMPLATES_DIR) / template_name
    if template_path.exists():
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    raise FileNotFoundError(f"Template bulunamadı: {template_path}")

def load_css_styles() -> str:
    return load_template("report_styles.css")

def load_html_template() -> str:
    return load_template("report_template.html")


class TriadJudgeOutput(BaseModel):
    relevance_score: float = Field(
        description="Cevabın sorulan soruyla ne kadar ilgili olduğu (0.0-1.0). "
                    "Soru ile cevap konusu uyuşmuyorsa düşük puan."
    )
    faithfulness_score: float = Field(
        description="Cevabın verilen context'e ne kadar sadık kaldığı (0.0-1.0). "
                    "Context dışı bilgi varsa düşük puan."
    )
    correctness_score: float = Field(
        description="Cevabın referans cevapla anlamsal ve hukuki örtüşmesi (0.0-1.0)."
    )
    completeness_score: float = Field(
        description="Cevabın beklenen bilgileri ne kadar kapsadığı (0.0-1.0). "
                    "Eksik kritik bilgi varsa düşük puan."
    )
    overall_score: float = Field(
        description="Tüm faktörler göz önüne alınarak genel kalite puanı (0.0-1.0)."
    )
    strengths: List[str] = Field(
        description="Cevabın güçlü yönleri (kısa maddeler halinde)."
    )
    weaknesses: List[str] = Field(
        description="Cevabın zayıf yönleri veya eksikleri (kısa maddeler halinde)."
    )
    reasoning: str = Field(
        description="Puanların gerekçesi (2-3 cümle)."
    )


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
            print(f"   📦 Semantic model yükleniyor: {cls._model_name}")
            cls._embedding_model = SentenceTransformer(cls._model_name)
            print("   ✓ Semantic model hazır")
        return cls._embedding_model
    
    @classmethod
    def compute_semantic_similarity(cls, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        
        model = cls.get_embedding_model()
        
        formatted_text1 = f"query: {text1}"
        formatted_text2 = f"passage: {text2}"
        
        embeddings = model.encode([formatted_text1, formatted_text2], normalize_embeddings=True)
        
        similarity = np.dot(embeddings[0], embeddings[1])
        
        return float(max(0.0, min(1.0, similarity)))
    
    @classmethod
    def exact_match_check(cls, text: str, keyword: str) -> bool:
        return keyword.lower() in text.lower()
    
    @classmethod
    def keyword_coverage(
        cls,
        generated: str, 
        must_include: List[str], 
        should_include: List[str],
        expected_answer: str = ""
    ) -> MetricResult:   
        if not must_include and not should_include and not expected_answer:
            return MetricResult(
                name="keyword_coverage",
                score=1.0,
                weight=METRIC_WEIGHTS["keyword_weight"],
                details={
                    "must_include_found": 0,
                    "must_include_total": 0,
                    "semantic_similarity": 1.0,
                    "exact_match_score": 1.0,
                    "missing_must": [],
                    "note": "No keywords expected"
                }
            )
        
        must_found = sum(1 for kw in must_include if cls.exact_match_check(generated, kw))
        should_found = sum(1 for kw in should_include if cls.exact_match_check(generated, kw))
        
        must_score = must_found / len(must_include) if must_include else 1.0
        should_score = should_found / len(should_include) if should_include else 1.0
        
        must_weight = MUST_SHOULD_WEIGHTS["must_weight"]
        should_weight = MUST_SHOULD_WEIGHTS["should_weight"]
        exact_match_score = (must_score * must_weight) + (should_score * should_weight)

        all_keywords = (must_include or []) + (should_include or [])
        combined_keywords_text = " ".join(all_keywords)
        
        if combined_keywords_text.strip():
            semantic_score = cls.compute_semantic_similarity(generated, combined_keywords_text)
        else:
            semantic_score = 1.0

        exact_weight = HYBRID_KEYWORD_WEIGHTS["exact_match"]
        semantic_weight = HYBRID_KEYWORD_WEIGHTS["semantic"]
        final_score = (exact_match_score * exact_weight) + (semantic_score * semantic_weight)
        
        return MetricResult(
            name="keyword_coverage",
            score=final_score,
            weight=METRIC_WEIGHTS["keyword_weight"],
            details={
                "must_include_found": must_found,
                "must_include_total": len(must_include),
                "should_include_found": should_found,
                "should_include_total": len(should_include),
                "exact_match_score": round(exact_match_score, 3),
                "semantic_similarity": round(semantic_score, 3),
                "missing_must": [kw for kw in must_include if not cls.exact_match_check(generated, kw)]
            }
        )
    
    @classmethod
    def semantic_correctness(
        cls,
        generated: str,
        expected: str,
        alternative_answers: List[str] = None
    ) -> MetricResult:
        if not expected:
            return MetricResult(
                name="semantic_correctness",
                score=1.0,
                weight=0.8,
                details={"note": "No expected answer provided"}
            )
        
        main_similarity = cls.compute_semantic_similarity(generated, expected)
        best_similarity = main_similarity
        best_match = "main_answer"

        alternative_scores = {}
        if alternative_answers:
            for i, alt_answer in enumerate(alternative_answers):
                if alt_answer and alt_answer.strip():
                    alt_sim = cls.compute_semantic_similarity(generated, alt_answer)
                    alternative_scores[f"alt_{i+1}"] = round(alt_sim, 3)
                    if alt_sim > best_similarity:
                        best_similarity = alt_sim
                        best_match = f"alternative_{i+1}"
        
        return MetricResult(
            name="semantic_correctness",
            score=best_similarity,
            weight=0.8,
            details={
                "main_answer_similarity": round(main_similarity, 3),
                "best_similarity": round(best_similarity, 3),
                "best_match_source": best_match,
                "alternative_scores": alternative_scores,
                "interpretation": cls._interpret_similarity(best_similarity)
            }
        )
    
    @staticmethod
    def _interpret_similarity(score: float) -> str:
        if score >= 0.9:
            return "Çok yüksek benzerlik - neredeyse aynı"
        elif score >= 0.75:
            return "Yüksek benzerlik - anlam korunmuş"
        elif score >= 0.6:
            return "Orta benzerlik - kısmi örtüşme"
        elif score >= 0.4:
            return "Düşük benzerlik - farklı içerik"
        else:
            return "Çok düşük benzerlik - alakasız"
    
    @classmethod
    def quote_presence(
        cls,
        context: str,
        source_quote: str
    ) -> MetricResult:
        if isinstance(context, list):
            context = "\n".join(str(c) for c in context if c)
        
        if not source_quote or not source_quote.strip():
            return MetricResult(
                name="quote_presence",
                score=1.0,  # Quote yoksa penalize etme
                weight=0.5,
                details={"note": "No source quote to check"}
            )
        
        if not context or not context.strip():
            return MetricResult(
                name="quote_presence",
                score=0.0,
                weight=0.5,
                details={"note": "Empty context", "quote_found": False}
            )
        
        context_lower = context.lower().strip()
        quote_lower = source_quote.lower().strip()
        
        if quote_lower in context_lower:
            return MetricResult(
                name="quote_presence",
                score=1.0,
                weight=0.5,
                details={
                    "quote_found": True,
                    "match_type": "exact",
                    "quote_preview": source_quote[:100] + "..." if len(source_quote) > 100 else source_quote
                }
            )
        
        import re
        context_sentences = re.split(r'[.!?\n]+', context)
        context_sentences = [s.strip() for s in context_sentences if len(s.strip()) > 20]
        
        best_similarity = 0.0
        best_sentence = ""
        
        for sentence in context_sentences[:30]:  # İlk 30 cümle ile sınırla (performans)
            sim = cls.compute_semantic_similarity(quote_lower, sentence.lower())
            if sim > best_similarity:
                best_similarity = sim
                best_sentence = sentence
        
        # Semantic threshold: 0.7 üzeri "bulundu" sayılır
        quote_found = best_similarity >= 0.7
        
        return MetricResult(
            name="quote_presence",
            score=best_similarity,
            weight=0.5,
            details={
                "quote_found": quote_found,
                "match_type": "semantic" if quote_found else "not_found",
                "semantic_similarity": round(best_similarity, 3),
                "best_matching_sentence": best_sentence[:100] + "..." if len(best_sentence) > 100 else best_sentence,
                "quote_preview": source_quote[:100] + "..." if len(source_quote) > 100 else source_quote
            }
        )
    
    @staticmethod
    def citation_accuracy(
        retrieved_articles: List[str],
        expected_article: str,
        expected_source: str
    ) -> MetricResult:
        """Doğru madde/kaynak getirildi mi? - Fuzzy matching ile geliştirilmiş"""
        import re
        
        if not expected_article:
            return MetricResult(
                name="citation_accuracy", 
                score=1.0, 
                weight=METRIC_WEIGHTS["citation_weight"]
            )
        
        expected_lower = expected_article.lower().strip()
        retrieved_lower = [a.lower().strip() for a in retrieved_articles if a]
        
        # Madde numarası çıkarma fonksiyonu (regex ile)
        def extract_article_numbers(text: str) -> List[str]:
            """Metinden madde numaralarını çıkar: 'Madde 14' -> ['14']"""
            patterns = [
                r'madde\s*(\d+(?:/[a-zA-Z])?)',  # Madde 14, Madde 14/A
                r'm\.?\s*(\d+(?:/[a-zA-Z])?)',    # m.14, m 14
                r'(\d+)(?:\.|\s)*madde',          # 14. madde, 14 madde
            ]
            numbers = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                numbers.extend(matches)
            return [n.lower() for n in numbers]
        
        expected_numbers = extract_article_numbers(expected_lower)

        exact_match = False
        partial_match = False
        number_match = False
        
        for art in retrieved_lower:
            if expected_lower in art:
                exact_match = True
                break
            
            # 2. Madde numarası eşleşmesi (Madde 14 != Madde 114)
            if expected_numbers:
                retrieved_numbers = extract_article_numbers(art)
                for exp_num in expected_numbers:
                    for ret_num in retrieved_numbers:
                        if exp_num == ret_num:
                            number_match = True
                            break
                    if number_match:
                        break
            

            if expected_source:
                source_lower = expected_source.lower()
                source_words = set(source_lower.split())
                art_words = set(art.split())
                common_words = source_words & art_words
                if len(common_words) >= len(source_words) * 0.5:
                    partial_match = True
        
        if exact_match:
            score = 1.0
        elif number_match:
            score = 0.9
        elif partial_match:
            score = 0.5
        else:
            score = 0.0
        
        return MetricResult(
            name="citation_accuracy",
            score=score,
            weight=METRIC_WEIGHTS["citation_weight"],
            details={
                "expected": expected_article,
                "expected_numbers": expected_numbers,
                "retrieved": retrieved_articles[:3],
                "source": expected_source,
                "exact_match": exact_match,
                "number_match": number_match,
                "partial_match": partial_match
            }
        )
    
    @staticmethod
    def response_quality(generated: str, question_type: str = "unknown") -> MetricResult:
        word_count = len(generated.split())
        
        WORD_COUNT_RANGES = {
            "factual": {"min": 15, "ideal_min": 30, "ideal_max": 150, "max": 300},
            "procedural": {"min": 40, "ideal_min": 80, "ideal_max": 400, "max": 600},
            "conceptual": {"min": 50, "ideal_min": 100, "ideal_max": 500, "max": 800},
            "comparative": {"min": 60, "ideal_min": 120, "ideal_max": 600, "max": 900},
            "calculation": {"min": 20, "ideal_min": 40, "ideal_max": 200, "max": 400},
            "unknown": {"min": 20, "ideal_min": 50, "ideal_max": 300, "max": 500}
        }
        
        ranges = WORD_COUNT_RANGES.get(question_type.lower(), WORD_COUNT_RANGES["unknown"])
        
        if word_count < ranges["min"]:
            length_score = 0.3  # Çok kısa
        elif word_count < ranges["ideal_min"]:
            # Minimum ile ideal_min arasında - lineer artış
            progress = (word_count - ranges["min"]) / (ranges["ideal_min"] - ranges["min"])
            length_score = 0.3 + (progress * 0.5)  # 0.3 -> 0.8
        elif word_count <= ranges["ideal_max"]:
            length_score = 1.0  # İdeal aralık
        elif word_count <= ranges["max"]:
            # ideal_max ile max arasında - hafif düşüş
            progress = (word_count - ranges["ideal_max"]) / (ranges["max"] - ranges["ideal_max"])
            length_score = 1.0 - (progress * 0.2)  # 1.0 -> 0.8
        else:
            length_score = 0.7  # Çok uzun
        
        # Yapısal öğeler
        structure_markers = ["1.", "2.", "-", "•", ": ", "\n-", "\n1.", "a)", "b)"]
        has_structure = any(marker in generated for marker in structure_markers)
        
        if question_type.lower() in ["procedural", "comparative"]:
            structure_weight = 0.5
        else:
            structure_weight = 0.3
        
        structure_score = 1.0 if has_structure else 0.75
        
        # "Bilmiyorum" kontrolü
        uncertainty_phrases = ["bilgi bulunamadı", "elimde yeterli", "mevcut değil", 
                             "bu konuda bilgi", "kapsamı dışında"]
        is_uncertain = any(phrase in generated.lower() for phrase in uncertainty_phrases)
        
        # Final skor hesapla
        length_weight = 1.0 - structure_weight
        final_score = (length_score * length_weight) + (structure_score * structure_weight)
        
        return MetricResult(
            name="response_quality",
            score=final_score,
            weight=METRIC_WEIGHTS["response_quality_weight"],
            details={
                "word_count": word_count,
                "question_type": question_type,
                "expected_range": f"{ranges['ideal_min']}-{ranges['ideal_max']}",
                "has_structure": has_structure,
                "is_uncertain": is_uncertain,
                "length_score": round(length_score, 2),
                "structure_score": round(structure_score, 2)
            }
        )
    
    @staticmethod
    def answer_consistency(
        generated: str, 
        expected: str,
        must_include: List[str] = None
    ) -> MetricResult:
        """Sayısal değer ve tarih tutarlılığı - Hallucination tespiti"""
        import re
        
        if not expected:
            return MetricResult(
                name="answer_consistency",
                score=1.0,
                weight=0.8,
                details={"note": "No expected answer provided"}
            )
        
        def extract_numbers(text: str) -> Dict[str, List[str]]:
            patterns = {
                "money": r'(\d{1,3}(?:\.\d{3})*(?:,\d{2})?\s*(?:TL|lira|kuruş))',
                "percentage": r'(%\s*\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s*%)',
                "days": r'(\d+)\s*(?:gün|günlük|günden)',
                "months": r'(\d+)\s*(?:ay|aylık|aydan)',
                "years": r'(\d+)\s*(?:yıl|yıllık|yıldan|sene)',
                "articles": r'(?:madde\s*)?(\d+)(?:\s*madde)?',
                "plain_numbers": r'\b(\d{1,6})\b'
            }
            results = {}
            for name, pattern in patterns.items():
                matches = re.findall(pattern, text.lower(), re.IGNORECASE)
                if matches:
                    results[name] = [str(m).strip() for m in matches]
            return results
        
        expected_numbers = extract_numbers(expected)
        generated_numbers = extract_numbers(generated)
        
        inconsistencies = []
        matches = 0
        total_checks = 0
        
        # Her kategori için kontrol
        for category in ["money", "percentage", "days", "months", "years"]:
            exp_vals = expected_numbers.get(category, [])
            gen_vals = generated_numbers.get(category, [])
            
            for exp_val in exp_vals:
                total_checks += 1
                # Normalize et ve karşılaştır
                exp_normalized = exp_val.replace(".", "").replace(",", ".").strip()
                found = False
                for gen_val in gen_vals:
                    gen_normalized = gen_val.replace(".", "").replace(",", ".").strip()
                    # Sayısal değerleri karşılaştır (sadece rakamları)
                    exp_digits = re.sub(r'[^\d.]', '', exp_normalized)
                    gen_digits = re.sub(r'[^\d.]', '', gen_normalized)
                    if exp_digits and gen_digits:
                        try:
                            if abs(float(exp_digits) - float(gen_digits)) < 0.01:
                                found = True
                                matches += 1
                                break
                        except ValueError:
                            if exp_digits == gen_digits:
                                found = True
                                matches += 1
                                break
                
                if not found and exp_val:
                    inconsistencies.append(f"{category}: expected '{exp_val}'")
        
        if total_checks == 0:
            score = 1.0  # Karşılaştırılacak sayısal değer yok
        else:
            score = matches / total_checks
            # Kritik değerler yanlışsa ağır penalize
            if len(inconsistencies) > 0:
                score = min(score, 0.6)  # En fazla 0.6 olabilir
        
        return MetricResult(
            name="answer_consistency",
            score=score,
            weight=0.8,
            details={
                "expected_numbers": expected_numbers,
                "generated_numbers": generated_numbers,
                "matches": matches,
                "total_checks": total_checks,
                "inconsistencies": inconsistencies[:5]  # İlk 5 tutarsızlık
            }
        )
    
    @staticmethod
    def latency_score(latency_ms: float, threshold_ms: float = LATENCY_THRESHOLD_MS) -> MetricResult:
        if latency_ms <= threshold_ms * 0.5:
            score = 1.0
        elif latency_ms <= threshold_ms:
            score = 0.8
        elif latency_ms <= threshold_ms * 2:
            score = 0.5
        else:
            score = 0.2
        
        return MetricResult(
            name="latency",
            score=score,
            weight=METRIC_WEIGHTS["latency_weight"],
            details={"latency_ms": latency_ms, "threshold_ms": threshold_ms}
        )


class RAGJudge:
    def __init__(self, model_name: str = EVALUATOR_MODEL_NAME):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0
        ).with_structured_output(TriadJudgeOutput)
        
        self.prompt = ChatPromptTemplate.from_template(ENHANCED_JUDGE_PROMPT)
        self.call_count = 0
        self.total_latency = 0
    
    def evaluate(
        self,
        question: str,
        expected: str,
        actual: str,
        context: List[str],
        must_include: List[str] = None,
        should_include: List[str] = None,
        source: str = ""
    ) -> TriadJudgeOutput:
        try:
            start = time.time()
            
            context_text = "\n\n---\n\n".join(context) if context else "❌ Context bulunamadı/boş."
            
            chain = self.prompt | self.llm
            result = chain.invoke({
                "question": question,
                "expected_answer": expected,
                "actual_answer": actual,
                "context": context_text,
                "must_include": ", ".join(must_include or []) or "Belirtilmemiş",
                "should_include": ", ".join(should_include or []) or "Belirtilmemiş",
                "source": source or "Belirtilmemiş"
            })
            
            self.call_count += 1
            self.total_latency += (time.time() - start) * 1000
            
            return result
            
        except Exception as e:
            print(f"    ⚠️ Judge hatası: {e}")
            return TriadJudgeOutput(
                relevance_score=0.0,
                faithfulness_score=0.0,
                correctness_score=0.0,
                completeness_score=0.0,
                overall_score=0.0,
                strengths=[],
                weaknesses=[f"Değerlendirme hatası: {str(e)}"],
                reasoning=f"Teknik hata oluştu: {str(e)}"
            )


@dataclass
class EvaluationResult:
    """Tek bir soru için değerlendirme sonucu"""
    question_id: int
    question: str
    category: str
    difficulty: str
    source: str
    
    generated_answer: str
    expected_answer: str
    
    # Retrieval bilgileri
    retrieved_contexts: List[str] = field(default_factory=list)
    retrieved_articles: List[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Metrikler
    heuristic_metrics: List[MetricResult] = field(default_factory=list)
    judge_result: Optional[TriadJudgeOutput] = None
    
    # Sonuç
    passed: bool = False
    final_score: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """JSON serializable dict"""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "difficulty": self.difficulty,
            "source": self.source,
            "generated_answer": self.generated_answer,
            "expected_answer": self.expected_answer,
            "retrieved_articles": self.retrieved_articles,
            "total_latency_ms": self.total_latency_ms,
            "heuristic_metrics": [asdict(m) for m in self.heuristic_metrics],
            "judge_scores": {
                "relevance": self.judge_result.relevance_score if self.judge_result else 0,
                "faithfulness": self.judge_result.faithfulness_score if self.judge_result else 0,
                "correctness": self.judge_result.correctness_score if self.judge_result else 0,
                "completeness": self.judge_result.completeness_score if self.judge_result else 0,
                "overall": self.judge_result.overall_score if self.judge_result else 0,
            },
            "judge_reasoning": self.judge_result.reasoning if self.judge_result else "",
            "strengths": self.judge_result.strengths if self.judge_result else [],
            "weaknesses": self.judge_result.weaknesses if self.judge_result else [],
            "passed": self.passed,
            "final_score": self.final_score,
            "failure_reasons": self.failure_reasons
        }


@dataclass
class TestSummary:
    """Test özet istatistikleri"""
    # Genel
    total_questions: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    
    # Skorlar
    avg_final_score: float = 0.0
    avg_relevance: float = 0.0
    avg_faithfulness: float = 0.0
    avg_correctness: float = 0.0
    avg_completeness: float = 0.0
    avg_overall_judge: float = 0.0
    
    # Performans
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    
    # Kategori bazlı
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_source: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Meta
    model_name: str = ""
    evaluator_model: str = ""
    dataset_path: str = ""
    retriever_k: int = 0
    timestamp: str = ""
    duration_seconds: float = 0.0
    
    # En iyi/kötü
    best_questions: List[Dict] = field(default_factory=list)
    worst_questions: List[Dict] = field(default_factory=list)
    
    # Hata analizi
    common_failure_reasons: Dict[str, int] = field(default_factory=dict)


@dataclass
class TestConfig:
    """Test yapılandırma ayarları"""
    dataset_path: str = DATASET_PATH
    model_name: str = MODEL_NAME
    evaluator_model: str = EVALUATOR_MODEL_NAME
    retriever_k: int = RETRIEVER_K
    delay_between_questions: float = TEST_CONFIG_DEFAULTS["delay_between_questions"]
    output_dir: str = TEST_CONFIG_DEFAULTS["output_dir"]
    run_name: Optional[str] = None
    pass_threshold: float = TEST_THRESHOLDS["pass_threshold"]
    relevance_threshold: float = TEST_THRESHOLDS["relevance_threshold"]
    faithfulness_threshold: float = TEST_THRESHOLDS["faithfulness_threshold"]
    citation_threshold: float = TEST_THRESHOLDS["citation_threshold"]
    save_contexts: bool = TEST_CONFIG_DEFAULTS["save_contexts"]
    verbose: bool = TEST_CONFIG_DEFAULTS["verbose"]
    
    def __post_init__(self):
        if self.run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"test_results_{timestamp}"


class HTMLReportGenerator:
    """HTML rapor oluşturma işlemlerini yönetir - Temiz Mimari"""
    
    def __init__(self, results: List[EvaluationResult], summary: TestSummary):
        self.results = results
        self.summary = summary
    
    def generate(self) -> str:
        """HTML raporu oluştur - Template + JSON Data yaklaşımı"""
        s = self.summary
        
        # CSS ve HTML template'i yükle
        css_styles = load_css_styles()
        html_template = load_html_template()
        
        # Score class helper
        def get_score_class(score: float) -> str:
            if score >= 0.8:
                return "success"
            elif score >= 0.6:
                return "warning"
            return "danger"
        
        # JavaScript için JSON data hazırla
        report_data = self._prepare_report_data()
        report_data_json = json.dumps(report_data, ensure_ascii=False, indent=2)
        
        # Template değişkenlerini hazırla (sadece basit değerler)
        template_vars = {
            "css_styles": css_styles,
            "timestamp_date": s.timestamp[:10] if s.timestamp else "",
            "model_name": s.model_name,
            "evaluator_model": s.evaluator_model,
            "retriever_k": s.retriever_k,
            "duration_seconds": f"{s.duration_seconds:.1f}",
            "pass_rate_class": get_score_class(s.pass_rate),
            "pass_rate_percent": f"{s.pass_rate*100:.1f}",
            "passed": s.passed,
            "failed": s.failed,
            "total_questions": s.total_questions,
            "avg_final_score": f"{s.avg_final_score:.2f}",
            "relevance_class": get_score_class(s.avg_relevance),
            "avg_relevance": f"{s.avg_relevance:.2f}",
            "faithfulness_class": get_score_class(s.avg_faithfulness),
            "avg_faithfulness": f"{s.avg_faithfulness:.2f}",
            "correctness_class": get_score_class(s.avg_correctness),
            "avg_correctness": f"{s.avg_correctness:.2f}",
            "avg_completeness": f"{s.avg_completeness:.2f}",
            "avg_overall_judge": f"{s.avg_overall_judge:.2f}",
            "avg_latency_sec": f"{s.avg_latency_ms/1000:.1f}",
            "p95_latency_sec": f"{s.p95_latency_ms/1000:.1f}",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report_data_json": report_data_json,
        }
        
        # Template'i doldur
        html = html_template
        for key, value in template_vars.items():
            html = html.replace("{{" + key + "}}", str(value))
        
        return html
    
    def _prepare_report_data(self) -> Dict[str, Any]:
        """JavaScript için tüm rapor verisini hazırla"""
        s = self.summary
        
        # Metrikler
        metrics = [
            {"name": "İlgililik (Relevance)", "score": s.avg_relevance},
            {"name": "Sadakat (Faithfulness)", "score": s.avg_faithfulness},
            {"name": "Doğruluk (Correctness)", "score": s.avg_correctness},
            {"name": "Tamlık (Completeness)", "score": s.avg_completeness},
            {"name": "Genel Judge Skoru", "score": s.avg_overall_judge},
        ]
        
        # Sorular
        questions = []
        for r in self.results:
            q_data = {
                "question_id": r.question_id,
                "question": r.question,
                "category": r.category,
                "difficulty": r.difficulty,
                "source": r.source,
                "generated_answer": r.generated_answer,
                "expected_answer": r.expected_answer,
                "total_latency_ms": r.total_latency_ms,
                "passed": r.passed,
                "final_score": r.final_score,
                "failure_reasons": r.failure_reasons,
                "judge_scores": None,
                "judge_reasoning": "",
                "strengths": [],
                "weaknesses": [],
            }
            
            if r.judge_result:
                q_data["judge_scores"] = {
                    "relevance": r.judge_result.relevance_score,
                    "faithfulness": r.judge_result.faithfulness_score,
                    "correctness": r.judge_result.correctness_score,
                    "completeness": r.judge_result.completeness_score,
                    "overall": r.judge_result.overall_score,
                }
                q_data["judge_reasoning"] = r.judge_result.reasoning
                q_data["strengths"] = r.judge_result.strengths
                q_data["weaknesses"] = r.judge_result.weaknesses
            
            questions.append(q_data)
        
        return {
            "metrics": metrics,
            "questions": questions,
            "by_difficulty": s.by_difficulty,
            "by_category": s.by_category,
            "failure_reasons": s.common_failure_reasons,
            "chart_data": {
                "triad": [s.avg_relevance, s.avg_faithfulness, s.avg_correctness, s.avg_completeness, s.avg_overall_judge],
                "pass_fail": [s.passed, s.failed],
            }
        }


class RAGTestRunner:
    """Ana test çalıştırıcı sınıf"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.pipeline: Optional[RAGPipeline] = None
        self.judge: Optional[RAGJudge] = None
        self.heuristic = HeuristicEvaluator()
        
        self.results: List[EvaluationResult] = []
        self.summary: Optional[TestSummary] = None
        
        self.start_time: Optional[datetime] = None
        self.questions_processed = 0
        self.total_questions = 0
        
        # Output dizini
        self.output_path = Path(config.output_dir) / config.run_name
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self._print_header()
    
    def _print_header(self):
        """Başlangıç bilgilerini yazdır"""
        print("\n" + "═" * 70)
        print("  🧪 RAG TEST RUNNER - Gelişmiş Değerlendirme Sistemi")
        print("═" * 70)
        print(f"  📁 Dataset    : {self.config.dataset_path}")
        print(f"  🤖 Model      : {self.config.model_name}")
        print(f"  ⚖️  Evaluator  : {self.config.evaluator_model}")
        print(f"  🔍 Retriever K: {self.config.retriever_k}")
        print(f"  📂 Output     : {self.output_path}")
        print("═" * 70 + "\n")
    
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Veri setini yükle"""
        print("📂 Veri seti yükleniyor...")
        
        with open(self.config.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        questions = data.get('questions', data) if isinstance(data, dict) else data
        self.total_questions = len(questions)
        
        # İstatistikler
        difficulties = defaultdict(int)
        categories = defaultdict(int)
        sources = defaultdict(int)
        
        for q in questions:
            difficulties[q.get('difficulty', 'unknown')] += 1
            categories[q.get('question_type', 'unknown')] += 1
            sources[q.get('source', 'unknown')] += 1
        
        print(f"   ✓ {len(questions)} soru yüklendi")
        print(f"   📊 Zorluk: {dict(difficulties)}")
        print(f"   📊 Kategori: {dict(categories)}")
        
        return questions
    
    def initialize(self):
        """Pipeline ve Judge'ı başlat"""
        print("\n🚀 Sistemler başlatılıyor...")
        
        # RAG Pipeline
        print("   ⏳ RAG Pipeline yükleniyor...")
        self.pipeline = RAGPipeline(
            model_name=self.config.model_name,
            retriever_k=self.config.retriever_k
        )
        # Warmup
        _ = self.pipeline.vectorstore
        _ = self.pipeline.llm
        print("   ✓ RAG Pipeline hazır")
        
        # LLM Judge
        print(f"   ⏳ LLM Judge yükleniyor ({self.config.evaluator_model})...")
        self.judge = RAGJudge(model_name=self.config.evaluator_model)
        print("   ✓ LLM Judge hazır")
        
        print("   ✓ Tüm sistemler hazır!\n")
    
    def _get_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """İlerleme çubuğu oluştur"""
        filled = int(width * current / total)
        bar = "█" * filled + "░" * (width - filled)
        percent = current / total * 100
        return f"[{bar}] {percent:.1f}%"
    
    def _get_eta(self) -> str:
        """Tahmini kalan süre"""
        if self.questions_processed == 0:
            return "Hesaplanıyor..."
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        avg_per_question = elapsed / self.questions_processed
        remaining = self.total_questions - self.questions_processed
        eta_seconds = avg_per_question * remaining
        
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.1f}m"
        else:
            return f"{eta_seconds/3600:.1f}h"
    
    def evaluate_single(
        self, 
        question_data: Dict[str, Any],
        rag_result: RAGResult
    ) -> EvaluationResult:
        """Tek bir soruyu değerlendir"""
        q_id = question_data.get('id', 0)
        question = question_data.get('question', '')
        category = question_data.get('question_type', 'unknown')
        difficulty = question_data.get('difficulty', 'unknown')
        source = question_data.get('source', '')
        
        answer_data = question_data.get('answer', {})
        if isinstance(answer_data, str):
            expected_answer = answer_data
            must_include = []
            should_include = []
            source_quote = ""
        else:
            expected_answer = answer_data.get('main_answer', '')
            key_entities = answer_data.get('key_entities', [])
            must_include = question_data.get('evaluation_criteria', {}).get('must_include', key_entities)
            should_include = question_data.get('evaluation_criteria', {}).get('should_include', [])
            source_quote = answer_data.get('source_quote', '')
        
        alternative_answers = question_data.get('alternative_correct_answers', [])
        source_details = question_data.get('source_details', {})
        expected_article = source_details.get('article', '')
        retrieved_articles = [m.get('article', '') for m in rag_result.retrieval.metadata_list]
        
        heuristic_metrics = []
        
        kw_metric = self.heuristic.keyword_coverage(
            rag_result.answer, must_include, should_include, 
            expected_answer=expected_answer
        )
        heuristic_metrics.append(kw_metric)
        
        semantic_metric = self.heuristic.semantic_correctness(
            rag_result.answer, expected_answer,
            alternative_answers=alternative_answers
        )
        heuristic_metrics.append(semantic_metric)
        
        quote_metric = self.heuristic.quote_presence(
            rag_result.retrieval.contexts, source_quote
        )
        heuristic_metrics.append(quote_metric)
        
        cit_metric = self.heuristic.citation_accuracy(
            retrieved_articles, expected_article, source
        )
        heuristic_metrics.append(cit_metric)
        
        qual_metric = self.heuristic.response_quality(rag_result.answer, question_type=category)
        heuristic_metrics.append(qual_metric)
        
        consistency_metric = self.heuristic.answer_consistency(
            rag_result.answer, expected_answer, must_include
        )
        heuristic_metrics.append(consistency_metric)
        
        lat_metric = self.heuristic.latency_score(rag_result.total_latency_ms)
        heuristic_metrics.append(lat_metric)
        
        judge_result = self.judge.evaluate(
            question=question,
            expected=expected_answer,
            actual=rag_result.answer,
            context=rag_result.retrieval.contexts,
            must_include=must_include,
            should_include=should_include,
            source=source
        )
        
        final_score, passed, failure_reasons = self._calculate_final_result(
            heuristic_metrics, judge_result, source
        )
        
        return EvaluationResult(
            question_id=q_id,
            question=question,
            category=category,
            difficulty=difficulty,
            source=source,
            generated_answer=rag_result.answer,
            expected_answer=expected_answer,
            retrieved_contexts=rag_result.retrieval.contexts if self.config.save_contexts else [],
            retrieved_articles=retrieved_articles,
            retrieval_latency_ms=rag_result.retrieval.latency_ms,
            generation_latency_ms=rag_result.generation.latency_ms,
            total_latency_ms=rag_result.total_latency_ms,
            heuristic_metrics=heuristic_metrics,
            judge_result=judge_result,
            passed=passed,
            final_score=final_score,
            failure_reasons=failure_reasons
        )
    
    def _calculate_final_result(
        self,
        heuristic_metrics: List[MetricResult],
        judge_result: TriadJudgeOutput,
        source: str
    ) -> Tuple[float, bool, List[str]]:
        """Final skor ve pass/fail kararı - Hibrit Mimari ile güncellenmiş"""
        
        failure_reasons = []
        is_out_of_scope = any(s in source.lower() for s in ['out_of_scope', 'edge_case'])
        
        rel = judge_result.relevance_score
        faith = judge_result.faithfulness_score
        corr = judge_result.correctness_score
        comp = judge_result.completeness_score
        
        cit_metric = next((m for m in heuristic_metrics if m.name == "citation_accuracy"), None)
        kw_metric = next((m for m in heuristic_metrics if m.name == "keyword_coverage"), None)
        qual_metric = next((m for m in heuristic_metrics if m.name == "response_quality"), None)
        consistency_metric = next((m for m in heuristic_metrics if m.name == "answer_consistency"), None)
        semantic_metric = next((m for m in heuristic_metrics if m.name == "semantic_correctness"), None)
        quote_metric = next((m for m in heuristic_metrics if m.name == "quote_presence"), None)
        
        cit_score = cit_metric.score if cit_metric else 1.0
        kw_score = kw_metric.score if kw_metric else 1.0
        qual_score = qual_metric.score if qual_metric else 1.0
        consistency_score = consistency_metric.score if consistency_metric else 1.0
        semantic_score = semantic_metric.score if semantic_metric else 1.0
        quote_score = quote_metric.score if quote_metric else 1.0
        
        judge_weight = SCORING_WEIGHTS["judge_weight"]
        heuristic_weight = SCORING_WEIGHTS["heuristic_weight"]
        
        judge_avg = (
            faith * JUDGE_SUBWEIGHTS["faithfulness"] +
            rel * JUDGE_SUBWEIGHTS["relevance"] +
            corr * JUDGE_SUBWEIGHTS["correctness"] +
            comp * JUDGE_SUBWEIGHTS["completeness"]
        )
        
        heuristic_avg = (
            semantic_score * HEURISTIC_SUBWEIGHTS["semantic_correctness"] +
            quote_score * HEURISTIC_SUBWEIGHTS["quote_presence"] +
            cit_score * HEURISTIC_SUBWEIGHTS["citation_accuracy"] +
            consistency_score * HEURISTIC_SUBWEIGHTS["answer_consistency"] +
            kw_score * HEURISTIC_SUBWEIGHTS["keyword_coverage"] +
            qual_score * HEURISTIC_SUBWEIGHTS["response_quality"]
        )
        
        final_score = (judge_avg * judge_weight) + (heuristic_avg * heuristic_weight)
        
        passed = True
        
        if rel < self.config.relevance_threshold:
            passed = False
            failure_reasons.append(f"Düşük ilgililik ({rel:.2f})")
        
        if faith < self.config.faithfulness_threshold:
            passed = False
            failure_reasons.append(f"Context'e sadakatsiz ({faith:.2f})")
        
        if cit_score < self.config.citation_threshold and not is_out_of_scope:
            passed = False
            failure_reasons.append(f"Yanlış/eksik atıf ({cit_score:.2f})")
        
        if consistency_score < TEST_THRESHOLDS["consistency_threshold"] and not is_out_of_scope:
            passed = False
            if consistency_metric and consistency_metric.details.get("inconsistencies"):
                failure_reasons.append(f"Sayısal tutarsızlık ({consistency_score:.2f})")
        
        if semantic_score < TEST_THRESHOLDS["semantic_threshold"] and not is_out_of_scope:
            passed = False
            failure_reasons.append(f"Düşük anlamsal benzerlik ({semantic_score:.2f})")
        
        if final_score < self.config.pass_threshold:
            passed = False
            if not failure_reasons:
                failure_reasons.append(f"Düşük genel skor ({final_score:.2f})")
        
        return final_score, passed, failure_reasons
    
    def run(self) -> List[EvaluationResult]:
        """Testi çalıştır"""
        questions = self.load_dataset()
        self.initialize()
        
        self.start_time = datetime.now()
        self.results = []
        
        print("=" * 70)
        print("  🧪 TEST BAŞLIYOR")
        print("=" * 70 + "\n")
        
        for i, q_data in enumerate(questions):
            self.questions_processed = i
            question = q_data.get('question', '')
            q_id = q_data.get('id', i + 1)
            
            progress = self._get_progress_bar(i + 1, len(questions))
            eta = self._get_eta()
            
            if self.config.verbose:
                print(f"\n{progress} ETA: {eta}")
                print(f"   Q{q_id}: {question[:55]}...")
            
            try:
                rag_result = self.pipeline.query(question)
                eval_result = self.evaluate_single(q_data, rag_result)
                self.results.append(eval_result)
                if self.config.verbose:
                    status = "✅ PASS" if eval_result.passed else "❌ FAIL"
                    j = eval_result.judge_result
                    print(f"   {status} | Score: {eval_result.final_score:.2f} | "
                          f"Rel: {j.relevance_score:.2f} | "
                          f"Faith: {j.faithfulness_score:.2f} | "
                          f"Corr: {j.correctness_score:.2f}")
                    
                    if not eval_result.passed and eval_result.failure_reasons:
                        print(f"   ⚠️ Sebep: {', '.join(eval_result.failure_reasons)}")
                
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
                self.results.append(EvaluationResult(
                    question_id=q_id,
                    question=question,
                    category=q_data.get('question_type', 'unknown'),
                    difficulty=q_data.get('difficulty', 'unknown'),
                    source=q_data.get('source', ''),
                    generated_answer=f"ERROR: {str(e)}",
                    expected_answer="",
                    passed=False,
                    final_score=0.0,
                    failure_reasons=[f"Exception: {str(e)}"]
                ))
            
            if i < len(questions) - 1:
                time.sleep(self.config.delay_between_questions)
        
        self.questions_processed = len(questions)
        self._generate_summary()
        
        return self.results
    
    def _generate_summary(self):
        """Özet istatistikleri oluştur"""
        if not self.results:
            return
        
        duration = (datetime.now() - self.start_time).total_seconds()
        
        # Temel istatistikler
        passed = sum(1 for r in self.results if r.passed)
        
        # Skor ortalamaları
        scores = [r.final_score for r in self.results]
        latencies = [r.total_latency_ms for r in self.results if r.total_latency_ms > 0]
        
        relevance_scores = [r.judge_result.relevance_score for r in self.results if r.judge_result]
        faithfulness_scores = [r.judge_result.faithfulness_score for r in self.results if r.judge_result]
        correctness_scores = [r.judge_result.correctness_score for r in self.results if r.judge_result]
        completeness_scores = [r.judge_result.completeness_score for r in self.results if r.judge_result]
        overall_scores = [r.judge_result.overall_score for r in self.results if r.judge_result]
        
        # Kategori bazlı analiz
        by_difficulty = defaultdict(lambda: {"total": 0, "passed": 0, "avg_score": 0, "scores": []})
        by_category = defaultdict(lambda: {"total": 0, "passed": 0, "avg_score": 0, "scores": []})
        by_source = defaultdict(lambda: {"total": 0, "passed": 0, "avg_score": 0, "scores": []})
        
        for r in self.results:
            for key, data_dict in [(r.difficulty, by_difficulty), 
                                    (r.category, by_category),
                                    (r.source[:30], by_source)]:
                data_dict[key]["total"] += 1
                data_dict[key]["passed"] += 1 if r.passed else 0
                data_dict[key]["scores"].append(r.final_score)
        
        # Ortalamaları hesapla
        for data_dict in [by_difficulty, by_category, by_source]:
            for key, val in data_dict.items():
                val["avg_score"] = sum(val["scores"]) / len(val["scores"]) if val["scores"] else 0
                val["pass_rate"] = val["passed"] / val["total"] if val["total"] else 0
                del val["scores"]  # Temizle
        
        # Failure analizi
        failure_reasons = defaultdict(int)
        for r in self.results:
            for reason in r.failure_reasons:
                # Normalize et
                if "ilgililik" in reason.lower():
                    failure_reasons["Düşük İlgililik"] += 1
                elif "sadakat" in reason.lower():
                    failure_reasons["Context Sadakatsizliği"] += 1
                elif "atıf" in reason.lower():
                    failure_reasons["Yanlış Atıf"] += 1
                else:
                    failure_reasons[reason.split("(")[0].strip()] += 1
        
        # En iyi / en kötü
        sorted_results = sorted(self.results, key=lambda x: x.final_score, reverse=True)
        best = [{"id": r.question_id, "score": r.final_score, "q": r.question[:50]} 
                for r in sorted_results[:5]]
        worst = [{"id": r.question_id, "score": r.final_score, "q": r.question[:50], 
                  "reasons": r.failure_reasons} 
                 for r in sorted_results[-5:] if not r.passed]
        
        self.summary = TestSummary(
            total_questions=len(self.results),
            passed=passed,
            failed=len(self.results) - passed,
            pass_rate=passed / len(self.results) if self.results else 0,
            avg_final_score=statistics.mean(scores) if scores else 0,
            avg_relevance=statistics.mean(relevance_scores) if relevance_scores else 0,
            avg_faithfulness=statistics.mean(faithfulness_scores) if faithfulness_scores else 0,
            avg_correctness=statistics.mean(correctness_scores) if correctness_scores else 0,
            avg_completeness=statistics.mean(completeness_scores) if completeness_scores else 0,
            avg_overall_judge=statistics.mean(overall_scores) if overall_scores else 0,
            avg_latency_ms=statistics.mean(latencies) if latencies else 0,
            min_latency_ms=min(latencies) if latencies else 0,
            max_latency_ms=max(latencies) if latencies else 0,
            p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 1 else 0,
            by_difficulty=dict(by_difficulty),
            by_category=dict(by_category),
            by_source=dict(by_source),
            model_name=self.config.model_name,
            evaluator_model=self.config.evaluator_model,
            dataset_path=self.config.dataset_path,
            retriever_k=self.config.retriever_k,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            best_questions=best,
            worst_questions=worst,
            common_failure_reasons=dict(failure_reasons)
        )
    
    def save_results(self):
        """Sonuçları kaydet"""
        if not self.results:
            return
        
        print(f"\n💾 Sonuçlar kaydediliyor: {self.output_path}")
        
        # 1. Summary JSON
        with open(self.output_path / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(asdict(self.summary), f, ensure_ascii=False, indent=2)
        print("   ✓ summary.json")
        
        # 2. Detailed Results JSON
        with open(self.output_path / "detailed_results.json", 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in self.results], f, ensure_ascii=False, indent=2)
        print("   ✓ detailed_results.json")
        
        # 3. CSV Export
        self._save_csv()
        print("   ✓ results.csv")
        
        # 4. HTML Report
        self._save_html_report()
        print("   ✓ report.html")
        
        print(f"\n📊 Rapor: file://{self.output_path.absolute()}/report.html")
    
    def _save_csv(self):
        """CSV formatında kaydet"""
        csv_path = self.output_path / "results.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'ID', 'Question', 'Category', 'Difficulty', 'Source',
                'Final Score', 'Passed',
                'Relevance', 'Faithfulness', 'Correctness', 'Completeness',
                'Semantic Sim', 'Keyword Cov',
                'Latency (ms)', 'Failure Reasons'
            ])
            
            # Rows
            for r in self.results:
                j = r.judge_result
                # Heuristic metriklerden semantic ve keyword skorlarını al
                semantic_m = next((m for m in r.heuristic_metrics if m.name == "semantic_correctness"), None)
                keyword_m = next((m for m in r.heuristic_metrics if m.name == "keyword_coverage"), None)
                
                writer.writerow([
                    r.question_id,
                    r.question[:100],
                    r.category,
                    r.difficulty,
                    r.source[:30],
                    f"{r.final_score:.3f}",
                    "Yes" if r.passed else "No",
                    f"{j.relevance_score:.2f}" if j else "N/A",
                    f"{j.faithfulness_score:.2f}" if j else "N/A",
                    f"{j.correctness_score:.2f}" if j else "N/A",
                    f"{j.completeness_score:.2f}" if j else "N/A",
                    f"{semantic_m.score:.2f}" if semantic_m else "N/A",
                    f"{keyword_m.score:.2f}" if keyword_m else "N/A",
                    f"{r.total_latency_ms:.0f}",
                    "; ".join(r.failure_reasons)
                ])
    
    def _save_html_report(self):
        """HTML raporu oluştur ve kaydet"""
        report_generator = HTMLReportGenerator(self.results, self.summary)
        html_content = report_generator.generate()
        
        with open(self.output_path / "report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def print_summary(self):
        """Özeti konsola yazdır"""
        if not self.summary:
            return
        
        s = self.summary
        
        print("\n" + "═" * 70)
        print("  📊 TEST SONUÇ ÖZETİ")
        print("═" * 70)
        
        # Başarı durumu
        status_emoji = "🎉" if s.pass_rate >= 0.8 else "⚠️" if s.pass_rate >= 0.6 else "❌"
        print(f"\n  {status_emoji} Başarı Oranı: {s.pass_rate*100:.1f}% ({s.passed}/{s.total_questions})")
        print(f"  📈 Ortalama Final Skor: {s.avg_final_score:.3f}")
        
        print("\n  ┌─────────────────────────────────────────┐")
        print("  │          RAG TRIAD METRİKLERİ           │")
        print("  ├─────────────────────────────────────────┤")
        print(f"  │  İlgililik (Relevance)    : {s.avg_relevance:.3f}        │")
        print(f"  │  Sadakat (Faithfulness)   : {s.avg_faithfulness:.3f}        │")
        print(f"  │  Doğruluk (Correctness)   : {s.avg_correctness:.3f}        │")
        print(f"  │  Tamlık (Completeness)    : {s.avg_completeness:.3f}        │")
        print(f"  │  Genel Judge Skoru        : {s.avg_overall_judge:.3f}        │")
        print("  └─────────────────────────────────────────┘")
        
        print("\n  ⚡ Performans:")
        print(f"     Ort. Yanıt Süresi : {s.avg_latency_ms:.0f}ms")
        print(f"     P95 Yanıt Süresi  : {s.p95_latency_ms:.0f}ms")
        print(f"     Test Süresi       : {s.duration_seconds:.1f}s")
        
        if s.common_failure_reasons:
            print("\n  ⚠️ En Yaygın Hatalar:")
            for reason, count in sorted(s.common_failure_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"     • {reason}: {count}x")
        
        print("\n" + "═" * 70)

def run_test(
    dataset_path: str = DATASET_PATH,
    model_name: str = MODEL_NAME,
    retriever_k: int = RETRIEVER_K,
    output_dir: str = TEST_CONFIG_DEFAULTS["output_dir"],
    verbose: bool = TEST_CONFIG_DEFAULTS["verbose"]
) -> TestSummary:    
    config = TestConfig(
        dataset_path=dataset_path,
        model_name=model_name,
        retriever_k=retriever_k,
        output_dir=output_dir,
        verbose=verbose
    )
    
    runner = RAGTestRunner(config)
    runner.run()
    runner.save_results()
    runner.print_summary()
    
    return runner.summary

if __name__ == "__main__":
    run_test()

