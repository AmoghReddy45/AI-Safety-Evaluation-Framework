"""Core type definitions for AI Safety Evaluation Framework."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class EvalType(Enum):
    """Enumeration of all supported evaluation types."""

    SYCOPHANCY_OPINION = "sycophancy_opinion"
    SYCOPHANCY_ANSWER_CHANGE = "sycophancy_answer_change"
    SYCOPHANCY_FEEDBACK = "sycophancy_feedback"
    SYCOPHANCY_MIMICRY = "sycophancy_mimicry"
    TRUTHFULNESS = "truthfulness"
    HALLUCINATION = "hallucination"
    SELF_AWARENESS = "self_awareness"
    OVER_REFUSAL = "over_refusal"
    UNDER_REFUSAL = "under_refusal"
    REFUSAL_QUALITY = "refusal_quality"
    DANGEROUS_BIO = "dangerous_bio"
    DANGEROUS_CYBER = "dangerous_cyber"
    DANGEROUS_CHEM = "dangerous_chem"


@dataclass
class EvaluationRun:
    """Represents a single evaluation run configuration and metadata."""

    run_id: str
    created_at: datetime
    eval_type: EvalType
    model_id: str
    model_version: str | None
    config: dict[str, Any]
    status: str


@dataclass
class EvaluationResult:
    """Represents the result of evaluating a single prompt."""

    result_id: str
    run_id: str
    prompt_id: str
    prompt_text: str
    response_text: str
    scores: dict[str, float]
    judge_reasoning: str | None
    latency_ms: float


@dataclass
class ClassificationResult:
    """Represents the result of classifying content as harmful or not."""

    is_harmful: bool
    confidence: float
    categories: list[str]
    latency_ms: float
