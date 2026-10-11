"""
[EVAL] Deterministic metrics. Pure functions, no I/O, no LLM; unit-tested in tests/test_evals.py.

LLM-as-a-judge (evals/judge.py) produces the *labels* (is this chunk relevant? is this
bullet grounded?); these functions turn labels into the numbers we report.
"""

from statistics import mean
from typing import Iterable, Sequence


def set_precision_recall_f1(expected: Iterable[str], predicted: Iterable[str]) -> dict:
    """Routing metric: which modules the planner chose vs. which it should have chosen."""
    expected, predicted = set(expected), set(predicted)
    true_pos = len(expected & predicted)
    precision = true_pos / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = true_pos / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "exact_match": expected == predicted,
    }


def precision_at_k(relevance_labels: Sequence[bool], k: int | None = None) -> float:
    """Retrieval metric: fraction of the top-k retrieved chunks judged relevant."""
    labels = list(relevance_labels)[:k] if k else list(relevance_labels)
    return round(sum(labels) / len(labels), 3) if labels else 0.0


def hit_at_k(relevance_labels: Sequence[bool], k: int | None = None) -> bool:
    """Retrieval metric: did at least one relevant chunk make it into the top-k?"""
    labels = list(relevance_labels)[:k] if k else list(relevance_labels)
    return any(labels)


def reciprocal_rank(relevance_labels: Sequence[bool]) -> float:
    """Retrieval metric: 1 / rank of the first relevant chunk (0 if none). Averaged -> MRR."""
    for rank, is_relevant in enumerate(relevance_labels, start=1):
        if is_relevant:
            return round(1 / rank, 3)
    return 0.0


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(mean(values), 3) if values else 0.0
