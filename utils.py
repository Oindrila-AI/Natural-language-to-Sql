"""Shared utilities for CodeT5 text-to-SQL fine-tuning."""

from typing import List


def normalize_sql(text: str) -> str:
    """Normalize SQL strings for exact-match comparison."""
    return " ".join(text.strip().lower().split())


def token_level_accuracy(predictions: List[str], references: List[str]) -> float:
    """Compute average token-level accuracy across prediction-reference pairs."""
    accuracies: List[float] = []
    for prediction, reference in zip(predictions, references):
        pred_tokens = prediction.strip().split()
        ref_tokens = reference.strip().split()
        max_len = max(len(pred_tokens), len(ref_tokens))
        if max_len == 0:
            accuracies.append(1.0)
            continue
        matches = sum(1 for pred_token, ref_token in zip(pred_tokens, ref_tokens) if pred_token.lower() == ref_token.lower())
        accuracies.append(matches / max_len)
    return float(sum(accuracies) / len(accuracies)) if accuracies else 0.0
