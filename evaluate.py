"""Evaluate the best CodeT5 checkpoint on the test split."""

import json
import os
from typing import Dict, List

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config
from dataset import get_raw_datasets
from utils import normalize_sql, token_level_accuracy


def generate_predictions(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    inputs: List[str],
    batch_size: int,
) -> List[str]:
    """Generate SQL predictions for a batch of input strings."""
    device = model.device
    predictions: List[str] = []

    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start : start + batch_size]
        encoded = tokenizer(
            batch_inputs,
            max_length=config.MAX_INPUT_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_length=config.MAX_TARGET_LENGTH,
                num_beams=config.EVAL_NUM_BEAMS,
                early_stopping=True,
            )
        predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    return predictions


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute EM, BLEU, and token-level accuracy."""
    normalized_preds = [normalize_sql(pred) for pred in predictions]
    normalized_refs = [normalize_sql(ref) for ref in references]
    exact_match = sum(pred == ref for pred, ref in zip(normalized_preds, normalized_refs)) / max(len(predictions), 1)
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    token_acc = token_level_accuracy(predictions, references)
    return {
        "exact_match_percent": round(exact_match * 100, 4),
        "bleu": round(float(bleu), 4),
        "token_accuracy_percent": round(token_acc * 100, 4),
    }


def main() -> None:
    """Load the best checkpoint, evaluate on test data, and save results."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(config.BEST_MODEL_DIR), use_fast=config.USE_FAST_TOKENIZER)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(config.BEST_MODEL_DIR))
    except Exception as exc:
        raise SystemExit(
            f"Failed to load best checkpoint from {config.BEST_MODEL_DIR}. Train the model first. Details: {exc}"
        ) from exc

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        raw_datasets = get_raw_datasets()
    except Exception as exc:
        raise SystemExit(f"Failed to load test data: {exc}") from exc

    test_inputs = raw_datasets["test"]["input_text"]
    test_targets = raw_datasets["test"]["target_text"]
    predictions = generate_predictions(model, tokenizer, test_inputs, config.EVAL_BATCH_SIZE)
    metrics = compute_metrics(predictions, test_targets)

    results = {
        "metrics": metrics,
        "sample_predictions": [
            {
                "input": test_inputs[index],
                "prediction": predictions[index],
                "target": test_targets[index],
            }
            for index in range(min(10, len(predictions)))
        ],
    }

    try:
        with open(config.EVAL_RESULTS_FILE, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise SystemExit(f"Failed to save evaluation results: {exc}") from exc

    print(f"Exact Match (%): {metrics['exact_match_percent']}")
    print(f"BLEU: {metrics['bleu']}")
    print(f"Token Accuracy (%): {metrics['token_accuracy_percent']}")
    print("\nSample predictions:")
    for sample in results["sample_predictions"]:
        print("-" * 80)
        print("Input:", sample["input"])
        print("Prediction:", sample["prediction"])
        print("Target:", sample["target"])


if __name__ == "__main__":
    main()
