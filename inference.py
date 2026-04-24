"""Inference utility for generating SQL with a fine-tuned CodeT5 model."""

from typing import List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config


def build_input_text(question: str, column_headers: List[str]) -> str:
    """Construct the model input from a natural-language question and headers."""
    header_text = " | ".join(header.strip() for header in column_headers if header.strip())
    return f"translate English to SQL: question: {question.strip()} context: {header_text}"


def load_model(checkpoint_path: str = str(config.BEST_MODEL_DIR)) -> Tuple:
    """Load model, tokenizer, and device once for repeated inference."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=config.USE_FAST_TOKENIZER)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {exc}") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_sql(
    question: str,
    column_headers: List[str],
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> str:
    """Generate SQL for one question using a preloaded model."""
    input_text = build_input_text(question, column_headers)
    encoded = tokenizer(
        input_text,
        max_length=config.MAX_INPUT_LENGTH,
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

    return tokenizer.decode(generated[0], skip_special_tokens=True)


if __name__ == "__main__":
    try:
        model, tokenizer, device = load_model()
        example_question = "What is the nationality of the player named Terrence Ross?"
        example_headers = ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"]
        prediction = generate_sql(example_question, example_headers, model, tokenizer, device)
        print("Question:", example_question)
        print("Headers:", example_headers)
        print("Generated SQL:", prediction)
    except Exception as exc:
        raise SystemExit(f"Inference failed: {exc}") from exc
