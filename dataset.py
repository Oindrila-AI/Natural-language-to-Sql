"""Dataset utilities for CodeT5 text-to-SQL fine-tuning."""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

import config


def _find_matching_parenthesis(text: str, start_index: int) -> int:
    """Find the matching closing parenthesis for a function call."""
    depth = 0
    in_string = False
    string_char = ""
    escaped = False

    for index in range(start_index, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_char:
                in_string = False
            continue

        if char in {"'", '"'}:
            in_string = True
            string_char = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index

    raise ValueError("Unbalanced parentheses while sanitizing array literal.")


def _split_top_level_dtype(text: str) -> str:
    """Strip a top-level dtype argument from an array call body."""
    bracket_depth = 0
    paren_depth = 0
    in_string = False
    string_char = ""
    escaped = False

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == string_char:
                in_string = False
            continue

        if char in {"'", '"'}:
            in_string = True
            string_char = char
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif text.startswith(", dtype=", index) and bracket_depth == 0 and paren_depth == 0:
            return text[:index]

    return text


def sanitize_literal_string(value: str) -> str:
    """Convert NumPy-style array wrappers into plain Python literals."""
    sanitized_parts: List[str] = []
    index = 0

    while index < len(value):
        if value.startswith("array(", index):
            array_start = index + len("array")
            array_end = _find_matching_parenthesis(value, array_start)
            inner = value[array_start + 1 : array_end]
            inner = _split_top_level_dtype(inner).strip()
            sanitized_parts.append(sanitize_literal_string(inner))
            index = array_end + 1
        else:
            sanitized_parts.append(value[index])
            index += 1

    return "".join(sanitized_parts)


def safe_literal_parse(value: str) -> Optional[dict]:
    """Safely parse a stringified Python dict using ast.literal_eval."""
    try:
        return ast.literal_eval(sanitize_literal_string(value))
    except (ValueError, SyntaxError):
        return None


def build_model_input(question: str, headers: List[str]) -> str:
    """Create the encoder input string from a question and table headers."""
    header_text = " | ".join(str(header).strip() for header in headers if str(header).strip())
    return f"question: {question.strip()} context: {header_text}"


def parse_example(row: pd.Series) -> Optional[Dict[str, str]]:
    """Parse one CSV row into model-ready source and target text."""
    table_obj = safe_literal_parse(str(row.get("table", "")))
    sql_obj = safe_literal_parse(str(row.get("sql", "")))

    if not table_obj or not sql_obj:
        return None

    headers = table_obj.get("header")
    target_sql = sql_obj.get("human_readable")
    question = str(row.get("question", "")).strip()

    if not isinstance(headers, list) or not question or not isinstance(target_sql, str) or not target_sql.strip():
        return None

    return {
        "input_text": build_model_input(question=question, headers=headers),
        "target_text": target_sql.strip(),
    }


def load_split(csv_path: Path) -> Dataset:
    """Load, validate, and convert one split from CSV into a Hugging Face Dataset."""
    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read dataset file: {csv_path}") from exc

    parsed_rows: List[Dict[str, str]] = []
    for _, row in frame.iterrows():
        parsed = parse_example(row)
        if parsed is not None:
            parsed_rows.append(parsed)

    if not parsed_rows:
        raise ValueError(f"No valid rows were parsed from {csv_path}.")

    return Dataset.from_pandas(pd.DataFrame(parsed_rows), preserve_index=False)


def tokenize_batch(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, List[List[int]]]:
    """Tokenize model inputs and labels for seq2seq training."""
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=config.MAX_INPUT_LENGTH,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=config.MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_raw_datasets() -> DatasetDict:
    """Load train, validation, and test splits without tokenization."""
    return DatasetDict(
        {
            "train": load_split(config.TRAIN_FILE),
            "validation": load_split(config.VALIDATION_FILE),
            "test": load_split(config.TEST_FILE),
        }
    )


def get_tokenized_datasets(tokenizer: AutoTokenizer) -> Tuple[DatasetDict, DatasetDict]:
    """Load raw datasets and return tokenized copies for training and evaluation."""
    raw_datasets = get_raw_datasets()
    tokenized = raw_datasets.map(
        lambda batch: tokenize_batch(batch, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=config.PREPROCESSING_NUM_WORKERS,
        desc="Tokenizing dataset",
    )
    return raw_datasets, tokenized


if __name__ == "__main__":
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        raw, tokenized = get_tokenized_datasets(tokenizer)
        print("Raw dataset sizes:", {split: len(ds) for split, ds in raw.items()})
        print("Tokenized columns:", tokenized["train"].column_names)
        print("Sample input:", raw["train"][0]["input_text"])
        print("Sample target:", raw["train"][0]["target_text"])
    except Exception as exc:
        raise SystemExit(f"Dataset preparation failed: {exc}") from exc
