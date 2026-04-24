"""Train CodeT5-base for text-to-SQL generation."""

import os
from typing import Dict

import numpy as np
import sacrebleu
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

import config
from dataset import get_tokenized_datasets
from utils import normalize_sql, token_level_accuracy


def print_gpu_memory() -> None:
    """Print GPU information before training begins."""
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device_index) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_index) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        print(
            f"GPU: {torch.cuda.get_device_name(device_index)} | "
            f"allocated={allocated:.2f} GB | reserved={reserved:.2f} GB | total={total:.2f} GB"
        )
    else:
        print("GPU is not available. Training will run on CPU, which will be very slow.")


def build_compute_metrics(tokenizer):
    """Create the compute_metrics function used by the trainer."""
    def compute_metrics(eval_preds) -> Dict[str, float]:
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        normalized_preds = [normalize_sql(pred) for pred in decoded_preds]
        normalized_labels = [normalize_sql(label) for label in decoded_labels]

        exact_match = sum(pred == label for pred, label in zip(normalized_preds, normalized_labels)) / max(
            len(normalized_preds), 1
        )
        bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels]).score
        token_acc = token_level_accuracy(decoded_preds, decoded_labels)

        return {
            "exact_match": round(exact_match * 100, 4),
            "bleu": round(float(bleu), 4),
            "token_accuracy": round(token_acc * 100, 4),
        }

    return compute_metrics


def main() -> None:
    """Run end-to-end training and save the best checkpoint."""
    os.environ.setdefault("WANDB_MODE", "offline")
    set_seed(config.SEED)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, use_fast=config.USE_FAST_TOKENIZER)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)
    except Exception as exc:
        raise SystemExit(f"Model/tokenizer loading failed: {exc}") from exc

    if config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled.")
        except Exception:
            print("torch.compile not available, continuing without it.")

    try:
        _, tokenized_datasets = get_tokenized_datasets(tokenizer)
    except Exception as exc:
        raise SystemExit(f"Dataset loading failed: {exc}") from exc

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    optim_name = "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        evaluation_strategy=config.EVAL_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        logging_steps=config.LOGGING_STEPS,
        fp16=config.FP16,
        predict_with_generate=config.PREDICT_WITH_GENERATE,
        generation_max_length=config.MAX_TARGET_LENGTH,
        generation_num_beams=config.TRAIN_NUM_BEAMS,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        report_to=config.REPORT_TO,
        run_name=config.WANDB_RUN_NAME,
        logging_dir=str(config.OUTPUT_DIR / "logs"),
        seed=config.SEED,
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=config.PIN_MEMORY,
        dataloader_persistent_workers=config.PERSISTENT_WORKERS,
        eval_accumulation_steps=config.EVAL_ACCUMULATION_STEPS,
        group_by_length=config.GROUP_BY_LENGTH,
        length_column_name="length",
        optim=optim_name,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
    )

    print_gpu_memory()

    try:
        trainer.train()
        trainer.save_model(str(config.BEST_MODEL_DIR))
        tokenizer.save_pretrained(str(config.BEST_MODEL_DIR))
        metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
        print("Validation metrics:", metrics)
    except Exception as exc:
        raise SystemExit(f"Training failed: {exc}") from exc


if __name__ == "__main__":
    main()
