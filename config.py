"""Central configuration for CodeT5 text-to-SQL fine-tuning."""

from pathlib import Path

import torch as _torch


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "archive (2)"
TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

MODEL_NAME = "Salesforce/codet5-base"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "codet5_text2sql"
BEST_MODEL_DIR = OUTPUT_DIR / "best_checkpoint"
EVAL_RESULTS_FILE = OUTPUT_DIR / "evaluation_results.json"
RAW_DATASET_CACHE_DIR = OUTPUT_DIR / "dataset_cache" / "raw"
TOKENIZED_DATASET_CACHE_DIR = OUTPUT_DIR / "dataset_cache" / "tokenized"

WANDB_PROJECT = "codet5-text2sql"
WANDB_RUN_NAME = "codet5-base-wikisql"
REPORT_TO = ["none"]

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
NUM_TRAIN_EPOCHS = 10
WEIGHT_DECAY = 0.01
TRAIN_NUM_BEAMS = 1
EVAL_NUM_BEAMS = 4
_GPU_NAME = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else ""
FP16 = _torch.cuda.is_available() and "P100" not in _GPU_NAME
GRADIENT_CHECKPOINTING = True
EARLY_STOPPING_PATIENCE = 3
SAVE_TOTAL_LIMIT = 2
LOGGING_STEPS = 50
EVAL_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
PREDICT_WITH_GENERATE = True
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "exact_match"
GREATER_IS_BETTER = True
SEED = 42
PREPROCESSING_NUM_WORKERS = 4
DATALOADER_NUM_WORKERS = 2
EVAL_ACCUMULATION_STEPS = 8
GROUP_BY_LENGTH = True
USE_FAST_TOKENIZER = True
PAD_TO_MAX_LENGTH = False
PIN_MEMORY = True
PERSISTENT_WORKERS = False
