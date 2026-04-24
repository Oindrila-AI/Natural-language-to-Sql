# Natural-language-to-Sql

Production-ready fine-tuning code for training `Salesforce/codet5-base` on WikiSQL-style CSV files for text-to-SQL generation.

## Project Files

- `config.py`: central place for all paths, hyperparameters, and runtime settings
- `dataset.py`: CSV loading, safe parsing, input construction, and tokenization
- `train.py`: training loop with `Seq2SeqTrainer`, mixed precision, gradient checkpointing, early stopping, and Weights & Biases logging
- `evaluate.py`: checkpoint evaluation on the test set with EM, BLEU, and token-level accuracy
- `inference.py`: single-example SQL generation helper
- `requirements.txt`: pinned dependencies

## Dataset Format

Expected CSV columns:

- `question`: natural language question
- `table`: stringified Python dict containing `header`, `rows`, and other metadata
- `sql`: stringified Python dict containing `human_readable`

This repo is already configured to read:

- [train.csv](C:\Users\Oindrila Mondal\OneDrive\Desktop\Dbms_proj\Natural-language-to-Sql\archive (2)\train.csv)
- [validation.csv](C:\Users\Oindrila Mondal\OneDrive\Desktop\Dbms_proj\Natural-language-to-Sql\archive (2)\validation.csv)
- [test.csv](C:\Users\Oindrila Mondal\OneDrive\Desktop\Dbms_proj\Natural-language-to-Sql\archive (2)\test.csv)

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Weights & Biases Setup

1. Create an account at [wandb.ai](https://wandb.ai/site).
2. Log in from your notebook or terminal:

```bash
wandb login
```

3. If you want a different project or run name, update `WANDB_PROJECT` and `WANDB_RUN_NAME` in [config.py](C:\Users\Oindrila Mondal\OneDrive\Desktop\Dbms_proj\Natural-language-to-Sql\config.py).

## Path Changes For Colab Or Kaggle

Only update the path section in [config.py](C:\Users\Oindrila Mondal\OneDrive\Desktop\Dbms_proj\Natural-language-to-Sql\config.py).

Example for Kaggle:

```python
PROJECT_ROOT = Path("/kaggle/working/Natural-language-to-Sql")
DATA_DIR = Path("/kaggle/input/your-dataset-folder")
```

Example for Colab:

```python
PROJECT_ROOT = Path("/content/Natural-language-to-Sql")
DATA_DIR = PROJECT_ROOT / "archive"
```

The rest of the code can stay unchanged.

## Run Order

1. Verify dataset parsing:

```bash
python dataset.py
```

2. Start training:

```bash
python train.py
```

3. Evaluate the saved best checkpoint on the test set:

```bash
python evaluate.py
```

4. Run single-example inference:

```bash
python inference.py
```

## What The Model Sees

Each example is converted to:

```text
question: {question} context: {col1} | {col2} | {col3} | ...
```

The target string is:

```text
sql["human_readable"]
```

## Outputs

Training artifacts are saved under:

- `outputs/codet5_text2sql/`

Important paths:

- best checkpoint: `outputs/codet5_text2sql/best_checkpoint`
- evaluation report: `outputs/codet5_text2sql/evaluation_results.json`

## Run Inference On A New Question

You can import the helper from [inference.py](C:\Users\Oindrila Mondal\OneDrive\Desktop\Dbms_proj\Natural-language-to-Sql\inference.py):

```python
from inference import generate_sql

question = "Which player has the position guard?"
headers = ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"]
sql = generate_sql(question, headers)
print(sql)
```

Or run the included example directly:

```bash
python inference.py
```
