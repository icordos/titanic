# Titanic Project

## Specs Summary
- **Spaceship Titanic | Kaggle.pdf**: Single-page snapshot of the Kaggle competition overview. Hosts: Kaggle; ~131k entrants, rolling leaderboard without medals; URL `https://www.kaggle.com/competitions/spaceship-titanic`.
- **titanic-presentation-1.pdf**: Slide deck describing a neural-network pipeline. Highlights: 13k-passenger anomaly scenario, dataset with 12 features (8700 train rows, 80/20 split, 4300 test), search space for feed-forward models (1–6 dense blocks, widths 16–512, ReLU/GELU/LeakyReLU, Adam/AdamW, LR/WD/batch/epoch tuning), and a genetic algorithm with elitist selection plus micro/macro mutations guided by eval/test accuracy fitness.

## Data Preparation Script
Use `scripts/prepare_data.py` to turn the Kaggle CSVs into fully numeric, normalized matrices:
```
python scripts/prepare_data.py \
  --input data/train.csv \
  --output data/train_prepared.csv \
  --exclude-columns Transported
```
- Non-numeric columns are one-hot encoded.
- Missing values are imputed with column means.
- Every feature is min-max scaled to [0, 1].
The script drops `PassengerId` and `Name` by default; override via `--drop-columns` if you need to remove other identifiers. Add `--exclude-columns` entries to keep labels or other fields untouched in the output. Run it once for `data/train.csv` (keeping `Transported`) and again for `data/test.csv` (omit `--exclude-columns`).

## Genetic Algorithm Neural Network Trainer
Use `scripts/train_ga_nn.py` to evolve fully connected networks on the prepared CSVs (defaults expect `data/train_prepared.csv` and `data/test_prepared.csv`):
```
python scripts/train_ga_nn.py \
  --train-prepared data/train_prepared.csv \
  --test-prepared data/test_prepared.csv \
  --raw-test data/test.csv \
  --generations 4 --population 8 --top-k 2
```
- Search space mirrors the specs deck: 1–6 dense layers (16–512 width), ReLU/GELU/LeakyReLU activations, dropout, Adam/AdamW with tunable LR, weight decay, batch size, and epoch caps.
- Each generation evaluates the population on an 80/20 validation split, keeps elites, and mutates new candidates (macro and micro changes).
- The top `--top-k` genomes retrain on the full dataset, generate Kaggle submissions under `submissions/`, save PyTorch weights under `models/`, and (if the Kaggle CLI is on PATH) auto-submit to `spaceship-titanic`. Use `--no-kaggle` to skip uploads. Training automatically picks CUDA, MPS, or CPU devices.
- After each submission the script polls `kaggle competitions submissions` (configurable via `--kaggle-score-timeout` / `--kaggle-score-interval`) and stores the public score plus the description tag inside `models/ga_search_summary.json` so next iterations can reuse leaderboard feedback even if you hit the daily submission cap.
