#!/usr/bin/env bash
set -euo pipefail

KAGGLE_COMPETITION="spaceship-titanic"
DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "[run_all] Downloading $KAGGLE_COMPETITION data via Kaggle CLI"
kaggle competitions download -c "$KAGGLE_COMPETITION" -p "$DATA_DIR"

cd "$DATA_DIR"
unzip -o "${KAGGLE_COMPETITION}.zip"
cd -

python scripts/prepare_data.py --input data/train.csv --output data/train_prepared.csv --exclude-columns Transported
python scripts/prepare_data.py --input data/test.csv --output data/test_prepared.csv
python scripts/add_baseline_predictions.py --train data/train_prepared.csv --test data/test_prepared.csv --target-column Transported
python scripts/train_ga_nn.py --train-prepared data/train_prepared.csv --test-prepared data/test_prepared.csv --raw-test data/test.csv --generations 1 --population 4 --top-k 1 --cv-folds 2 --no-kaggle
