# Repository Guidelines

## Project Structure & Module Organization
The `data/` directory contains the Kaggle CSVs (`train.csv`, `test.csv`, `sample_submission.csv`) and should remain immutable; place any derived artifacts under `data/processed/` (create it when needed) so reviewers can track reproducible steps. The `specs/` folder stores research PDFs for reference only. Put reusable Python modules in `src/` and exploratory notebooks in `notebooks/`; keep notebook cells lightweight and move finalized logic into versioned scripts so experiments can be invoked headlessly.

## Build, Test, and Development Commands
Set up a local environment before touching the data:
```
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt     # dependency lockfile belongs at repo root
python -m pip install -e .                    # optional, if you expose src/ as a package
python scripts/train_model.py --data data/train.csv --out models/baseline.pkl
```
Use short, task-focused scripts under `scripts/` to reproduce notebook findings; keep CLI flags explicit so runs are traceable in pull requests.

## Coding Style & Naming Conventions
Prefer Python 3.11+, enforce `ruff` + `black` (`ruff check`, `ruff format`) with a 120-character line limit, and annotate public functions. Name modules with lowercase snake_case (`src/features/build_cabin_groups.py`), classes in PascalCase, and configuration files as `*.yaml` grouped under `configs/`. Store secrets in `.env` files excluded via `.gitignore` and access them through `os.environ`.

## Testing Guidelines
Adopt `pytest` for unit and integration coverage. Mirror the runtime package layout under `tests/` (e.g., `tests/features/test_build_cabin_groups.py`) and rely on fixtures that read from lightweight samples in `data/sample_submission.csv` instead of the full training set. Gate new work behind `pytest -q` and add regression tests whenever a notebook outcome graduates into `src/`.

## Commit & Pull Request Guidelines
Write Conventional Commit subjects (`feat: add cabin grouping features`, `fix: guard missing cryo deck data`) and keep bodies focused on intent plus verification (`pytest -q`, sample command lines). Each PR should link the motivating issue or Kaggle discussion, describe data dependencies (new files, expected shapes), attach before/after metrics, and include a reproducibility section listing every command needed to rebuild artifacts referenced in the change log.
