#!/usr/bin/env python3
"""Execute Day 3 feature ablation workflow with log-loss guard."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence

KAGGLE_COMPETITION = "spaceship-titanic"


def run_cmd(command: Sequence[str], cwd: Path | None = None) -> None:
    pretty = " ".join(command)
    print(f"[day3] $ {pretty}")
    subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)


def load_best_log_loss(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fp:
            entries = json.load(fp)
    except json.JSONDecodeError:
        return None
    values = []
    for entry in entries:
        try:
            values.append(float(entry["val_log_loss"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return None
    return min(values)


def load_best_entry(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fp:
            entries = json.load(fp)
    except json.JSONDecodeError:
        return None
    if not isinstance(entries, list) or not entries:
        return None
    entries_sorted = sorted(entries, key=lambda e: e.get("val_log_loss", float("inf")))
    return entries_sorted[0]


def ensure_raw_csvs(data_dir: Path) -> None:
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Expected {train_csv} and {test_csv}. Place Kaggle CSVs or re-run download before Day 3."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Day 3 preprocessing ablation and guarded GA search.")
    parser.add_argument("--data-dir", default="data", help="Directory with raw Kaggle CSVs.")
    parser.add_argument("--stack-folds", type=int, default=5)
    parser.add_argument("--stack-seed", type=int, default=2026)
    parser.add_argument("--stack-top-n", type=int, default=12)
    parser.add_argument("--ga-generations", type=int, default=10)
    parser.add_argument("--ga-population", type=int, default=12)
    parser.add_argument("--ga-top-k", type=int, default=2)
    parser.add_argument("--ga-cv-folds", type=int, default=5)
    parser.add_argument("--ga-seed", type=int, default=1701)
    parser.add_argument(
        "--reference-summary",
        default="models/ga_search_summary.json",
        help="Path to the previous GA summary for baseline log loss (read before overwriting).",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.003,
        help="Minimum log-loss improvement required before submitting a Day 3 model.",
    )
    parser.add_argument(
        "--submit-if-better",
        action="store_true",
        help="Submit the best new GA model automatically if improvement threshold is met.",
    )
    parser.add_argument(
        "--kaggle-message",
        default="Day3 sanity check",
        help="Submission message used if --submit-if-better triggers.",
    )
    parser.add_argument(
        "--select-output",
        default="models/day3_top_configs.json",
        help="Destination JSON for selected elite configs.",
    )
    parser.add_argument(
        "--stack-output-dir",
        default="models/day3_stack",
        help="Directory to store rebuilt Day 3 stacking artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    ensure_raw_csvs(data_dir)
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    train_prepared = data_dir / "train_prepared.csv"
    test_prepared = data_dir / "test_prepared.csv"

    reference_best = load_best_log_loss(Path(args.reference_summary))
    if reference_best is None:
        print("[day3] No baseline log loss detected; any improvement will qualify.")
    else:
        print(f"[day3] Baseline val_log_loss: {reference_best:.4f}")

    run_cmd(
        [
            "python",
            "scripts/prepare_data.py",
            "--input",
            str(train_csv),
            "--output",
            str(train_prepared),
            "--exclude-columns",
            "Transported",
        ]
    )
    run_cmd(
        [
            "python",
            "scripts/prepare_data.py",
            "--input",
            str(test_csv),
            "--output",
            str(test_prepared),
        ]
    )
    run_cmd(
        [
            "python",
            "scripts/add_baseline_predictions.py",
            "--train",
            str(train_prepared),
            "--test",
            str(test_prepared),
            "--target-column",
            "Transported",
        ]
    )

    select_output = Path(args.select_output)
    run_cmd(
        [
            "python",
            "scripts/select_top_submissions.py",
            "--summaries-dir",
            "models",
            "--pattern",
            "ga_search_summary*.json",
            "--top",
            str(args.stack_top_n),
            "--output",
            str(select_output),
            "--include-missing",
        ]
    )

    run_cmd(
        [
            "python",
            "scripts/build_stack_meta.py",
            "--train",
            str(train_prepared),
            "--test",
            str(test_prepared),
            "--raw-test",
            str(test_csv),
            "--top-configs",
            str(select_output),
            "--folds",
            str(args.stack_folds),
            "--seed",
            str(args.stack_seed),
            "--output-dir",
            args.stack_output_dir,
        ]
    )

    run_cmd(
        [
            "python",
            "scripts/train_ga_nn.py",
            "--train-prepared",
            str(train_prepared),
            "--test-prepared",
            str(test_prepared),
            "--raw-test",
            str(test_csv),
            "--generations",
            str(args.ga_generations),
            "--population",
            str(args.ga_population),
            "--top-k",
            str(args.ga_top_k),
            "--cv-folds",
            str(args.ga_cv_folds),
            "--resume-summary",
            str(select_output),
            "--resume-top-n",
            str(args.stack_top_n),
            "--seed",
            str(args.ga_seed),
            "--no-kaggle",
        ]
    )

    new_summary_path = Path("models") / "ga_search_summary.json"
    new_best_entry = load_best_entry(new_summary_path)
    if not new_best_entry:
        raise RuntimeError(f"GA summary missing or empty: {new_summary_path}")
    new_best = float(new_best_entry["val_log_loss"])
    print(f"[day3] New GA best val_log_loss: {new_best:.4f}")

    improvement = None
    if reference_best is not None:
        improvement = reference_best - new_best
        print(f"[day3] Improvement over baseline: {improvement:.4f}")
    else:
        print("[day3] Treating new model as baseline (no comparison).")

    should_submit = reference_best is None or (improvement is not None and improvement >= args.improvement_threshold)
    if should_submit:
        print("[day3] Improvement threshold satisfied. You may submit the new GA model.")
        if args.submit_if_better:
            submission_path = new_best_entry.get("submission_path")
            if not submission_path:
                raise RuntimeError("Best entry missing submission_path; cannot submit automatically.")
            submission_file = Path(submission_path)
            if not submission_file.exists():
                raise RuntimeError(f"Submission CSV not found: {submission_file}")
            run_cmd(
                [
                    "kaggle",
                    "competitions",
                    "submit",
                    "-c",
                    KAGGLE_COMPETITION,
                    "-f",
                    str(submission_file),
                    "-m",
                    args.kaggle_message,
                ]
            )
    else:
        print(
            "[day3] Improvement is below threshold; skip Kaggle submissions for this ablation to conserve the quota."
        )


if __name__ == "__main__":
    main()
