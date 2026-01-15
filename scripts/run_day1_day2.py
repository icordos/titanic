#!/usr/bin/env python3
"""Orchestrate Day 1 & Day 2 workflow from codex-strategy."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


KAGGLE_COMPETITION = "spaceship-titanic"


def run_cmd(command: list[str], cwd: Path | None = None) -> None:
    pretty = " ".join(command)
    print(f"[day12] $ {pretty}")
    subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Day 1 + Day 2 Codex strategy steps.")
    parser.add_argument("--data-dir", default="data", help="Directory with raw Kaggle CSVs.")
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download/refresh Kaggle data before preprocessing (requires kaggle CLI).",
    )
    parser.add_argument("--stack-folds", type=int, default=5, help="OOF folds for stacking models.")
    parser.add_argument("--stack-seed", type=int, default=2025, help="Random seed for stacking OOF splits.")
    parser.add_argument("--stack-top-n", type=int, default=12, help="Number of top genomes to keep for stacking/resume.")
    parser.add_argument("--ga-generations", type=int, default=15, help="Generations for the GA search.")
    parser.add_argument("--ga-population", type=int, default=15, help="Population size for GA.")
    parser.add_argument("--ga-cv-folds", type=int, default=8, help="CV folds for GA fitness evaluation.")
    parser.add_argument("--ga-top-k", type=int, default=3, help="Number of genomes to retrain after GA search.")
    parser.add_argument("--ga-seed", type=int, default=1347, help="Base random seed for GA search.")
    parser.add_argument("--resume-pattern", default="ga_search_summary*.json", help="Glob for GA summary inputs.")
    parser.add_argument(
        "--select-output",
        default="models/day1_top_configs.json",
        help="Output JSON for selected top genomes.",
    )
    parser.add_argument(
        "--stack-output-dir",
        default="models/day1_stack",
        help="Directory for stacking artifacts (OOF/test/meta-model).",
    )
    parser.add_argument(
        "--skip-ga",
        action="store_true",
        help="Stop after building the stacking meta-model (skip GA run).",
    )
    return parser.parse_args()


def maybe_download_data(data_dir: Path) -> None:
    run_cmd(["kaggle", "competitions", "download", "-c", KAGGLE_COMPETITION, "-p", str(data_dir)])
    zip_path = data_dir / f"{KAGGLE_COMPETITION}.zip"
    if zip_path.exists():
        run_cmd(["unzip", "-o", zip_path.name], cwd=data_dir)


def ensure_raw_csvs(data_dir: Path) -> None:
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Expected {train_csv} and {test_csv}. Run with --download-data or place the Kaggle files manually."
        )


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.download_data:
        maybe_download_data(data_dir)
    ensure_raw_csvs(data_dir)

    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"
    train_prepared = data_dir / "train_prepared.csv"
    test_prepared = data_dir / "test_prepared.csv"

    # Day 1: preprocessing + stacking pipeline
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
            args.resume_pattern,
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

    if args.skip_ga:
        print("[day12] Skipping GA search (--skip-ga). Day 1 artifacts are ready.")
        return

    # Day 2: upgraded GA exploration seeded by Day 1 elites
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
    print("[day12] Day 2 GA run complete. Review models/ga_search_summary.json for fresh elites.")


if __name__ == "__main__":
    main()
