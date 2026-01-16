#!/usr/bin/env python3
"""Day 5 orchestration: refresh stacker, run boosted GA search, and prep final submissions."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence


def run_cmd(command: Sequence[str], cwd: Optional[Path] = None) -> None:
    pretty = " ".join(command)
    print(f"[day5] $ {pretty}")
    subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)


def load_best_entry(summary_path: Path) -> dict:
    if not summary_path.exists():
        raise FileNotFoundError(f"GA summary not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as fp:
        entries = json.load(fp)
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"No entries in GA summary: {summary_path}")
    best = min(entries, key=lambda e: float(e.get("val_log_loss", float("inf"))))
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute Day 5 pushing workflow.")
    parser.add_argument("--data-dir", default="data", help="Directory containing train/test CSVs.")
    parser.add_argument("--train-prepared", default="data/train_prepared.csv")
    parser.add_argument("--test-prepared", default="data/test_prepared.csv")
    parser.add_argument("--raw-test", default="data/test.csv")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Re-run prepare_data.py and add_baseline_predictions.py before stacking/GA.",
    )
    parser.add_argument("--stacker-submission", required=True, help="Path to latest stacker CSV for blending.")
    parser.add_argument(
        "--stack-top-n",
        type=int,
        default=15,
        help="Number of elite configs to seed the GA and stacker.",
    )
    parser.add_argument(
        "--stack-seed",
        type=int,
        default=2027,
        help="Random seed for Day 5 stacking OOF splits.",
    )
    parser.add_argument("--stack-folds", type=int, default=5)
    parser.add_argument("--stack-output-dir", default="models/day5_stack")
    parser.add_argument("--select-output", default="models/day5_top_configs.json")
    parser.add_argument("--ga-generations", type=int, default=20)
    parser.add_argument("--ga-population", type=int, default=18)
    parser.add_argument("--ga-cv-folds", type=int, default=8)
    parser.add_argument("--ga-top-k", type=int, default=5)
    parser.add_argument("--ga-seed", type=int, default=2045)
    parser.add_argument(
        "--resume-summary",
        default="models/ga_search_summary.json",
        help="Existing GA summary to seed the Day 5 population.",
    )
    parser.add_argument("--output-dir", default="models/day5_push")
    parser.add_argument(
        "--submit-ga",
        action="store_true",
        help="Allow scripts/train_ga_nn.py to submit top genomes to Kaggle.",
    )
    parser.add_argument(
        "--meta-weights",
        type=float,
        nargs=3,
        metavar=("GA", "LR", "GB"),
        default=(0.75, 0.15, 0.10),
        help="Weights for GA/LR/GB meta ensemble.",
    )
    return parser.parse_args()


def ensure_data_files(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    for name in ("train.csv", "test.csv"):
        candidate = data_dir / name
        if not candidate.exists():
            raise FileNotFoundError(f"Missing {candidate}; download Kaggle data first.")


def maybe_prepare_data(args: argparse.Namespace) -> None:
    if not args.prepare_data:
        return
    ensure_data_files(args)
    run_cmd(
        [
            "python",
            "scripts/prepare_data.py",
            "--input",
            str(Path(args.data_dir) / "train.csv"),
            "--output",
            args.train_prepared,
            "--exclude-columns",
            "Transported",
        ]
    )
    run_cmd(
        [
            "python",
            "scripts/prepare_data.py",
            "--input",
            str(Path(args.data_dir) / "test.csv"),
            "--output",
            args.test_prepared,
        ]
    )
    run_cmd(
        [
            "python",
            "scripts/add_baseline_predictions.py",
            "--train",
            args.train_prepared,
            "--test",
            args.test_prepared,
            "--target-column",
            "Transported",
        ]
    )


def main() -> None:
    args = parse_args()
    maybe_prepare_data(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

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
            args.select_output,
            "--include-missing",
        ]
    )

    run_cmd(
        [
            "python",
            "scripts/build_stack_meta.py",
            "--train",
            args.train_prepared,
            "--test",
            args.test_prepared,
            "--raw-test",
            args.raw_test,
            "--top-configs",
            args.select_output,
            "--folds",
            str(args.stack_folds),
            "--seed",
            str(args.stack_seed),
            "--allowed-types",
            "histgb,rf,extratrees",
            "--output-dir",
            args.stack_output_dir,
        ]
    )

    ga_cmd = [
        "python",
        "scripts/train_ga_nn.py",
        "--train-prepared",
        args.train_prepared,
        "--test-prepared",
        args.test_prepared,
        "--raw-test",
        args.raw_test,
        "--generations",
        str(args.ga_generations),
        "--population",
        str(args.ga_population),
        "--top-k",
        str(args.ga_top_k),
        "--cv-folds",
        str(args.ga_cv_folds),
        "--resume-summary",
        args.resume_summary,
        "--resume-top-n",
        str(args.stack_top_n),
        "--seed",
        str(args.ga_seed),
    ]
    if not args.submit_ga:
        ga_cmd.append("--no-kaggle")
    run_cmd(ga_cmd)

    summary_path = Path("models") / "ga_search_summary.json"
    best_entry = load_best_entry(summary_path)
    best_submission = best_entry.get("submission_path")
    if not best_submission:
        raise RuntimeError("Best GA entry missing submission_path.")
    best_submission_path = Path(best_submission)
    if not best_submission_path.exists():
        raise FileNotFoundError(f"Best submission not found: {best_submission_path}")

    # Copy GA best into output dir for bookkeeping
    ga_copy = Path(args.output_dir) / f"day5_ga_best.csv"
    shutil.copy(best_submission_path, ga_copy)
    print(f"[day5] Copied GA best submission to {ga_copy}")

    # Meta ensemble with LR/GB
    meta_output = Path(args.output_dir) / "day5_meta_lrgb.csv"
    run_cmd(
        [
            "python",
            "scripts/meta_ensemble.py",
            "--ga-submission",
            str(best_submission_path),
            "--test-prepared",
            args.test_prepared,
            "--output",
            str(meta_output),
            "--weights",
            *(str(w) for w in args.meta_weights),
        ]
    )

    # Blend GA with stacker
    stacker_path = Path(args.stacker_submission)
    if not stacker_path.exists():
        raise FileNotFoundError(f"Stacker submission not found: {stacker_path}")
    ga_stacker_output = Path(args.output_dir) / "day5_ga_stacker_blend.csv"
    run_cmd(
        [
            "python",
            "scripts/ensemble_submissions.py",
            str(best_submission_path),
            str(stacker_path),
            "--output",
            str(ga_stacker_output),
        ]
    )

    summary_out = Path(args.output_dir) / "day5_summary.json"
    payload = {
        "best_val_log_loss": float(best_entry.get("val_log_loss", 0.0)),
        "best_submission": str(best_submission_path),
        "meta_ensemble": str(meta_output),
        "ga_stacker_blend": str(ga_stacker_output),
    }
    with summary_out.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    print(f"[day5] Wrote summary to {summary_out}")
    print("[day5] Ready to review GA best, meta LR/GB blend, and GA+stacker blend before Kaggle submissions.")


if __name__ == "__main__":
    main()
