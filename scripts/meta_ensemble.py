#!/usr/bin/env python3
"""Blend GA submission probabilities with baseline LR/GB predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meta-ensemble GA submission with LR/GB baselines.")
    parser.add_argument("--ga-submission", required=True, help="Path to GA submission CSV.")
    parser.add_argument("--test-prepared", default="data/test_prepared.csv", help="Prepared test CSV with LR/GB columns.")
    parser.add_argument("--output", required=True, help="Destination CSV for the blended submission.")
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        metavar=("GA", "LR", "GB"),
        default=(0.6, 0.2, 0.2),
        help="Weights for GA, LR_pred, and GB_pred probabilities (must sum to 1).",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for Transported label.")
    parser.add_argument("--keep-prob", action="store_true", help="Include blended probability column in output CSV.")
    return parser.parse_args()


def load_probabilities(ga_path: Path, test_path: Path) -> tuple[pd.Series, np.ndarray]:
    ga_df = pd.read_csv(ga_path)
    if "PassengerId" not in ga_df.columns or "Transported" not in ga_df.columns:
        raise ValueError(f"{ga_path} missing PassengerId/Transported columns.")
    if ga_df["Transported"].dtype == bool:
        ga_probs = ga_df["Transported"].astype(float).to_numpy()
    else:
        ga_probs = ga_df["Transported"].to_numpy()
    passenger_ids = ga_df["PassengerId"]

    test_df = pd.read_csv(test_path)
    if "LR_pred" not in test_df.columns or "GB_pred" not in test_df.columns:
        raise ValueError(f"{test_path} missing LR_pred and/or GB_pred columns.")
    lr_probs = test_df["LR_pred"].to_numpy()
    gb_probs = test_df["GB_pred"].to_numpy()
    return passenger_ids, np.vstack([ga_probs, lr_probs, gb_probs])


def main() -> None:
    args = parse_args()
    weights = np.array(args.weights, dtype=np.float64)
    if weights.shape != (3,):
        raise ValueError("Provide exactly three weights for GA, LR_pred, and GB_pred.")
    weight_sum = weights.sum()
    if not np.isclose(weight_sum, 1.0):
        weights = weights / weight_sum

    ga_path = Path(args.ga_submission)
    test_path = Path(args.test_prepared)
    passenger_ids, stacked = load_probabilities(ga_path, test_path)
    blended = np.clip(np.dot(weights, stacked), 0.0, 1.0)

    output = pd.DataFrame(
        {
            "PassengerId": passenger_ids,
            "Transported": blended >= args.threshold,
        }
    )
    if args.keep_prob:
        output["TransportedProb"] = blended
    output_path = Path(args.output)
    output.to_csv(output_path, index=False)
    print(f"Wrote meta-ensemble submission to {output_path}")


if __name__ == "__main__":
    main()
