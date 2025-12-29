#!/usr/bin/env python3
"""Average multiple Kaggle submission CSVs into an ensemble prediction."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble multiple GA submission CSVs.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Submission CSV paths to average. Columns must include PassengerId and Transported.",
    )
    parser.add_argument("--output", required=True, help="Destination CSV for the ensembled submission.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for Transported label (default: 0.5).",
    )
    return parser.parse_args()


def read_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "PassengerId" not in df.columns or "Transported" not in df.columns:
        raise ValueError(f"{path} missing PassengerId/Transported columns.")
    if df["Transported"].dtype == bool:
        df["Transported"] = df["Transported"].astype(float)
    return df


def main() -> None:
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]
    if len(input_paths) < 2:
        raise ValueError("Provide at least two submission files for ensembling.")

    frames: List[pd.DataFrame] = []
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Submission file not found: {path}")
        frame = read_submission(path)
        frames.append(frame.sort_values("PassengerId").reset_index(drop=True))

    passenger_ids = frames[0]["PassengerId"]
    for frame in frames[1:]:
        if not passenger_ids.equals(frame["PassengerId"]):
            raise ValueError("PassengerId ordering mismatch between submissions.")

    stacked = pd.concat([frame["Transported"] for frame in frames], axis=1)
    mean_probs = stacked.mean(axis=1)
    ensemble = pd.DataFrame(
        {
            "PassengerId": passenger_ids,
            "Transported": (mean_probs >= args.threshold),
        }
    )
    # Store averaged probability for debugging purposes.
    ensemble["TransportedProb"] = mean_probs
    output_path = Path(args.output)
    ensemble.to_csv(output_path, index=False)
    print(f"Wrote ensemble submission to {output_path}")


if __name__ == "__main__":
    main()
