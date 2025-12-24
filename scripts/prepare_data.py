#!/usr/bin/env python3
"""Utility to preprocess Spaceship Titanic tabular data.

Steps:
1. One-hot encode every non-numeric column.
2. Impute missing values with the mean of their columns.
3. Min-max normalize every resulting numeric feature to [0, 1].

By default the script transforms all columns, but you can keep specific fields
(such as the target label) untouched via --exclude-columns.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Titanic dataset features.")
    parser.add_argument("--input", required=True, help="Path to the raw CSV file to transform.")
    parser.add_argument("--output", required=True, help="Destination CSV for the processed data.")
    parser.add_argument(
        "--exclude-columns",
        nargs="*",
        default=(),
        help="Column names to preserve unchanged (e.g., Transported).",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=("PassengerId", "Name"),
        help="Columns to remove entirely before processing (default: PassengerId, Name).",
    )
    return parser.parse_args()


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all non-numeric columns to one-hot encoded numeric features."""
    encoded = pd.get_dummies(df, drop_first=False)
    return encoded.astype(float)


def fill_missing_with_means(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN values with column means (works on numeric-only frames)."""
    means = df.mean(numeric_only=True)
    # Reindex ensures columns with non-numeric types (if any remain) still get something.
    means = means.reindex(df.columns, fill_value=0.0)
    return df.fillna(means)


def min_max_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Scale every column to the [0, 1] range (no-ops for constant columns)."""
    mins = df.min()
    ranges = df.max() - mins
    ranges = ranges.replace(0, 1.0)
    return (df - mins) / ranges


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_df = pd.read_csv(input_path)

    excluded_columns: Sequence[str] = args.exclude_columns
    drop_columns: Sequence[str] = args.drop_columns
    preserved_df = raw_df.loc[:, raw_df.columns.intersection(excluded_columns)]
    drop_set = set(excluded_columns) | set(drop_columns)
    features_df = raw_df.drop(columns=drop_set, errors="ignore")

    encoded_df = one_hot_encode(features_df)
    imputed_df = fill_missing_with_means(encoded_df)
    normalized_df = min_max_normalize(imputed_df)

    processed_df = pd.concat([normalized_df, preserved_df], axis=1)
    processed_df.to_csv(output_path, index=False)

    print(f"Wrote processed data to {output_path}")


if __name__ == "__main__":
    main()
