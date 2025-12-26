#!/usr/bin/env python3
"""Utility to preprocess Spaceship Titanic tabular data.

Steps:
1. Engineer richer domain features (passenger groups, cabin splits, spending aggregates).
2. Impute missing values with column-aware strategies (medians for numeric, modes for categoricals).
3. One-hot encode any categorical leftovers and min-max normalize numeric features to [0, 1].

By default the script transforms all columns, but you can keep specific fields
(such as the target label) untouched via --exclude-columns.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def parse_bool_like(series: pd.Series) -> pd.Series:
    """Normalize Kaggle's string booleans into binary floats."""
    mapping = {
        "True": 1.0,
        True: 1.0,
        "False": 0.0,
        False: 0.0,
    }
    return series.map(mapping)


def add_cabin_features(df: pd.DataFrame) -> None:
    if "Cabin" not in df.columns:
        return
    cabin_parts = df["Cabin"].str.split("/", expand=True)
    df["CabinDeck"] = cabin_parts[0]
    df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    df["CabinSide"] = cabin_parts[2]


def add_passenger_group_features(df: pd.DataFrame) -> None:
    if "PassengerId" not in df.columns:
        return
    passenger_parts = df["PassengerId"].str.split("_", expand=True)
    group_ids = passenger_parts[0]
    member_numbers = passenger_parts[1]
    df["GroupId"] = pd.to_numeric(group_ids, errors="coerce")
    df["GroupMemberNumber"] = pd.to_numeric(member_numbers, errors="coerce")
    group_sizes = group_ids.map(group_ids.value_counts())
    df["GroupSize"] = group_sizes.astype(float)
    df["IsGroupSolo"] = (df["GroupSize"] == 1).astype(float)


def add_family_features(df: pd.DataFrame) -> None:
    if "Name" not in df.columns:
        return
    surnames = df["Name"].str.split().str[-1]
    family_sizes = surnames.map(surnames.value_counts())
    df["FamilySize"] = family_sizes.astype(float)
    df["IsFamilySolo"] = (df["FamilySize"] == 1).astype(float)


def add_spending_features(df: pd.DataFrame) -> None:
    existing_spend_cols = [col for col in SPEND_COLUMNS if col in df.columns]
    if not existing_spend_cols:
        return
    spend_df = df[existing_spend_cols].fillna(0.0)
    df["TotalSpend"] = spend_df.sum(axis=1)
    df["LogTotalSpend"] = np.log1p(df["TotalSpend"])
    df["NonZeroSpendCount"] = (spend_df > 0).sum(axis=1)
    df["AllZeroSpend"] = (df["TotalSpend"] == 0).astype(float)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for col in ("CryoSleep", "VIP"):
        if col in enriched.columns:
            enriched[col] = parse_bool_like(enriched[col])
    add_cabin_features(enriched)
    add_passenger_group_features(enriched)
    add_family_features(enriched)
    add_spending_features(enriched)
    return enriched


def impute_targeted(features_df: pd.DataFrame) -> None:
    if "HomePlanet" in features_df.columns and "Age" in features_df.columns:
        planet_medians = features_df.groupby("HomePlanet")["Age"].transform("median")
        features_df["Age"] = features_df["Age"].fillna(planet_medians)
        features_df["Age"] = features_df["Age"].fillna(features_df["Age"].median())

    if "CabinDeck" in features_df.columns and "CabinNum" in features_df.columns:
        deck_medians = features_df.groupby("CabinDeck")["CabinNum"].transform("median")
        features_df["CabinNum"] = features_df["CabinNum"].fillna(deck_medians)
        features_df["CabinNum"] = features_df["CabinNum"].fillna(features_df["CabinNum"].median())

    if "Destination" in features_df.columns:
        destination = features_df["Destination"]
    else:
        destination = None
    for col in SPEND_COLUMNS:
        if col not in features_df.columns:
            continue
        if destination is not None:
            dest_medians = features_df.groupby("Destination")[col].transform("median")
            features_df[col] = features_df[col].fillna(dest_medians)
        features_df[col] = features_df[col].fillna(features_df[col].median())

    for col in ("GroupId", "GroupMemberNumber", "GroupSize", "FamilySize"):
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(features_df[col].median())

    for col in ("CryoSleep", "VIP", "IsGroupSolo", "IsFamilySolo", "AllZeroSpend"):
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(0.0)

    for col in ("CabinDeck", "CabinSide", "HomePlanet", "Destination"):
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna("Unknown")


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
    enriched_df = engineer_features(raw_df)

    excluded_columns: Sequence[str] = args.exclude_columns
    drop_columns: Sequence[str] = args.drop_columns
    preserved_df = enriched_df.loc[:, enriched_df.columns.intersection(excluded_columns)]
    drop_set = set(excluded_columns) | set(drop_columns)
    features_df = enriched_df.drop(columns=drop_set, errors="ignore")

    impute_targeted(features_df)

    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    categorical_cols = [col for col in features_df.columns if col not in numeric_cols]
    if numeric_cols.any():
        features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].median())
    for col in categorical_cols:
        mode_series = features_df[col].mode(dropna=True)
        fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        features_df[col] = features_df[col].fillna(fill_value)

    encoded_df = one_hot_encode(features_df)
    imputed_df = fill_missing_with_means(encoded_df)
    normalized_df = min_max_normalize(imputed_df)

    processed_df = pd.concat([normalized_df, preserved_df], axis=1)
    processed_df.to_csv(output_path, index=False)

    print(f"Wrote processed data to {output_path}")


if __name__ == "__main__":
    main()
