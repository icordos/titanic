#!/usr/bin/env python3
"""Utility to preprocess Spaceship Titanic tabular data.

Steps:
1. Engineer richer domain features (passenger groups, cabin splits, spending aggregates).
2. Clip extreme spending values and impute missing entries with column-aware strategies.
3. One-hot encode a small set of low-cardinality categoricals and z-score scale all numeric features.

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
ONE_HOT_CATEGORICALS = ("HomePlanet", "Destination", "CabinDeck", "CabinSide")
HIGH_CARDINALITY_DROP = {"Cabin"}


def clip_spend_inputs(df: pd.DataFrame, quantile: float = 0.995) -> None:
    existing = [col for col in SPEND_COLUMNS if col in df.columns]
    for col in existing:
        upper = df[col].quantile(quantile)
        if pd.isna(upper):
            continue
        df[col] = df[col].clip(upper=upper)


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
    clip_spend_inputs(enriched)
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


def encode_limited_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode only the configured low-cardinality categoricals."""
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    cat_cols = [col for col in ONE_HOT_CATEGORICALS if col in df.columns]
    if not cat_cols:
        return numeric_df
    encoded = pd.get_dummies(df[cat_cols], drop_first=False, dtype=float)
    return pd.concat([numeric_df, encoded], axis=1)


def standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply z-score scaling column-wise, leaving constant columns untouched."""
    means = df.mean()
    stds = df.std(ddof=0).replace(0, 1.0)
    return (df - means) / stds


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
    drop_set = set(excluded_columns) | set(drop_columns) | HIGH_CARDINALITY_DROP
    features_df = enriched_df.drop(columns=drop_set, errors="ignore")

    impute_targeted(features_df)

    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    categorical_cols = [col for col in features_df.columns if col not in numeric_cols]
    if len(numeric_cols) > 0:
        features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].median())
    for col in categorical_cols:
        mode_series = features_df[col].mode(dropna=True)
        fill_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        features_df[col] = features_df[col].fillna(fill_value)

    encoded_df = encode_limited_categoricals(features_df)
    standardized_df = standardize_features(encoded_df)

    processed_df = pd.concat([standardized_df, preserved_df], axis=1)
    processed_df.to_csv(output_path, index=False)

    print(f"Wrote processed data to {output_path}")


if __name__ == "__main__":
    main()
