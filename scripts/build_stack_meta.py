#!/usr/bin/env python3
"""Build a stacked meta-model from top GA tree genomes without touching Kaggle."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CALIBRATED_TYPES = {"rf", "extratrees", "gbstump", "histgb"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stacking meta-model on top GA tree genomes.")
    parser.add_argument("--train", default="data/train_prepared.csv", help="Path to prepared train CSV.")
    parser.add_argument("--test", default="data/test_prepared.csv", help="Path to prepared test CSV.")
    parser.add_argument("--raw-test", default="data/test.csv", help="Raw Kaggle test CSV (for PassengerId).")
    parser.add_argument("--target-column", default="Transported", help="Binary target column name.")
    parser.add_argument("--top-configs", required=True, help="JSON file produced by select_top_submissions.py.")
    parser.add_argument("--folds", type=int, default=5, help="Number of OOF folds.")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--allowed-types",
        default="rf,extratrees,gbstump,histgb",
        help="Comma-separated model types to include (others are skipped).",
    )
    parser.add_argument("--output-dir", default="models/day1_stack", help="Directory for stack artifacts.")
    parser.add_argument(
        "--calibration-folds",
        type=int,
        default=3,
        help="Inner folds for CalibratedClassifierCV when fitting tree models.",
    )
    parser.add_argument(
        "--meta-model",
        choices=["logistic"],
        default="logistic",
        help="Choice of level-2 learner (currently only logistic is supported).",
    )
    parser.add_argument(
        "--baseline-columns",
        default="LR_pred,GB_pred",
        help="Comma-separated baseline probability columns to include in the meta features.",
    )
    return parser.parse_args()


def sanitize_name(raw: str, idx: int) -> str:
    text = raw.strip() or f"model_{idx}"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    if not slug:
        slug = f"model_{idx}"
    return slug


@dataclass
class ModelSpec:
    name: str
    model_type: str
    params: Dict[str, object]

    def build_estimator(self, seed: int):
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=int(self.params.get("rf_trees", 200)),
                max_depth=self.params.get("rf_max_depth"),
                max_features=float(self.params.get("rf_max_features", 0.7)),
                min_samples_split=int(self.params.get("rf_min_samples_split", 20)),
                max_samples=self.params.get("rf_sample_ratio"),
                random_state=seed,
                n_jobs=-1,
            )
        if self.model_type == "extratrees":
            return ExtraTreesClassifier(
                n_estimators=int(self.params.get("rf_trees", 200)),
                max_depth=self.params.get("rf_max_depth"),
                max_features=float(self.params.get("rf_max_features", 0.7)),
                min_samples_split=int(self.params.get("rf_min_samples_split", 20)),
                max_samples=self.params.get("rf_sample_ratio"),
                bootstrap=True,
                random_state=seed,
                n_jobs=-1,
            )
        if self.model_type == "gbstump":
            return GradientBoostingClassifier(
                n_estimators=int(self.params.get("gb_estimators", 150)),
                learning_rate=float(self.params.get("gb_learning_rate", 0.2)),
                min_samples_leaf=int(self.params.get("gb_min_leaf", 100)),
                max_depth=1,
                random_state=seed,
            )
        if self.model_type == "histgb":
            max_depth = self.params.get("hist_max_depth")
            return HistGradientBoostingClassifier(
                learning_rate=float(self.params.get("hist_learning_rate", 0.05)),
                max_depth=None if max_depth in (None, "None") else int(max_depth),
                max_leaf_nodes=int(self.params.get("hist_max_leaf_nodes", 63)),
                min_samples_leaf=int(self.params.get("hist_min_samples_leaf", 50)),
                l2_regularization=float(self.params.get("hist_l2_regularization", 0.1)),
                random_state=seed,
            )
        raise ValueError(f"Unsupported model_type for stacking: {self.model_type}")


def load_top_configs(path: Path, allowed: Sequence[str]) -> List[ModelSpec]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    specs: List[ModelSpec] = []
    for idx, entry in enumerate(payload, 1):
        genome = entry.get("genome") or {}
        model_type = genome.get("model_type") or entry.get("model_type")
        if not model_type or model_type not in allowed:
            continue
        label = entry.get("kaggle_description") or entry.get("name") or f"{model_type}_{idx}"
        specs.append(ModelSpec(name=sanitize_name(label, idx), model_type=model_type, params=genome))
    return specs


def generate_oof_predictions(
    spec: ModelSpec,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    folds: int,
    seed: int,
    calibration_folds: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    test_fold_preds = np.zeros((folds, X_test.shape[0]), dtype=np.float64)
    fold_losses: List[float] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        estimator = spec.build_estimator(seed + fold_idx)
        if spec.model_type in CALIBRATED_TYPES:
            model = CalibratedClassifierCV(estimator, method="sigmoid", cv=calibration_folds)
        else:
            model = estimator
        model.fit(X[train_idx], y[train_idx])
        val_probs = model.predict_proba(X[val_idx])[:, 1]
        oof[val_idx] = val_probs
        test_fold_preds[fold_idx] = model.predict_proba(X_test)[:, 1]
        fold_losses.append(log_loss(y[val_idx], np.clip(val_probs, 1e-7, 1 - 1e-7)))
    test_mean = test_fold_preds.mean(axis=0)
    avg_loss = float(np.mean(fold_losses))
    return oof, test_mean, avg_loss


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_types = {token.strip() for token in args.allowed_types.split(",") if token.strip()}

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    raw_test = pd.read_csv(args.raw_test)
    if args.target_column not in train_df.columns:
        raise ValueError(f"Target column '{args.target_column}' missing from {args.train}")

    baseline_cols = [col.strip() for col in args.baseline_columns.split(",") if col.strip()]
    ensure_columns(train_df, baseline_cols + [args.target_column])
    ensure_columns(test_df, baseline_cols)

    feature_cols = [col for col in train_df.columns if col != args.target_column]
    X = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df.reindex(columns=feature_cols, fill_value=0.0).to_numpy(dtype=np.float32)
    y = train_df[args.target_column].astype(int).to_numpy()

    config_path = Path(args.top_configs)
    specs = load_top_configs(config_path, allowed=allowed_types)
    if not specs:
        raise ValueError(f"No compatible genomes found in {config_path} for allowed types {allowed_types}.")
    print(f"[stack] Loaded {len(specs)} configs from {config_path}")

    stack_train = pd.DataFrame(index=train_df.index)
    stack_test = pd.DataFrame(index=test_df.index)
    for col in baseline_cols:
        stack_train[col] = train_df[col].to_numpy()
        stack_test[col] = test_df[col].to_numpy()

    model_summaries: List[Dict[str, object]] = []
    for spec in specs:
        print(f"[stack] Generating OOF predictions for {spec.name} ({spec.model_type})")
        oof, test_mean, avg_loss = generate_oof_predictions(
            spec,
            X,
            y,
            X_test,
            folds=args.folds,
            seed=args.seed,
            calibration_folds=args.calibration_folds,
        )
        stack_train[spec.name] = oof
        stack_test[spec.name] = test_mean
        model_summaries.append(
            {
                "name": spec.name,
                "model_type": spec.model_type,
                "avg_log_loss": avg_loss,
            }
        )

    meta_features = stack_train.to_numpy(dtype=np.float32)
    meta_test = stack_test.to_numpy(dtype=np.float32)
    if args.meta_model == "logistic":
        meta_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=500,
                        C=1.0,
                        class_weight=None,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported meta-model '{args.meta_model}'")

    meta_model.fit(meta_features, y)
    train_probs = meta_model.predict_proba(meta_features)[:, 1]
    meta_log_loss = log_loss(y, np.clip(train_probs, 1e-7, 1 - 1e-7))
    meta_accuracy = accuracy_score(y, train_probs >= 0.5)
    test_probs = meta_model.predict_proba(meta_test)[:, 1]

    stack_train_out = stack_train.copy()
    stack_train_out[args.target_column] = y
    stack_train_path = output_dir / "stack_train_features.csv"
    stack_test_path = output_dir / "stack_test_features.csv"
    stack_train_out.to_csv(stack_train_path, index=False)
    stack_test.to_csv(stack_test_path, index=False)

    passenger_ids = raw_test["PassengerId"]
    submission = pd.DataFrame({"PassengerId": passenger_ids, "Transported": test_probs >= 0.5})
    submission["TransportedProb"] = test_probs
    submission_path = output_dir / "stack_meta_submission.csv"
    submission.to_csv(submission_path, index=False)

    model_path = output_dir / "stack_meta_model.joblib"
    joblib.dump(meta_model, model_path)

    summary_path = output_dir / "stack_summary.json"
    summary_payload = {
        "meta_model": args.meta_model,
        "meta_accuracy": meta_accuracy,
        "meta_log_loss": meta_log_loss,
        "baseline_columns": baseline_cols,
        "models": model_summaries,
        "folds": args.folds,
    }
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)

    print(f"[stack] Saved train features to {stack_train_path}")
    print(f"[stack] Saved test features to {stack_test_path}")
    print(f"[stack] Saved meta-model to {model_path}")
    print(f"[stack] Saved submission preview to {submission_path}")


if __name__ == "__main__":
    main()
