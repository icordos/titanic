#!/usr/bin/env python3
"""Train lightweight baseline models and append their predictions as features.

This script reads the prepared train/test CSVs, fits two deterministic models
(logistic regression and a shallow gradient boosting ensemble of decision stumps),
and writes their predicted probabilities back as `LR_pred` / `GB_pred` columns.
Run it after `scripts/prepare_data.py` so the GA can consume the extra signals.
"""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


def sigmoid_array(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class TorchLogisticRegression:
    """Simple full-batch logistic regression implemented in PyTorch."""

    def __init__(self, epochs: int, learning_rate: float, weight_decay: float, device: torch.device) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.weights: Optional[Tensor] = None
        self.bias: Optional[Tensor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().unsqueeze(1).to(self.device)

        weights = torch.zeros((n_features, 1), device=self.device, requires_grad=True)
        bias = torch.zeros(1, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([weights, bias], lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            logits = X_tensor @ weights + bias
            loss = criterion(logits, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.weights = weights.detach().cpu()
        self.bias = bias.detach().cpu()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been fit yet.")
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            logits = X_tensor @ self.weights + self.bias
            probs = torch.sigmoid(logits).squeeze()
        return probs.cpu().numpy()


@dataclass
class DecisionStump:
    feature_index: int = 0
    threshold: float = 0.0
    left_value: float = 0.0
    right_value: float = 0.0
    min_leaf: int = 20

    def fit(self, X: np.ndarray, residuals: np.ndarray) -> None:
        n_samples, n_features = X.shape
        best_loss = math.inf
        best_params: Tuple[int, float, float, float] | None = None

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            order = np.argsort(feature_values, kind="mergesort")
            sorted_vals = feature_values[order]
            sorted_res = residuals[order]
            sorted_sq = sorted_res ** 2

            cum_sum = np.cumsum(sorted_res)
            cum_sq = np.cumsum(sorted_sq)
            total_sum = cum_sum[-1]
            total_sq = cum_sq[-1]

            for split_idx in range(self.min_leaf, n_samples - self.min_leaf):
                if sorted_vals[split_idx] == sorted_vals[split_idx - 1]:
                    continue
                left_count = split_idx
                right_count = n_samples - split_idx
                left_sum = cum_sum[split_idx - 1]
                right_sum = total_sum - left_sum
                left_sq = cum_sq[split_idx - 1]
                right_sq = total_sq - left_sq
                left_mean = left_sum / left_count
                right_mean = right_sum / right_count
                left_loss = left_sq - (left_sum ** 2) / left_count
                right_loss = right_sq - (right_sum ** 2) / right_count
                loss = left_loss + right_loss
                if loss < best_loss:
                    threshold = (sorted_vals[split_idx] + sorted_vals[split_idx - 1]) / 2.0
                    best_loss = loss
                    best_params = (feature_idx, threshold, left_mean, right_mean)

        if best_params is None:
            self.feature_index = 0
            self.threshold = 0.0
            self.left_value = 0.0
            self.right_value = 0.0
        else:
            self.feature_index, self.threshold, self.left_value, self.right_value = best_params

    def predict(self, X: np.ndarray) -> np.ndarray:
        feature_values = X[:, self.feature_index]
        mask = feature_values <= self.threshold
        return np.where(mask, self.left_value, self.right_value)


class GradientBoostingStumps:
    """Logistic gradient boosting using decision stumps as weak learners."""

    def __init__(self, n_estimators: int, learning_rate: float, min_leaf: int) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_leaf = min_leaf
        self.base_logit: float = 0.0
        self.estimators: list[DecisionStump] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        prob = np.clip(y.mean(), 1e-6, 1.0 - 1e-6)
        self.base_logit = math.log(prob / (1.0 - prob))
        logits = np.full(len(y), self.base_logit, dtype=np.float64)
        self.estimators.clear()

        for _ in range(self.n_estimators):
            preds = sigmoid_array(logits)
            residuals = y - preds
            stump = DecisionStump(min_leaf=self.min_leaf)
            stump.fit(X, residuals)
            logits += self.learning_rate * stump.predict(X)
            self.estimators.append(stump)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.full(X.shape[0], self.base_logit, dtype=np.float64)
        for stump in self.estimators:
            logits += self.learning_rate * stump.predict(X)
        return sigmoid_array(logits)


def align_test_features(train_features: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    return test_df.reindex(columns=train_features.columns, fill_value=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append baseline model predictions as new features.")
    parser.add_argument("--train", default="data/train_prepared.csv", help="Path to prepared training CSV.")
    parser.add_argument("--test", default="data/test_prepared.csv", help="Path to prepared test CSV.")
    parser.add_argument("--train-out", default=None, help="Optional output path for augmented training CSV.")
    parser.add_argument("--test-out", default=None, help="Optional output path for augmented test CSV.")
    parser.add_argument("--target-column", default="Transported")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-epochs", type=int, default=300)
    parser.add_argument("--lr-learning-rate", type=float, default=0.05)
    parser.add_argument("--lr-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gb-estimators", type=int, default=100)
    parser.add_argument("--gb-learning-rate", type=float, default=0.3)
    parser.add_argument("--gb-min-leaf", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    test_path = Path(args.test)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Train/test CSVs must exist before running this script.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if args.target_column not in train_df.columns:
        raise ValueError(f"Target column '{args.target_column}' missing from {args.train}")

    feature_df = train_df.drop(columns=[args.target_column])
    test_features = align_test_features(feature_df, test_df)
    X_train = feature_df.to_numpy(dtype=np.float32)
    X_test = test_features.to_numpy(dtype=np.float32)
    y = train_df[args.target_column].astype(int).to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[baseline] Training logistic regression on device {device}.")
    logistic = TorchLogisticRegression(
        epochs=args.lr_epochs,
        learning_rate=args.lr_learning_rate,
        weight_decay=args.lr_weight_decay,
        device=device,
    )
    logistic.fit(X_train, y)
    train_df["LR_pred"] = logistic.predict_proba(X_train)
    test_df["LR_pred"] = logistic.predict_proba(X_test)

    print("[baseline] Training gradient boosting stumps.")
    gbst = GradientBoostingStumps(
        n_estimators=args.gb_estimators,
        learning_rate=args.gb_learning_rate,
        min_leaf=args.gb_min_leaf,
    )
    gbst.fit(X_train, y)
    train_df["GB_pred"] = gbst.predict_proba(X_train)
    test_df["GB_pred"] = gbst.predict_proba(X_test)

    train_out = Path(args.train_out) if args.train_out else train_path
    test_out = Path(args.test_out) if args.test_out else test_path
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"Wrote augmented training data to {train_out}")
    print(f"Wrote augmented test data to {test_out}")


if __name__ == "__main__":
    main()
