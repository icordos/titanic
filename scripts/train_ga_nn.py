#!/usr/bin/env python3
"""Genetic-algorithm search over feed-forward neural nets for Spaceship Titanic.

Workflow:
1. Read the prepared training CSV (output of scripts/prepare_data.py with the
   target column preserved) and split it into train/validation folds.
2. Sample a population of fully-connected neural network configurations and let
   a simple GA evolve them for a fixed number of generations.
3. Retrain the top performers on the full training data, score them on the
   validation holdout, and generate Kaggle submissions using the prepared test
   CSV. When the Kaggle CLI is available, submissions are uploaded automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from kaggle_submissions import list_kaggle_submissions

KAGGLE_COMPETITION = "spaceship-titanic"
ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.1),
}
OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
}


@dataclass
class Genome:
    model_type: str
    num_layers: int
    layer_widths: List[int]
    activation: str
    dropout: float
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    gb_estimators: int
    gb_learning_rate: float
    gb_min_leaf: int
    rf_trees: int
    rf_max_depth: int
    rf_max_features: float
    rf_min_samples_split: int
    rf_sample_ratio: float
    rf_random_thresholds: bool
    fitness: float = field(default=-math.inf)

    def clone(self) -> "Genome":
        return Genome(
            model_type=self.model_type,
            num_layers=self.num_layers,
            layer_widths=list(self.layer_widths),
            activation=self.activation,
            dropout=self.dropout,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            epochs=self.epochs,
            gb_estimators=self.gb_estimators,
            gb_learning_rate=self.gb_learning_rate,
            gb_min_leaf=self.gb_min_leaf,
            rf_trees=self.rf_trees,
            rf_max_depth=self.rf_max_depth,
            rf_max_features=self.rf_max_features,
            rf_min_samples_split=self.rf_min_samples_split,
            rf_sample_ratio=self.rf_sample_ratio,
            rf_random_thresholds=self.rf_random_thresholds,
            fitness=self.fitness,
        )


# --- Utility helpers ------------------------------------------------------- #

def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def split_train_val(X: np.ndarray, y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    val_size = max(1, int(len(X) * val_ratio))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def stratified_kfold_indices(y: np.ndarray, folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if folds < 2:
        raise ValueError("folds must be >= 2 for stratified_kfold_indices")
    rng = np.random.default_rng(seed)
    y = y.astype(int)
    unique_classes = np.unique(y)
    fold_bins: List[List[np.ndarray]] = [[] for _ in range(folds)]
    for cls in unique_classes:
        class_indices = np.where(y == cls)[0]
        rng.shuffle(class_indices)
        splits = np.array_split(class_indices, folds)
        for fold_idx, split in enumerate(splits):
            if split.size > 0:
                fold_bins[fold_idx].append(split)
    folds_list: List[Tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(y))
    for fold_idx in range(folds):
        if fold_bins[fold_idx]:
            val_idx = np.concatenate(fold_bins[fold_idx])
        else:
            val_idx = np.array([], dtype=int)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_indices[train_mask]
        folds_list.append((train_idx, val_idx))
    return folds_list


def build_cv_splits(
    X: np.ndarray,
    y: np.ndarray,
    folds: int,
    val_ratio: float,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if folds <= 1:
        X_train, y_train, X_val, y_val = split_train_val(X, y, val_ratio=val_ratio, seed=seed)
        return [(X_train, y_train, X_val, y_val)]
    splits = []
    for train_idx, val_idx in stratified_kfold_indices(y, folds, seed):
        splits.append((X[train_idx], y[train_idx], X[val_idx], y[val_idx]))
    return splits


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    features = torch.from_numpy(X).float()
    labels = torch.from_numpy(y).float().unsqueeze(1)
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_model(input_dim: int, genome: Genome) -> nn.Module:
    layers: List[nn.Module] = []
    in_features = input_dim
    activation_cls = ACTIVATIONS[genome.activation]
    for width in genome.layer_widths[: genome.num_layers]:
        layers.append(nn.Linear(in_features, width))
        layers.append(activation_cls())
        if genome.dropout > 0:
            layers.append(nn.Dropout(genome.dropout))
        in_features = width
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)


def train_candidate(
    genome: Genome,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> Tuple[float, float]:
    if genome.model_type == "mlp":
        train_loader = make_loader(X_train, y_train, genome.batch_size, shuffle=True)
        val_loader = make_loader(X_val, y_val, batch_size=1024, shuffle=False)
        model = build_model(X_train.shape[1], genome).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer_cls = OPTIMIZERS[genome.optimizer]
        optimizer = optimizer_cls(model.parameters(), lr=genome.learning_rate, weight_decay=genome.weight_decay)
        for _ in range(genome.epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        return evaluate_metrics(model, val_loader, device)

    if genome.model_type == "gbstump":
        booster = GradientBoostingStumps(
            n_estimators=genome.gb_estimators,
            learning_rate=genome.gb_learning_rate,
            min_leaf=genome.gb_min_leaf,
            seed=random.randint(0, 1_000_000),
        )
        booster.fit(X_train, y_train)
        probs = booster.predict_proba(X_val)
        return compute_numpy_metrics(y_val, probs)

    forest = RandomForestEnsemble(
        n_estimators=genome.rf_trees,
        max_depth=genome.rf_max_depth,
        min_samples_split=genome.rf_min_samples_split,
        max_features_fraction=genome.rf_max_features,
        sample_ratio=genome.rf_sample_ratio,
        random_thresholds=genome.model_type == "extratrees",
        seed=random.randint(0, 1_000_000),
    )
    forest.fit(X_train, y_train)
    probs = forest.predict_proba(X_val)
    return compute_numpy_metrics(y_val, probs)


def evaluate_genome_cv(
    genome: Genome,
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device,
) -> Tuple[float, List[float], List[float]]:
    accuracies: List[float] = []
    log_losses: List[float] = []
    for X_train, y_train, X_val, y_val in cv_splits:
        acc, log_loss = train_candidate(genome, X_train, y_train, X_val, y_val, device)
        accuracies.append(acc)
        log_losses.append(log_loss)
    if not log_losses:
        return math.inf, [], []
    avg_log_loss = float(np.mean(log_losses))
    return avg_log_loss, accuracies, log_losses


def evaluate_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    log_loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            pred_labels = probs >= 0.5
            correct += (pred_labels.float() == yb).sum().item()
            total += yb.numel()
            probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)
            log_loss = -(yb * torch.log(probs) + (1 - yb) * torch.log(1 - probs))
            log_loss_sum += log_loss.sum().item()
    accuracy = correct / max(1, total)
    avg_log_loss = log_loss_sum / max(1, total)
    return accuracy, avg_log_loss


def compute_numpy_metrics(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    probs = probs.reshape(-1)
    preds = probs >= 0.5
    accuracy = float((preds == y_true).mean())
    clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    log_loss = float(-(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)).mean())
    return accuracy, log_loss


@dataclass
class DecisionStump:
    feature_index: int = 0
    threshold: float = 0.0
    left_value: float = 0.0
    right_value: float = 0.0
    min_leaf: int = 20

    def fit(self, X: np.ndarray, residuals: np.ndarray) -> None:
        best_loss = math.inf
        best_params: Optional[Tuple[int, float, float, float]] = None
        n_samples, n_features = X.shape
        for feat in range(n_features):
            feature_values = X[:, feat]
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
                left_loss = left_sq - (left_sum**2) / left_count
                right_loss = right_sq - (right_sum**2) / right_count
                loss = left_loss + right_loss
                if loss < best_loss:
                    threshold = (sorted_vals[split_idx] + sorted_vals[split_idx - 1]) / 2.0
                    best_loss = loss
                    best_params = (feat, threshold, left_mean, right_mean)
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
    def __init__(self, n_estimators: int, learning_rate: float, min_leaf: int, seed: int) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_leaf = min_leaf
        self.seed = seed
        self.base_logit = 0.0
        self.estimators: List[DecisionStump] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        prob = np.clip(y.mean(), 1e-6, 1.0 - 1e-6)
        self.base_logit = math.log(prob / (1.0 - prob))
        logits = np.full(len(y), self.base_logit, dtype=np.float64)
        self.estimators.clear()
        for _ in range(self.n_estimators):
            preds = 1.0 / (1.0 + np.exp(-logits))
            residuals = y - preds
            stump = DecisionStump(min_leaf=self.min_leaf)
            stump.fit(X, residuals)
            logits += self.learning_rate * stump.predict(X)
            self.estimators.append(stump)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.full(X.shape[0], self.base_logit, dtype=np.float64)
        for stump in self.estimators:
            logits += self.learning_rate * stump.predict(X)
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass
class TreeNode:
    prob: float
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


class RandomTreeClassifier:
    def __init__(
        self,
        max_depth: int,
        min_samples_split: int,
        max_features_fraction: float,
        n_thresholds: int,
        random_thresholds: bool,
        seed: int,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_fraction = max_features_fraction
        self.n_thresholds = n_thresholds
        self.random_thresholds = random_thresholds
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.root: Optional[TreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        prob = float(y.mean())
        node = TreeNode(prob=prob)
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or prob == 0.0
            or prob == 1.0
        ):
            return node
        n_features = X.shape[1]
        max_features = max(1, int(self.max_features_fraction * n_features))
        feature_indices = self.rng.choice(n_features, size=max_features, replace=False)
        best_feature = None
        best_threshold = None
        best_score = math.inf
        for feat in feature_indices:
            values = X[:, feat]
            min_val, max_val = float(values.min()), float(values.max())
            if min_val == max_val:
                continue
            thresholds = []
            if self.random_thresholds:
                thresholds.append(float(self.rng.uniform(min_val, max_val)))
            else:
                quantiles = np.linspace(0.1, 0.9, self.n_thresholds)
                thresholds = list(np.quantile(values, quantiles))
            for thresh in thresholds:
                left_mask = values <= thresh
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue
                left_y = y[left_mask]
                right_y = y[right_mask]
                left_imp = self._gini(left_y)
                right_imp = self._gini(right_y)
                score = (len(left_y) * left_imp + len(right_y) * right_imp) / len(y)
                if score < best_score:
                    best_score = score
                    best_feature = feat
                    best_threshold = thresh
        if best_feature is None or best_threshold is None:
            return node
        values = X[:, best_feature]
        left_mask = values <= best_threshold
        right_mask = ~left_mask
        node.feature_index = best_feature
        node.threshold = best_threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    @staticmethod
    def _gini(labels: np.ndarray) -> float:
        if len(labels) == 0:
            return 0.0
        p = labels.mean()
        return 2.0 * p * (1.0 - p)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Tree not fitted.")
        probs = np.empty(X.shape[0], dtype=np.float64)
        for idx, row in enumerate(X):
            node = self.root
            while node and node.feature_index is not None and node.threshold is not None:
                if row[node.feature_index] <= node.threshold:
                    node = node.left or node
                else:
                    node = node.right or node
            probs[idx] = node.prob if node else 0.5
        return probs


class RandomForestEnsemble:
    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        min_samples_split: int,
        max_features_fraction: float,
        sample_ratio: float,
        random_thresholds: bool,
        seed: int,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_fraction = max_features_fraction
        self.sample_ratio = sample_ratio
        self.random_thresholds = random_thresholds
        self.seed = seed
        self.trees: List[RandomTreeClassifier] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        n_samples = len(X)
        self.trees.clear()
        for est_idx in range(self.n_estimators):
            sample_size = max(1, int(self.sample_ratio * n_samples))
            indices = rng.choice(n_samples, size=sample_size, replace=True)
            tree = RandomTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features_fraction=self.max_features_fraction,
                n_thresholds=8,
                random_thresholds=self.random_thresholds,
                seed=int(rng.integers(0, 1_000_000)),
            )
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Forest not fitted.")
        preds = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self.trees:
            preds += tree.predict_proba(X)
        return preds / len(self.trees)


# --- Genetic Algorithm ----------------------------------------------------- #

WIDTH_CHOICES = [16, 32, 64, 128, 256, 512]
BATCH_CHOICES = [64, 128, 256, 512]
EPOCH_CHOICES = [10, 15, 20, 25, 30]
LEARNING_RATES = [5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
WEIGHT_DECAY = [0.0, 1e-5, 1e-4, 5e-4]
DROPOUT = [0.0, 0.1, 0.2, 0.3, 0.4]
MODEL_TYPES = ["mlp", "gbstump", "rf", "extratrees"]
GB_ESTIMATORS = [100, 150, 200, 300]
GB_LEARNING_RATE = [0.1, 0.2, 0.3]
GB_MIN_LEAF = [50, 100, 150, 200]
RF_TREES = [100, 200, 300, 400]
RF_MAX_DEPTH = [4, 5, 6, 7, 8]
RF_MAX_FEATURES = [0.5, 0.7, 0.9]
RF_MIN_SAMPLES_SPLIT = [10, 20, 40]
RF_SAMPLE_RATIO = [0.6, 0.8, 1.0]


def random_genome() -> Genome:
    model_type = random.choice(MODEL_TYPES)
    num_layers = random.randint(1, 6)
    widths = [random.choice(WIDTH_CHOICES) for _ in range(num_layers)]
    return Genome(
        model_type=model_type,
        num_layers=num_layers,
        layer_widths=widths,
        activation=random.choice(list(ACTIVATIONS.keys())),
        dropout=random.choice(DROPOUT),
        optimizer=random.choice(list(OPTIMIZERS.keys())),
        learning_rate=random.choice(LEARNING_RATES),
        weight_decay=random.choice(WEIGHT_DECAY),
        batch_size=random.choice(BATCH_CHOICES),
        epochs=random.choice(EPOCH_CHOICES),
        gb_estimators=random.choice(GB_ESTIMATORS),
        gb_learning_rate=random.choice(GB_LEARNING_RATE),
        gb_min_leaf=random.choice(GB_MIN_LEAF),
        rf_trees=random.choice(RF_TREES),
        rf_max_depth=random.choice(RF_MAX_DEPTH),
        rf_max_features=random.choice(RF_MAX_FEATURES),
        rf_min_samples_split=random.choice(RF_MIN_SAMPLES_SPLIT),
        rf_sample_ratio=random.choice(RF_SAMPLE_RATIO),
        rf_random_thresholds=model_type == "extratrees",
    )


def mutate(genome: Genome, macro_rate: float) -> Genome:
    child = genome.clone()
    if random.random() < macro_rate:
        child.model_type = random.choice(MODEL_TYPES)
        child.num_layers = random.randint(1, 6)
        child.layer_widths = [random.choice(WIDTH_CHOICES) for _ in range(child.num_layers)]
        child.activation = random.choice(list(ACTIVATIONS.keys()))
    else:
        idx = random.randrange(child.num_layers)
        child.layer_widths[idx] = random.choice(WIDTH_CHOICES)
    # micro tweaks
    child.dropout = max(0.0, min(0.5, child.dropout + random.uniform(-0.05, 0.05)))
    child.learning_rate = max(5e-4, min(5e-3, child.learning_rate * random.uniform(0.8, 1.2)))
    child.weight_decay = max(0.0, min(1e-3, child.weight_decay * random.uniform(0.5, 1.5) + random.uniform(-1e-5, 1e-5)))
    child.batch_size = random.choice(BATCH_CHOICES)
    child.epochs = random.choice(EPOCH_CHOICES)
    child.optimizer = random.choice(list(OPTIMIZERS.keys()))
    child.gb_estimators = random.choice(GB_ESTIMATORS)
    child.gb_learning_rate = random.choice(GB_LEARNING_RATE)
    child.gb_min_leaf = random.choice(GB_MIN_LEAF)
    child.rf_trees = random.choice(RF_TREES)
    child.rf_max_depth = random.choice(RF_MAX_DEPTH)
    child.rf_max_features = random.choice(RF_MAX_FEATURES)
    child.rf_min_samples_split = random.choice(RF_MIN_SAMPLES_SPLIT)
    child.rf_sample_ratio = random.choice(RF_SAMPLE_RATIO)
    child.rf_random_thresholds = child.model_type == "extratrees"
    return child


def evolve(
    population: List[Genome],
    cv_splits: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    device: torch.device,
    generations: int,
    macro_rate: float,
    elite_count: int,
) -> List[Genome]:
    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")
        for genome in population:
            avg_log_loss, fold_accs, fold_losses = evaluate_genome_cv(genome, cv_splits, device)
            genome.fitness = -avg_log_loss  # Higher fitness = lower log loss
            fold_desc = ", ".join(f"loss={loss:.4f}/acc={acc:.4f}" for acc, loss in zip(fold_accs, fold_losses))
            if genome.model_type == "mlp":
                arch_desc = (
                    f"layers={genome.layer_widths[:genome.num_layers]} act={genome.activation}"
                    f" lr={genome.learning_rate:.4f} batch={genome.batch_size} epochs={genome.epochs}"
                )
            elif genome.model_type == "gbstump":
                arch_desc = (
                    f"gbstump estimators={genome.gb_estimators} lr={genome.gb_learning_rate}"
                    f" min_leaf={genome.gb_min_leaf}"
                )
            else:
                arch_desc = (
                    f"{genome.model_type} trees={genome.rf_trees} depth={genome.rf_max_depth}"
                    f" max_feat={genome.rf_max_features:.2f} sample_ratio={genome.rf_sample_ratio:.2f}"
                )
            print(f"  [{genome.model_type}] {arch_desc} -> avg_log_loss={avg_log_loss:.4f} (folds: {fold_desc})")
        population.sort(key=lambda g: g.fitness, reverse=True)
        print(f"  best avg log loss this gen: {-population[0].fitness:.4f}")
        elites = [g.clone() for g in population[:elite_count]]
        offspring: List[Genome] = []
        while len(elites) + len(offspring) < len(population):
            parent = random.choice(population[: max(elite_count * 2, 2)])
            offspring.append(mutate(parent, macro_rate))
        population = elites + offspring
    return population


# --- Submission helpers ---------------------------------------------------- #


def align_test_features(test_df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    # Reindex ensures the same column order and inserts any missing columns
    # filled with zeros, while dropping extras automatically.
    return test_df.reindex(columns=feature_cols, fill_value=0.0)


def retrain_full_mlp(genome: Genome, X: np.ndarray, y: np.ndarray, device: torch.device) -> nn.Module:
    loader = make_loader(X, y, genome.batch_size, shuffle=True)
    model = build_model(X.shape[1], genome).to(device)
    optimizer = OPTIMIZERS[genome.optimizer](
        model.parameters(), lr=genome.learning_rate, weight_decay=genome.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(genome.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model


def predict_probabilities(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    loader = DataLoader(torch.from_numpy(X).float(), batch_size=1024)
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            preds.append(probs)
    return np.vstack(preds).squeeze()


def save_submission(passenger_ids: Sequence[str], probs: np.ndarray, threshold: float, path: Path) -> None:
    df = pd.DataFrame({"PassengerId": passenger_ids, "Transported": probs >= threshold})
    df.to_csv(path, index=False)


def try_kaggle_submit(file_path: Path, message: str) -> bool:
    kaggle_cli = shutil.which("kaggle")
    if kaggle_cli is None:
        print("[warn] Kaggle CLI not found in PATH; skipping automatic submission.")
        return False
    cmd = f"kaggle competitions submit -c {KAGGLE_COMPETITION} -f {file_path} -m \"{message}\""
    print(f"Submitting via Kaggle CLI: {cmd}")
    result = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    if result.returncode != 0:
        print("[warn] Kaggle submission failed:")
        print(result.stdout)
        print(result.stderr)
        return False
    print(result.stdout)
    return True

def extract_score(row: Dict[str, str]) -> Optional[float]:
    for key in ("publicScore", "PublicScore"):
        score_str = row.get(key)
        if score_str and score_str not in {"", "None", "null"}:
            try:
                return float(score_str)
            except ValueError:
                return None
    return None


def wait_for_kaggle_score(
    description: str,
    timeout_seconds: int,
    poll_interval: int,
) -> Optional[float]:
    if timeout_seconds <= 0:
        return None
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        submissions = list_kaggle_submissions(KAGGLE_COMPETITION)
        if submissions is None:
            print("[warn] Skipping score polling due to Kaggle CLI/network issue.")
            return None
        for row in submissions:
            desc = row.get("description") or row.get("Description")
            if desc == description:
                score = extract_score(row)
                if score is not None:
                    return score
        print("Waiting for Kaggle score...")
        time.sleep(poll_interval)
    print("[warn] Timed out waiting for Kaggle to score the submission.")
    return None


def load_genomes_from_summary(path: Path, limit: Optional[int]) -> List[Genome]:
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    genomes: List[Genome] = []
    for entry in payload[: limit]:
        gdict = entry.get("genome", {})
        fitness_value: float
        if "val_log_loss" in entry:
            fitness_value = -float(entry["val_log_loss"])
        elif "val_accuracy" in entry:
            fitness_value = float(entry["val_accuracy"])
        elif "fitness" in entry:
            fitness_value = float(entry["fitness"])
        elif "fitness" in gdict:
            fitness_value = float(gdict["fitness"])
        else:
            fitness_value = -math.inf
        genome = Genome(
            model_type=gdict.get("model_type", entry.get("model_type", "mlp")),
            num_layers=gdict.get("num_layers", len(gdict.get("layer_widths", [])) or 1),
            layer_widths=list(gdict.get("layer_widths", [])),
            activation=gdict.get("activation", "relu"),
            dropout=float(gdict.get("dropout", 0.0)),
            optimizer=gdict.get("optimizer", "adam"),
            learning_rate=float(gdict.get("learning_rate", 1e-3)),
            weight_decay=float(gdict.get("weight_decay", 0.0)),
            batch_size=int(gdict.get("batch_size", 128)),
            epochs=int(gdict.get("epochs", 20)),
            gb_estimators=int(gdict.get("gb_estimators", entry.get("gb_estimators", 150))),
            gb_learning_rate=float(gdict.get("gb_learning_rate", entry.get("gb_learning_rate", 0.2))),
            gb_min_leaf=int(gdict.get("gb_min_leaf", entry.get("gb_min_leaf", 100))),
            rf_trees=int(gdict.get("rf_trees", entry.get("rf_trees", 200))),
            rf_max_depth=int(gdict.get("rf_max_depth", entry.get("rf_max_depth", 6))),
            rf_max_features=float(gdict.get("rf_max_features", entry.get("rf_max_features", 0.7))),
            rf_min_samples_split=int(gdict.get("rf_min_samples_split", entry.get("rf_min_samples_split", 20))),
            rf_sample_ratio=float(gdict.get("rf_sample_ratio", entry.get("rf_sample_ratio", 0.8))),
            rf_random_thresholds=bool(gdict.get("rf_random_thresholds", entry.get("rf_random_thresholds", False))),
            fitness=fitness_value,
        )
        if "val_log_loss" in entry:
            genome.fitness = -float(entry["val_log_loss"])
        elif "val_accuracy" in entry and genome.fitness is None:
            genome.fitness = float(entry["val_accuracy"])
        if genome.fitness is None:
            genome.fitness = -math.inf
        genomes.append(genome)
    return genomes


# --- Main entry point ------------------------------------------------------ #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GA-tuned neural nets on prepared Titanic data.")
    parser.add_argument("--train-prepared", default="data/train_prepared.csv")
    parser.add_argument("--test-prepared", default="data/test_prepared.csv")
    parser.add_argument("--raw-test", default="data/test.csv")
    parser.add_argument("--target-column", default="Transported")
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--macro-mutation-rate", type=float, default=0.3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=1,
        help="Number of stratified CV folds to average for GA fitness (1 = single holdout).",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Number of top genomes to retrain and submit.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--no-kaggle", action="store_true", help="Skip Kaggle submissions even if CLI exists.")
    parser.add_argument(
        "--resume-summary",
        type=str,
        help="Optional path to ga_search_summary.json from a previous run to seed the population.",
    )
    parser.add_argument(
        "--resume-top-n",
        type=int,
        default=None,
        help="When resuming, number of genomes to load from the summary (default: all available).",
    )
    parser.add_argument(
        "--kaggle-score-timeout",
        type=int,
        default=180,
        help="Seconds to wait for Kaggle to provide a public score (0 to disable).",
    )
    parser.add_argument(
        "--kaggle-score-interval",
        type=int,
        default=15,
        help="Seconds between leaderboard polls while waiting for a score.",
    )
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_df = pd.read_csv(args.train_prepared)
    if args.target_column not in train_df.columns:
        raise ValueError(f"Target column '{args.target_column}' missing from {args.train_prepared}")

    y = train_df[args.target_column].astype(int).to_numpy()
    feature_df = train_df.drop(columns=[args.target_column])
    feature_cols = feature_df.columns.tolist()
    X = feature_df.to_numpy(dtype=np.float32)

    test_df = pd.read_csv(args.test_prepared)
    test_features = align_test_features(test_df, feature_cols).to_numpy(dtype=np.float32)
    raw_test = pd.read_csv(args.raw_test)
    passenger_ids = raw_test["PassengerId"].tolist()
    if len(passenger_ids) != len(test_features):
        raise ValueError("Mismatch between raw test rows and prepared test features")

    cv_splits = build_cv_splits(X, y, folds=args.cv_folds, val_ratio=args.val_ratio, seed=args.seed)
    if args.cv_folds > 1:
        print(f"Using stratified {args.cv_folds}-fold CV for GA fitness.")
    else:
        val_count = cv_splits[0][2].shape[0]
        print(f"Using single holdout with {val_count} validation rows (val_ratio={args.val_ratio:.2f}).")
    device = resolve_device()
    print(f"Using device: {device}")

    population: List[Genome] = []
    if args.resume_summary:
        summary_path = Path(args.resume_summary)
        if not summary_path.exists():
            raise FileNotFoundError(f"Resume summary not found: {summary_path}")
        limit = args.resume_top_n
        resumed = load_genomes_from_summary(summary_path, limit)
        if not resumed:
            print(f"[warn] No genomes recovered from {summary_path}; starting fresh.")
        else:
            print(f"Loaded {len(resumed)} genomes from {summary_path} to seed the population.")
            population.extend(resumed[: args.population])
    while len(population) < args.population:
        population.append(random_genome())
    evolved = evolve(
        population,
        cv_splits=cv_splits,
        device=device,
        generations=args.generations,
        macro_rate=args.macro_mutation_rate,
        elite_count=args.elite_count,
    )
    evolved.sort(key=lambda g: g.fitness, reverse=True)

    ensure_dirs(Path("models"), Path("submissions"))
    top_genomes = evolved[: args.top_k]
    metadata: List[Dict[str, object]] = []
    for idx, genome in enumerate(top_genomes, 1):
        print(f"Retraining top genome #{idx} ({genome.model_type}) with val_log_loss={-genome.fitness:.4f}")
        if genome.model_type == "mlp":
            model = retrain_full_mlp(genome, X, y, device)
            probs = predict_probabilities(model, test_features, device)
            artifact = {"state_dict": model.state_dict(), "genome": genome.__dict__}
        elif genome.model_type == "gbstump":
            model = GradientBoostingStumps(
                n_estimators=genome.gb_estimators,
                learning_rate=genome.gb_learning_rate,
                min_leaf=genome.gb_min_leaf,
                seed=args.seed + idx,
            )
            model.fit(X, y)
            probs = model.predict_proba(test_features)
            artifact = {"model": model, "genome": genome.__dict__}
        else:
            model = RandomForestEnsemble(
                n_estimators=genome.rf_trees,
                max_depth=genome.rf_max_depth,
                min_samples_split=genome.rf_min_samples_split,
                max_features_fraction=genome.rf_max_features,
                sample_ratio=genome.rf_sample_ratio,
                random_thresholds=genome.model_type == "extratrees",
                seed=args.seed + idx,
            )
            model.fit(X, y)
            probs = model.predict_proba(test_features)
            artifact = {"model": model, "genome": genome.__dict__}
        timestamp = int(time.time())
        model_path = Path("models") / f"ga_nn_{idx}_{timestamp}.pt"
        torch.save(artifact, model_path)
        submission_path = Path("submissions") / f"ga_nn_{idx}_{timestamp}.csv"
        save_submission(passenger_ids, probs, threshold=0.5, path=submission_path)
        print(f"Saved model to {model_path} and submission to {submission_path}")
        description = f"GA rank {idx} ts {timestamp} logloss {-genome.fitness:.4f}"
        score: Optional[float] = None
        submitted = False
        if not args.no_kaggle:
            submitted = try_kaggle_submit(submission_path, message=description)
            if submitted:
                score = wait_for_kaggle_score(
                    description,
                    timeout_seconds=args.kaggle_score_timeout,
                    poll_interval=args.kaggle_score_interval,
                )
        metadata.append(
            {
                "rank": idx,
                "val_log_loss": -genome.fitness,
                "genome": genome.__dict__,
                "model_path": str(model_path),
                "submission_path": str(submission_path),
                "kaggle_description": description,
                "kaggle_submitted": submitted,
                "kaggle_score": score,
            }
        )

    summary_path = Path("models") / "ga_search_summary.json"
    backup_path = None
    if summary_path.exists():
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = summary_path.with_name(f"ga_search_summary_{timestamp}.json")
        summary_path.replace(backup_path)
        print(f"Backed up previous summary to {backup_path}")
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
    print(f"Wrote GA summary to {summary_path}")


if __name__ == "__main__":
    main()
