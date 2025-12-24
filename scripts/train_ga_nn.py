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
    num_layers: int
    layer_widths: List[int]
    activation: str
    dropout: float
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    fitness: float = field(default=-math.inf)

    def clone(self) -> "Genome":
        return Genome(
            num_layers=self.num_layers,
            layer_widths=list(self.layer_widths),
            activation=self.activation,
            dropout=self.dropout,
            optimizer=self.optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            epochs=self.epochs,
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
) -> float:
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

    return evaluate_accuracy(model, val_loader, device)


def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = torch.sigmoid(model(xb)) >= 0.5
            correct += (preds.float() == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)


# --- Genetic Algorithm ----------------------------------------------------- #

WIDTH_CHOICES = [16, 32, 64, 128, 256, 512]
BATCH_CHOICES = [64, 128, 256, 512]
EPOCH_CHOICES = [10, 15, 20, 25, 30]
LEARNING_RATES = [5e-4, 1e-3, 2e-3, 3e-3, 5e-3]
WEIGHT_DECAY = [0.0, 1e-5, 1e-4, 5e-4]
DROPOUT = [0.0, 0.1, 0.2, 0.3, 0.4]


def random_genome() -> Genome:
    num_layers = random.randint(1, 6)
    widths = [random.choice(WIDTH_CHOICES) for _ in range(num_layers)]
    return Genome(
        num_layers=num_layers,
        layer_widths=widths,
        activation=random.choice(list(ACTIVATIONS.keys())),
        dropout=random.choice(DROPOUT),
        optimizer=random.choice(list(OPTIMIZERS.keys())),
        learning_rate=random.choice(LEARNING_RATES),
        weight_decay=random.choice(WEIGHT_DECAY),
        batch_size=random.choice(BATCH_CHOICES),
        epochs=random.choice(EPOCH_CHOICES),
    )


def mutate(genome: Genome, macro_rate: float) -> Genome:
    child = genome.clone()
    if random.random() < macro_rate:
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
    return child


def evolve(
    population: List[Genome],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    generations: int,
    macro_rate: float,
    elite_count: int,
) -> List[Genome]:
    for gen in range(generations):
        print(f"Generation {gen + 1}/{generations}")
        for genome in population:
            genome.fitness = train_candidate(genome, X_train, y_train, X_val, y_val, device)
            print(
                f"  candidate layers={genome.layer_widths[:genome.num_layers]} act={genome.activation}"
                f" lr={genome.learning_rate:.4f} batch={genome.batch_size} epochs={genome.epochs}"
                f" -> val_acc={genome.fitness:.4f}"
            )
        population.sort(key=lambda g: g.fitness, reverse=True)
        print(f"  best accuracy this gen: {population[0].fitness:.4f}")
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


def retrain_full_model(genome: Genome, X: np.ndarray, y: np.ndarray, device: torch.device) -> nn.Module:
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
        genome = Genome(
            num_layers=gdict.get("num_layers", len(gdict.get("layer_widths", [])) or 1),
            layer_widths=list(gdict.get("layer_widths", [])),
            activation=gdict.get("activation", "relu"),
            dropout=float(gdict.get("dropout", 0.0)),
            optimizer=gdict.get("optimizer", "adam"),
            learning_rate=float(gdict.get("learning_rate", 1e-3)),
            weight_decay=float(gdict.get("weight_decay", 0.0)),
            batch_size=int(gdict.get("batch_size", 128)),
            epochs=int(gdict.get("epochs", 20)),
            fitness=float(entry.get("val_accuracy", gdict.get("fitness", -math.inf))),
        )
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

    X_train, y_train, X_val, y_val = split_train_val(X, y, val_ratio=args.val_ratio, seed=args.seed)
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
        X_train,
        y_train,
        X_val,
        y_val,
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
        print(f"Retraining top genome #{idx} with val_acc={genome.fitness:.4f}")
        model = retrain_full_model(genome, X, y, device)
        probs = predict_probabilities(model, test_features, device)
        timestamp = int(time.time())
        model_path = Path("models") / f"ga_nn_{idx}_{timestamp}.pt"
        torch.save({"state_dict": model.state_dict(), "genome": genome.__dict__}, model_path)
        submission_path = Path("submissions") / f"ga_nn_{idx}_{timestamp}.csv"
        save_submission(passenger_ids, probs, threshold=0.5, path=submission_path)
        print(f"Saved model to {model_path} and submission to {submission_path}")
        description = f"GA rank {idx} ts {timestamp} acc {genome.fitness:.4f}"
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
                "val_accuracy": genome.fitness,
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
