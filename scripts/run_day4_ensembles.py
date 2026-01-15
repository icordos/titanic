#!/usr/bin/env python3
"""Day 4 workflow: cross-run ensembles and correlation diagnostics."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class SubmissionGroup:
    name: str
    files: List[str]
    passenger_ids: pd.Series
    probs: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Day-4 ensembles from historical submissions.")
    parser.add_argument(
        "--day2",
        nargs="+",
        required=True,
        help="Submission CSVs from Day 2 (new GA models) to blend.",
    )
    parser.add_argument(
        "--historical",
        nargs="*",
        default=[],
        help="High-scoring historical submission CSVs to blend.",
    )
    parser.add_argument(
        "--stacker",
        required=True,
        help="Stacked/meta submission CSV (e.g., Day 1 stacker).",
    )
    parser.add_argument(
        "--output-dir",
        default="models/day4_ensembles",
        help="Directory where ensemble CSVs and summary JSON will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for Transported label.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        metavar=("DAY2", "HIST", "STACK"),
        default=(0.6, 0.2, 0.2),
        help="Weights used for the three-way hybrid blend (unused components are ignored).",
    )
    parser.add_argument(
        "--keep-prob",
        action="store_true",
        help="Include the blended probability column (TransportedProb) in output CSVs.",
    )
    return parser.parse_args()


def load_submission_probs(path: Path) -> tuple[pd.Series, np.ndarray]:
    df = pd.read_csv(path)
    if "PassengerId" not in df.columns or "Transported" not in df.columns:
        raise ValueError(f"{path} missing PassengerId/Transported columns.")
    df = df.sort_values("PassengerId").reset_index(drop=True)
    if "TransportedProb" in df.columns:
        probs = df["TransportedProb"].astype(float).to_numpy()
    else:
        transported = df["Transported"]
        if transported.dtype == bool:
            probs = transported.astype(float).to_numpy()
        else:
            try:
                probs = transported.astype(float).to_numpy()
            except ValueError:
                mapping = {"True": 1.0, "False": 0.0}
                probs = transported.map(mapping).astype(float).to_numpy()
    passenger_ids = df["PassengerId"]
    return passenger_ids, probs


def aggregate_group(name: str, file_paths: Sequence[str]) -> Optional[SubmissionGroup]:
    if not file_paths:
        return None
    ids_ref: Optional[pd.Series] = None
    probs_stack: List[np.ndarray] = []
    for file_path in file_paths:
        ids, probs = load_submission_probs(Path(file_path))
        if ids_ref is None:
            ids_ref = ids
        elif not ids_ref.equals(ids):
            raise ValueError(f"PassengerId mismatch detected when processing {file_path}")
        probs_stack.append(probs)
    assert ids_ref is not None
    mean_probs = np.mean(np.vstack(probs_stack), axis=0)
    return SubmissionGroup(name=name, files=[str(p) for p in file_paths], passenger_ids=ids_ref, probs=mean_probs)


def ensure_ids_match(groups: Sequence[SubmissionGroup]) -> None:
    ids_ref: Optional[pd.Series] = None
    for group in groups:
        if group is None:
            continue
        if ids_ref is None:
            ids_ref = group.passenger_ids
            continue
        if not ids_ref.equals(group.passenger_ids):
            raise ValueError(f"PassengerId mismatch detected between groups (e.g., {group.name})")


def pearson_corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a is None or b is None:
        return None
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def write_submission(
    passenger_ids: pd.Series,
    probs: np.ndarray,
    path: Path,
    threshold: float,
    keep_prob: bool,
) -> None:
    clipped = np.clip(probs, 0.0, 1.0)
    df = pd.DataFrame({"PassengerId": passenger_ids, "Transported": clipped >= threshold})
    if keep_prob:
        df["TransportedProb"] = clipped
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[day4] wrote ensemble to {path}")


def build_weighted_blend(
    components: List[tuple[str, np.ndarray]],
    weights: Sequence[float],
) -> tuple[np.ndarray, List[str]]:
    active_probs: List[np.ndarray] = []
    active_names: List[str] = []
    active_weights: List[float] = []
    for (name, probs), weight in zip(components, weights):
        if probs is None:
            continue
        active_probs.append(probs)
        active_names.append(name)
        active_weights.append(weight)
    if not active_probs:
        raise ValueError("No components available for weighted blend.")
    weights_np = np.array(active_weights, dtype=np.float64)
    weights_np = np.clip(weights_np, 0.0, None)
    if weights_np.sum() == 0:
        weights_np[:] = 1.0
    weights_np /= weights_np.sum()
    stacked = np.vstack(active_probs)
    blend = np.average(stacked, axis=0, weights=weights_np)
    return blend, active_names


def choose_low_corr_pair(groups: Sequence[SubmissionGroup]) -> Optional[tuple[str, str, np.ndarray, np.ndarray]]:
    available = [g for g in groups if g is not None]
    if len(available) < 2:
        return None
    best_pair: Optional[tuple[str, str, np.ndarray, np.ndarray]] = None
    best_corr = float("inf")
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            gi = available[i]
            gj = available[j]
            corr = pearson_corr(gi.probs, gj.probs)
            if corr is None:
                continue
            if corr < best_corr:
                best_corr = corr
                best_pair = (gi.name, gj.name, gi.probs, gj.probs)
    return best_pair


def main() -> None:
    args = parse_args()
    day2_group = aggregate_group("day2", args.day2)
    hist_group = aggregate_group("historical", args.historical)
    stacker_group = aggregate_group("stacker", [args.stacker])
    ensure_ids_match([g for g in (day2_group, hist_group, stacker_group) if g is not None])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "groups": [
            {
                "name": g.name,
                "num_files": len(g.files),
                "files": g.files,
            }
            for g in (day2_group, hist_group, stacker_group)
            if g is not None
        ],
        "ensembles": [],
        "correlations": [],
    }

    # Pairwise correlations for diagnostics
    pairs = [
        ("day2", day2_group, "historical", hist_group),
        ("day2", day2_group, "stacker", stacker_group),
        ("historical", hist_group, "stacker", stacker_group),
    ]
    for name_a, group_a, name_b, group_b in pairs:
        if group_a is None or group_b is None:
            continue
        corr = pearson_corr(group_a.probs, group_b.probs)
        summary["correlations"].append(
            {"pair": f"{name_a}__{name_b}", "pearson": corr}
        )

    passenger_ids = None
    for group in (day2_group, hist_group, stacker_group):
        if group is not None:
            passenger_ids = group.passenger_ids
            break
    if passenger_ids is None:
        raise ValueError("No submissions provided.")

    # Day2 + stacker blend
    if day2_group is not None:
        blend_probs = (day2_group.probs + stacker_group.probs) / 2.0
        path = output_dir / "ensemble_day2_stacker.csv"
        write_submission(passenger_ids, blend_probs, path, args.threshold, args.keep_prob)
        summary["ensembles"].append(
            {
                "name": "ensemble_day2_stacker",
                "path": str(path),
                "components": ["day2", "stacker"],
            }
        )

    # Historical + stacker blend
    if hist_group is not None:
        blend_probs = (hist_group.probs + stacker_group.probs) / 2.0
        path = output_dir / "ensemble_historical_stacker.csv"
        write_submission(passenger_ids, blend_probs, path, args.threshold, args.keep_prob)
        summary["ensembles"].append(
            {
                "name": "ensemble_historical_stacker",
                "path": str(path),
                "components": ["historical", "stacker"],
            }
        )

    # Weighted hybrid
    components = [
        ("day2", day2_group.probs if day2_group is not None else None),
        ("historical", hist_group.probs if hist_group is not None else None),
        ("stacker", stacker_group.probs if stacker_group is not None else None),
    ]
    blend, active_names = build_weighted_blend(components, args.weights)
    path = output_dir / "ensemble_weighted_hybrid.csv"
    write_submission(passenger_ids, blend, path, args.threshold, args.keep_prob)
    summary["ensembles"].append(
        {
            "name": "ensemble_weighted_hybrid",
            "path": str(path),
            "components": active_names,
            "weights": args.weights,
        }
    )

    # Diversity-focused pair
    pair = choose_low_corr_pair([day2_group, hist_group, stacker_group])
    if pair is not None:
        name_a, name_b, probs_a, probs_b = pair
        blend_probs = (probs_a + probs_b) / 2.0
        path = output_dir / f"ensemble_lowcorr_{name_a}_{name_b}.csv"
        write_submission(passenger_ids, blend_probs, path, args.threshold, args.keep_prob)
        summary["ensembles"].append(
            {
                "name": f"ensemble_lowcorr_{name_a}_{name_b}",
                "path": str(path),
                "components": [name_a, name_b],
                "note": "Average of the least-correlated pair.",
            }
        )

    summary_path = output_dir / "day4_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[day4] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
