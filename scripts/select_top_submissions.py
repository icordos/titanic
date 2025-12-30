#!/usr/bin/env python3
"""Aggregate GA summary files and export the top Kaggle-scoring entries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan ga_search_summary*.json files and export the top submissions by Kaggle score."
    )
    parser.add_argument("--summaries-dir", default="models", help="Directory containing ga_search_summary*.json files.")
    parser.add_argument("--pattern", default="ga_search_summary*.json", help="Glob pattern for summary files.")
    parser.add_argument("--top", type=int, default=5, help="Number of top entries to keep.")
    parser.add_argument("--output", required=True, help="Path to JSON file where the results are saved.")
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include entries with missing Kaggle scores (sorted after scored entries).",
    )
    return parser.parse_args()


def load_entries(path: Path) -> List[dict]:
    try:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if isinstance(payload, list):
            return payload
        return []
    except json.JSONDecodeError:
        print(f"[warn] Skipping malformed summary file: {path}")
        return []


def entry_score(entry: dict) -> Optional[float]:
    score = entry.get("kaggle_score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    base_dir = Path(args.summaries_dir)
    summary_paths = sorted(base_dir.glob(args.pattern))
    all_entries: List[dict] = []
    for path in summary_paths:
        entries = load_entries(path)
        for entry in entries:
            meta = entry.copy()
            meta["source_file"] = str(path)
            all_entries.append(meta)

    scored = [entry for entry in all_entries if entry_score(entry) is not None]
    unscored = [entry for entry in all_entries if entry_score(entry) is None]
    scored.sort(key=lambda e: entry_score(e), reverse=True)
    results: List[dict] = scored[: args.top]
    if args.include_missing and len(results) < args.top:
        remaining = args.top - len(results)
        results.extend(unscored[:remaining])

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"Wrote top {len(results)} entries to {output_path}")


if __name__ == "__main__":
    main()
