#!/usr/bin/env python3
"""Utilities for inspecting recent Kaggle submissions.

This module exports `list_kaggle_submissions` for reuse in training scripts and
can also be invoked directly to print the submission table for debugging.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from typing import Dict, List, Optional


def list_kaggle_submissions(competition: str) -> Optional[List[Dict[str, str]]]:
    kaggle_cli = shutil.which("kaggle")
    if kaggle_cli is None:
        print("[warn] Kaggle CLI not found in PATH.")
        return None
    cmd = f"kaggle competitions submissions -c {competition} --csv"
    result = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
    if result.returncode != 0:
        print("[warn] Unable to fetch Kaggle submissions list:")
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())
        return None
    if not result.stdout.strip():
        return []
    reader = csv.DictReader(result.stdout.splitlines())
    return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description="List Kaggle submissions for debugging.")
    parser.add_argument("--competition", default="spaceship-titanic")
    args = parser.parse_args()
    submissions = list_kaggle_submissions(args.competition)
    if submissions is None:
        return
    if not submissions:
        print("No submissions returned.")
        return
    print(f"Found {len(submissions)} submissions (latest first):")
    for row in submissions:
        desc = row.get("description") or row.get("Description")
        score = row.get("publicScore") or row.get("PublicScore")
        date = row.get("date") or row.get("Date")
        print(f"- {date}: score={score} desc={desc}")


if __name__ == "__main__":
    main()
