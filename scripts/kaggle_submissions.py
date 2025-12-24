#!/usr/bin/env python3
"""Utilities for inspecting recent Kaggle submissions.

Uses the Kaggle Python API directly so we can avoid CLI regressions (e.g.,
page_number keyword errors) and reuse the logic from other scripts.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

from kaggle.api.kaggle_api_extended import KaggleApi


def _submission_to_dict(submission) -> Dict[str, str]:
    status = getattr(submission.status, "name", "") if submission.status else ""
    date_val = submission.date.isoformat() if submission.date else ""
    return {
        "ref": str(submission.ref),
        "description": submission.description,
        "publicScore": submission.public_score,
        "privateScore": submission.private_score,
        "date": date_val,
        "status": status,
        "fileName": submission.file_name,
        "teamName": submission.team_name,
        "submittedBy": submission.submitted_by,
    }


def list_kaggle_submissions(competition: str, page_size: int = 50) -> Optional[List[Dict[str, str]]]:
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - depends on local config
        print(f"[warn] Kaggle API authentication failed: {exc}")
        return None
    try:
        submissions = api.competition_submissions(competition, page_size=page_size)
    except Exception as exc:  # pragma: no cover - network errors
        print(f"[warn] Unable to fetch Kaggle submissions list: {exc}")
        return None
    return [_submission_to_dict(sub) for sub in submissions]


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
