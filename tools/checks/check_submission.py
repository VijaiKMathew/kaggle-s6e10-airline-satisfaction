#!/usr/bin/env python3
"""Submission Verification Gate for Airline Satisfaction (Playground S6E10)."""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    submission_path = ROOT / "submissions" / "submission.csv"
    if not submission_path.exists():
        submission_path = ROOT / "submission.csv"

    if not submission_path.exists():
        print(f"[SKIP] No submission found at {submission_path} yet.")
        return 0

    df = pd.read_csv(submission_path)
    print(f"Checking {submission_path} (rows: {len(df)})...")

    expected_cols = ["id", "satisfaction"]
    if list(df.columns) != expected_cols:
        print(f"[FAIL] Column mismatch: Expected {expected_cols}, got {list(df.columns)}")
        return 1

    if df.isnull().any().any():
        print("[FAIL] Null values found in submission.")
        return 1

    if (df["satisfaction"] < 0.0).any() or (df["satisfaction"] > 1.0).any():
        print("[FAIL] Predictions out of probability bounds [0, 1].")
        return 1

    print("[PASS] Submission is valid and ready for scoring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
