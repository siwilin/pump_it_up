"""Validate generated submission files against the competition schema."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_pump_it_up_data
from src.submissions import submission_summary, validate_submission


def main() -> None:
    """Validate every generated submission under ``reports/submissions/``."""
    _, _, test_values = load_pump_it_up_data(PROJECT_ROOT / "data")
    expected_ids = test_values["id"]
    files = sorted(
        (PROJECT_ROOT / "reports" / "submissions").glob("submission_*.csv")
    )
    if not files:
        raise FileNotFoundError(
            "No submission_*.csv files found under reports/submissions/"
        )

    for file_path in files:
        submission = pd.read_csv(file_path)
        validate_submission(submission, expected_ids=expected_ids)
        summary = submission_summary(submission)
        distribution = ", ".join(
            f"{row.status_group}={row.proportion:.1%}"
            for row in summary.itertuples()
        )
        print(f"VALID: {file_path.name} ({len(submission)} rows; {distribution})")


if __name__ == "__main__":
    main()
