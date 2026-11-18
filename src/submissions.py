"""Validation helpers for Pump It Up competition submissions."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


EXPECTED_COLUMNS = ("id", "status_group")
VALID_CLASSES = frozenset(
    {"functional", "functional needs repair", "non functional"}
)


def validate_submission(
    submission: pd.DataFrame,
    *,
    expected_ids: Iterable[int] | None = None,
) -> None:
    """Raise ``ValueError`` when a submission violates the competition schema."""
    if tuple(submission.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected columns {EXPECTED_COLUMNS}, got {tuple(submission.columns)}"
        )
    if submission.empty:
        raise ValueError("Submission must contain at least one row")
    if submission["id"].isna().any() or submission["status_group"].isna().any():
        raise ValueError("Submission contains missing ids or predictions")
    if submission["id"].duplicated().any():
        raise ValueError("Submission contains duplicated ids")

    invalid_classes = set(submission["status_group"]) - VALID_CLASSES
    if invalid_classes:
        raise ValueError(f"Unknown status_group values: {sorted(invalid_classes)}")

    if expected_ids is not None:
        expected = set(expected_ids)
        actual = set(submission["id"])
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                f"Submission ids do not match test ids; missing={missing[:5]}, "
                f"unexpected={unexpected[:5]}"
            )


def submission_summary(submission: pd.DataFrame) -> pd.DataFrame:
    """Return prediction counts and proportions by status group."""
    validate_submission(submission)
    summary = submission["status_group"].value_counts().rename("count").to_frame()
    summary["proportion"] = summary["count"] / len(submission)
    return summary.reset_index(names="status_group")
