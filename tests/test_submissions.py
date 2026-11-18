"""Tests for submission validation helpers."""

import pandas as pd
import pytest

from src.submissions import submission_summary, validate_submission


def test_validate_submission_accepts_expected_schema() -> None:
    submission = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "status_group": [
                "functional",
                "functional needs repair",
                "non functional",
            ],
        }
    )

    validate_submission(submission, expected_ids=[1, 2, 3])


def test_validate_submission_rejects_unknown_class() -> None:
    submission = pd.DataFrame({"id": [1], "status_group": ["unknown"]})

    with pytest.raises(ValueError, match="Unknown status_group"):
        validate_submission(submission)


def test_submission_summary_returns_proportions() -> None:
    submission = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "status_group": ["functional", "functional", "non functional"],
        }
    )

    summary = submission_summary(submission).set_index("status_group")

    assert summary.loc["functional", "count"] == 2
    assert summary.loc["functional", "proportion"] == pytest.approx(2 / 3)
