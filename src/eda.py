"""Reusable exploratory summaries for the Pump It Up datasets."""

from __future__ import annotations

import pandas as pd


def missingness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages, ordered by missing count."""
    result = pd.DataFrame(
        {
            "missing_count": frame.isna().sum(),
            "missing_pct": frame.isna().mean().mul(100),
        }
    )
    return result.loc[result["missing_count"].gt(0)].sort_values(
        "missing_count", ascending=False
    )


def target_distribution(frame: pd.DataFrame, target: str = "status_group") -> pd.DataFrame:
    """Return class counts and percentages for the target column."""
    if target not in frame.columns:
        raise KeyError(f"Target column not found: {target}")

    counts = frame[target].value_counts(dropna=False).rename("count")
    result = counts.to_frame()
    result["percentage"] = result["count"].div(result["count"].sum()).mul(100)
    return result


def column_type_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise numeric and categorical columns without mutating the frame."""
    numeric = frame.select_dtypes(include=["number"]).columns
    categorical = frame.select_dtypes(include=["object", "string", "category"]).columns
    return pd.DataFrame(
        {
            "column_type": ["numeric"] * len(numeric) + ["categorical"] * len(categorical),
        },
        index=list(numeric) + list(categorical),
    )
