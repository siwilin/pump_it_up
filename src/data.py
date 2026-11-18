"""Data loading and validation utilities for the Pump It Up project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_TRAIN_VALUE_COLUMNS = {"id", "date_recorded"}
REQUIRED_LABEL_COLUMNS = {"id", "status_group"}
REQUIRED_TEST_VALUE_COLUMNS = {"id", "date_recorded"}


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    """Read a CSV file and fail with a useful message when its schema is wrong."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {path.name}: {missing_names}")

    return frame


def load_pump_it_up_data(data_dir: str | Path = "data") -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Load training values, training labels and test values.

    Expected files are the original Pump It Up competition CSVs inside
    ``data_dir``. The function validates the minimum schema before returning
    the three dataframes.
    """
    root = Path(data_dir)
    train_values = _read_csv(
        root / "training_set_values.csv", REQUIRED_TRAIN_VALUE_COLUMNS
    )
    train_labels = _read_csv(
        root / "training_set_labels.csv", REQUIRED_LABEL_COLUMNS
    )
    test_values = _read_csv(root / "test_set_values.csv", REQUIRED_TEST_VALUE_COLUMNS)

    if not train_values["id"].is_unique:
        raise ValueError("training_set_values.csv contains duplicated ids")
    if not train_labels["id"].is_unique:
        raise ValueError("training_set_labels.csv contains duplicated ids")
    if not test_values["id"].is_unique:
        raise ValueError("test_set_values.csv contains duplicated ids")

    return train_values, train_labels, test_values
