from pathlib import Path

import pandas as pd
import pytest

from src.data import load_pump_it_up_data


def test_loader_reports_missing_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="training_set_values.csv"):
        load_pump_it_up_data(tmp_path)


def test_loader_reads_and_validates_minimum_schema(tmp_path: Path) -> None:
    pd.DataFrame({"id": [1], "date_recorded": ["2013-01-01"]}).to_csv(
        tmp_path / "training_set_values.csv", index=False
    )
    pd.DataFrame({"id": [1], "status_group": ["functional"]}).to_csv(
        tmp_path / "training_set_labels.csv", index=False
    )
    pd.DataFrame({"id": [2], "date_recorded": ["2013-01-02"]}).to_csv(
        tmp_path / "test_set_values.csv", index=False
    )

    train_values, train_labels, test_values = load_pump_it_up_data(tmp_path)

    assert list(train_values["id"]) == [1]
    assert list(train_labels["status_group"]) == ["functional"]
    assert list(test_values["id"]) == [2]
