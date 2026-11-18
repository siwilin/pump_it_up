import pandas as pd
import pytest

from src.eda import column_type_summary, missingness_summary, target_distribution


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "status_group": ["functional", "non functional", "functional"],
            "population": [10, None, 30],
            "region": pd.Series(["north", "south", "north"], dtype="string"),
        }
    )


def test_missingness_summary(sample_frame: pd.DataFrame) -> None:
    result = missingness_summary(sample_frame)
    assert result.loc["population", "missing_count"] == 1
    assert result.loc["population", "missing_pct"] == pytest.approx(100 / 3 * 1)


def test_target_distribution(sample_frame: pd.DataFrame) -> None:
    result = target_distribution(sample_frame)
    assert result.loc["functional", "count"] == 2
    assert result["percentage"].sum() == pytest.approx(100)


def test_column_type_summary(sample_frame: pd.DataFrame) -> None:
    result = column_type_summary(sample_frame)
    assert result.loc["population", "column_type"] == "numeric"
    assert result.loc["region", "column_type"] == "categorical"
