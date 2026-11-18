import numpy as np
import pandas as pd

from src.preprocessing import build_preprocessing_pipeline, engineer_features


def test_engineer_features_drops_identifiers_and_builds_date_features() -> None:
    frame = pd.DataFrame(
        {
            "id": [1],
            "recorded_by": ["x"],
            "scheme_name": [None],
            "wpt_name": ["Pump A"],
            "date_recorded": ["2013-05-10"],
            "construction_year": [2000],
            "population": [0],
        }
    )

    result = engineer_features(frame)

    assert "id" not in result.columns
    assert "date_recorded" not in result.columns
    assert result.loc[0, "year_recorded"] == 2013
    assert result.loc[0, "month_recorded"] == 5
    assert result.loc[0, "pump_age"] == 13
    assert np.isnan(result.loc[0, "population"])


def test_preprocessor_handles_missing_and_unseen_categories() -> None:
    train = pd.DataFrame(
        {
            "population": [10.0, np.nan],
            "region": ["north", True],
        }
    )
    validation = pd.DataFrame(
        {
            "population": [np.nan],
            "region": ["new-region"],
        }
    )

    preprocessor = build_preprocessing_pipeline(train)
    transformed_train = preprocessor.fit_transform(train)
    transformed_validation = preprocessor.transform(validation)

    assert transformed_train.shape == (2, 2)
    assert transformed_validation.shape == (1, 2)
    assert not np.isnan(transformed_validation).any()
    assert transformed_validation[0, 1] == -1
