"""Leakage-safe feature engineering and preprocessing for Pump It Up."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


DROP_COLUMNS = ("id", "recorded_by", "scheme_name", "wpt_name")


def _coerce_categories_to_string(values: pd.DataFrame) -> pd.DataFrame:
    """Make mixed boolean/string categorical columns encoder-compatible."""
    string_values = values.astype("string")
    return string_values.astype(object).where(string_values.notna(), np.nan)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready raw features without fitting on the data.

    The function is deliberately deterministic: learned statistics such as
    medians and category mappings are handled later by the sklearn pipeline.
    """
    result = frame.copy()
    columns_to_drop = [column for column in DROP_COLUMNS if column in result.columns]
    result = result.drop(columns=columns_to_drop)

    if "date_recorded" in result.columns:
        recorded = pd.to_datetime(result.pop("date_recorded"), errors="coerce")
        result["year_recorded"] = recorded.dt.year
        result["month_recorded"] = recorded.dt.month

    if "construction_year" in result.columns:
        construction_year = result["construction_year"].replace(0, np.nan)
        result["construction_year"] = construction_year
        if "year_recorded" in result.columns:
            result["pump_age"] = result["year_recorded"] - construction_year

    if "population" in result.columns:
        result["population"] = result["population"].replace(0, np.nan)

    return result


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    """Build a transformer whose imputers and encoders learn only on fit."""
    numeric_columns = frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = frame.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("to_string", FunctionTransformer(_coerce_categories_to_string)),
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessing_pipeline(
    frame: pd.DataFrame,
    *,
    sparse_output: bool = False,
) -> ColumnTransformer:
    """Return a fitted-ready preprocessing object for a feature frame.

    ``sparse_output`` is retained as an explicit API option for future models;
    the current ordinal representation is dense by design.
    """
    del sparse_output
    return build_preprocessor(frame)
