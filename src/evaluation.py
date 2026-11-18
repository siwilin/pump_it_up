"""Evaluation helpers for multiclass maintenance models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


REPAIR_CLASS = "functional needs repair"


def business_cost(
    target: Sequence[str],
    predictions: Sequence[str],
    *,
    repair_miss_cost: float = 5.0,
    other_error_cost: float = 1.0,
) -> float:
    """Return average error cost, penalizing missed repair cases more heavily."""
    target_array = np.asarray(target)
    prediction_array = np.asarray(predictions)
    incorrect = target_array != prediction_array
    missed_repair = (target_array == REPAIR_CLASS) & incorrect
    costs = np.where(incorrect, other_error_cost, 0.0)
    costs[missed_repair] = repair_miss_cost
    return float(costs.mean())


def confusion_matrix_frame(
    matrix: np.ndarray,
    labels: Sequence[str],
) -> pd.DataFrame:
    """Convert an aggregated confusion matrix into a labelled dataframe."""
    return pd.DataFrame(matrix, index=labels, columns=labels)
