"""Model construction and evaluation utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessing_pipeline


def build_random_forest_pipeline(
    features: pd.DataFrame,
    *,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    class_weight: str | dict[str, float] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build a leakage-safe preprocessing plus Random Forest pipeline."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    return build_classifier_pipeline(features, model)


def build_classifier_pipeline(
    features: pd.DataFrame,
    estimator: ClassifierMixin,
    *,
    scale_features: bool = False,
) -> Pipeline:
    """Attach the shared leakage-safe preprocessor to any classifier."""
    steps: list[tuple[str, object]] = [
        ("preprocessor", build_preprocessing_pipeline(features))
    ]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(
        steps=steps
    )


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Adapt classifiers that require consecutive integer target labels."""

    def __init__(self, estimator: ClassifierMixin) -> None:
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> "LabelEncodedClassifier":
        self.label_encoder_ = LabelEncoder()
        encoded_target = self.label_encoder_.fit_transform(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, encoded_target)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X: Any) -> Any:
        encoded_predictions = self.estimator_.predict(X)
        return self.label_encoder_.inverse_transform(encoded_predictions.astype(int))


def portfolio_model_specs(random_state: int = 42) -> dict[str, ClassifierMixin]:
    """Return diverse classifiers for the portfolio model benchmark."""
    return {
        "logistic_regression_balanced": LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=random_state
        ),
        "decision_tree_balanced": DecisionTreeClassifier(
            class_weight="balanced", max_depth=18, random_state=random_state
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=50, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "extra_trees_balanced": ExtraTreesClassifier(
            n_estimators=50, class_weight="balanced", random_state=random_state, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=random_state
        ),
        "xgboost": LabelEncodedClassifier(
            XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "lightgbm_balanced": LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        ),
        "catboost_balanced": CatBoostClassifier(
            iterations=150,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            loss_function="MultiClass",
            random_seed=random_state,
            verbose=False,
            thread_count=-1,
        ),
    }


def stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target while preserving class proportions."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def classification_metrics(
    target: pd.Series,
    predictions: pd.Series,
) -> dict[str, Any]:
    """Return portfolio-friendly summary metrics and the full class report."""
    return {
        "accuracy": accuracy_score(target, predictions),
        "macro_f1": f1_score(target, predictions, average="macro"),
        "classification_report": classification_report(
            target, predictions, output_dict=True
        ),
    }
