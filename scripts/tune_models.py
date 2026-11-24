"""Evaluate focused hyperparameter configurations on one fixed split."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import recall_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_pump_it_up_data
from src.modeling import (
    build_classifier_pipeline,
    classification_metrics,
    LabelEncodedClassifier,
    stratified_split,
)


def model_configs() -> dict[str, object]:
    """Return focused configurations for the strongest model families."""
    return {
        "rf_balanced_100": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "rf_balanced_depth20": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "rf_balanced_leaf2": RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "rf_unbalanced_100": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "rf_unbalanced_300": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "rf_unbalanced_leaf2": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "rf_unbalanced_leaf2_400": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "rf_unbalanced_leaf2_600": RandomForestClassifier(
            n_estimators=600,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "rf_unbalanced_leaf2_800": RandomForestClassifier(
            n_estimators=800,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "rf_unbalanced_leaf2_features70": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees_balanced_100": ExtraTreesClassifier(
            n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "extra_trees_balanced_leaf2": ExtraTreesClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees_unbalanced_100": ExtraTreesClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "extra_trees_unbalanced_features70": ExtraTreesClassifier(
            n_estimators=300,
            max_features=0.7,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees_unbalanced_leaf2": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees_unbalanced_features70_leaf2": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.7,
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees_balanced_features70": ExtraTreesClassifier(
            n_estimators=100,
            max_features=0.7,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting_lr005": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        ),
        "hist_gradient_boosting_regularized": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        "xgb_depth4": LabelEncodedClassifier(
            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )
        ),
        "xgb_depth8": LabelEncodedClassifier(
            XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            )
        ),
        "lgbm_balanced_leaves31": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=30,
            class_weight="balanced",
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
        "lgbm_balanced_leaves63": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=50,
            class_weight="balanced",
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
        "lgbm_unbalanced_leaves31": LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=31,
            min_child_samples=30,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        ),
        "catboost_balanced_depth6": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.04,
            l2_leaf_reg=5.0,
            auto_class_weights="Balanced",
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
            thread_count=-1,
        ),
        "catboost_balanced_depth8": CatBoostClassifier(
            iterations=250,
            depth=8,
            learning_rate=0.04,
            l2_leaf_reg=8.0,
            auto_class_weights="Balanced",
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
            thread_count=-1,
        ),
        "catboost_unbalanced_depth6": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.04,
            l2_leaf_reg=5.0,
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
            thread_count=-1,
        ),
    }


def main() -> None:
    train_values, train_labels, _ = load_pump_it_up_data(PROJECT_ROOT / "data")
    train = train_values.merge(train_labels, on="id", validate="one_to_one")
    features = train.drop(columns=["status_group"])
    target = train["status_group"]
    X_train, X_validation, y_train, y_validation = stratified_split(features, target)

    rows: list[dict[str, float | str]] = []
    for name, estimator in model_configs().items():
        pipeline = build_classifier_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_validation)
        metrics = classification_metrics(y_validation, predictions)
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "repair_recall": recall_score(
                    y_validation,
                    predictions,
                    labels=["functional needs repair"],
                    average=None,
                )[0],
            }
        )

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
