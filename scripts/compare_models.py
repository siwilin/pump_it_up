"""Compare the portfolio classifier families on a fixed validation split."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_pump_it_up_data
from src.modeling import (
    build_classifier_pipeline,
    classification_metrics,
    portfolio_model_specs,
    stratified_split,
)


def main() -> None:
    train_values, train_labels, _ = load_pump_it_up_data(PROJECT_ROOT / "data")
    train = train_values.merge(train_labels, on="id", validate="one_to_one")

    features = train.drop(columns=["status_group"])
    target = train["status_group"]
    X_train, X_validation, y_train, y_validation = stratified_split(
        features, target
    )

    rows: list[dict[str, float | str]] = []
    for name, estimator in portfolio_model_specs().items():
        pipeline = build_classifier_pipeline(
            X_train,
            estimator,
            scale_features=name.startswith("logistic"),
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_validation)
        metrics = classification_metrics(y_validation, predictions)
        repair_recall = recall_score(
            y_validation,
            predictions,
            labels=["functional needs repair"],
            average=None,
        )[0]
        rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "repair_recall": repair_recall,
            }
        )

    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
