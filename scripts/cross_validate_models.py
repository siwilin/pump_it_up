"""Run five-fold cross-validation for every configured portfolio model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tune_models import model_configs
from src.data import load_pump_it_up_data
from src.evaluation import REPAIR_CLASS, business_cost
from src.modeling import build_classifier_pipeline, stratified_split


def main() -> None:
    train_values, train_labels, _ = load_pump_it_up_data(PROJECT_ROOT / "data")
    train = train_values.merge(train_labels, on="id", validate="one_to_one")
    features = train.drop(columns=["status_group"])
    target = train["status_group"]
    labels = sorted(target.unique())
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows: list[dict[str, float | int | str]] = []
    confusion_results: dict[str, list[list[int]]] = {}

    for model_name, estimator in model_configs().items():
        fold_rows: list[dict[str, float | int | str]] = []
        aggregate_matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for fold_number, (train_index, validation_index) in enumerate(
            folds.split(features, target), start=1
        ):
            X_train = features.iloc[train_index]
            X_validation = features.iloc[validation_index]
            y_train = target.iloc[train_index]
            y_validation = target.iloc[validation_index]
            pipeline = build_classifier_pipeline(X_train, estimator)
            pipeline.fit(X_train, y_train)
            predictions = np.asarray(pipeline.predict(X_validation)).reshape(-1)
            aggregate_matrix += confusion_matrix(
                y_validation, predictions, labels=labels
            )
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    "accuracy": float((predictions == y_validation).mean()),
                    "macro_f1": float(
                        f1_score(y_validation, predictions, average="macro")
                    ),
                    "repair_recall": float(
                        recall_score(
                            y_validation,
                            predictions,
                            labels=[REPAIR_CLASS],
                            average=None,
                        )[0]
                    ),
                    "business_cost": business_cost(y_validation, predictions),
                }
            )

        fold_frame = pd.DataFrame(fold_rows)
        rows.append(
            {
                "model": model_name,
                "accuracy_mean": fold_frame["accuracy"].mean(),
                "accuracy_std": fold_frame["accuracy"].std(ddof=1),
                "macro_f1_mean": fold_frame["macro_f1"].mean(),
                "macro_f1_std": fold_frame["macro_f1"].std(ddof=1),
                "repair_recall_mean": fold_frame["repair_recall"].mean(),
                "repair_recall_std": fold_frame["repair_recall"].std(ddof=1),
                "business_cost_mean": fold_frame["business_cost"].mean(),
                "business_cost_std": fold_frame["business_cost"].std(ddof=1),
            }
        )
        confusion_results[model_name] = aggregate_matrix.tolist()
        print(f"Completed {model_name}", flush=True)

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    results = pd.DataFrame(rows).sort_values("macro_f1_mean", ascending=False)
    results.to_csv(reports_dir / "cross_validation_results.csv", index=False)
    (reports_dir / "cross_validation_confusion_matrices.json").write_text(
        json.dumps({"labels": labels, "matrices": confusion_results}, indent=2),
        encoding="utf-8",
    )
    print("\nCross-validation summary:\n")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
