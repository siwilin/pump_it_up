"""Train representative models on all labelled data and create predictions."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tune_models import model_configs
from src.data import load_pump_it_up_data
from src.modeling import build_classifier_pipeline


def select_models(results: pd.DataFrame) -> list[str]:
    """Select a portfolio set covering cost, macro F1, and repair recall."""
    names = set(results.nsmallest(3, "business_cost_mean")["model"])
    names.update(results.nlargest(2, "macro_f1_mean")["model"])
    names.update(results.nlargest(2, "repair_recall_mean")["model"])
    return sorted(names)


def main() -> None:
    reports_dir = PROJECT_ROOT / "reports"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    results = pd.read_csv(reports_dir / "cross_validation_results.csv")
    selected = select_models(results)

    train_values, train_labels, test_values = load_pump_it_up_data(
        PROJECT_ROOT / "data"
    )
    train = train_values.merge(train_labels, on="id", validate="one_to_one")
    features = train.drop(columns=["status_group"])
    target = train["status_group"]
    test_ids = test_values["id"]
    configs = model_configs()

    default_model = results.nsmallest(1, "business_cost_mean").iloc[0]["model"]
    selection_report = {"selected_models": selected, "default_model": default_model}
    for name in selected:
        pipeline = build_classifier_pipeline(features, configs[name])
        pipeline.fit(features, target)
        predictions = np.asarray(pipeline.predict(test_values)).reshape(-1)
        joblib.dump(pipeline, models_dir / f"{name}.joblib")
        submission = pd.DataFrame({"id": test_ids, "status_group": predictions})
        submission.to_csv(reports_dir / f"submission_{name}.csv", index=False)
        print(f"Trained and exported {name}", flush=True)

    (reports_dir / "model_selection.json").write_text(
        json.dumps(selection_report, indent=2), encoding="utf-8"
    )
    print(f"Selected models: {', '.join(selected)}")


if __name__ == "__main__":
    main()
