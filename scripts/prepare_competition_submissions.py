"""Train three competition-oriented models and export valid submission files."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tune_models import model_configs
from src.data import load_pump_it_up_data
from src.modeling import build_classifier_pipeline
from src.submissions import validate_submission


COMPETITION_CANDIDATES = (
    "rf_unbalanced_100",
    "extra_trees_unbalanced_100",
    "lgbm_unbalanced_leaves31",
)


def main() -> None:
    reports_dir = PROJECT_ROOT / "reports"
    train_values, train_labels, test_values = load_pump_it_up_data(
        PROJECT_ROOT / "data"
    )
    train = train_values.merge(train_labels, on="id", validate="one_to_one")
    features = train.drop(columns=["status_group"])
    target = train["status_group"]
    test_ids = test_values["id"]
    configs = model_configs()

    for model_name in COMPETITION_CANDIDATES:
        pipeline = build_classifier_pipeline(features, configs[model_name])
        pipeline.fit(features, target)
        predictions = np.asarray(pipeline.predict(test_values)).reshape(-1)
        submission = pd.DataFrame(
            {"id": test_ids, "status_group": predictions}
        )
        output_path = reports_dir / f"submission_{model_name}.csv"
        submission.to_csv(output_path, index=False)
        validate_submission(submission, expected_ids=test_ids)
        distribution = submission["status_group"].value_counts(normalize=True)
        print(f"Created {output_path.name}")
        print(distribution.round(4).to_dict())


if __name__ == "__main__":
    main()
