"""Train selected model configurations and export competition submissions."""

from __future__ import annotations

import argparse
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


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "submissions"


def parse_args() -> argparse.Namespace:
    """Parse a future-proof model selection command."""
    parser = argparse.ArgumentParser(
        description="Train named configurations and create DrivenData submissions."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Registered model configuration names to train.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available configurations and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where submission CSVs are written.",
    )
    return parser.parse_args()


def build_submission(
    model_name: str,
    features: pd.DataFrame,
    target: pd.Series,
    test_values: pd.DataFrame,
    test_ids: pd.Series,
) -> pd.DataFrame:
    """Fit one registered configuration and return a validated submission."""
    estimator = model_configs()[model_name]
    pipeline = build_classifier_pipeline(features, estimator)
    pipeline.fit(features, target)
    predictions = np.asarray(pipeline.predict(test_values)).reshape(-1)
    submission = pd.DataFrame({"id": test_ids, "status_group": predictions})
    validate_submission(submission, expected_ids=test_ids)
    return submission


def main() -> None:
    args = parse_args()
    configs = model_configs()

    if args.list_models:
        print("\n".join(sorted(configs)))
        return
    if not args.models:
        raise SystemExit("Provide --models or use --list-models.")

    unknown = sorted(set(args.models) - set(configs))
    if unknown:
        raise SystemExit(
            f"Unknown model configuration(s): {', '.join(unknown)}. "
            "Use --list-models to inspect the registry."
        )

    train_values, train_labels, test_values = load_pump_it_up_data(
        PROJECT_ROOT / "data"
    )
    train = train_values.merge(train_labels, on="id", validate="one_to_one")
    features = train.drop(columns=["status_group"])
    target = train["status_group"]
    test_ids = test_values["id"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        submission = build_submission(
            model_name, features, target, test_values, test_ids
        )
        output_path = args.output_dir / f"submission_{model_name}.csv"
        submission.to_csv(output_path, index=False)
        distribution = submission["status_group"].value_counts(normalize=True)
        print(f"Created {output_path}")
        print(distribution.round(4).to_dict())


if __name__ == "__main__":
    main()
