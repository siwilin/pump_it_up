import pandas as pd

from src.modeling import (
    build_random_forest_pipeline,
    classification_metrics,
    portfolio_model_specs,
    stratified_split,
)


def _sample_data() -> tuple[pd.DataFrame, pd.Series]:
    features = pd.DataFrame(
        {
            "population": [10, 20, 30, 40, 50, 60, 70, 80],
            "region": ["north", "north", "south", "south"] * 2,
        }
    )
    target = pd.Series(["functional", "non functional"] * 4, name="status_group")
    return features, target


def test_stratified_split_preserves_observations() -> None:
    features, target = _sample_data()
    x_train, x_val, y_train, y_val = stratified_split(features, target, test_size=0.25)

    assert len(x_train) + len(x_val) == len(features)
    assert len(y_train) + len(y_val) == len(target)
    assert set(y_train) == set(target)
    assert set(y_val) == set(target)


def test_random_forest_pipeline_can_fit_small_dataset() -> None:
    features, target = _sample_data()
    pipeline = build_random_forest_pipeline(features, n_estimators=10)
    pipeline.fit(features, target)

    predictions = pipeline.predict(features)
    metrics = classification_metrics(target, predictions)

    assert len(predictions) == len(target)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1


def test_portfolio_model_specs_are_diverse() -> None:
    specs = portfolio_model_specs()

    assert len(specs) == 8
    assert "logistic_regression_balanced" in specs
    assert "extra_trees_balanced" in specs
    assert "hist_gradient_boosting" in specs
    assert "xgboost" in specs
    assert "lightgbm_balanced" in specs
    assert "catboost_balanced" in specs
