"""Interactive Streamlit demo for pump maintenance prioritisation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_LABELS = {
    "rf_balanced_leaf2": "Random Forest - lower business cost",
    "rf_balanced_100": "Random Forest - strongest macro F1",
    "rf_balanced_depth20": "Random Forest - deeper regularisation",
    "extra_trees_balanced_leaf2": "Extra Trees - strong repair recall",
    "catboost_balanced_depth6": "CatBoost - balanced repair focus",
    "catboost_balanced_depth8": "CatBoost - strongest repair recall",
}

STATUS_LABELS = {
    "functional": "Functional",
    "functional needs repair": "Needs repair",
    "non functional": "Non-functional",
}

st.set_page_config(
    page_title="Pump Maintenance Advisor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        .hero { padding: 1.4rem 1.6rem; border-radius: 1rem;
                background: linear-gradient(135deg, #0f766e, #155e75);
                color: white; margin-bottom: 1.25rem; }
        .hero h1 { color: white; margin-bottom: 0.25rem; }
        .hero p { color: #d9f7f4; margin-bottom: 0; }
        .status-card { padding: 1rem 1.2rem; border-radius: 0.8rem;
                       border: 1px solid #d7e5e7; background: #f8fbfb; }
        .status-card h2 { margin: 0.2rem 0; }
        .status-card p { margin: 0; color: #52666b; }
        .small-label { color: #60777d; font-size: 0.82rem; text-transform: uppercase;
                       letter-spacing: 0.06em; font-weight: 700; }
        .recommendation { padding: 1rem 1.2rem; border-left: 5px solid #0f766e;
                          background: #f0fdfa; border-radius: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_test_data() -> pd.DataFrame:
    """Load the competition test records."""
    return pd.read_csv(DATA_DIR / "test_set_values.csv")


@st.cache_data
def load_cv_results() -> pd.DataFrame:
    """Load cross-validation metrics used in the model comparison view."""
    return pd.read_csv(REPORTS_DIR / "cross_validation_results.csv")


@st.cache_data
def load_model_selection() -> dict[str, object]:
    """Load the portfolio model-selection metadata."""
    return json.loads((REPORTS_DIR / "model_selection.json").read_text())


@st.cache_resource
def load_model(model_name: str):
    """Load a trained model once per session."""
    return joblib.load(MODELS_DIR / f"{model_name}.joblib")


def percentage(value: float | int | None) -> str:
    """Format a metric as a percentage."""
    return "-" if value is None else f"{float(value):.1%}"


def prediction_message(prediction: str) -> tuple[str, str]:
    """Return a short operational interpretation for a predicted status."""
    if prediction == "non functional":
        return "Critical maintenance priority", "Schedule an inspection or intervention as soon as possible."
    if prediction == "functional needs repair":
        return "Maintenance priority", "Include this water point in the preventive maintenance queue."
    return "Routine monitoring", "The model does not flag an immediate maintenance issue."


test_data = load_test_data()
cv_results = load_cv_results()
selection = load_model_selection()
available_models = [
    name
    for name in MODEL_LABELS
    if (MODELS_DIR / f"{name}.joblib").exists()
    and name in set(cv_results["model"])
]

if not available_models:
    st.error("No trained models found. Run scripts/train_selected_models.py first.")
    st.stop()

default_model = str(selection.get("default_model", available_models[0]))
default_model = default_model if default_model in available_models else available_models[0]

with st.sidebar:
    st.markdown("### Controls")
    model_name = st.selectbox(
        "Model",
        available_models,
        index=available_models.index(default_model),
        format_func=lambda name: MODEL_LABELS[name],
    )
    record_position = st.slider(
        "Test record",
        min_value=0,
        max_value=len(test_data) - 1,
        value=0,
        help="Choose a record from the competition test set.",
    )
    st.divider()
    st.caption("Portfolio prototype")
    st.caption("Use predictions to prioritise inspections, not to replace field verification.")

model = load_model(model_name)
record = test_data.iloc[[record_position]]
prediction = str(model.predict(record)[0])
prediction_title, recommendation = prediction_message(prediction)
model_row = cv_results.loc[cv_results["model"] == model_name].iloc[0]

st.markdown(
    """
    <div class="hero">
        <h1>Pump Maintenance Advisor</h1>
        <p>A decision-support prototype for prioritising water-pump maintenance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"### Record {int(record.iloc[0]['id'])}")
st.caption(f"Prediction generated with {MODEL_LABELS[model_name]}")

metric_columns = st.columns(4)
metric_columns[0].metric("Validation accuracy", percentage(model_row["accuracy_mean"]))
metric_columns[1].metric("Macro F1", percentage(model_row["macro_f1_mean"]))
metric_columns[2].metric("Repair recall", percentage(model_row["repair_recall_mean"]))
metric_columns[3].metric("Business cost", f"{model_row['business_cost_mean']:.3f}")

st.divider()

prediction_column, profile_column = st.columns([1, 1.35], gap="large")
with prediction_column:
    st.markdown(
        f"""
        <div class="status-card">
            <div class="small-label">Predicted condition</div>
            <h2>{STATUS_LABELS.get(prediction, prediction)}</h2>
            <p>{prediction_title}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='recommendation'><b>Recommended action</b><br>{recommendation}</div>", unsafe_allow_html=True)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(record)[0]
        classes = getattr(model, "classes_", [])
        probability_frame = pd.DataFrame(
            {
                "Condition": [STATUS_LABELS.get(str(item), str(item)) for item in classes],
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False)
        st.markdown("#### Model confidence")
        st.dataframe(
            probability_frame.assign(Probability=probability_frame["Probability"].map(percentage)),
            hide_index=True,
            use_container_width=True,
        )
        st.bar_chart(probability_frame.set_index("Condition"), y="Probability", height=180)

with profile_column:
    st.markdown("#### Asset profile")
    profile_columns = [
        "id", "region", "district_code", "gps_height", "population",
        "construction_year", "waterpoint_type", "source_type", "management",
        "date_recorded",
    ]
    profile_columns = [column for column in profile_columns if column in record.columns]
    profile = record.iloc[0][profile_columns].rename("Value").to_frame()
    profile.index = [column.replace("_", " ").title() for column in profile.index]
    st.dataframe(profile, use_container_width=True)

tab_comparison, tab_maintenance, tab_method = st.tabs(
    ["Model comparison", "Maintenance lens", "Methodology"]
)

with tab_comparison:
    st.markdown("#### Cross-validation comparison")
    comparison = cv_results.copy()
    comparison["Model"] = comparison["model"].map(MODEL_LABELS).fillna(comparison["model"])
    comparison = comparison.sort_values("accuracy_mean", ascending=False)
    display_columns = {
        "Model": "Model",
        "accuracy_mean": "Accuracy",
        "macro_f1_mean": "Macro F1",
        "repair_recall_mean": "Repair recall",
        "business_cost_mean": "Business cost",
    }
    display = comparison[list(display_columns)].rename(columns=display_columns)
    for column in ["Accuracy", "Macro F1", "Repair recall"]:
        display[column] = display[column].map(percentage)
    display["Business cost"] = display["Business cost"].map(lambda value: f"{value:.3f}")
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption("Metrics are five-fold stratified cross-validation means. Lower business cost is better.")

with tab_maintenance:
    st.markdown("#### Why the minority class matters")
    st.write(
        "A pump classified as functional when it actually needs repair can delay an intervention. "
        "For that reason, the portfolio reports repair recall and a provisional maintenance cost "
        "alongside competition accuracy."
    )
    insight_columns = st.columns(3)
    insight_columns[0].metric("Selected model repair recall", percentage(model_row["repair_recall_mean"]))
    insight_columns[1].metric("Selected model business cost", f"{model_row['business_cost_mean']:.3f}")
    best_accuracy = cv_results.loc[cv_results["accuracy_mean"].idxmax()]
    insight_columns[2].metric("Best CV accuracy", percentage(best_accuracy["accuracy_mean"]))
    st.info("The model with the best competition score is not necessarily the model with the best maintenance profile.")

with tab_method:
    st.markdown("#### Reproducible workflow")
    st.markdown(
        "1. Raw competition data is loaded from `data/`.\n"
        "2. Feature engineering and imputation are learned inside a scikit-learn pipeline.\n"
        "3. Candidate models are compared with stratified five-fold cross-validation.\n"
        "4. The selected model is trained on all labelled data and used here for inference."
    )
    st.caption("This is a portfolio demonstration, not an automated maintenance dispatch system.")
