# Pump It Up — Predictive Maintenance for Water Pumps

An end-to-end machine learning project for predicting the operational condition of water pumps and supporting maintenance prioritisation.

This repository turns the original competition exercise into a reproducible portfolio project: the code is modular, preprocessing is learned inside a scikit-learn pipeline, evaluation makes class imbalance visible, and the project includes a small decision-support demo.

The intended audience is technical reviewers: the repository includes executable notebooks, reusable Python modules, automated tests, a model card, and a CI workflow.

## Business question

Given the technical, geographical and operational information available for a water point, can we identify whether its pump is:

- **Functional**
- **Functional, needs repair**
- **Non-functional**

The minority class is especially important operationally: missing a pump that needs repair can delay maintenance and increase the risk of failure.

## Evaluation strategy

This project separates two evaluation objectives:

1. **Competition performance:** accuracy is the primary metric for the DrivenData submission.
2. **Maintenance decision support:** macro F1, recall for `functional needs repair`, and an explicit business cost make the operational trade-offs visible.

The first reproducible benchmark uses the same stratified validation split for all model families:

| Model | Accuracy | Macro F1 | Recall — needs repair |
| --- | ---: | ---: | ---: |
| Logistic Regression, balanced | 0.5229 | 0.4653 | 0.5261 |
| Decision Tree, balanced | 0.6833 | 0.5974 | **0.5295** |
| Random Forest, balanced | 0.7989 | **0.6963** | 0.4299 |
| Extra Trees, balanced | **0.8028** | 0.6829 | 0.3221 |
| HistGradientBoosting | 0.7882 | 0.6465 | 0.2225 |
| XGBoost | 0.7755 | 0.6110 | 0.1553 |
| LightGBM, balanced | 0.7119 | 0.6356 | 0.6941 |
| CatBoost, balanced | 0.6672 | 0.5962 | **0.7045** |

The class-weighted models show the central trade-off: tree ensembles retain overall accuracy, while LightGBM and CatBoost identify more of the minority class. A competition submission should be selected by validation accuracy; a maintenance deployment may reasonably select a different model after reviewing error costs.

### Focused tuning round

The strongest configurations from the next round were:

| Configuration | Accuracy | Macro F1 | Recall — needs repair |
| --- | ---: | ---: | ---: |
| Random Forest, `min_samples_leaf=2` | 0.7903 | **0.7011** | 0.5550 |
| Extra Trees, `min_samples_leaf=2` | 0.7866 | 0.6977 | **0.5724** |
| LightGBM, balanced, `num_leaves=31` | 0.7227 | 0.6450 | 0.6848 |
| LightGBM, balanced, `num_leaves=63` | 0.7571 | 0.6728 | 0.6211 |
| CatBoost, balanced, depth 8 | 0.6951 | 0.6194 | 0.6929 |
| XGBoost, depth 8 | 0.7941 | 0.6463 | 0.2086 |

This shows why model selection should not rely on accuracy alone. CatBoost and LightGBM offer higher minority-class recall, while Random Forest retains the strongest macro F1. The final model should be chosen after cross-validation, external leaderboard feedback, and an explicit error-cost discussion.

### Five-fold validation

The full benchmark is also evaluated with five stratified folds. Using a provisional cost of 5 for missing a pump that needs repair and 1 for other errors, the lowest mean business cost was `rf_balanced_leaf2` at `0.3381`. The strongest macro F1 was `rf_balanced_100` at `0.7010`, while CatBoost and LightGBM reached the highest repair recall, above `0.71` in cross-validation.

### External leaderboard feedback

The first external submissions exposed an important trade-off between the competition objective and the maintenance objective:

| Submission | DrivenData score |
| --- | ---: |
| Random Forest, balanced leaf 2 | 0.7962 |
| Extra Trees, balanced leaf 2 | 0.7919 |
| CatBoost, balanced depth 8 | 0.6912 |

The results are documented rather than hidden: class balancing improved repair recall locally but over-predicted the minority class for the leaderboard metric. The next competition iteration will benchmark unweighted models separately.

## Project structure

```text
app/          Streamlit decision-support demo
data/         Local competition data; excluded from version control
notebooks/    Exploratory analysis and model experiments
reports/      Executed notebooks and generated evaluation artifacts
src/          Reusable data, EDA, preprocessing and modelling code
tests/        Automated tests for project utilities
```

Key files:

- `notebooks/01_exploration.ipynb` — data quality, missingness and target analysis.
- `notebooks/02_modeling.ipynb` — reproducible baseline and confusion matrix.
- `notebooks/03_experiments.ipynb` — comparison of baseline and class-weighted models.
- `notebooks/04_model_benchmark.ipynb` — benchmark of eight model families/configurations.
- `notebooks/05_model_tuning.ipynb` — focused hyperparameter configurations.
- `notebooks/06_cross_validation.ipynb` — five-fold validation, cost analysis and aggregated confusion matrices.
- `src/preprocessing.py` — feature engineering and leakage-safe transformations.
- `src/modeling.py` — stratified split, Random Forest pipeline and evaluation metrics.
- `model_card.md` — intended use, evaluation, limitations and monitoring recommendations.
- `.github/workflows/ci.yml` — automated tests and notebook validation.

## Reproduce locally

1. Create or activate a Python environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   This installs the project in editable mode together with development dependencies from `pyproject.toml`.

3. Place these competition files in `data/`:

   ```text
   training_set_values.csv
   training_set_labels.csv
   test_set_values.csv
   ```

4. Run the tests:

   ```bash
   pytest -q
   ```

5. Validate all generated submissions:

   ```bash
   python scripts/validate_submissions.py
   ```

6. Open the notebooks from the repository root and run them in order. Executed copies are stored under `reports/`.

7. To train the selected portfolio models and prepare the demo artifacts:

   ```bash
   python scripts/train_selected_models.py
   streamlit run app/app.py
   ```

The trained model files are intentionally excluded from Git because several tree ensembles are large. Run the training script locally before launching the app.

## Roadmap

- Add frequency encoding and rare-category grouping for high-cardinality features.
- Benchmark unweighted models specifically for the competition accuracy objective.
- Add feature importance and model interpretation to the demo.
- Add CI checks for tests, notebook validity, and submission schema validation.
- Publish a concise model card describing intended use and limitations.

## Data and acknowledgement

The project is based on the Pump It Up: Data Mining the Water Table competition dataset. Raw competition files are not committed to this repository; users should obtain them from the original competition source and place them locally under `data/`.
