# Model Card: Pump Condition Classification

## Model purpose

This project predicts the operational condition of water pumps from technical, geographical, management, and maintenance-related attributes. It was developed as a portfolio project based on the Pump It Up: Data Mining the Water Table competition.

The predictions are intended to support maintenance prioritisation and model-comparison analysis. They are not intended to replace field inspections, engineering judgement, or local maintenance planning.

## Target classes

- `functional`
- `functional needs repair`
- `non functional`

The `functional needs repair` class is operationally important but under-represented in the labelled training data.

## Data and preprocessing

The project uses the original competition files supplied locally under `data/`. Raw data is excluded from version control.

The reusable preprocessing pipeline:

- removes identifiers and selected high-cardinality fields;
- derives recording year, recording month, and pump age;
- treats selected zero values as missing values;
- imputes numeric values using training-set statistics;
- encodes categorical values with an unknown-category strategy;
- fits transformations inside the model pipeline to reduce leakage risk.

## Evaluation

The project reports two complementary objectives:

1. **Competition objective:** validation accuracy, aligned with the DrivenData leaderboard.
2. **Maintenance objective:** macro F1, recall for `functional needs repair`, and a provisional business cost that penalises missed repair cases more heavily.

Five-fold stratified cross-validation provides the main internal comparison. The current strongest internal configurations include:

- Random Forest with 100 trees: accuracy mean `0.8002`, macro F1 mean `0.7010`.
- Random Forest with `min_samples_leaf=2`: business cost mean `0.3381` and macro F1 mean `0.7008`.
- CatBoost balanced depth 8: repair recall mean `0.7132`, with a substantial accuracy trade-off.

External leaderboard results observed during development were:

| Submission | Score |
| --- | ---: |
| Random Forest, unbalanced | **0.8171** |
| Extra Trees, unbalanced | 0.8134 |
| LightGBM, unbalanced | 0.7890 |
| Random Forest, balanced leaf 2 | 0.7962 |
| Extra Trees, balanced leaf 2 | 0.7919 |
| CatBoost, balanced depth 8 | 0.6912 |

The unbalanced Random Forest is the current competition baseline. These external scores show that the model optimised for maintenance recall is not automatically the best competition model.

## Limitations and risks

- The training data represents a historical competition dataset and may not reflect current infrastructure conditions.
- The provisional business cost is an analytical assumption, not an approved operational cost model.
- High-cardinality features such as pump names and funders may contain useful signal that is not fully exploited in the current baseline.
- The application does not establish causality or diagnose the physical cause of a pump failure.
- Predictions should be reviewed alongside field observations and local context.

## Monitoring recommendations

For a real deployment, monitor class frequencies, missingness, feature drift, calibration, repair recall, false negatives, and performance across geographic or administrative groups. Retraining should be triggered by documented performance or data-quality thresholds rather than by leaderboard score alone.
