# PRD: V2 — Supervised ML Training

## Problem Statement

The V1 EDA report tells the user what their data looks like — column types, distributions, outliers, correlations, and feature engineering candidates. What it cannot tell them is how well that data actually predicts an outcome. Data scientists and analysts routinely follow EDA with a quick baseline model to answer "is there signal here at all?" — but doing this manually means writing boilerplate train/test split code, fitting multiple model types, computing evaluation metrics, and interpreting feature importances, all before getting back to the question they actually care about. Automated EDA without automated baseline modeling leaves the hardest part of the first-pass workflow still manual.

## Solution

Extend the agent with an opt-in supervised ML mode — activated via `--target` or `--mode ml` — that runs after EDA, trains a fixed comparison set of models (logistic regression + random forest for classification; linear regression + random forest for regression), evaluates them on a held-out test set, and adds a "Modeling" section to the existing report with metrics, feature importances, and chart artifacts. The trained model artifacts are saved alongside the report for downstream use. The same uncertainty mechanism that governs EDA (human checkpoint / flagged assumption) handles new ML-specific triggers: severe class imbalance and datasets too small for a reliable split. V1 EDA-only runs are unchanged.

## User Stories

1. As an ad-hoc analyst, I want to run the agent in ML mode by specifying a target column, so that I get a baseline model trained and evaluated without writing any code.
2. As a data science power user, I want the agent to infer a target column from EDA results when I haven't specified one, so that I can get a first model without knowing which column to predict.
3. As a data science power user, I want to be asked to confirm an inferred target column when the agent isn't confident, so that I don't train on the wrong outcome.
4. As a pipeline operator, I want an ambiguous target column to fall back to a documented default with a clear flag in the report, so that my unattended run still completes.
5. As an ad-hoc analyst, I want to see both a logistic regression and a random forest trained on my classification problem, so that I can compare a simple baseline against a stronger model.
6. As an ad-hoc analyst, I want to see both a linear regression and a random forest trained on my regression problem, for the same reason.
7. As a data science power user, I want to pin to a single model type via `--model`, so that I can trade comparison breadth for speed on a large dataset.
8. As any ML user, I want F1 (macro), ROC-AUC, and a confusion matrix for classification results, so that I get a fair picture even on imbalanced classes.
9. As any ML user, I want RMSE and R² for regression results, so that I understand both absolute error and explained variance.
10. As any ML user, I want evaluation metrics narrated in plain language in the Modeling section, so that I can interpret results without a statistics background.
11. As a data science power user, I want a feature importance bar chart for every model trained, so that I can see which columns drove the predictions.
12. As a data science power user, I want a confusion matrix heatmap saved as a chart artifact for classification results, so that I can visually inspect the error pattern.
13. As a data science power user, I want a residual plot saved as a chart artifact for regression results, so that I can validate model assumptions at a glance.
14. As a pipeline operator, I want the trained model saved as a `.pkl` file alongside the report, so that I can load and use it downstream without retraining.
15. As any ML user, I want the report to note which columns were imputed or encoded internally before training, so that the preprocessing isn't a black box.
16. As a data science power user, I want severe class imbalance (majority class >80%) to trigger the standard uncertainty mechanism, so that I'm warned before misleading metrics are produced.
17. As any ML user, I want a dataset too small for a reliable train/test split to trigger the standard uncertainty mechanism, so that I know my evaluation results are shaky.
18. As a pipeline operator, I want both new ML uncertainty triggers to fall back to flagged assumptions in non-interactive runs, so that automated pipelines always complete.
19. As a data science power user, I want the uncertainty thresholds for class imbalance and minimum split size to be configurable, consistent with V1 threshold configuration.
20. As any ML user, I want training to complete in under 5 minutes for datasets up to 100k rows, so that I get results fast enough to stay in flow.
21. As any ML user, I want to be informed when training used a row sample rather than the full dataset, so that I understand the scope of the model's training data.
22. As a pipeline operator, I want ML mode to be strictly opt-in, so that existing V1-style EDA-only runs are not affected by V2 changes.
23. As a developer extending the agent, I want the training tool to be a standalone, directly-callable function with no LLM dependency, so that I can unit-test it against fixture DataFrames.
24. As a developer testing the agent end-to-end, I want ML graph wiring and Modeling section rendering covered by the existing fake-LLM integration-test pattern, so that deterministic tests validate the orchestration without real API calls.

## Implementation Decisions

- **Opt-in activation**: ML mode runs only when `--target <column>` or `--mode ml` is passed. Without these flags, the agent runs V1 EDA only. `--target` implies `--mode ml`; `--mode ml` without `--target` triggers target inference.
- **Target column inference**: When no target is specified, the agent infers a candidate from EDA signal (low-cardinality columns for classification; high-variance continuous columns for regression) and routes through the standard uncertainty-trigger mechanism — human checkpoint (interactive) or flagged assumption (non-interactive) — when confidence is low.
- **Task type detection**: If the target column's inferred type is categorical/boolean, the agent runs classification; if numeric continuous, regression. Ambiguous cases (e.g. a numeric column with low cardinality) go through the uncertainty mechanism.
- **Model comparison set**: Classification trains logistic regression + random forest; regression trains linear regression + random forest. An optional `--model {logistic_regression,random_forest,linear_regression}` flag pins to a single model.
- **Internal preprocessing (ADR-0001/ADR-0002 compliant)**: The training tool applies median imputation (numerics) and ordinal encoding (categoricals) internally before fitting. This never surfaces to the LLM; the tool's aggregate output reports which columns were imputed or encoded and how many were dropped (e.g. high-cardinality text columns). Implement with a `skip_internal_preprocessing=False` parameter so V3's automated feature application can bypass it cleanly.
- **Train/test split**: 80/20 stratified split by default. Configurable via `--test-size <float>` (overrides split ratio) or `--cv-folds <int>` (switches to k-fold cross-validation). The agent proactively suggests k-fold via the uncertainty mechanism when the dataset is below the existing "too small for statistical conclusions" threshold.
- **Row sampling**: When the training dataset exceeds 50k rows, the training tool automatically draws a stratified sample of 50k rows for model fitting only. EDA still runs on the full dataset. The sample size and seed are reported in the Modeling section.
- **Evaluation metrics**: Classification — F1 (macro), ROC-AUC, confusion matrix. Regression — RMSE, R². All returned as column-level aggregates (no raw prediction rows), consistent with ADR-0002.
- **Model artifact persistence**: Each trained model is serialized to `models/<model_type>_<timestamp>.pkl` in the output directory and referenced by relative path in the Modeling section. The training tool handles serialization; the LLM never sees the binary.
- **Chart artifacts**: Feature importance bar chart (both task types), confusion matrix heatmap (classification only), residual plot (regression only) — all saved as PNGs in `charts/`, referenced by relative markdown image links in the Modeling section. Follows the V1 chart pattern exactly.
- **New uncertainty triggers**: Two new triggers plug into the existing mechanism with no new architecture: (1) majority class >80% of rows (classification) — default threshold configurable; (2) dataset row count below minimum-split-size threshold (configurable, default TBD by implementation). Both appear in the Data Quality Scorecard as flagged assumptions when non-interactive.
- **Report structure**: V2 adds a 6th canonical section — "Modeling" — after "Feature Engineering Recommendations." CLAUDE.md must be updated when this is implemented to replace "All 5 report sections" with "All 6 report sections." When ML mode is not activated, the Modeling section is omitted (the 6-section rule applies only to ML runs; EDA-only runs retain the 5-section structure).
- **Issue sequencing**: (1) `train_model` tool — pure Python, unit-testable, no LLM. (2) Chart generation tools for modeling artifacts. (3) Graph wiring: target inference, new uncertainty triggers, ML node in LangGraph. (4) Modeling section renderer + report integration. (5) CLAUDE.md update.
- **Glossary terms in play** (defined in `CONTEXT.md`): Tool, Uncertainty trigger, Human checkpoint, Flagged assumption.

## Testing Decisions

- Follows the V1 pattern established in `docs/prd/v1-eda-and-report.md`: unit tests for every new tool directly against fixture DataFrames; integration tests via fake LLM client.
- **Training tool unit tests**: Fixture DataFrames covering — a clean classification dataset, a clean regression dataset, a classification dataset with >80% majority class, a dataset below the minimum split size, a DataFrame with mixed types and NaNs (validates internal preprocessing), and a dataset exceeding 50k rows (validates row sampling). Assert on returned metric keys, `.pkl` file existence, and which columns were reported as encoded/imputed.
- **Chart artifact tests**: Assert that the correct PNG files appear in `charts/` for each task type; assert that the Modeling section markdown references them by relative path.
- **ML uncertainty trigger tests**: At least one fixture per new trigger type, exercised through both interactive and non-interactive paths. Assert that class imbalance and small-split triggers produce a Human checkpoint (interactive fake) or a Flagged assumption entry in the Data Quality Scorecard (non-interactive fake).
- **Integration tests**: Drive full ML-mode runs with fake LLM client; assert Modeling section present with expected subsections, all chart files exist, `.pkl` files exist in `models/`, and trace log contains one entry per tool call.
- **EDA-only regression**: At least one integration test confirms that a run without `--target` / `--mode ml` produces exactly 5 sections and no `models/` directory.

## Out of Scope

- Automated feature transformation applied before training (V3) — V2 trains on data as-is with internal preprocessing only
- Natural language Q&A / conversational follow-up (V3)
- Unsupervised learning: clustering, dimensionality reduction (V3+)
- Hyperparameter tuning / AutoML (V3+)
- Residual plots for classification or feature importance plots beyond bar charts (V3+)
- Model deployment or serving endpoints
- SHAP or other model-explanation frameworks (V3+)
- Non-CSV inputs (V3+)
- Jupyter notebook export (V3)
- FastAPI service wrapper (V3)
