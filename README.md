# ML-Lab
Hands-on machine learning implementations - core algorithms written from scratch first,
then the same problem solved with production tooling to understand what the libraries do.

Every notebook runs top to bottom in Colab. Results below are measured, not estimated.

## Results

| Notebook | Focus | Result |
|---|---|---|
| `day_01_titanic_eda.ipynb` | EDA, pandas, missing-value handling | Survival drivers identified: sex (74% vs 19%), class (63% vs 24%) |
| `day_02_linear_regression.ipynb` | Gradient descent implemented in NumPy | Recovers true parameters (w=3.0, b=4.0) from noisy data; sklearn baseline R² 0.58 on California Housing |
| `day_03_logistic_metrics.ipynb` | Classification metrics, threshold tuning | ROC-AUC 0.85; precision/recall trade-off measured at thresholds 0.5 vs 0.3 |
| `day_04_overfitting_regularization.ipynb` | Bias-variance, Ridge/Lasso, k-fold CV | Polynomial degrees 1/4/15 compared; test error divergence quantified |
| `day_05_xgboost_house_prices.ipynb` | Gradient boosting, Kaggle submission | **0.145 RMSLE** (numeric features only) |
| `day_06_pipelines.ipynb` | ColumnTransformer + Pipeline + GridSearchCV | **0.1407 RMSLE** — leakage-safe preprocessing, categorical features recovered |

## What each notebook demonstrates

**From scratch, no libraries:** gradient descent (forward pass, MSE loss, analytic gradients,
parameter updates) and the metric definitions behind `classification_report` — hand-computed
confusion matrices verified against sklearn.

**Production practice:** all preprocessing lives inside an sklearn `Pipeline` fitted on training
data only, so imputation and encoding parameters never leak from the validation or test split.
The Day 5 → Day 6 improvement comes entirely from correctly handling the 43 categorical columns
that a numeric-only baseline discards.

**Evaluation discipline:** models are compared with k-fold cross-validation rather than a single
split, and every result in the table above is a held-out or leaderboard score.

## Stack

Python · NumPy · pandas · scikit-learn · XGBoost · Matplotlib · Jupyter/Colab

## Running it
Open any notebook directly in Google Colab - datasets download themselves
(`fetch_california_housing`, `kagglehub`, seaborn) with no manual setup.

## Roadmap

PyTorch training loops · transformer fine-tuning (HuggingFace) · retrieval-augmented
generation from scratch · model serving with FastAPI + Docker

---

Part of a structured self-study track. Related repositories:
[GameAI](https://github.com/nurgissas/GameAI) (open-source RL inference engine)
[github-code-reviewer](https://github.com/nurgissas/github-code-reviewer)
