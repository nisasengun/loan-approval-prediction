# Loan Approval Prediction — Classification Models

A binary classification project that predicts whether a bank loan application will be **approved** or
**rejected**, based on an applicant's demographic and financial profile. Nine classic ML algorithms
(plus a bonus XGBoost model) were trained, tuned with `GridSearchCV`, and compared on the same
train/test split.

> Originally built as the final project for a *Statistical Software II* course; reorganized here as a
> standalone, run-anywhere repository.

## Problem

Given an applicant's income, requested loan amount, term, credit history (CIBIL score), and asset
values, predict `loan_status ∈ {Approved, Rejected}`.

## Dataset

[**Loan Approval Prediction Dataset**](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) (Kaggle)
— 4,269 rows × 13 columns, no missing values.

| Column | Description |
|---|---|
| `no_of_dependents` | Number of dependents |
| `education` | Graduate / Not Graduate |
| `self_employed` | Yes / No |
| `income_annum` | Annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan term (years) |
| `cibil_score` | Credit score (300–900) |
| `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value` | Asset values |
| `loan_status` | **Target** — Approved / Rejected |

The CSV is not included in this repo (Kaggle's terms) — download it from the link above and place it in
the project root as `loan_approval_dataset.csv` before running the notebook.

## Approach

1. **Preprocessing** — dropped the non-predictive `loan_id`, binary-encoded `education`,
   `self_employed`, and the target `loan_status`, then scaled all features with `StandardScaler`
   (required for the linear/distance-based models).
2. **EDA** — class balance check (62% approved / 38% rejected — no resampling needed), a correlation
   heatmap, and CIBIL-score-vs-approval and income-vs-loan-amount visualizations.
3. **Modeling** — trained a default model, then ran `GridSearchCV` (5–10 fold CV) to tune each
   algorithm, evaluated on an 80/20 train/test split:
   - Logistic Regression (+ `statsmodels` for coefficient significance)
   - Naive Bayes
   - K-Nearest Neighbors
   - Linear SVM
   - RBF-kernel SVM
   - Multi-Layer Perceptron (Neural Network)
   - Decision Tree (CART)
   - Random Forest
   - Gradient Boosting
   - *Bonus:* XGBoost, for comparison against the course's Gradient Boosting model
4. **Comparison** — ranked all models by test-set Accuracy and F1-score.

## Results

| Rank | Model | Accuracy | F1-Score |
|---|---|---|---|
| 1 | **Gradient Boosting** | **0.9848** | **0.9879** |
| 2 | Random Forest | 0.9731 | 0.9786 |
| 3 | Decision Tree (CART) | 0.9696 | 0.9760 |
| 4 | Neural Network (MLP) | 0.9660 | 0.9730 |
| 5 | RBF SVM | 0.9520 | 0.9618 |
| 6 | Naive Bayes | 0.9368 | 0.9490 |
| 7 | Linear SVM | 0.9169 | 0.9331 |
| 8 | K-NN | 0.9063 | 0.9248 |
| 9 | Logistic Regression | 0.9052 | 0.9248 |

*(Bonus XGBoost, tuned: 0.9836 accuracy / 0.9870 F1 — a close second to GBM, with a faster, more
aggressive tree structure.)*

**Best model: Gradient Boosting** — 13 misclassifications out of 854 test cases (8 false positives,
5 false negatives), AUC 0.9974.

### Key findings

- **`cibil_score` dominates.** It alone explains over 80% of feature importance in every tree-based
  model and is the only variable with a strong (0.77) correlation with the target. Every algorithm,
  regardless of architecture, converges on the same signal.
- **The decision boundary is non-linear.** Linear models (Logistic Regression, Linear SVM, K-NN) plateau
  around 90–92% accuracy, while models that can carve flexible boundaries (RBF SVM, MLP, tree ensembles)
  clear 95%+ — evidence the approval rule isn't a straight line in feature space.
- **Demographics don't matter here.** `education`, `self_employed`, and `no_of_dependents` carry
  near-zero importance in every model — the decision is driven by financial variables, not personal
  attributes.
- **Ensembles win, and boosting wins hardest.** Random Forest and Gradient Boosting both landed in the
  top 3; GBM's sequential error-correction gave it the edge over RF's parallel voting.

## Repo structure

```
.
├── README.md
├── requirements.txt
└── loan_approval_prediction.ipynb   # full pipeline: EDA → preprocessing → 9 models + XGBoost → comparison
```

## Running it

```bash
pip install -r requirements.txt
# download loan_approval_dataset.csv from the Kaggle link above into this folder
jupyter notebook loan_approval_prediction.ipynb
```

## Tech stack

Python · pandas · NumPy · scikit-learn · statsmodels · XGBoost · matplotlib · seaborn
