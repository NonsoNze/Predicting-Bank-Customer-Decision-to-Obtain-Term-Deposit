# Term Deposit Subscription Prediction
 
A supervised machine learning classification project predicting whether a bank customer
will subscribe to a term deposit, built on the UCI Bank Marketing dataset. The project
covers the full analytical pipeline from exploratory data analysis through feature
engineering, model comparison, hyperparameter tuning, and honest evaluation with
explicit attention to data quality, leakage prevention, and transparent reporting of
where the model performs well and where it does not.
 
---
 
## Dataset
 
| Property | Value |
|---|---|
| Source | UCI Bank Marketing Dataset |
| Rows | 11,162 |
| Target | `deposit` — whether the client subscribed (yes/no) |
| Class balance | ~53% no / ~47% yes |
| Features | 16 raw features (demographic, financial, campaign contact) |
 
The dataset contains information about direct marketing campaigns run by a Portuguese
bank. Each row represents one customer contact, with features covering the customer's
age, job, marital status, account balance, previous campaign outcomes, and contact
details.
 
The relatively balanced class distribution (53/47) means severe imbalance techniques
are not required, though the baseline and evaluation methodology still account for it.
 
---
 
## Pipeline Structure
 
| Section | Contents |
|---|---|
| 1. Load & Inspect | Data loading, shape, dtypes, null check |
| 2. Target Variable | Class distribution, fraud rate, countplot |
| 3. Duration Investigation | Leakage identification and removal |
| 4. Categorical EDA | Fraud rates by job, education, poutcome, contact |
| 5. pdays & previous | Sentinel value handling, recoding strategy |
| 6. Balance & Outliers | Distribution analysis, negative balance handling |
| 7. Cleaning Summary | Documented decisions for every column |
| 8. Feature Engineering | New derived features |
| 9. Feature Matrix | One-hot encoding, train/val split, scaling |
| 10. Baseline | Majority-class accuracy for context |
| 11. Model Comparison | Cross-validated ROC-AUC across five models |
| 12. Hyperparameter Tuning | RandomizedSearchCV on winning model |
| 13. Final Evaluation | ROC-AUC, classification report, confusion matrix |
| 14. Feature Importance | Permutation importance on validation set |
| 15. Additional Feature Engineering | Cyclical encoding, interaction terms, log balance |
| 16. Rebuild Feature Matrix | Updated feature set with new additions |
| 17. Extended Tuning | Wider hyperparameter grid |
| 18. Stacking Ensemble | HGBC + Random Forest + Logistic Regression |
| 19. Before/After Comparison | ROC-AUC across all model versions |
| 20. Summary | Honest findings and next steps |
 
---
 
## Key Design Decisions
 
**Why `duration` was removed**
 
`duration` is the length in seconds of the last phone call with the customer. It is
one of the strongest predictors in the raw dataset — but it is only known after the
call ends, which is after the outcome (subscription or not) is already determined.
A longer call almost always means the customer said yes, because there is more to
discuss. Including this variable would mean the model is reading the answer after
the fact rather than predicting it beforehand.
 
Removing it is the single most important decision in the project. The resulting metrics
are lower, but the model reflects a system that could actually be deployed before a
call is made — which is the real prediction problem.
 
**Why `poutcome` was kept**
 
`poutcome` (outcome of the previous campaign) has a high proportion of `'unknown'`
values. A naive approach would drop it as too sparse. However, a crosstab reveals it
is one of the strongest signals in the dataset: customers with a prior `'success'`
outcome subscribe to the current campaign approximately 91% of the time, versus
around 41% for `'unknown'`. Dropping it would discard a genuinely valuable feature.
It is retained with `'unknown'` kept as its own category.
 
**Why `pdays` was recoded**
 
`pdays` represents days since the customer was last contacted in a previous campaign,
with `-1` used as a sentinel meaning "never previously contacted." Treating `-1` as
a literal numeric value would mislead the model — a value of `-1` is numerically
smaller than all positive values, implying these customers were contacted before
everyone else, which is the opposite of the truth.
 
The fix: a binary `was_contacted_before` flag captures the real signal, and `pdays`
is recoded to 0 for never-contacted customers, making the numeric value meaningful
only where it actually applies.
 
**Why `'unknown'` categories were retained**
 
Dropping rows where job or education is `'unknown'` discards data and assumes those
customers are randomly distributed — which may not be true. Customers with unknown
job or education could systematically differ in their subscription behaviour.
`'unknown'` is kept as a category in its own right and one-hot encoded alongside
the other values, letting the model learn from it.
 
**Why scaling is applied only to continuous columns**
 
Scaling binary 0/1 dummy columns (from one-hot encoding) is unnecessary and makes
coefficients harder to interpret. StandardScaler is applied only to genuinely
continuous features: age, balance, day, campaign, pdays_clean, previous, and the
engineered features. Tree-based models do not need scaling at all; it is applied
here only for the benefit of the linear models in the comparison.
 
---
 
## Feature Engineering
 
| Feature | Description | Rationale |
|---|---|---|
| `was_contacted_before` | Binary flag: pdays != -1 | Recodes the -1 sentinel into a meaningful signal |
| `pdays_clean` | pdays with -1 replaced by 0 | Makes the numeric value interpretable |
| `poutcome_success_flag` | Binary: poutcome == 'success' | Isolates the strongest poutcome category |
| `success_x_previous` | poutcome_success_flag x previous | Interaction: prior success combined with high contact count signals an engaged, responsive customer |
| `month_sin` / `month_cos` | Cyclical encoding of month | Preserves seasonal adjacency — December and January are treated as close rather than maximally distant |
| `balance_log` | Signed log transform of balance | Compresses the heavy right skew and outliers in account balance |
| `job_grouped` | Rare job categories merged to 'other' | Reduces sparse, noisy dummy columns from low-frequency categories |
 
---
 
## Models Compared
 
Five models are evaluated via 6-fold stratified cross-validation on the training set,
scored by ROC-AUC:
 
- Logistic Regression
- Ridge Classifier
- Decision Tree Classifier
- HistGradientBoosting Classifier
- K-Nearest Neighbours
HistGradientBoosting consistently outperforms the others and is selected for
hyperparameter tuning.
 
---
 
## Final Model and Ensemble
 
Two versions of the final model are evaluated:
 
**V1 — Tuned HGBC (original features)**
Randomized hyperparameter search (n_iter=50) on the original cleaned feature set.
 
**V2 — Tuned HGBC (extended features)**
Same search applied to the extended feature matrix including the cyclical encoding,
interaction term, log balance, and grouped job categories.
 
**V2 — Stacking Ensemble**
A StackingClassifier combining:
- HistGradientBoosting Classifier (best params from V2 search)
- Random Forest (n_estimators=200, max_depth=12)
- Logistic Regression
with a Logistic Regression meta-learner and 5-fold internal cross-validation.
 
---
 
## Results
 
| Model | Test ROC-AUC |
|---|---|
| Baseline (majority class) | 0.50 |
| V1: Tuned HGBC (original features) | 0.7998 |
| V2: Tuned HGBC (extended features) | 0.7955 |
| V2: Stacking Ensemble | 0.7957 |
| V2 features + V1 hyperparameters (diagnostic) | ~0.798 |
 
**Key finding from the diagnostic:** The V2 model's slightly lower score compared to V1
was caused by a search-budget confound, not by the new features. V2 used a wider
hyperparameter grid with the same number of iterations (n_iter=25), meaning each
region of the grid was explored less thoroughly. When the V2 feature set was evaluated
using V1's already-tuned hyperparameters, it performed at approximately the same level
as V1. This demonstrates that the new features did not hurt performance and that a
larger search budget on the V2 grid would likely exceed V1's score.
 
The honest conclusion: the new features (interaction term, cyclical encoding, log
balance) carry genuine signal — confirmed by permutation importance — but require a
properly resourced hyperparameter search to demonstrate it in the headline metric.
 
---
 
## Performance
 
| Metric | Value |
|---|---|
| Baseline accuracy | 52.6% |
| Model accuracy | 73.98% |
| ROC-AUC | 0.7998 |
| Recall (Yes class) | 0.62 |
| Precision (Yes class) | 0.79 |
 
The model's weakest point is recall on the "yes" class — it misses 38% of actual
subscribers. This is the area most worth improving in future iterations.
 
---
 
## Requirements
 
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
 
## Running the Notebook
 
1. Place `bank.csv` in the same directory as the notebook
2. Run all cells top to bottom
The full pipeline including hyperparameter search runs in approximately 5–10 minutes.
To reduce runtime, lower `n_iter` in Sections 12 and 17.
 
---
 
## Limitations and Next Steps
 
- **n_iter constraints** — the hyperparameter searches were run with limited iterations
  to keep notebook runtime manageable. Running with n_iter=100+ on the V2 feature set
  would likely produce a cleaner comparison and potentially exceed V1's ROC-AUC
- **Threshold tuning not explored** — the default 0.5 threshold is not optimal for
  this problem; tuning toward higher recall on the "yes" class would better serve a
  campaign targeting use case where missing a potential subscriber is costly
- **XGBoost / LightGBM not tested** — unavailable in the notebook environment; either
  would be a natural next comparison given their strong performance on similar tabular
  classification problems
- **SHAP values** — permutation importance gives overall feature rankings but not
  per-prediction explanations; SHAP would give richer insight into individual customer
  scoring
- **Time-based validation** — the train/test split is random; a chronological split
  (train on earlier campaigns, validate on later ones) would give a more realistic
  estimate of how the model performs on future campaigns
---
 
## References
 
Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success
of bank telemarketing. Decision Support Systems, 62, 22–31.
 
UCI Machine Learning Repository — Bank Marketing Dataset.
https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
 
