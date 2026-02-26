# Gold Recovery Optimization 

A Machine Learning project to **predict gold recovery** in an industrial process (*rougher* and *final* stages) to support **production optimization** and better decision-making.

---

## Goal 
Predict:
- `rougher.output.recovery`
- `final.output.recovery`

Using process features (concentrations, particle sizes, feed rates, etc.).

---

## Data 
Files used:
- `gold_recovery_train.csv` → **(16860, 86)**
- `gold_recovery_test.csv` → **(5856, 52)**
- `gold_recovery_full.csv` → **(22716, 86)**

⚠️ The **test set has fewer columns**, so features are aligned using the **intersection** of columns between train and test (52 shared features).

---

## Recovery Formula Check 
I validated the project’s recovery formula against `rougher.output.recovery`:

- **MAE ≈ 9.30e-15** (basically 0)  
This confirms the target column is consistent with the expected calculation.

---

## Metric 
Evaluation uses **sMAPE** and a weighted final score:

- `sMAPE(rougher)`
- `sMAPE(final)`
- `sMAPE_final = 0.25 * sMAPE(rougher) + 0.75 * sMAPE(final)`

---

## Approach 
1. Load + explore the datasets
2. Clean data (missing values, consistency checks, distribution review)
3. Align features between train and test
4. Train a **multi-output** model (2 targets)
5. Evaluate with train/validation split + cross-validation
6. Hyperparameter tuning with `GridSearchCV` (nested-CV style)

---

## Results 
Main model: **RandomForestRegressor (multi-output)**

- Baseline (no tuning):  
  - `sMAPE_final ≈ 10.1897`

- Best tuned model:  
  - `n_estimators=700, max_depth=None, min_samples_split=2`  
  - `sMAPE(rougher) ≈ 8.035`  
  - `sMAPE(final) ≈ 6.9762`  
  - **sMAPE_final ≈ 7.2409**

---
 
