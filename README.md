## Gold Extraction & Flotation

Built an end-to-end machine learning pipeline to **optimize gold recovery from ore** by predicting two key outputs:  
- `rougher.output.recovery` (recovery after flotation)  
- `final.output.recovery` (final recovery after purification)

### Data & Setup
- Datasets: `gold_recovery_train.csv`, `gold_recovery_test.csv`, `gold_recovery_full.csv`
- Time-series indexed by `date` (used as the DataFrame index)
- Train/test feature alignment using the intersection of columns (test has fewer features)

### Workflow
1. **Data validation & quality checks** (missing values, duplicates, timestamp integrity)
2. **Recovery formula verification** for rougher stage and error check using **MAE**
3. **Exploratory analysis & visualization** of metal concentrations (Au/Ag/Pb) across processing stages
4. **Preprocessing** with missing-value handling and **mean imputation**
5. **Modeling** using a **multi-output RandomForestRegressor**
6. **Evaluation** with a custom metric:
   - sMAPE for each target
   - Final score = `0.25 * sMAPE(rougher) + 0.75 * sMAPE(final)`
7. **Cross-validation & hyperparameter tuning** via `KFold` + `GridSearchCV`

### Result
Achieved a **final sMAPE ≈ 7.24** (lower is better), enabling accurate recovery predictions to support process optimization without running expensive physical experiments.

### Tools
Python, pandas, NumPy, matplotlib, scikit-learn (SimpleImputer, RandomForestRegressor, KFold, GridSearchCV, custom scoring)
