# Optimización de recuperación de oro 🏭✨

Proyecto de *Machine Learning* para **predecir la recuperación de oro** en un proceso industrial (etapas *rougher* y *final*) con el fin de apoyar la **optimización de producción** y una mejor toma de decisiones.

---

## Objetivo 🎯
Predecir:
- `rougher.output.recovery`
- `final.output.recovery`

Usando variables del proceso (concentraciones, tamaños de partícula, tasas de alimentación, etc.).

---

## Datos 📦
Archivos utilizados:
- `gold_recovery_train.csv` → **(16860, 86)**
- `gold_recovery_test.csv` → **(5856, 52)**
- `gold_recovery_full.csv` → **(22716, 86)**

⚠️ El **conjunto de prueba tiene menos columnas**, así que las features se alinean usando la **intersección** entre train y test (52 features compartidas).

---

## Verificación de fórmula de recuperación ✅
Validé la fórmula de recuperación del proyecto contra `rougher.output.recovery`:

- **MAE ≈ 9.30e-15** (prácticamente 0)  
Esto confirma que la columna objetivo es consistente con el cálculo esperado.

---

## Métrica 📏
La evaluación usa **sMAPE** y un score final ponderado:

- `sMAPE(rougher)`
- `sMAPE(final)`
- `sMAPE_final = 0.25 * sMAPE(rougher) + 0.75 * sMAPE(final)`

---

## Enfoque 🧠
1. Cargar y explorar los datasets
2. Limpiar datos (valores faltantes, checks de consistencia, revisión de distribuciones)
3. Alinear features entre train y test
4. Entrenar un modelo **multi-output** (2 targets)
5. Evaluar con split train/validación + cross-validation
6. Ajuste de hiperparámetros con `GridSearchCV` (estilo nested-CV)

---

## Resultados 📈
Modelo principal: **RandomForestRegressor (multi-output)**

- Baseline (sin tuning):  
  - `sMAPE_final ≈ 10.1897`

- Mejor modelo ajustado:  
  - `n_estimators=700, max_depth=None, min_samples_split=2`  
  - `sMAPE(rougher) ≈ 8.035`  
  - `sMAPE(final) ≈ 6.9762`  
  - ✅ **sMAPE_final ≈ 7.2409**

---

## Estructura sugerida del repositorio 🗂️

```text
.
├── data/
│   ├── gold_recovery_train.csv
│   ├── gold_recovery_test.csv
│   └── gold_recovery_full.csv
├── notebooks/
│   └── gold-recovery-optimization.ipynb
├── src/                      # (opcional) funciones reutilizables
│   ├── metrics.py
│   └── utils.py
├── README.md
└── requirements.txt
