# Optimización de recuperación de oro (Sprint 13)

Modelo de Machine Learning para **predecir la recuperación de oro** en un proceso industrial (etapas *rougher* y *final*), con el objetivo de **optimizar la producción** y apoyar la toma de decisiones en planta.

---

## Objetivo
Predecir:
- `rougher.output.recovery`
- `final.output.recovery`

A partir de variables del proceso (concentraciones, tamaños de partícula, tasas de alimentación, etc.).

---

## Datos
Se utilizan tres archivos:
- `gold_recovery_train.csv` → **(16860, 86)**
- `gold_recovery_test.csv` → **(5856, 52)**
- `gold_recovery_full.csv` → **(22716, 86)**

> Nota: el conjunto **test tiene menos columnas**, por lo que se alinean *features* usando la **intersección** de columnas entre train y test (52 variables).

---

## Validación del cálculo de recuperación
Se comprobó el cálculo de `rougher.output.recovery` con la fórmula del proyecto y se comparó contra el valor real:

- **MAE ≈ 9.30e-15** (prácticamente 0)  
Esto confirma que el cálculo y la columna objetivo son consistentes.

---

## Métrica
Se evalúa con **sMAPE** y un **sMAPE final** ponderado:

- `sMAPE(rougher)`
- `sMAPE(final)`
- `sMAPE_final = 0.25 * sMAPE(rougher) + 0.75 * sMAPE(final)`

---

## Enfoque
1. Carga y exploración de datos.
2. Limpieza (valores nulos, consistencia, revisión de distribuciones).
3. Alineación de features entre train y test.
4. Entrenamiento **multi-output** (dos targets).
5. Evaluación con split train/valid y validación cruzada.
6. Búsqueda de hiperparámetros con `GridSearchCV` y enfoque tipo *nested CV*.

---

## Resultados
Modelo principal: **RandomForestRegressor (multi-output)**

- Baseline (sin tuning):  
  - `sMAPE_final ≈ 10.1897`

- Mejor modelo (tuning):  
  - `n_estimators=700, max_depth=None, min_samples_split=2`  
  - `sMAPE(rougher) ≈ 8.035`  
  - `sMAPE(final) ≈ 6.9762`  
  - ✅ **sMAPE_final ≈ 7.2409**

---

## Estructura sugerida del repo
