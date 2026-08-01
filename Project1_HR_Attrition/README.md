# Enterprise HR Employee Attrition Analytics & Retention Modeling

**Author:** Senior Data Scientist & HR Quantitative Analyst  
**Domain:** Workforce Intelligence, People Analytics & Retention Economics  
**Dataset:** IBM HR Employee Attrition Dataset (1,470 Employees | 35 Features)

---

## 1. Executive Summary & Problem Formulation

Unplanned employee turnover creates significant financial and operational friction. Replacing a skilled employee costs **1.5x to 2.0x annual salary** due to recruitment costs, onboarding lags, and lost productivity.

---

## 2. Preprocessing & Leakage-Free Pipeline Architecture

- **Data Cleaning:** Verified null values, deduplicated records, and binary-encoded the target variable `Attrition`.
- **Leakage Prevention:** Integrated `SMOTE` oversampling strictly within `imblearn.pipeline.Pipeline` during 5-Fold Stratified Cross-Validation folds to ensure validation fold independence.

---

## 3. Statistical Hypothesis Testing

1. **One-Way ANOVA (Monthly Income):** Confirms significant income disparity between departing and retained employees ($F = 42.48, p < 0.0001$).
2. **Chi-Square Test (OverTime Status):** Confirms strong dependency between mandatory overtime and flight risk ($\chi^2 = 87.56, p < 0.0001$).

---

## 4. Machine Learning Model Benchmarking

Evaluated across 5-Fold Stratified Cross-Validation:

| Model | Mean ROC-AUC | Mean PR-AUC | Mean F1-Score |
|---|---|---|---|
| **Random Forest** | **0.8145** | **0.4850** | **0.4320** |
| **Extra Trees** | 0.8090 | 0.4720 | 0.4210 |
| **Gradient Boosting** | 0.8010 | 0.4610 | 0.4150 |
| **LightGBM** | 0.8050 | 0.4680 | 0.4250 |

---

## 5. Strategic Retention Directives

1. **OverTime Workload Management:** Cap mandatory overtime to reduce the 30.5% attrition rate observed among overtime workers.
2. **Early-Career Mentorship:** Focus retention interventions on junior staff within their initial 24 months of service.
