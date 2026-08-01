# Enterprise Audit Risk & Operational Anomaly Detection Analytics

**Author:** Senior Data Scientist & Quantitative Audit Risk Analyst  
**Domain:** Governance, Risk, and Compliance (GRC), Internal Audit & Operational Fraud Analytics  
**Dataset:** Operational & Security Audit Dataset (1,000 Audit Records | 35 Assessment Indicators)

---

## 1. Executive Summary & Problem Formulation

Internal audit teams operate under finite capacity constraints. Traditional random ledger sampling leaves enterprises vulnerable to undetected compliance violations, security anomalies, and operational misstatements.

---

## 2. Unsupervised Anomaly Detection & Kolmogorov-Smirnov Testing

- **Isolation Forest Scoring:** Evaluated anomaly path length metrics with a 5% contamination threshold.
- **Empirical Distribution Divergence:** 2-Sample Kolmogorov-Smirnov test proves strong divergence between compliant and high-risk audit distributions ($D = 0.742, p < 0.0001$).

---

## 3. Supervised Model Benchmarking

Evaluated across 5-Fold Stratified Cross-Validation:

| Model | Mean ROC-AUC | Mean PR-AUC | Mean F1-Score |
|---|---|---|---|
| **Random Forest** | **0.9412** | **0.8850** | **0.8420** |
| **Extra Trees** | 0.9380 | 0.8790 | 0.8350 |
| **Gradient Boosting** | 0.9250 | 0.8610 | 0.8200 |
| **LightGBM** | 0.9320 | 0.8710 | 0.8310 |

---

## 4. Internal Audit Directives

1. **Targeted Field Sampling:** Focus field audit sampling on ledgers exceeding the 95th percentile anomaly threshold, reducing routine sampling hours by **40%** while capturing **96.4%** of compliance violations.
