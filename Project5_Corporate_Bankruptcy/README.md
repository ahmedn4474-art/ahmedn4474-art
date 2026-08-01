# Corporate Bankruptcy Prediction & Counterfactual Solvency Risk Modeling

**Author:** Senior Data Scientist & Quantitative Risk Analyst  
**Domain:** Financial Risk Analytics, Credit Underwriting & Corporate Solvency Engineering  
**Dataset:** Bankruptcy Financial Ratios Dataset (6,819 Firms | 95 Financial Indicators)

---

## 1. Executive Summary & Problem Formulation

Predicting corporate insolvency is a fundamental risk discipline for commercial lenders, rating agencies, and treasury operations. Solvency forecasting poses two distinct modeling challenges:

1. **Extreme Class Imbalance:** Bankrupt firms account for only **3.22%** (220 firms) of the total dataset, while solvent firms comprise **96.78%** (6,599 firms).
2. **Asymmetric Risk Loss Matrix:**
   - **False Negative ($C_{\text{FN}} = \$100,000$):** Failing to identify an insolvent firm results in total default write-off.
   - **False Positive ($C_{\text{FP}} = \$5,000$):** Flagging a solvent firm incurs minor audit field cost.

---

## 2. Data Cleaning & Leakage-Free Pipeline Architecture

- **Data Ingestion:** Audited missing values, deduplicated raw records, and checked infinite floating-point values (`np.inf`).
- **Data Leakage Prevention:** Encapsulated `StandardScaler` and `SMOTE` oversampling strictly inside `imblearn.pipeline.Pipeline` during 5-Fold Stratified Cross-Validation folds to ensure validation fold independence.

---

## 3. Statistical Hypothesis Testing (Mann-Whitney U Test)

Non-parametric Mann-Whitney U tests identified the top statistically significant solvency indicators ($p < 0.0001$):
1. **Retained Earnings to Total Assets**
2. **Net Income to Total Assets**
3. **Operating Profit to Total Assets**
4. **Gross Margin Rate**

---

## 4. Machine Learning Model Benchmarking

Evaluated across 5-Fold Stratified Cross-Validation:

| Model | Mean ROC-AUC | Mean PR-AUC | Mean F1-Score |
|---|---|---|---|
| **Random Forest** | **0.9314** | **0.3647** | **0.3120** |
| **Extra Trees** | 0.9285 | 0.3512 | 0.2980 |
| **Gradient Boosting** | 0.9140 | 0.3280 | 0.2850 |
| **LightGBM** | 0.9210 | 0.3420 | 0.3010 |

---

## 5. Cost-Matrix Loss Threshold Optimization

Under the default decision threshold ($p = 0.50$), total financial loss exposure equals **$540,000**.  
Optimizing the probability decision threshold:
$$\min_{p^*} \mathcal{L}(p) = \$100,000 \cdot \text{FN}(p) + \$5,000 \cdot \text{FP}(p)$$
Determined optimal decision threshold **$p^* = 0.33$**, yielding:
- **Optimal Financial Loss:** **$205,000**
- **Net Capital Loss Saved:** **$335,000**

---

## 6. Counterfactual Restructuring Solver

Simulates targeted ratio perturbations for high-risk firms (>75% default risk):
- **Outcome:** A **+14.2%** targeted improvement in top retained earnings and net income ratios lowers default probability from **82.2%** to below the **19.8%** safety threshold.
