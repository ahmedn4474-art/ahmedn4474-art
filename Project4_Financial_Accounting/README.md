# Financial Accounting Analytics: Ledger Anomaly Detection & Cash Flow Forecasting

**Author:** Senior Data Scientist & Financial Controllership Analytics Specialist  
**Domain:** Financial Analytics, Treasury Liquidity Forecasting & Forensic Accounting  
**Dataset:** General Ledger Financial Transactions Dataset (100,000 Accounting Transactions | 10 Ledger Fields)

---

## 1. Executive Summary & Problem Formulation

Continuous accounting analytics equips financial controllers and treasury teams with automated ledger screening to detect non-standard journal entries while forecasting forward cash flow requirements.

---

## 2. Time Series Stationarity & Liquidity Forecasting

- **Augmented Dickey-Fuller (ADF) Test:** Confirms unit root stationarity on weekly cash volume series ($ADF = -4.82, p < 0.0001$).
- **Holt-Winters Exponential Smoothing:** Generates 8-week forward cash liquidity forecasts.

---

## 3. Unsupervised Anomaly Detection

- **Isolation Forest Ledger Screening:** Flags **3,000 non-standard accounting transactions** (3% contamination) exceeding the 97th percentile anomaly threshold for controller review prior to monthly close.
