"""
Enterprise Audit Risk & Operational Anomaly Detection Pipeline
--------------------------------------------------------------
Author: Senior Data Scientist / Financial Data Specialist
Description: Statistical hypothesis testing, Isolation Forest anomaly scoring,
             and PCA dimensionality reduction for internal audit targeting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import xgboost as xgb

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

sns.set_theme(style="whitegrid", context="notebook")


def main():
    print("==========================================================================")
    print("   ENTERPRISE AUDIT RISK & OPERATIONAL ANOMALY DETECTION PIPELINE")
    print("==========================================================================")

    # 1. Load Data
    data_path = r"D:\download\protfolio\archive (2)\full_audit_dataset_with_security_operational.xlsx"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", "archive (2)", "full_audit_dataset_with_security_operational.xlsx")

    print(f"[Data Load] Loading operational audit dataset from {data_path}...")
    df = pd.read_excel(data_path, engine='openpyxl')

    target = 'RiskLevel'
    if target in df.columns and df[target].dtype == 'object':
        df['Target_Risk'] = df[target].apply(lambda x: 1 if str(x).strip().lower() == 'high' else 0)
        target = 'Target_Risk'

    le = LabelEncoder()
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if col != 'RiskLevel':
            df[col + '_Enc'] = le.fit_transform(df[col].astype(str))

    print(f"[Dataset Summary] Records: {df.shape[0]} | Columns: {df.shape[1]}")
    if target in df.columns:
        print(f"[Class Distribution] High Risk Audits: {df[target].mean():.2%}")

    # 2. Hypothesis Testing (Mann-Whitney U Test)
    if target in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors='ignore')
        risk_0 = df[df[target] == 0]
        risk_1 = df[df[target] == 1]

        p_values = {}
        for col in numeric_cols:
            if SCIPY_AVAILABLE and len(risk_0[col].dropna()) > 0 and len(risk_1[col].dropna()) > 0:
                stat, p = stats.mannwhitneyu(risk_0[col].dropna(), risk_1[col].dropna(), alternative='two-sided')
                p_values[col] = p

        if p_values:
            sorted_p = sorted(p_values.items(), key=lambda x: x[1])
            print("\n[Statistical Significance] Top 5 Differentiating Variables (Mann-Whitney U Test):")
            for col_name, p_val in sorted_p[:5]:
                print(f"   - {col_name:30s} | p-value: {p_val:.4e}")

    # 3. Isolation Forest Anomaly Detection
    feature_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols].fillna(0))

    iso = IsolationForest(contamination=0.05, random_state=42)
    anomaly_preds = iso.fit_predict(X_scaled)
    anomaly_scores = -iso.score_samples(X_scaled)
    df['Anomaly_Score'] = anomaly_scores
    df['Is_Anomaly'] = (anomaly_preds == -1).astype(int)

    print(f"\n[Anomaly Detection] Flagged {df['Is_Anomaly'].sum()} anomalous audit entries out of {len(df)} records.")

    # 4. Dimensionality Reduction (PCA)
    pca = PCA(n_components=3)
    pca_coords = pca.fit_transform(X_scaled)
    print(f"[PCA Analysis] 3 Principal Components explain {pca.explained_variance_ratio_.sum():.2%} of total variance.")

    print("\n==========================================================================")
    print("   AUDIT RISK PIPELINE EXECUTED SUCCESSFULLY")
    print("==========================================================================")


if __name__ == "__main__":
    main()
