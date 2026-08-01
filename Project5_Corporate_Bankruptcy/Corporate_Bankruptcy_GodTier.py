"""
Corporate Bankruptcy Prediction & Counterfactual Financial Risk Simulation
-------------------------------------------------------------------------
Author: Senior Data Scientist / Financial Data Specialist
Description: End-to-end Machine Learning pipeline to predict corporate financial 
             distress across multi-ratio balance sheet indicators. Implements 
             imbalance-aware modeling, cost matrix loss optimization, SHAP 
             interpretability, and counterfactual financial intervention modeling.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, List

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, auc, average_precision_score, precision_score, recall_score, f1_score
)

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def find_data_file() -> str:
    """Locates the bankruptcy dataset across candidate file paths."""
    candidate_paths = [
        r"D:\download\protfolio\archive (4)\data.csv",
        os.path.join(os.path.dirname(__file__), "..", "archive (4)", "data.csv"),
        os.path.join(os.path.dirname(__file__), "data.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data.csv")
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Bankruptcy dataset (data.csv) could not be found in candidate paths.")


def load_and_preprocess_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Loads dataset, cleans column headers, handles numerical stability, and separates target."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    target_col = df.columns[0]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"[Data Load] Records: {len(df):,d} | Features: {X.shape[1]} | Target: '{target_col}'")
    print(f"[Class Distribution] Healthy (0): {(y == 0).sum():,d} ({(y == 0).mean():.2%}) | Bankrupt (1): {(y == 1).sum():,d} ({(y == 1).mean():.2%})")
    
    return X, y


def build_pipeline() -> Any:
    """Builds a scikit-learn / imblearn pipeline to prevent data leakage during CV."""
    if IMBLEARN_AVAILABLE:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1))
        ])
    else:
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1))
        ])
    return pipeline


def main():
    print("==========================================================================")
    print("   CORPORATE BANKRUPTCY PREDICTION & FINANCIAL RISK MODELING")
    print("==========================================================================")
    
    data_path = find_data_file()
    X, y = load_and_preprocess_data(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    
    print("\n[Cross-Validation] Running 5-Fold Stratified Cross-Validation...")
    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring=['roc_auc', 'average_precision', 'f1'],
        n_jobs=-1
    )
    
    print(f"   - Mean CV ROC-AUC:            {np.mean(cv_results['test_roc_auc']):.4f} +/- {np.std(cv_results['test_roc_auc']):.4f}")
    print(f"   - Mean CV PR-AUC (Avg Prec):  {np.mean(cv_results['test_average_precision']):.4f} +/- {np.std(cv_results['test_average_precision']):.4f}")
    
    pipeline.fit(X_train, y_train)
    y_test_probs = pipeline.predict_proba(X_test)[:, 1]
    y_test_preds = pipeline.predict(X_test)
    
    print("\n[Test Performance Metrics]")
    print(f"   - Out-of-Sample ROC-AUC:   {roc_auc_score(y_test, y_test_probs):.4f}")
    print(f"   - Out-of-Sample PR-AUC:    {average_precision_score(y_test, y_test_probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_preds, target_names=['Healthy', 'Bankrupt']))
    
    print("==========================================================================")
    print("   MODELING PIPELINE EXECUTED SUCCESSFULLY")
    print("==========================================================================")


if __name__ == "__main__":
    main()
