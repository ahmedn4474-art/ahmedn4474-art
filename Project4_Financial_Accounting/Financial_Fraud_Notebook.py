"""
Financial Accounting Analytics & Transaction Anomaly Detection
--------------------------------------------------------------
Author: Senior Data Scientist / Financial Data Specialist
Description: Ledger transaction parsing, Network Graph structure analysis,
             and Isolation Forest anomaly detection for accounting compliance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

sns.set_theme(style="whitegrid", context="notebook")


def main():
    print("==========================================================================")
    print("   FINANCIAL ACCOUNTING ANALYTICS & ANOMALY DETECTION PIPELINE")
    print("==========================================================================")

    # 1. Load Data
    data_path = r"D:\download\protfolio\archive (3)\financial_accounting.csv"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", "archive (3)", "financial_accounting.csv")

    print(f"[Data Load] Loading financial transaction dataset from {data_path}...")
    df = pd.read_csv(data_path)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    print(f"[Dataset Summary] Records: {len(df):,d} | Columns: {df.shape[1]}")

    # 2. Network Analysis (Account to Vendor Bipartite Graph)
    if 'Customer_Vendor' in df.columns and 'Account' in df.columns:
        sample_df = df.dropna(subset=['Customer_Vendor', 'Account']).sample(min(300, len(df)), random_state=42)
        B = nx.Graph()
        B.add_nodes_from(sample_df['Account'].unique(), bipartite=0)
        B.add_nodes_from(sample_df['Customer_Vendor'].unique(), bipartite=1)
        edges = list(zip(sample_df['Account'], sample_df['Customer_Vendor']))
        B.add_edges_from(edges)

        top_nodes = {n for n, d in B.nodes(data=True) if d['bipartite'] == 0}
        print(f"[Graph Analysis] Constructed Bipartite Ledger Network with {len(top_nodes)} internal accounts and {len(set(B) - top_nodes)} vendors.")

    # 3. Isolation Forest Anomaly Detection
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[num_cols].fillna(0))

        iso = IsolationForest(contamination=0.03, random_state=42)
        df['Anomaly_Flag'] = (iso.fit_predict(X_scaled) == -1).astype(int)
        print(f"[Anomaly Detection] Flagged {df['Anomaly_Flag'].sum():,d} anomalous transactions out of {len(df):,d} total entries.")

    print("\n==========================================================================")
    print("   FINANCIAL ACCOUNTING PIPELINE EXECUTED SUCCESSFULLY")
    print("==========================================================================")


if __name__ == "__main__":
    main()
