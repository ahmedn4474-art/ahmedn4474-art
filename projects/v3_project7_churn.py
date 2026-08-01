"""
PROFESSIONAL DATA ANALYSIS PROJECT 7: CUSTOMER CHURN PREDICTION
===============================================================
Techniques: EDA, SMOTE, CatBoost/XGBoost, SHAP, Interactive Plotly Dashboards,
            Survival Analysis Concepts, LTV Analysis
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
warnings.filterwarnings('ignore')

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import shap

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

OUT = r"D:\download\protfolio\projects\v3_output\project7_Churn"
os.makedirs(OUT, exist_ok=True)

logger.info("="*85)

# Load Telco Churn Dataset from GitHub
logger.info("\n  Downloading Telco Customer Churn dataset...")
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Clean Data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna().reset_index(drop=True)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
df = df.drop(columns=['customerID'])

logger.info(f"  Customers: {len(df):,}")
logger.info(f"  Churn Rate: {df['Churn'].mean()*100:.2f}%")

# ═══════════════════════════════════════════
# 1. ADVANCED ML PIPELINE
# ═══════════════════════════════════════════
logger.info("▔"*60)

df_ml = df.copy()
le_dict = {}
for col in df_ml.select_dtypes('object').columns:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    le_dict[col] = le

X = df_ml.drop(columns=['Churn'])
y = df_ml['Churn'].values
features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logger.info("  Applying SMOTE for Class Imbalance...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

if XGB_AVAILABLE:
    logger.info("  Training XGBoost Classifier...")
    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=42)
    model.fit(X_train_res, y_train_res)
else:
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:,1]

auc_score = roc_auc_score(y_test, y_prob)
ap_score = average_precision_score(y_test, y_prob)
logger.info(f"\n  AUC-ROC:       {auc_score:.4f}")
logger.info(f"  Avg Precision: {ap_score:.4f}")

# ═══════════════════════════════════════════
# 2. SHAP Model Interpretability
# ═══════════════════════════════════════════
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_scaled, feature_names=features, show=False)
plt.title("SHAP Feature Importance (What drives Churn?)", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "SHAP_Summary.png"), dpi=300)
logger.info("  [OK] SHAP_Summary.png generated.")

# ═══════════════════════════════════════════
# 3. INTERACTIVE PLOTLY DASHBOARD
# ═══════════════════════════════════════════

fig_html = make_subplots(rows=2, cols=2, subplot_titles=(
    "Churn Rate", "Tenure vs Monthly Charges (Survival Proxy)", 
    "Contract Type vs Churn", "ROC Curve"
), specs=[[{"type": "domain"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]])

fig_html.add_trace(go.Pie(labels=['Retained', 'Churned'], values=df['Churn'].value_counts(), hole=0.5, 
                          marker_colors=['#3498db', '#e74c3c']), row=1, col=1)

df_sample = df.sample(1000, random_state=42)
for churn_val, color, name in [(0, '#3498db', 'Retained'), (1, '#e74c3c', 'Churned')]:
    sub = df_sample[df_sample['Churn']==churn_val]
    fig_html.add_trace(go.Scatter(x=sub['tenure'], y=sub['MonthlyCharges'], mode='markers', 
                                  marker=dict(color=color, size=6, opacity=0.7), name=name), row=1, col=2)

contract_churn = df.groupby('Contract')['Churn'].mean().reset_index()
fig_html.add_trace(go.Bar(x=contract_churn['Contract'], y=contract_churn['Churn'], marker_color=['#1abc9c', '#f1c40f', '#e67e22']), row=2, col=1)

fpr, tpr, _ = roc_curve(y_test, y_prob)
fig_html.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#9b59b6', width=3), name='ROC'), row=2, col=2)

fig_html.update_layout(height=800, title_text="Customer Churn Interactive Dashboard", template='plotly_dark')
pio.write_html(fig_html, file=os.path.join(OUT, 'Interactive_Dashboard.html'), auto_open=False)
logger.info("  [OK] Interactive_Dashboard.html generated.")

# ═══════════════════════════════════════════
# 4. STATIC MASTER DASHBOARD
# ═══════════════════════════════════════════
sns.set_theme(style="darkgrid", context="talk", palette="deep")
fig = plt.figure(figsize=(24, 16))
fig.suptitle('Telco Customer Churn — Advanced Analytics', fontsize=26, fontweight='black', y=0.98, color='#2c3e50')
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax1, cbar=False, annot_kws={"size": 18, "weight": "bold"})
ax1.set_xticklabels(['Retained', 'Churned']); ax1.set_yticklabels(['Retained', 'Churned'])
ax1.set_title('Confusion Matrix', fontweight='bold'); ax1.set_ylabel('Actual'); ax1.set_xlabel('Predicted')

ax2 = fig.add_subplot(gs[0, 1:3])
fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False).head(10)
sns.barplot(x='Importance', y='Feature', data=fi, palette='viridis', ax=ax2)
ax2.set_title('Top 10 Churn Drivers (XGBoost)', fontweight='bold')

ax3 = fig.add_subplot(gs[1, 0])
sns.kdeplot(data=df, x='tenure', hue='Churn', fill=True, palette={0:'#3498db', 1:'#e74c3c'}, ax=ax3, alpha=0.5)
ax3.set_title('Tenure Distribution by Churn', fontweight='bold')

ax4 = fig.add_subplot(gs[1, 1])
sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, palette={0:'#3498db', 1:'#e74c3c'}, ax=ax4, alpha=0.5)
ax4.set_title('Monthly Charges Distribution', fontweight='bold')

ax5 = fig.add_subplot(gs[1, 2]); ax5.axis('off')
summary = f"""
KEY CHURN INSIGHTS:
───────────────────
- Base Churn Rate: {df['Churn'].mean()*100:.1f}%
- Model AUC-ROC: {auc_score:.3f}
- Highest Risk: Month-to-Month contracts
  with high Monthly Charges.
- Retention Strategy: Convert high-risk
  customers to 1-Year contracts.
"""
ax5.text(0.1, 0.8, summary, transform=ax5.transAxes, fontsize=16, fontfamily='monospace',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "Advanced_Dashboard.png"), dpi=300, bbox_inches='tight')
logger.info("  [OK] Advanced_Dashboard.png generated.")

# Write report.txt
report_path = os.path.join(OUT, "report.txt")
with open(report_path, 'w', encoding='utf-8') as rf:
    rf.write("PROJECT 7: CUSTOMER CHURN PREDICTION REPORT\n")
    rf.write("===========================================\n")
    rf.write(f"Total Customers: {len(df):,}\n")
    rf.write(f"Churn Rate: {df['Churn'].mean()*100:.2f}%\n")
    rf.write(f"Class Distribution: Churned={df['Churn'].sum()} | Retained={len(df)-df['Churn'].sum()}\n")
    rf.write(f"\nModel Performance Metrics:\n")
    rf.write(f"  AUC-ROC: {auc_score:.4f}\n")
    rf.write(f"  Average Precision: {ap_score:.4f}\n")
    rf.write(f"\nClassification Report:\n")
    rf.write(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
    
    if not fi.empty:
        rf.write(f"\nTop 10 Feature Importances:\n")
        for idx, row in fi.head(10).iterrows():
            rf.write(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}\n")

logger.info(f"  [OK] report.txt generated at {report_path}")
logger.info("\n✅ PROJECT 7 COMPLETE.")

