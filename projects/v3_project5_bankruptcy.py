"""
Data Analysis and Machine Learning Pipeline
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import shap
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier

OUT = r"D:\download\protfolio\projects\v3_output\project5_Bankruptcy"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("D:\\download\\protfolio\\archive (4)\\data.csv")
df.columns = [c.strip() for c in df.columns]
logger.info(f"\n  Companies: {len(df):,}")
logger.info(f"  Financial Ratios: {len(df.columns)-1}")

target_col = [c for c in df.columns if 'bankrupt' in c.lower() or 'class' in c.lower() or 'status' in c.lower()]
y_name = target_col[0] if target_col else df.columns[0]
if df[y_name].dtype in ['object','bool']:
    df[y_name] = (df[y_name].astype(str).str.lower().str.contains('yes|true|1|bankrupt')).astype(int)

bkr = df[y_name].sum(); tot = len(df)
logger.info(f"\n  Bankrupt: {bkr} / {tot} ({bkr/tot*100:.2f}%)")

# ═══════════════════════════════════════════
# 1. ADVANCED ML PIPELINE (SMOTE + XGBOOST)
# ═══════════════════════════════════════════

features = [c for c in df.columns if c!= y_name]
X = df[features].select_dtypes(include=[np.number]).fillna(0)
y = df[y_name].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logger.info("\n  Applying SMOTE for Imbalanced Classes...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

if XGB_AVAILABLE:
    logger.info("  Training XGBoost Classifier...")
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, 
                          eval_metric='logloss', use_label_encoder=False, random_state=42)
    model.fit(X_train_res, y_train_res)
else:
    logger.info("  Training Random Forest...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_res, y_train_res)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:,1]

auc_score = roc_auc_score(y_test, y_prob)
ap_score = average_precision_score(y_test, y_prob)
logger.info(f"\n  Final Model AUC-ROC:       {auc_score:.4f}")
logger.info(f"  Final Model Avg Precision: {ap_score:.4f}")
logger.info(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Stable', 'Bankrupt'])}")

# ═══════════════════════════════════════════
# 2. SHAP Model Interpretability
# ═══════════════════════════════════════════
try:
    explainer = shap.TreeExplainer(model)
    # Use a sample for SHAP to avoid memory issues
    X_sample = X_test_scaled[:500] if len(X_test_scaled)>500 else X_test_scaled
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=features, show=False)
    plt.title("SHAP Feature Importance (Top Predictors of Bankruptcy)", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "SHAP_Summary.png"), dpi=300)
    logger.info("  [OK] SHAP_Summary.png generated.")
except Exception as e:
    logger.info(f"  SHAP Error: {e}")

# ═══════════════════════════════════════════
# 3. INTERACTIVE PLOTLY DASHBOARD
# ═══════════════════════════════════════════
# Feature importance
if hasattr(model, 'feature_importances_'):
    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False).head(15)
else:
    fi = pd.DataFrame()

fig_html = make_subplots(rows=2, cols=2, subplot_titles=(
    "Bankruptcy Rate", "Top 15 Predictors of Bankruptcy", 
    "ROC Curve", "Precision-Recall Curve"
), specs=[[{"type": "domain"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]])

fig_html.add_trace(go.Pie(labels=['Stable', 'Bankrupt'], values=[tot-bkr, bkr], hole=0.5, 
                          marker_colors=['#2ecc71', '#e74c3c']), row=1, col=1)

if not fi.empty:
    fig_html.add_trace(go.Bar(x=fi['Importance'][::-1], y=fi['Feature'].str[:30][::-1], orientation='h', marker_color='#9b59b6'), row=1, col=2)

fpr, tpr, _ = roc_curve(y_test, y_prob)
fig_html.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={auc_score:.2f}', line=dict(color='#e74c3c', width=3)), row=2, col=1)
fig_html.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(color='gray', dash='dash')), row=2, col=1)

prec, rec, _ = precision_recall_curve(y_test, y_prob)
fig_html.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name=f'AP={ap_score:.2f}', line=dict(color='#3498db', width=3)), row=2, col=2)

fig_html.update_layout(height=800, title_text="Bankruptcy Prediction Interactive Dashboard", showlegend=False, template='plotly_dark')
pio.write_html(fig_html, file=os.path.join(OUT, 'Interactive_Dashboard.html'), auto_open=False)
logger.info("  [OK] Interactive_Dashboard.html generated.")

# ═══════════════════════════════════════════
# 4. PROFESSIONAL STATIC MASTER DASHBOARD
# ═══════════════════════════════════════════
sns.set_theme(style="darkgrid", context="talk", palette="deep")
fig = plt.figure(figsize=(24, 16))
fig.suptitle('Bankruptcy Prediction — Advanced Analytics', fontsize=26, fontweight='black', y=0.98, color='#2c3e50')
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

# 1. Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax1, cbar=False, annot_kws={"size": 18, "weight": "bold"})
ax1.set_xticklabels(['Stable', 'Bankrupt']); ax1.set_yticklabels(['Stable', 'Bankrupt'])
ax1.set_title('Confusion Matrix (XGBoost)', fontweight='bold'); ax1.set_ylabel('Actual'); ax1.set_xlabel('Predicted')

# 2. Feature Importance
ax2 = fig.add_subplot(gs[0, 1:3])
if not fi.empty:
    sns.barplot(x='Importance', y='Feature', data=fi.head(10), palette='viridis', ax=ax2)
    ax2.set_yticklabels(fi.head(10)['Feature'].str[:40])
ax2.set_title('Top 10 Risk Factors (XGBoost Importance)', fontweight='bold')

# 3. ROC Curve
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'ROC (AUC = {auc_score:.3f})')
ax3.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
ax3.set_title('ROC Curve', fontweight='bold'); ax3.legend()

# 4. PR Curve
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(rec, prec, color='#8e44ad', lw=3, label=f'PR (AP = {ap_score:.3f})')
ax4.set_title('Precision-Recall Curve', fontweight='bold'); ax4.legend()

# 5. Top feature distribution
ax5 = fig.add_subplot(gs[1, 2])
if not fi.empty:
    top_feat = fi.iloc[0]['Feature']
    sns.kdeplot(data=df, x=top_feat, hue=y_name, fill=True, palette={0:'#2ecc71', 1:'#e74c3c'}, ax=ax5, clip=(df[top_feat].quantile(0.01), df[top_feat].quantile(0.99)))
    ax5.set_title(f'Distribution of Top Predictor\n({top_feat[:30]}...)', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "Advanced_Dashboard.png"), dpi=300, bbox_inches='tight')
logger.info("  [OK] Advanced_Dashboard.png generated.")

# Write report.txt
report_path = os.path.join(OUT, "report.txt")
with open(report_path, 'w', encoding='utf-8') as rf:
    rf.write("PROJECT 5: CORPORATE BANKRUPTCY PREDICTION REPORT\n")
    rf.write("================================================\n")
    rf.write(f"Total Companies analyzed: {tot:,}\n")
    rf.write(f"Bankrupt Companies: {bkr} ({bkr/tot*100:.2f}%)\n")
    rf.write(f"Stable Companies: {tot-bkr} ({(tot-bkr)/tot*100:.2f}%)\n")
    rf.write(f"\nModel Performance Metrics:\n")
    rf.write(f"  AUC-ROC: {auc_score:.4f}\n")
    rf.write(f"  Average Precision: {ap_score:.4f}\n")
    rf.write(f"\nClassification Report:\n")
    rf.write(classification_report(y_test, y_pred, target_names=['Stable', 'Bankrupt']))
    if not fi.empty:
        rf.write(f"\nTop 10 Bankruptcy Predictors:\n")
        for idx, row in fi.head(10).iterrows():
            rf.write(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}\n")

logger.info(f"  [OK] report.txt generated at {report_path}")
logger.info("\n✅ PROJECT 5 COMPLETE.")
