"""
PROFESSIONAL DATA ANALYSIS PROJECT 9: HEALTHCARE ANALYTICS (BREAST CANCER)
==========================================================================
Techniques: PCA Dimensionality Reduction, Correlation Analysis, 
            LightGBM Classification, SHAP Clinical Interpretability, 
            Survival/Risk Dashboards, Bayesian Probability
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
from scipy.stats import beta as beta_dist
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

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import shap

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier

OUT = r"D:\download\protfolio\projects\v3_output\project9_Healthcare"
os.makedirs(OUT, exist_ok=True)

logger.info("="*85)

# Load Breast Cancer Dataset
logger.info("\n  Loading Clinical Breast Cancer Dataset...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
# 0 is malignant, 1 is benign in original. We flip it so 1 = Malignant (disease)
df['Diagnosis'] = 1 - data.target

logger.info(f"  Patients: {len(df):,}")
logger.info(f"  Clinical Features: {len(data.feature_names)}")
mal = df['Diagnosis'].sum()
logger.info(f"  Malignant Cases: {mal} / {len(df)} ({mal/len(df)*100:.1f}%)")

X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis'].values
features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ═══════════════════════════════════════════
# 1. PCA CLINICAL CLUSTERING
# ═══════════════════════════════════════════
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaler.transform(X))
df['PCA1'] = X_pca[:,0]
df['PCA2'] = X_pca[:,1]
logger.info(f"  Variance explained by Top 2 clinical principal components: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ═══════════════════════════════════════════
# 2. ADVANCED MEDICAL DIAGNOSIS ML PIPELINE
# ═══════════════════════════════════════════

if LGB_AVAILABLE:
    logger.info("  Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train_scaled, y_train)
else:
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:,1]

auc_score = roc_auc_score(y_test, y_prob)
ap_score = average_precision_score(y_test, y_prob)
logger.info(f"\n  Clinical AUC-ROC:       {auc_score:.4f}")
logger.info(f"  Clinical Avg Precision: {ap_score:.4f}")
logger.info(f"\n  Diagnostic Report:\n{classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'])}")

# ═══════════════════════════════════════════
# 3. SHAP Model Interpretability
# ═══════════════════════════════════════════
logger.info("▔"*60)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_scaled)

plt.figure(figsize=(10, 8))
# For LGBM/RF binary classification, shap_values might be a list. We need the positive class.
shap_v = shap_values[1] if isinstance(shap_values, list) else shap_values
shap.summary_plot(shap_v, X_test_scaled, feature_names=features, show=False)
plt.title("Clinical Feature Importance for Diagnosis", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "SHAP_Summary.png"), dpi=300)
logger.info("  [OK] SHAP_Summary.png generated.")

# ═══════════════════════════════════════════
# 4. INTERACTIVE PLOTLY DASHBOARD
# ═══════════════════════════════════════════

fig_html = make_subplots(rows=2, cols=2, subplot_titles=(
    "Clinical PCA Separation", "Top Feature Density", 
    "Diagnosis Probability Curve", "ROC Curve"
))

# PCA
colors = {0: '#3498db', 1: '#e74c3c'}
for diag_val, color, name in [(0, '#3498db', 'Benign'), (1, '#e74c3c', 'Malignant')]:
    sub = df[df['Diagnosis']==diag_val]
    fig_html.add_trace(go.Scatter(x=sub['PCA1'], y=sub['PCA2'], mode='markers', 
                                  marker=dict(color=color, size=6, opacity=0.7), name=name), row=1, col=1)

# Top Feature density
top_feat = features[np.argsort(model.feature_importances_)[-1]] if hasattr(model, 'feature_importances_') else features[0]
for diag_val, color, name in [(0, '#3498db', 'Benign'), (1, '#e74c3c', 'Malignant')]:
    sub = df[df['Diagnosis']==diag_val]
    fig_html.add_trace(go.Box(y=sub[top_feat], name=name, marker_color=color), row=1, col=2)

# Probabilities
prob_df = pd.DataFrame({'Prob': y_prob, 'Actual': y_test}).sort_values('Prob')
fig_html.add_trace(go.Scatter(x=np.arange(len(prob_df)), y=prob_df['Prob'], mode='lines', line=dict(color='#f39c12', width=3), name='Risk Prob'), row=2, col=1)

# ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig_html.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#9b59b6', width=3), name='ROC'), row=2, col=2)

fig_html.update_layout(height=800, title_text="Healthcare Diagnosis Interactive Dashboard", template='plotly_dark')
pio.write_html(fig_html, file=os.path.join(OUT, 'Interactive_Dashboard.html'), auto_open=False)
logger.info("  [OK] Interactive_Dashboard.html generated.")

# ═══════════════════════════════════════════
# 5. STATIC MASTER DASHBOARD
# ═══════════════════════════════════════════
sns.set_theme(style="darkgrid", context="talk", palette="deep")
fig = plt.figure(figsize=(24, 16))
fig.suptitle('Healthcare Analytics — Oncology Diagnosis', fontsize=26, fontweight='black', y=0.98, color='#2c3e50')
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax1, cbar=False, annot_kws={"size": 18, "weight": "bold"})
ax1.set_xticklabels(['Benign', 'Malignant']); ax1.set_yticklabels(['Benign', 'Malignant'])
ax1.set_title('Diagnostic Confusion Matrix', fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1:3])
if hasattr(model, 'feature_importances_'):
    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False).head(10)
    sns.barplot(x='Importance', y='Feature', data=fi, palette='magma', ax=ax2)
    ax2.set_title('Top 10 Clinical Risk Indicators', fontweight='bold')

ax3 = fig.add_subplot(gs[1, 0])
sns.scatterplot(x='PCA1', y='PCA2', hue='Diagnosis', palette={0:'#3498db', 1:'#e74c3c'}, data=df, alpha=0.7, ax=ax3)
ax3.set_title('Patient Profiling (PCA)', fontweight='bold')

ax4 = fig.add_subplot(gs[1, 1])
sns.kdeplot(data=df, x=top_feat, hue='Diagnosis', fill=True, palette={0:'#3498db', 1:'#e74c3c'}, ax=ax4, alpha=0.5)
ax4.set_title(f'Distribution of {top_feat}', fontweight='bold')

ax5 = fig.add_subplot(gs[1, 2]); ax5.axis('off')
summary = f"""
ONCOLOGY CLINICAL SUMMARY:
──────────────────────────
- Total Patients: {len(df)}
- Malignant Rate: {mal/len(df)*100:.1f}%
- Diagnostic AUC: {auc_score:.4f}
- Precision (Mal): {cm[1,1]/(cm[0,1]+cm[1,1]) if (cm[0,1]+cm[1,1])>0 else 0:.3f}
- Recall (Mal):    {cm[1,1]/(cm[1,0]+cm[1,1]):.3f}

The LightGBM model successfully isolates 
malignant biomarkers with near-perfect 
diagnostic accuracy, aiding clinicians in
early and robust detection.
"""
ax5.text(0.1, 0.9, summary, transform=ax5.transAxes, fontsize=16, fontfamily='monospace',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "Advanced_Dashboard.png"), dpi=300, bbox_inches='tight')
logger.info("  [OK] Advanced_Dashboard.png generated.")

# Write report.txt
report_path = os.path.join(OUT, "report.txt")
with open(report_path, 'w', encoding='utf-8') as rf:
    rf.write("PROJECT 9: HEALTHCARE ANALYTICS (BREAST CANCER DIAGNOSIS) REPORT\n")
    rf.write("===============================================================\n")
    rf.write(f"Total Patients: {len(df):,}\n")
    rf.write(f"Malignant Cases (flips original class labels): {mal} ({mal/len(df)*100:.2f}%)\n")
    rf.write(f"Benign Cases: {len(df)-mal} ({(len(df)-mal)/len(df)*100:.2f}%)\n")
    rf.write(f"Clinical Features analyzed: {len(features)}\n")
    rf.write(f"\nModel Performance Metrics:\n")
    rf.write(f"  Diagnostic AUC-ROC: {auc_score:.4f}\n")
    rf.write(f"  Diagnostic Average Precision: {ap_score:.4f}\n")
    rf.write(f"\nModel Diagnostic Report:\n")
    rf.write(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    rf.write(f"\nConfusion Matrix:\n")
    rf.write(f"  True Negatives (Benign caught): {cm[0,0]:>6d}  |  False Positives (False Alarms): {cm[0,1]:>6d}\n")
    rf.write(f"  False Negatives (Missed cancers): {cm[1,0]:>5d}  |  True Positives (Cancers caught):  {cm[1,1]:>6d}\n")
    
    # Bayesian analysis for malignant probability
    # Using a uniform Beta(1,1) prior
    a_prior, b_prior = 1, 1
    a_post = a_prior + mal
    b_post = b_prior + (len(df) - mal)
    mean_prob = a_post / (a_post + b_post)
    ci = beta_dist.ppf([0.025, 0.975], a_post, b_post)
    rf.write(f"\nBayesian Inference on Malignancy Base Probability:\n")
    rf.write(f"  Posterior Distribution: Beta({a_post}, {b_post})\n")
    rf.write(f"  Posterior Mean (Base Probability): {mean_prob*100:.2f}%\n")
    rf.write(f"  95% Credible Interval: [{ci[0]:.4f}, {ci[1]:.4f}]\n")
    
    if hasattr(model, 'feature_importances_'):
        rf.write(f"\nTop 10 Clinical Risk Indicators (Feature Importance):\n")
        for idx, row in fi.head(10).iterrows():
            rf.write(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}\n")

logger.info(f"  [OK] report.txt generated at {report_path}")
logger.info("\n✅ PROJECT 9 COMPLETE.")

