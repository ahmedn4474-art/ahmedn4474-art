"""
Data Analysis and Machine Learning Pipeline
"""
import pandas as pd, numpy as np, os, warnings, subprocess, sys
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

# Reconfigure console output to UTF-8
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


OUT = r"D:\download\protfolio\projects\v3_output\project3_Audit"
os.makedirs(OUT, exist_ok=True)

report_path = os.path.join(OUT, "report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("PROJECT 3: AUDIT & SECURITY RISK ANALYSIS REPORT\n")
    f.write("==============================================\n\n")

def w(text=""):
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(str(text) + "\n")
    print(text)

# ── Load Data ──
logger.info("\nLoading audit dataset via PowerShell extraction...")
ps = '''
$e=New-Object -ComObject Excel.Application;$e.Visible=$false
$w=$e.Workbooks.Open("D:\\download\\protfolio\\archive (2)\\full_audit_dataset_with_security_operational.xlsx")
$s=$w.Sheets.Item(1);$p=[System.IO.Path]::GetTempFileName()+".csv"
$s.SaveAs($p,6);$w.Close($false);$e.Quit()
[Runtime.Interopservices.Marshal]::ReleaseComObject($e)|Out-Null
Write-Host $p
'''
res = subprocess.run(['powershell','-NoProfile','-Command',ps],capture_output=True,text=True)
cp = res.stdout.strip().split('\n')[-1].strip()
df = pd.read_csv(cp); os.remove(cp)
logger.info(f"  Loaded: {len(df)} audits x {len(df.columns)} cols")

cat_feats = ['AuditType','RiskLevel','AuditStatus']
num_feats = ['AuditScore','Variance','Duration','ErrorRate','CompletionPercentage','AuditCost','RiskFactor']
sec_cols = [c for c in df.columns if c not in num_feats + cat_feats + ['AuditID','DataValue','Timestamp']]

# ═══════════════════════════════════════════
# 1. ADVANCED RISK SCORING & CLUSTERING
# ═══════════════════════════════════════════

# 1.1 Custom Risk Score
df['risk_score'] = (df['AuditScore'].rank(pct=True) * 0.3 +
                    df['Variance'].rank(pct=True) * 0.2 +
                    (1-df['CompletionPercentage'].rank(pct=True)) * 0.2 +
                    df['AuditCost'].rank(pct=True) * 0.15 +
                    df['ErrorRate'].rank(pct=True) * 0.15)
df['risk_tier'] = pd.cut(df['risk_score'], bins=[0, 0.25, 0.5, 0.75, 1], labels=['Low','Medium','High','Critical'])

# 1.2 PCA on Security Controls
le = LabelEncoder()
sec_df = df[sec_cols].apply(lambda col: le.fit_transform(col.astype(str)))
scaler = StandardScaler()
sec_scaled = scaler.fit_transform(sec_df)

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(sec_scaled)
df['PCA1'] = pca_fit[:, 0]
df['PCA2'] = pca_fit[:, 1]
df['PCA3'] = pca_fit[:, 2]

# 1.3 K-Means Clustering for Risk Profiles
kmeans = KMeans(n_clusters=4, random_state=42)
df['Risk_Cluster'] = kmeans.fit_predict(pca_fit)
df['Risk_Cluster'] = 'Profile ' + df['Risk_Cluster'].astype(str)
logger.info(f"  K-Means Clustering complete. Identified {df['Risk_Cluster'].nunique()} distinct audit profiles.")

# 1.4 Isolation Forest for Anomaly Detection (Fraud/High Risk Flags)
iso = IsolationForest(contamination=0.05, random_state=42)
df['Anomaly'] = iso.fit_predict(scaler.fit_transform(df[num_feats].fillna(0)))
df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})
logger.info(f"  Isolation Forest flagged {(df['Anomaly']=='Anomaly').sum()} audits as severe anomalies.")

# ═══════════════════════════════════════════
# 2. INTERACTIVE PLOTLY DASHBOARD
# ═══════════════════════════════════════════

fig_html = make_subplots(rows=2, cols=2, subplot_titles=(
    "3D PCA - Security Control Profiles", "Audit Risk Score Distribution", 
    "Anomaly Detection (Cost vs Variance)", "Audit Status by Risk Level"
), specs=[[{"type": "scatter3d"}, {"type": "xy"}], [{"type": "xy"}, {"type": "domain"}]])

# 1. 3D PCA
scatter_3d = go.Scatter3d(x=df['PCA1'], y=df['PCA2'], z=df['PCA3'], mode='markers',
                          marker=dict(size=4, color=df['risk_score'], colorscale='Viridis', opacity=0.8),
                          text=df['RiskLevel'])
fig_html.add_trace(scatter_3d, row=1, col=1)

# 2. Risk Score Histogram
fig_html.add_trace(go.Histogram(x=df['risk_score'], nbinsx=30, marker_color='#9b59b6'), row=1, col=2)

# 3. Anomaly Scatter
anom_colors = {'Normal': '#3498db', 'Anomaly': '#e74c3c'}
for anom in ['Normal', 'Anomaly']:
    sub = df[df['Anomaly']==anom]
    fig_html.add_trace(go.Scatter(x=sub['Variance'], y=sub['AuditCost'], mode='markers',
                                  name=anom, marker_color=anom_colors[anom]), row=2, col=1)

# 4. Sunburst (Approximated as Pie for subplot compatibility, or we just do Bar)
fig_html.add_trace(go.Pie(labels=df['RiskLevel'].value_counts().index, values=df['RiskLevel'].value_counts().values,
                          hole=0.4), row=2, col=2)

fig_html.update_layout(height=900, title_text="Audit & Security Interactive Dashboard", template='plotly_dark')
pio.write_html(fig_html, file=os.path.join(OUT, 'Interactive_Dashboard.html'), auto_open=False)
logger.info("  [OK] Interactive_Dashboard.html generated.")

# ═══════════════════════════════════════════
# 3. PROFESSIONAL STATIC DASHBOARD ()
# ═══════════════════════════════════════════
sns.set_theme(style="darkgrid", context="talk", palette="deep")
fig = plt.figure(figsize=(24, 18))
fig.suptitle('Audit & Security Analytics — Advanced Risk Assessment', fontsize=26, fontweight='black', y=0.98, color='#2c3e50')
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# 1. K-Means Clusters on PCA
ax1 = fig.add_subplot(gs[0, 0])
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Risk_Cluster', palette='Set1', s=80, alpha=0.7, ax=ax1)
ax1.set_title('K-Means Clusters on Security Controls (PCA)', fontweight='bold')

# 2. Anomalies (Isolation Forest)
ax2 = fig.add_subplot(gs[0, 1])
sns.scatterplot(data=df, x='ErrorRate', y='AuditCost', hue='Anomaly', palette={'Normal':'#2ecc71', 'Anomaly':'#e74c3c'}, s=80, alpha=0.8, ax=ax2)
ax2.set_title('Anomaly Detection (Isolation Forest)', fontweight='bold')

# 3. Risk Tiers
ax3 = fig.add_subplot(gs[0, 2])
sns.countplot(data=df, x='risk_tier', palette='YlOrRd', ax=ax3, edgecolor='k')
ax3.set_title('Composite Risk Tiers', fontweight='bold')

# 4. Audit Score by Risk Level Violin
ax4 = fig.add_subplot(gs[1, 0:2])
sns.violinplot(data=df, x='RiskLevel', y='AuditScore', palette='pastel', inner='quartile', ax=ax4)
ax4.set_title('Audit Score Distribution by Risk Level', fontweight='bold')

# 5. Bayesian Risk by Type
ax5 = fig.add_subplot(gs[1, 2])
xx = np.linspace(0, 1, 500)
colors = sns.color_palette("husl", len(df['AuditType'].unique()))
for i, at in enumerate(df['AuditType'].unique()[:5]):  # limit to 5
    k = int((df[df.AuditType==at].RiskLevel=='High').sum())
    n = len(df[df.AuditType==at])
    ax5.plot(xx, beta_dist.pdf(xx, 1+k, 1+n-k), lw=3, label=at, color=colors[i])
ax5.set_title('Bayesian: P(High Risk) by Audit Type', fontweight='bold')
ax5.legend(fontsize=10)

# 6. Correlation Heatmap
ax6 = fig.add_subplot(gs[2, 0:2])
corr = df[num_feats].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, ax=ax6, cbar_kws={'shrink':0.8})
ax6.set_title('Numeric Features Correlation Matrix', fontweight='bold')

# 7. Summary Text
ax7 = fig.add_subplot(gs[2, 2]); ax7.axis('off')
summary = f"""
KEY ML INSIGHTS:
────────────────
- Anomalies Detected: {(df['Anomaly']=='Anomaly').sum()}
- Clustering identified {df['Risk_Cluster'].nunique()}
  distinct security profiles.
- Top PC1 driver: Security configuration
- Highest Bayesian Risk Type:
  {df.groupby('AuditType').apply(lambda x: (x.RiskLevel=='High').mean()).idxmax()}
"""
ax7.text(0.1, 0.9, summary, transform=ax7.transAxes, fontsize=16, fontfamily='monospace',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "Advanced_Dashboard.png"), dpi=300, bbox_inches='tight')
logger.info("  [OK] Advanced_Dashboard.png generated.")

# ═══════════════════════════════════════════
# 4. EXECUTIVE SUMMARY
# ═══════════════════════════════════════════
w(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                       EXECUTIVE SUMMARY                  │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  🎯 ADVANCED ANALYTICS DEPLOYED                                     │
  │     • PCA & K-Means Clustering: Reduced dimensional complexity of   │
  │       security controls, identifying {df['Risk_Cluster'].nunique()} core profiles.            │
  │     • Anomaly Detection (Isolation Forest): Flagged {(df['Anomaly']=='Anomaly').sum()} high-risk  │
  │       audits that deviate significantly from standard operational   │
  │       norms (e.g., highly unusual Cost-Variance ratios).            │
  │                                                                     │
  │  📊 DELIVERABLES                                                    │
  │     • 1x Interactive HTML Dashboard (Plotly, 3D PCA)                │
  │     • 1x High-Resolution Static Dashboard (Violin plots, Bayesian)  │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
""")

print(f"\n✅ PROJECT 3 () COMPLETE → {OUT}")
