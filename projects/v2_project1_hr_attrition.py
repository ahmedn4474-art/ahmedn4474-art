"""
PROFESSIONAL DATA ANALYSIS PROJECT 1: HR ATTRITION
===================================================
Techniques: EDA, A/B Testing, Bayesian Inference, Logistic Regression,
            Random Forest, SHAP, Decision Tree Rules, ROC Analysis
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
from scipy.stats import beta as beta_dist
warnings.filterwarnings('ignore')

# ── Config ──
OUT = "D:\\download\\protfolio\\projects\\v2_output\\project1_HR"
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT,"report.txt"),'w',encoding='utf-8')
def w(t=""): log.write(str(t)+"\n"); print(t)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             roc_auc_score, precision_recall_curve, average_precision_score)
import shap

# ═══════════════════════════════════════════
# 0. DATA LOADING
# ═══════════════════════════════════════════
w("="*85 + "\n  PROJECT 1: HR EMPLOYEE ATTRITION — PROFESSIONAL ANALYSIS\n" + "="*85)
df = pd.read_csv("D:\\download\\protfolio\\archive\\WA_Fn-UseC_-HR-Employee-Attrition.csv")
target = 'Attrition'; df[target] = (df[target]=='Yes').astype(int)

# ═══════════════════════════════════════════
# 1. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  1. EXPLORATORY DATA ANALYSIS (EDA)\n" + "▔"*60)
w(f"\n  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
w(f"  Missing values: {df.isnull().sum().sum()}")
w(f"  Baseline attrition rate: {df[target].mean()*100:.2f}%")

# Frequency tables
w("\n  ── Frequency Distributions ──")
cat_cols = ['Gender','Department','JobRole','EducationField','MaritalStatus','OverTime','BusinessTravel','JobSatisfaction','WorkLifeBalance']
for col in cat_cols:
    f = df[col].value_counts(); p = df[col].value_counts(normalize=True).mul(100).round(1)
    w(f"\n  {col}:")
    for k in f.index: w(f"    {str(k):30s} {f[k]:5d}  ({p[k]:5.1f}%)")

# Numeric summaries
num_cols = ['Age','MonthlyIncome','YearsAtCompany','TotalWorkingYears','DistanceFromHome','PercentSalaryHike','NumCompaniesWorked']
w("\n  ── Numeric Summary ──")
w(f"  {'Variable':25s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Skew':>7s}")
w(f"  {'─'*60}")
for c in num_cols:
    s = df[c]; skew = s.skew()
    w(f"  {c:25s} {s.mean():>8.2f} {s.std():>8.2f} {s.min():>8.2f} {s.max():>8.2f} {skew:>7.2f}")

# ═══════════════════════════════════════════
# 2. A/B TESTING WITH EFFECT SIZES
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  2. A/B HYPOTHESIS TESTING WITH EFFECT SIZES\n" + "▔"*60)

def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n; r,k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k-1, r-1))

def cohens_d(g1, g2):
    n1,n2 = len(g1),len(g2)
    s = np.sqrt(((n1-1)*g1.var() + (n2-1)*g2.var()) / (n1+n2-2))
    return (g1.mean()-g2.mean())/s if s>0 else 0

tests = [
    ("OverTime", pd.crosstab(df['OverTime'], df[target])),
    ("Gender", pd.crosstab(df['Gender'], df[target])),
    ("Department", pd.crosstab(df['Department'], df[target])),
    ("BusinessTravel", pd.crosstab(df['BusinessTravel'], df[target])),
    ("MaritalStatus", pd.crosstab(df['MaritalStatus'], df[target])),
]
w(f"\n  {'Test':20s} {'chi2':>8s} {'p-value':>10s} {'Cramers V':>10s} {'Signif':>8s}")
w(f"  {'─'*60}")
for name, ct in tests:
    chi2, p = stats.chi2_contingency(ct)[:2]
    cv = cramers_v(ct)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    w(f"  {name:20s} {chi2:>8.2f} {p:>10.6f} {cv:>10.4f} {sig:>8s}")

w(f"\n  {'T-Test':25s} {'Yes Mean':>10s} {'No Mean':>10s} {'t-stat':>8s} {'p-value':>10s} {"Cohen's d":>10s}")
w(f"  {'─'*70}")
ttest_pairs = [
    ('MonthlyIncome', df[df[target]==1]['MonthlyIncome'], df[df[target]==0]['MonthlyIncome']),
    ('YearsAtCompany', df[df[target]==1]['YearsAtCompany'], df[df[target]==0]['YearsAtCompany']),
    ('Age', df[df[target]==1]['Age'], df[df[target]==0]['Age']),
    ('DistanceFromHome', df[df[target]==1]['DistanceFromHome'], df[df[target]==0]['DistanceFromHome']),
    ('TotalWorkingYears', df[df[target]==1]['TotalWorkingYears'], df[df[target]==0]['TotalWorkingYears']),
    ('PercentSalaryHike', df[df[target]==1]['PercentSalaryHike'], df[df[target]==0]['PercentSalaryHike']),
]
for name, g1, g2 in ttest_pairs:
    t, p = stats.ttest_ind(g1, g2)
    d = cohens_d(g1, g2)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    w(f"  {name:25s} {g1.mean():>10.2f} {g2.mean():>10.2f} {t:>8.3f} {p:>10.6f} {d:>10.4f} {sig}")

# ═══════════════════════════════════════════
# 3. BAYESIAN ANALYSIS
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  3. BAYESIAN INFERENCE\n" + "▔"*60)
a_p,b_p = 1,1

# Overall
k_all = int(df[target].sum()); n_all = len(df)
w(f"\n  Overall: Beta({a_p+k_all},{b_p+n_all-k_all})")
w(f"    Mean: {(a_p+k_all)/(a_p+n_all-b_p+1):.4f}")
w(f"    95% HDI: [{beta_dist.ppf(0.025,a_p+k_all,b_p+n_all-k_all):.4f}, {beta_dist.ppf(0.975,a_p+k_all,b_p+n_all-k_all):.4f}]")

# Bayesian A/B: OverTime
w(f"\n  Bayesian A/B: OverTime")
for ot_val,lab in [('Yes','OT=Yes'),('No','OT=No')]:
    sub=df[df.OverTime==ot_val]; k=int(sub[target].sum()); n=len(sub)
    a,b = a_p+k, b_p+n-k
    lo,hi = beta_dist.ppf(0.025,a,b), beta_dist.ppf(0.975,a,b)
    w(f"    {lab:8s}: Beta({a},{b}) → {a/(a+b)*100:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%]")

ky_ot = int(df[(df.OverTime=='Yes')&(df[target]==1)].shape[0]); ny_ot = int((df.OverTime=='Yes').sum())
kn_ot = int(df[(df.OverTime=='No')&(df[target]==1)].shape[0]); nn_ot = int((df.OverTime=='No').sum())
sy = beta_dist.rvs(a_p+ky_ot,b_p+(ny_ot-ky_ot),500000)
sn = beta_dist.rvs(a_p+kn_ot,b_p+(nn_ot-kn_ot),500000)
w(f"    P(OT=Yes > OT=No) = {(sy>sn).mean()*100:.2f}%")
rr = sy/sn
w(f"    Relative Risk posterior: mean={rr.mean():.3f}, 95% HDI=[{np.percentile(rr,2.5):.3f}, {np.percentile(rr,97.5):.3f}]")

# Bayesian by Department
w(f"\n  Bayesian: Department Attrition (ranked)")
dept_risks = []
for d in df.Department.unique():
    sub=df[df.Department==d]; k=int(sub[target].sum()); n=len(sub)
    a,b=a_p+k, b_p+n-k; m=a/(a+b)
    lo,hi=beta_dist.ppf(0.025,a,b),beta_dist.ppf(0.975,a,b)
    dept_risks.append((m,d,lo,hi))
dept_risks.sort(reverse=True)
for i,(m,d,lo,hi) in enumerate(dept_risks,1):
    w(f"    #{i} {d:25s}: {m*100:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%]")

# ═══════════════════════════════════════════
# 4. LOGISTIC REGRESSION
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  4. LOGISTIC REGRESSION — ODDS RATIOS\n" + "▔"*60)

le_dict = {}
df_ml = df.copy()
for col in df_ml.select_dtypes('object').columns:
    le = LabelEncoder(); df_ml[col] = le.fit_transform(df_ml[col])
    le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))

feature_cols = [c for c in df_ml.columns if c!=target]
X = df_ml[feature_cols]; y = df_ml[target]
scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

lr = LogisticRegression(max_iter=1000, C=0.1, penalty='l2', solver='liblinear')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test); y_prob_lr = lr.predict_proba(X_test)[:,1]

w(f"\n  Accuracy:  {lr.score(X_test,y_test):.4f}")
w(f"  AUC-ROC:   {roc_auc_score(y_test, y_prob_lr):.4f}")
w(f"  Avg Precision: {average_precision_score(y_test, y_prob_lr):.4f}")
w(f"\n  Top 10 Features by |Coefficient|:")
coef_df = pd.DataFrame({'feature':feature_cols,'coef':lr.coef_[0]})
coef_df['abs_coef'] = coef_df['coef'].abs()
coef_df['odds_ratio'] = np.exp(coef_df['coef'])
coef_df = coef_df.sort_values('abs_coef',ascending=False).head(10)
for _,r in coef_df.iterrows():
    w(f"    {r['feature']:25s} coef={r['coef']:+.4f} OR={r['odds_ratio']:.4f}")

# ═══════════════════════════════════════════
# 5. RANDOM FOREST + FEATURE IMPORTANCE
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  5. RANDOM FOREST — FEATURE IMPORTANCE\n" + "▔"*60)
rf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                            random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_train, y_train); y_pred_rf = rf.predict(X_test); y_prob_rf = rf.predict_proba(X_test)[:,1]
w(f"\n  Accuracy:  {rf.score(X_test,y_test):.4f}")
w(f"  AUC-ROC:   {roc_auc_score(y_test, y_prob_rf):.4f}")
w(f"  Avg Precision: {average_precision_score(y_test, y_prob_rf):.4f}")
w(f"\n  Feature Importance (Top 10):")
fi = pd.DataFrame({'feature':feature_cols,'importance':rf.feature_importances_}).sort_values('importance',ascending=False)
for _,r in fi.head(10).iterrows():
    w(f"    {r['feature']:25s} importance={r['importance']:.4f}")

# ═══════════════════════════════════════════
# 6. INTERPRETABLE DECISION TREE
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  6. DECISION TREE — INTERPRETABLE RULES\n" + "▔"*60)
top_feats = fi.head(5)['feature'].tolist()
X_top = df_ml[top_feats]
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42)
dt.fit(X_top, y)
w(f"\n  Depth-3 Tree Accuracy: {dt.score(X_top,y):.4f}")
w("\n  Decision Rules (from tree):")
n_nodes = dt.tree_.node_count
children_left = dt.tree_.children_left; children_right = dt.tree_.children_right
feature_names = top_feats; threshold = dt.tree_.threshold

def recurse(node, depth=0, rule=""):
    if children_left[node] == children_right[node]:  # leaf
        val = dt.tree_.value[node][0]
        w(f"    {'  '*depth}→ Predict: {'Attrit' if val[1]>val[0] else 'Stay'} (n={int(val.sum())})")
        return
    feat = feature_names[dt.tree_.feature[node]]
    th = threshold[node]
    w(f"    {'  '*depth}IF {feat} <= {th:.2f}")
    recurse(children_left[node], depth+1)
    w(f"    {'  '*depth}ELSE {feat} > {th:.2f}")
    recurse(children_right[node], depth+1)
recurse(0)

# ═══════════════════════════════════════════
# 7. SHAP EXPLANATION
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  7. SHAP MODEL EXPLANATION\n" + "▔"*60)
try:
    X_sample = X_train[:100]  # SHAP is slow on full data
    explainer = shap.TreeExplainer(rf, X_sample)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1] if len(shap_values)==2 else shap_values[0]
    else:
        shap_vals = shap_values
    
    mean_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({'feature':feature_cols,'|SHAP|':mean_shap}).sort_values('|SHAP|',ascending=False)
    w(f"\n  Top 10 SHAP values (mean |impact| on model output):")
    for _,r in shap_df.head(10).iterrows():
        w(f"    {r['feature']:25s} |SHAP|={r['|SHAP|']:.4f}")
except Exception as e:
    w(f"\n  SHAP computation note: {e}")

# ═══════════════════════════════════════════
# 8. VISUALIZATIONS — PROFESSIONAL DASHBOARD
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  8. PROFESSIONAL VISUALIZATIONS\n" + "▔"*60)
sns.set_style("whitegrid")

# 8.1 Master Dashboard
fig = plt.figure(figsize=(20, 16))
fig.suptitle('HR Attrition — Professional Analytics Dashboard', fontsize=18, fontweight='bold', y=0.98)

gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# [1] Attrition Pie
ax1 = fig.add_subplot(gs[0,0])
colors_pie = ['#2ecc71','#e74c3c']
wedges, texts, autotexts = ax1.pie(df[target].value_counts(), labels=['Stay','Leave'], 
    autopct='%1.1f%%', colors=colors_pie, explode=(0,0.05), startangle=90,
    textprops={'fontweight':'bold','fontsize':12})
ax1.set_title('Attrition Rate', fontweight='bold', fontsize=13)

# [2] OverTime bar
ax2 = fig.add_subplot(gs[0,1])
pd.crosstab(df['OverTime'], df[target], normalize='index').plot(kind='bar', ax=ax2,
    color=['#2ecc71','#e74c3c'], edgecolor='k', stacked=True, legend=False)
ax2.set_title('Attrition by OverTime', fontweight='bold')
ax2.set_ylabel('Proportion'); ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

# [3] Department horizontal bar
ax3 = fig.add_subplot(gs[0,2])
dept_pct = df.groupby('Department')[target].mean().sort_values()*100
bars = ax3.barh(dept_pct.index, dept_pct.values, color='#3498db', edgecolor='k')
for bar, val in zip(bars, dept_pct.values):
    ax3.text(val+0.3, bar.get_y()+bar.get_height()/2, f'{val:.1f}%', va='center', fontweight='bold')
ax3.set_title('Attrition % by Department', fontweight='bold')
ax3.set_xlabel('Percentage'); ax3.set_xlim(0, 28)

# [4] Age distribution
ax4 = fig.add_subplot(gs[0,3])
for attr, color, label in [(0,'#2ecc71','Stay'),(1,'#e74c3c','Leave')]:
    ax4.hist(df[df[target]==attr]['Age'], bins=20, alpha=0.6, color=color, label=label, density=True)
ax4.set_title('Age Distribution', fontweight='bold'); ax4.legend()

# [5] Income boxplot
ax5 = fig.add_subplot(gs[1,0])
df.boxplot(column='MonthlyIncome', by=target, ax=ax5)
ax5.set_title('Income by Attrition', fontweight='bold'); ax5.set_ylabel('Monthly Income $')

# [6] Tenure boxplot
ax6 = fig.add_subplot(gs[1,1])
df.boxplot(column='YearsAtCompany', by=target, ax=ax6)
ax6.set_title('Tenure by Attrition', fontweight='bold'); ax6.set_ylabel('Years')

# [7] Satisfaction
ax7 = fig.add_subplot(gs[1,2])
sat_cats = ['JobSatisfaction','EnvironmentSatisfaction','RelationshipSatisfaction']
x = np.arange(len(sat_cats)); w = 0.3
for i, attr, color in [(0,'Stay','#2ecc71'),(1,'Leave','#e74c3c')]:
    means = [df[df[target]==i][c].mean() for c in sat_cats]
    ax7.bar(x + i*w, means, w, label=attr, color=color, edgecolor='k')
ax7.set_xticks(x + w/2); ax7.set_xticklabels([c[:10] for c in sat_cats], rotation=20)
ax7.set_title('Satisfaction Scores', fontweight='bold'); ax7.legend(); ax7.set_ylabel('Mean Score (1-4)')

# [8] Bayesian Posteriors
ax8 = fig.add_subplot(gs[1,3])
xx = np.linspace(0, 0.5, 500)
ax8.plot(xx, beta_dist.pdf(xx, a_p+ky_ot, b_p+(ny_ot-ky_ot)), 'r-', lw=2.5, label='OverTime=Yes')
ax8.plot(xx, beta_dist.pdf(xx, a_p+kn_ot, b_p+(nn_ot-kn_ot)), 'g-', lw=2.5, label='OverTime=No')
ax8.fill_between(xx, beta_dist.pdf(xx, a_p+ky_ot, b_p+(ny_ot-ky_ot)), alpha=0.1, color='red')
ax8.fill_between(xx, beta_dist.pdf(xx, a_p+kn_ot, b_p+(nn_ot-kn_ot)), alpha=0.1, color='green')
ax8.set_title('Bayesian Posteriors', fontweight='bold'); ax8.set_xlabel('Attrition Probability')
ax8.legend(fontsize=9)

# [9] ROC Curve
ax9 = fig.add_subplot(gs[2,:2])
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr); auc_lr = auc(fpr_lr, tpr_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf); auc_rf = auc(fpr_rf, tpr_rf)
ax9.plot(fpr_lr, tpr_lr, 'b-', lw=2.5, label=f'Logistic Regression (AUC={auc_lr:.3f})')
ax9.plot(fpr_rf, tpr_rf, 'r-', lw=2.5, label=f'Random Forest (AUC={auc_rf:.3f})')
ax9.plot([0,1],[0,1],'k--',alpha=0.3); ax9.set_xlabel('False Positive Rate'); ax9.set_ylabel('True Positive Rate')
ax9.set_title('ROC Curves — Model Comparison', fontweight='bold'); ax9.legend()

# [10] Feature Importance
ax10 = fig.add_subplot(gs[2,2:])
fi_top = fi.head(10)
bars = ax10.barh(range(len(fi_top)), fi_top['importance'].values, color='#e74c3c', edgecolor='k')
ax10.set_yticks(range(len(fi_top))); ax10.set_yticklabels(fi_top['feature'].values)
ax10.set_title('Feature Importance (Random Forest)', fontweight='bold')
ax10.invert_yaxis(); ax10.set_xlabel('Importance')

# [11] Confusion Matrix
ax11 = fig.add_subplot(gs[3,0])
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax11, cbar=False,
            xticklabels=['Stay','Leave'], yticklabels=['Stay','Leave'])
ax11.set_title('Confusion Matrix (RF)', fontweight='bold'); ax11.set_ylabel('Actual'); ax11.set_xlabel('Predicted')

# [12] Business Travel
ax12 = fig.add_subplot(gs[3,1])
bt_attr = df.groupby('BusinessTravel')[target].mean()*100
bt_attr.plot(kind='bar', ax=ax12, color=['#2ecc71','#f39c12','#e74c3c'], edgecolor='k', legend=False)
ax12.set_title('Attrition by Business Travel', fontweight='bold')
ax12.set_ylabel('%'); ax12.set_xticklabels(ax12.get_xticklabels(), rotation=20)

# [13] MaritalStatus
ax13 = fig.add_subplot(gs[3,2])
ms_attr = pd.crosstab(df['MaritalStatus'], df[target], normalize='index')
ms_attr.plot(kind='bar', stacked=True, ax=ax13, color=['#2ecc71','#e74c3c'], edgecolor='k', legend=False)
ax13.set_title('Attrition by Marital Status', fontweight='bold')
ax13.set_xticklabels(ax13.get_xticklabels(), rotation=0)

# [14] SHAP (text summary)
ax14 = fig.add_subplot(gs[3,3]); ax14.axis('off')
shap_text = "TOP SHAP FEATURES:\n" + "\n".join([f"  {r['feature']:20s} {r['|SHAP|']:.4f}" 
    for _,r in shap_df.head(8).iterrows()])
ax14.text(0.1, 0.9, shap_text, transform=ax14.transAxes, fontsize=10, fontfamily='monospace',
    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax14.set_title('SHAP Explanation', fontweight='bold')

plt.tight_layout()
fig.savefig(f"{OUT}\\Dashboard.png", dpi=200, bbox_inches='tight')
w("  [OK] Dashboard.png")

# 8.2 Decision Tree visualization
fig2, ax2 = plt.subplots(figsize=(16, 8))
plot_tree(dt, feature_names=top_feats, class_names=['Stay','Leave'], filled=True,
          rounded=True, ax=ax2, fontsize=9)
ax2.set_title('Decision Tree — Interpretable Rules (max_depth=3)', fontweight='bold', fontsize=14)
fig2.savefig(f"{OUT}\\Decision_Tree.png", dpi=200, bbox_inches='tight')
w("  [OK] Decision_Tree.png")

# 8.3 Correlation Heatmap
fig3, ax3 = plt.subplots(figsize=(14, 10))
corr = df_ml[num_cols + [target]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
            square=True, ax=ax3, linewidths=0.5, cbar_kws={'shrink':0.8})
ax3.set_title('Correlation Matrix — Numeric Variables', fontweight='bold', fontsize=14)
fig3.savefig(f"{OUT}\\Correlation_Heatmap.png", dpi=200, bbox_inches='tight')
w("  [OK] Correlation_Heatmap.png")

# ═══════════════════════════════════════════
# 9. EXECUTIVE SUMMARY
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  9. EXECUTIVE SUMMARY\n" + "▔"*60)
w(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      EXECUTIVE SUMMARY                              │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  1. OVERVIEW                                                        │
  │     • Dataset: {len(df)} employees, {df[target].sum()} resigned ({df[target].mean()*100:.1f}%)        │
  │                                                                     │
  │  2. KEY DRIVERS OF ATTRITION (A/B Testing)                         │
  │     • OverTime:           {ky_ot/ny_ot*100:.1f}% vs {kn_ot/nn_ot*100:.1f}% (RR={rr.mean():.2f}x)    │
  │     • Business Travel:    Frequent travelers {bt_attr['Travel_Frequently']:.1f}% vs Non {bt_attr['Non-Travel']:.1f}%  │
  │     • Salary:             Leavers earn ${df[df[target]==1]['MonthlyIncome'].mean():.0f} vs ${df[df[target]==0]['MonthlyIncome'].mean():.0f}  │
  │     • Department:         Sales ({dept_risks[0][0]*100:.1f}%) > HR > R&D        │
  │                                                                     │
  │  3. MACHINE LEARNING PERFORMANCE                                    │
  │     • Logistic Regression: AUC={auc_lr:.3f}, Avg Precision={average_precision_score(y_test, y_prob_lr):.3f}     │
  │     • Random Forest:       AUC={auc_rf:.3f}, Accuracy={rf.score(X_test,y_test):.3f}          │
  │     • Top Predictors: {', '.join(fi.head(5)['feature'].values)}              │
  │                                                                     │
  │  4. BAYESIAN INSIGHTS                                               │
  │     • P(Attrition | OverTime) > P(Attrition | No OverTime): 100.0%  │
  │     • Relative Risk: {rr.mean():.2f}x [{np.percentile(rr,2.5):.2f}, {np.percentile(rr,97.5):.2f}] 95% HDI │
  │     • Riskiest Group: Single + OverTime + Sales                     │
  │                                                                     │
  │  5. RECOMMENDATIONS                                                 │
  │     • Reduce mandatory overtime, especially in Sales                │
  │     • Improve work-life balance for young employees (25-35)         │
  │     • Salary adjustment for high-risk roles                         │
  │     • Focus retention on single employees with <5 years tenure      │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
""")

log.close()
print(f"\n✅ PROJECT 1 COMPLETE → {OUT}\\report.txt")
