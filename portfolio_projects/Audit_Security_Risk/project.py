"""
Audit Security Risk Analysis
============================
Multi-dimensional analysis of audit risk, IT security compliance,
and inventory management data. Includes statistical testing,
anomaly detection, and predictive modeling with cost-sensitive evaluation.
"""

import sys, os, warnings, textwrap
from itertools import cycle

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, kruskal, f_oneway
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings('ignore')

try: HERE = os.path.dirname(os.path.abspath(__file__))
except NameError: HERE = os.getcwd()
DATA_PATH = os.path.join(HERE, 'data', 'audit_security_data.xlsx')
CHARTS_DIR = os.path.join(HERE, 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.figsize': (10, 6), 'legend.fontsize': 10,
})
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
sns.set_palette(sns.color_palette(COLORS))
RISK_PALETTE = {'Low': '#27AE60', 'Medium': '#F39C12', 'High': '#E74C3C'}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_chart(fig, fname):
    path = os.path.join(CHARTS_DIR, fname)
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='w')
    plt.show()
    plt.close(fig)
    print(f'  Saved: {fname}')


def write_report(path, content, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as f:
        f.write(content)
        try:
            print(content)
        except UnicodeEncodeError:
            print(f"[Report: content saved to file]")
        try:
            print(content)
        except UnicodeEncodeError:
            print(f"[Report: content saved to file]")
    print(f'  Report written: {os.path.basename(path)}')


# ---------------------------------------------------------------------------
# 1. Load & Explore
# ---------------------------------------------------------------------------

print('=' * 60)
print('LOADING DATA')
print('=' * 60)

df = pd.read_excel(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
print(f'Shape: {df.shape}')
print(f'Columns ({len(df.columns)}): {list(df.columns)}')

target = 'RiskLevel'
print(f'Target: {target} | Classes: {df[target].value_counts().to_dict()}')

print('\nMissing values:')
null_info = df.isna().sum()
null_info = null_info[null_info > 0]
if len(null_info):
    for col, n in null_info.items():
        print(f'  {col}: {n} ({n / len(df) * 100:.1f}%)')
else:
    print('  None')

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'AuditID' in num_cols:
    num_cols.remove('AuditID')
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if target in cat_cols:
    cat_cols.remove(target)

print(f'\nNumerical features: {len(num_cols)}')
print(f'Categorical features: {len(cat_cols)}')

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

print('\n' + '=' * 60)
print('PREPROCESSING')
print('=' * 60)

# Fill missing categoricals with mode
for col in cat_cols:
    if df[col].isna().any():
        mode_val = df[col].mode().iloc[0] if len(df[col].mode()) else 'Unknown'
        df[col] = df[col].fillna(mode_val)
        print(f'  Filled {col} with mode ({mode_val})')

# Fill missing numericals with median
for col in num_cols:
    if df[col].isna().any():
        med_val = df[col].median()
        df[col] = df[col].fillna(med_val)
        print(f'  Filled {col} with median ({med_val:.2f})')

# Build encoded dataframe for modeling
df_encoded = df.copy()

risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
df_encoded[target] = df_encoded[target].map(risk_map)
y = df_encoded[target].values
print(f'  Encoded target: {dict(zip(risk_map.values(), risk_map.keys()))}')

label_encodings = {}
for col in cat_cols:
    vals = df_encoded[col].astype('category')
    mapping = {v: i for i, v in enumerate(vals.cat.categories)}
    df_encoded[col] = vals.cat.codes
    label_encodings[col] = mapping

print(f'  Encoded {len(cat_cols)} categorical predictors')

# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------

print('\n' + '=' * 60)
print('EXPLORATORY DATA ANALYSIS')
print('=' * 60)

# Target distribution
fig, ax = plt.subplots(figsize=(8, 5))
counts = df[target].value_counts()
bars = ax.bar(counts.index, counts.values,
              color=[RISK_PALETTE.get(l, '#888888') for l in counts.index],
              edgecolor='white', linewidth=1.5, width=0.6)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)
ax.set_title('Distribution of Risk Levels', fontweight='bold', pad=15)
ax.set_ylabel('Number of Records')
ax.set_xlabel('Risk Level')
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
save_chart(fig, '01_target_dist.png')

# AuditType vs RiskLevel cross-tabulation
fig, ax = plt.subplots(figsize=(10, 6))
ct = pd.crosstab(df['AuditType'], df[target], normalize='index')
ct.plot(kind='bar', stacked=True, ax=ax,
        color=[RISK_PALETTE[c] for c in ct.columns],
        edgecolor='white', linewidth=0.5)
ax.set_title('Audit Type Composition by Risk Level', fontweight='bold', pad=15)
ax.set_ylabel('Proportion')
ax.set_xlabel('Audit Type')
ax.legend(title='Risk Level', bbox_to_anchor=(1.02, 1), loc='upper left')
ax.spines[['top', 'right']].set_visible(False)
for container in ax.containers:
    for patch in container.patches:
        w, h = patch.get_width(), patch.get_height()
        if h > 0.03:
            ax.text(patch.get_x() + w / 2, patch.get_y() + h / 2,
                    f'{h:.0%}', ha='center', va='center', fontsize=8, color='white',
                    fontweight='bold')
save_chart(fig, '02_crosstab.png')

# Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 11))
corr = df_encoded[num_cols + [target]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
            fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.75},
            square=True, ax=ax, annot_kws={'fontsize': 7})
ax.set_title('Feature Correlation Matrix', fontweight='bold', pad=20)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
save_chart(fig, '03_corr_heatmap.png')

# Distribution plots by RiskLevel
key_metrics = ['AuditScore', 'Variance', 'Duration', 'DataValue']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(key_metrics):
    ax = axes[i]
    for risk_lvl in ['Low', 'Medium', 'High']:
        subset = df[df[target] == risk_lvl][col].dropna()
        sns.kdeplot(subset, ax=ax, label=risk_lvl,
                    color=RISK_PALETTE[risk_lvl], fill=True, alpha=0.3,
                    linewidth=2)
    ax.set_title(f'{col} by Risk Level', fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Density')
    ax.legend(title='Risk Level')
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(pad=3)
save_chart(fig, '04_distributions.png')

# ---------------------------------------------------------------------------
# 3. Statistical Analysis
# ---------------------------------------------------------------------------

print('\n' + '=' * 60)
print('STATISTICAL ANALYSIS')
print('=' * 60)

# Kruskal-Wallis: do numerical features differ by RiskLevel?
print('\nKruskal-Wallis H-test (numerical features vs RiskLevel):')
kw_results = []
for col in num_cols:
    groups = [df_encoded[df_encoded[target] == g][col].dropna().values
              for g in sorted(df_encoded[target].unique())]
    if all(len(g) > 1 for g in groups):
        h_stat, p_val = kruskal(*groups)
        kw_results.append((col, h_stat, p_val))
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f'  {col:30s} H={h_stat:8.2f}  p={p_val:.6f}  {sig}')

# ANOVA as complementary test
print('\nANOVA (numerical features vs RiskLevel):')
anova_results = []
for col in num_cols:
    groups = [df_encoded[df_encoded[target] == g][col].dropna().values
              for g in sorted(df_encoded[target].unique())]
    if all(len(g) > 1 for g in groups):
        f_stat, p_val = f_oneway(*groups)
        anova_results.append((col, f_stat, p_val))
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        print(f'  {col:30s} F={f_stat:8.2f}  p={p_val:.6f}  {sig}')

# Chi-square tests for categorical associations
print('\nChi-square tests (categorical vs RiskLevel):')
chi2_results = []
for col in cat_cols:
    ct = pd.crosstab(df_encoded[col], df_encoded[target])
    chi2, p_val, dof, expected = chi2_contingency(ct)
    chi2_results.append((col, chi2, p_val, dof))
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    cramer_v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1)))
    print(f'  {col:30s} chi2={chi2:8.2f}  p={p_val:.6f}  V={cramer_v:.3f}  {sig}')

# Benford's Law on DataValue
print('\nBenford\'s Law Analysis (DataValue first-digit distribution):')
dv = df['DataValue'].dropna()
digits = dv.astype(str).str[0]
digits = digits[digits.str.isdigit()].astype(int)
if len(digits) > 0:
    observed = np.array([(digits == d).mean() for d in range(1, 10)])
    benford_expected = np.log10(1 + 1 / np.arange(1, 10))
    chi2_b, p_b = stats.chisquare(observed * len(digits),
                                  benford_expected * len(digits))
    benford_pass = p_b > 0.05
    print(f'  Chi-square: {chi2_b:.2f}  p-value: {p_b:.6f}')
    print(f'  Verdict: {"PASS (conforms to Benford)" if benford_pass else "FAIL (deviates from Benford)"}')

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(1, 10)
    bars = ax.bar(x, observed, width=0.6, alpha=0.75,
                  color='#2E86AB', edgecolor='white', label='Observed')
    ax.plot(x, benford_expected, 'o-', color='#C73E1D', linewidth=2.5,
            markersize=8, label='Benford Expected')
    for i, (o, e) in enumerate(zip(observed, benford_expected)):
        ax.text(x[i], max(o, e) + 0.01, f'{o:.2%}', ha='center', fontsize=8,
                fontweight='bold')
    ax.set_title('Benford\'s Law: First-Digit Frequency of DataValue',
                 fontweight='bold', pad=15)
    ax.set_xlabel('First Digit')
    ax.set_ylabel('Proportion')
    ax.legend()
    ax.set_xticks(x)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, max(observed.max(), benford_expected.max()) * 1.25)
    save_chart(fig, '05_benford.png')
else:
    benford_pass = None
    chi2_b = p_b = 0

# ---------------------------------------------------------------------------
# 4. Anomaly Detection (Ensemble)
# ---------------------------------------------------------------------------

print('\n' + '=' * 60)
print('ANOMALY DETECTION')
print('=' * 60)

# Prepare feature matrix for anomaly detection
anom_feats = [c for c in num_cols if c != target]
anom_X = df_encoded[anom_feats].copy()
# Add encoded categoricals
for col in cat_cols:
    anom_X[col] = df_encoded[col].values

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(anom_X.fillna(0))

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Isolation Forest
iso = IsolationForest(n_estimators=300, contamination=0.05,
                      random_state=42, n_jobs=-1)
iso_pred = iso.fit_predict(X_scaled)
iso_score = iso.decision_function(X_scaled)

# LOF
lof = LocalOutlierFactor(n_neighbors=25, contamination=0.05)
lof_pred = lof.fit_predict(X_scaled)
lof_score = -lof.negative_outlier_factor_

# Ensemble consensus
anomaly_ensemble = (iso_pred == -1).astype(int) + (lof_pred == -1).astype(int)
consensus_anom = anomaly_ensemble >= 2

n_if = int((iso_pred == -1).sum())
n_lof = int((lof_pred == -1).sum())
n_both = int(consensus_anom.sum())
print(f'  Isolation Forest anomalies: {n_if} ({n_if / len(df) * 100:.1f}%)')
print(f'  LOF anomalies:              {n_lof} ({n_lof / len(df) * 100:.1f}%)')
print(f'  Both methods agree:         {n_both} ({n_both / len(df) * 100:.1f}%)')

# PCA visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# PCA by anomaly consensus
ax = axes[0]
colors_anom = ['#2E86AB' if not c else '#C73E1D' for c in consensus_anom]
sizes_anom = [30 if not c else 60 for c in consensus_anom]
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_anom, s=sizes_anom,
                alpha=0.6, edgecolors='grey', linewidth=0.3)
ax.set_title(f'Anomaly Detection (Consensus)\n{n_both} flagged',
             fontweight='bold')
ax.set_xlabel(f'PC1 ({var_exp[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({var_exp[1]:.1%} var)')
ax.spines[['top', 'right']].set_visible(False)
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB',
                          markersize=8, label='Normal'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#C73E1D',
                          markersize=8, label='Anomaly')]
ax.legend(handles=legend_elements, loc='best')

# PCA by RiskLevel
ax = axes[1]
risk_colors = [RISK_PALETTE.get(df[target].iloc[i], '#888')
               for i in range(len(df))]
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=risk_colors, s=25,
           alpha=0.6, edgecolors='grey', linewidth=0.3)
ax.set_title('PCA Colored by Risk Level', fontweight='bold')
ax.set_xlabel(f'PC1 ({var_exp[0]:.1%} var)')
ax.set_ylabel(f'PC2 ({var_exp[1]:.1%} var)')
ax.spines[['top', 'right']].set_visible(False)
for lvl, color in RISK_PALETTE.items():
    ax.scatter([], [], c=color, label=lvl)
ax.legend(title='Risk Level')

# Anomaly score distributions
ax = axes[2]
ax.hist(iso_score, bins=40, alpha=0.6, color='#2E86AB', edgecolor='white',
        label=f'IF Score (n={n_if})')
ax.hist(lof_score, bins=40, alpha=0.6, color='#F18F01', edgecolor='white',
        label=f'LOF Score (n={n_lof})')
ax.axvline(np.percentile(iso_score, 5), color='#2E86AB', ls='--', alpha=0.7)
ax.axvline(np.percentile(lof_score, 95), color='#F18F01', ls='--', alpha=0.7)
ax.set_title('Anomaly Score Distributions', fontweight='bold')
ax.set_xlabel('Score')
ax.set_ylabel('Frequency')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
save_chart(fig, '06_anomaly_detection.png')

# Risk profile of anomalies
if n_both > 0:
    anom_risk_dist = df.loc[consensus_anom, target].value_counts()
    print(f'  Risk profile of consensus anomalies:')
    for lvl, cnt in anom_risk_dist.items():
        print(f'    {lvl}: {cnt} ({cnt / n_both * 100:.1f}%)')

# ---------------------------------------------------------------------------
# 5. Machine Learning
# ---------------------------------------------------------------------------

print('\n' + '=' * 60)
print('MACHINE LEARNING')
print('=' * 60)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_auc_score, roc_curve, classification_report)

# Prepare feature matrix
ml_cols = [c for c in anom_X.columns if c != target]
X = df_encoded[ml_cols].fillna(0).values
y_ml = y  # already encoded 0,1,2

X_train, X_test, y_train, y_test = train_test_split(
    X, y_ml, test_size=0.2, random_state=42, stratify=y_ml
)

print(f'Train: {X_train.shape[0]} | Test: {X_test.shape[0]}')

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=2000, C=1.0, solver='lbfgs', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42
    ),
}

results = []
best_f1 = -1
best_model = None
best_name = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # ROC AUC one-vs-rest
    try:
        rocauc_ovr = roc_auc_score(y_test, y_prob, multi_class='ovr')
    except Exception:
        rocauc_ovr = np.nan

    results.append({
        'Model': name,
        'Accuracy': acc,
        'F1 (macro)': f1_macro,
        'F1 (weighted)': f1_weighted,
        'ROC AUC (OvR)': rocauc_ovr,
    })

    print(f'\n  {name}:')
    print(f'    Accuracy:     {acc:.4f}')
    print(f'    F1 (macro):   {f1_macro:.4f}')
    print(f'    F1 (weighted):{f1_weighted:.4f}')
    print(f'    ROC AUC OvR:  {rocauc_ovr:.4f}' if not np.isnan(rocauc_ovr) else '    ROC AUC OvR: N/A')
    print(f'    {classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"])}')

    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_model = model
        best_name = name

results_df = pd.DataFrame(results).sort_values('F1 (macro)', ascending=False)
print(f'\n  Best model: {best_name} (F1 macro = {best_f1:.4f})')

# Confusion matrix for best model
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            ax=ax, linewidths=1, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
ax.set_title(f'Confusion Matrix — {best_name}', fontweight='bold', pad=15)
ax.set_xlabel('Predicted Risk Level', fontweight='bold')
ax.set_ylabel('Actual Risk Level', fontweight='bold')
save_chart(fig, '07_confusion_matrix.png')

# ROC curves (one-vs-rest)
fig, ax = plt.subplots(figsize=(10, 8))
class_names = ['Low', 'Medium', 'High']
class_colors = ['#27AE60', '#F39C12', '#E74C3D']
for i, (cl_name, cl_color) in enumerate(zip(class_names, class_colors)):
    y_test_bin = (y_test == i).astype(int)
    fpr, tpr, _ = roc_curve(y_test_bin, y_prob_best[:, i])
    auc_val = roc_auc_score(y_test_bin, y_prob_best[:, i])
    ax.plot(fpr, tpr, lw=2.5, color=cl_color,
            label=f'{cl_name} (AUC = {auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
ax.set_title(f'ROC Curves (One-vs-Rest) — {best_name}', fontweight='bold', pad=15)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
save_chart(fig, '08_roc_curves.png')

# Feature importance
print('\nFeature Importance:')
imp = None
if hasattr(best_model, 'feature_importances_'):
    imp = best_model.feature_importances_
    imp_label = 'Importance (Gini)'
elif hasattr(best_model, 'coef_'):
    imp = np.max(np.abs(best_model.coef_), axis=0)
    imp_label = '|Coefficient| (max across classes)'

top_features = []
if imp is not None:
    fi_df = pd.DataFrame({
        'Feature': ml_cols,
        'Importance': imp
    }).sort_values('Importance', ascending=False)

    n_show = min(20, len(fi_df))
    fi_show = fi_df.head(n_show).iloc[::-1]
    top_features = fi_df.head(5)['Feature'].tolist()

    print(f'  Top 10 predictors:')
    for j, row in fi_df.head(10).iterrows():
        print(f'    {row["Feature"]:35s} {row["Importance"]:.4f}')

    fi_display_names = []
    for f in fi_show['Feature']:
        if f in label_encodings:
            fi_display_names.append(f)
        else:
            fi_display_names.append(f)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors_imp = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(fi_show)))
    bars = ax.barh(range(len(fi_show)), fi_show['Importance'].values,
                   color=colors_imp, edgecolor='grey', linewidth=0.5)
    ax.set_yticks(range(len(fi_show)))
    ax.set_yticklabels(fi_display_names, fontsize=9)
    ax.set_title(f'Feature Importance — {best_name}', fontweight='bold', pad=15)
    ax.set_xlabel(imp_label)
    ax.spines[['top', 'right']].set_visible(False)
    for bar, val in zip(bars, fi_show['Importance'].values):
        ax.text(bar.get_width() + max(fi_show['Importance'].values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)
    save_chart(fig, '09_feature_importance.png')
else:
    top_features = ['AuditScore', 'Variance', 'DataValue', 'Duration', 'ErrorRate']

# ---------------------------------------------------------------------------
# 6. Cost-Sensitive Analysis
# ---------------------------------------------------------------------------

print('\n' + '=' * 60)
print('COST-SENSITIVE ANALYSIS')
print('=' * 60)

# Cost matrix: rows=actual, cols=predicted (0=Low, 1=Medium, 2=High)
# Predicting Low when actually High is most costly
cost_matrix = np.array([
    [0, 1, 5],    # actual Low
    [3, 0, 8],    # actual Medium
    [10, 6, 0],   # actual High
])

cost_labels = {
    (0, 0): 'Correct (Low)',
    (0, 1): 'Low→Medium',
    (0, 2): 'Low→High',
    (1, 0): 'Medium→Low',
    (1, 1): 'Correct (Medium)',
    (1, 2): 'Medium→High',
    (2, 0): 'High→Low',
    (2, 1): 'High→Medium',
    (2, 2): 'Correct (High)',
}

print('Cost Matrix (actual \\ predicted):')
print(f'           {"Low":>8s} {"Medium":>8s} {"High":>8s}')
for i, lvl in enumerate(['Low', 'Medium', 'High']):
    print(f'  {lvl:10s} {cost_matrix[i, 0]:8d} {cost_matrix[i, 1]:8d} {cost_matrix[i, 2]:8d}')

cost_results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    cm_test = confusion_matrix(y_test, y_pred)
    total_cost = (cm_test * cost_matrix).sum()
    cost_results.append({'Model': name, 'Total Misclassification Cost': total_cost})
    print(f'  {name:25s} Total cost: {total_cost:.0f}')

cost_results_df = pd.DataFrame(cost_results).sort_values('Total Misclassification Cost')

# Cost analysis visualization
fig, ax = plt.subplots(figsize=(10, 6))
colors_cost = ['#27AE60' if r['Model'] == best_name else '#A23B72'
               for _, r in cost_results_df.iterrows()]
bars = ax.bar(cost_results_df['Model'], cost_results_df['Total Misclassification Cost'],
              color=colors_cost, edgecolor='white', width=0.5)
for bar, val in zip(bars, cost_results_df['Total Misclassification Cost']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_title('Cost-Sensitive Model Comparison', fontweight='bold', pad=15)
ax.set_ylabel('Total Misclassification Cost')
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
save_chart(fig, '10_cost_analysis.png')

# Per-class cost breakdown for best model
cm_best = confusion_matrix(y_test, y_pred_best)
per_class_cost = (cm_best * cost_matrix).sum(axis=1)

fig, ax = plt.subplots(figsize=(8, 5))
class_labels = ['Low (actual)', 'Medium (actual)', 'High (actual)']
bar_colors = [RISK_PALETTE['Low'], RISK_PALETTE['Medium'], RISK_PALETTE['High']]
bars = ax.bar(class_labels, per_class_cost, color=bar_colors,
              edgecolor='white', width=0.5)
for bar, val in zip(bars, per_class_cost):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
ax.set_title(f'Misclassification Cost by Actual Risk — {best_name}',
             fontweight='bold', pad=15)
ax.set_ylabel('Total Cost')
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
save_chart(fig, '11_cost_by_class.png')

# ---------------------------------------------------------------------------
# 7. Generate Reports
# ---------------------------------------------------------------------------

print('\n' + '=' * 60)
print('GENERATING REPORTS')
print('=' * 60)

# Collect key statistics for reports
n_records = len(df)
n_low = int((df[target] == 'Low').sum())
n_med = int((df[target] == 'Medium').sum())
n_high = int((df[target] == 'High').sum())
best_acc = results_df.iloc[0]['Accuracy']
best_f1_val = results_df.iloc[0]['F1 (macro)']
best_rocauc = results_df.iloc[0]['ROC AUC (OvR)']

# Significantly different features (Kruskal p < 0.05)
sig_features = [r[0] for r in kw_results if r[2] < 0.05]
top_sig = sig_features[:5]

# Significant categorical associations
sig_cat = [r[0] for r in chi2_results if r[2] < 0.05]

benford_verdict_ar = 'مطابق' if benford_pass else 'غير مطابق'
benford_verdict_en = 'CONFORMS' if benford_pass else 'DEVIATES'

best_cost_model = cost_results_df.iloc[0]['Model']
best_cost_val = cost_results_df.iloc[0]['Total Misclassification Cost']

# Model comparison table (text)
model_table_lines = []
model_table_lines.append(f'{"Model":<25s} {"Accuracy":>10s} {"F1 Macro":>10s} {"ROC AUC":>10s} {"Cost":>10s}')
model_table_lines.append('-' * 65)
for _, row in results_df.iterrows():
    cost_row = cost_results_df[cost_results_df['Model'] == row['Model']]
    cost_val = cost_row.iloc[0]['Total Misclassification Cost'] if len(cost_row) else 0
    model_table_lines.append(
        f'{row["Model"]:<25s} {row["Accuracy"]:>10.4f} {row["F1 (macro)"]:>10.4f} '
        f'{row["ROC AUC (OvR)"]:>10.4f} {cost_val:>10.0f}'
    )
model_table = '\n'.join(model_table_lines)

# ---------- SHORT ARABIC ----------

ar_short = f"""تقرير تدقيق وتحليل المخاطر الأمنية - ملخص تنفيذي
{"=" * 45}

إجمالي السجلات: {n_records}
توزيع المخاطر: منخفض {n_low} | متوسط {n_med} | مرتفع {n_high}

نتائج التحليل:
• أفضل نموذج: {best_name}
  - الدقة: {best_acc:.4f}
  - متوسط F1 الكلي: {best_f1_val:.4f}

• كشف الشذوذ: تم اكتشاف {n_both} سجلاً شاذة بالتوافق بين Isolation Forest و LOF

• الميزات الأكثر تأثيراً: {", ".join(top_features[:5])}

• تحليل بنفورد: {benford_verdict_ar} (p={p_b:.4f})

• التكلفة الإجمالية للتصنيف الخاطئ: {best_cost_val:.0f}

التوصيات الرئيسية:
- مراجعة السجلات الشاذة التي تم اكتشافها
- تحسين ضوابط الأمن السيبراني للمخاطر العالية
- الاستفادة من النموذج التنبؤي ({best_name}) لتصنيف المخاطر مسبقاً
"""

write_report(os.path.join(HERE, 'Audit_Report_AR_short.txt'), ar_short)

# ---------- SHORT ENGLISH ----------

en_short = f"""AUDIT SECURITY RISK ANALYSIS - EXECUTIVE SUMMARY
{"=" * 50}

Total Records: {n_records}
Risk Distribution: Low {n_low} | Medium {n_med} | High {n_high}

Key Findings:
• Best Model: {best_name}
  - Accuracy: {best_acc:.4f}
  - Macro F1: {best_f1_val:.4f}

• Anomaly Detection: {n_both} records flagged by both Isolation Forest and LOF

• Top Risk Factors: {", ".join(top_features[:5])}

• Benford's Law: {benford_verdict_en} (p={p_b:.4f})

• Total Misclassification Cost: {best_cost_val:.0f}

Primary Recommendations:
- Investigate all consensus-anomaly records
- Strengthen cybersecurity controls for high-risk segments
- Deploy {best_name} for proactive risk classification
"""

write_report(os.path.join(HERE, 'Audit_Report_EN_short.txt'), en_short)

# ---------- FULL ARABIC ----------

# Kruskal-Wallis summary table
kw_lines = []
kw_lines.append(f'{"المتغير":<25s} {"قيمة H":<10s} {"p-value":<10s} {"الدلالة":<8s}')
kw_lines.append('-' * 53)
for col, h, p in kw_results[:10]:
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    kw_lines.append(f'{col:<25s} {h:<10.2f} {p:<10.6f} {sig:<8s}')
kw_table = '\n'.join(kw_lines)

# Chi-square summary
chi2_lines = []
chi2_lines.append(f'{"المتغير":<25s} {"Chi2":<10s} {"p-value":<10s} {"Cramer V":<10s}')
chi2_lines.append('-' * 55)
for col, chi, p, dof in chi2_results[:10]:
    cramer = np.sqrt(chi / (n_records * (min(2, dof + 1) - 1))) if (n_records * min(2, dof + 1) - 1) > 0 else 0
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    chi2_lines.append(f'{col:<25s} {chi:<10.2f} {p:<10.6f} {cramer:<10.4f}')
chi2_table = '\n'.join(chi2_lines)

# Confusion matrix in text
cm_text = '\n'.join([
    f'             {"Low":>8s} {"Medium":>8s} {"High":>8s}',
    f'  Low      {cm[0,0]:8d} {cm[0,1]:8d} {cm[0,2]:8d}',
    f'  Medium   {cm[1,0]:8d} {cm[1,1]:8d} {cm[1,2]:8d}',
    f'  High     {cm[2,0]:8d} {cm[2,1]:8d} {cm[2,2]:8d}',
])

ar_full = f"""تقرير تدقيق وتحليل المخاطر الأمنية - تقرير كامل
{"=" * 55}

1. ملخص البيانات
{"-" * 30}
إجمالي السجلات: {n_records}
عدد المتغيرات: {len(df.columns)}
المتغير المستهدف: RiskLevel (مستوى المخاطرة)
توزيع المخاطر: منخفض {n_low} ({n_low / n_records * 100:.1f}%) | متوسط {n_med} ({n_med / n_records * 100:.1f}%) | مرتفع {n_high} ({n_high / n_records * 100:.1f}%)

المتغيرات الرقمية ({len(num_cols)}):
{", ".join(num_cols)}

المتغيرات الفئوية ({len(cat_cols)}):
{", ".join(cat_cols)}

2. تحليل البيانات الاستكشافي
{"-" * 30}
- توزيع المخاطر متوازن نسبياً (أنحرف طفيف نحو الفئة المتوسطة)
- أكثر أنواع التدقيق المرتبطة بالمخاطر العالية يتم تحليلها في جدول التقاطع
- مصفوفة الارتباط متاحة في المخططات البيانية

3. الاختبارات الإحصائية
{"-" * 30}

اختبار كروسكال-واليس (فروق المتغيرات الرقمية حسب مستوى المخاطرة):
{kw_table}

اختبار كاي تربيع (الارتباطات الفئوية):
{chi2_table}

قانون بنفورد:
- المتغير: DataValue
- Chi-square: {chi2_b:.2f}
- p-value: {p_b:.6f}
- النتيجة: {benford_verdict_ar}

4. كشف الشذوذ
{"-" * 30}
- Isolation Forest: {n_if} سجلاً شاذاً ({n_if / n_records * 100:.1f}%)
- Local Outlier Factor: {n_lof} سجلاً شاذاً ({n_lof / n_records * 100:.1f}%)
- بالتوافق بين الطريقتين: {n_both} سجلاً ({n_both / n_records * 100:.1f}%)

5. نماذج التعلم الآلي
{"-" * 30}

مقارنة النماذج:
{model_table}

أفضل نموذج: {best_name}
مصفوفة الارتباك ({best_name}):
{cm_text}

6. أهم الميزات المؤثرة:
{"-" * 30}
{chr(10).join(f'  {i+1}. {f}' for i, f in enumerate(top_features[:10]))}

7. تحليل التكاليف
{"-" * 30}
التكلفة الإجمالية للتصنيف الخاطئ:
{chr(10).join(f'  {r["Model"]}: {r["Total Misclassification Cost"]}' for _, r in cost_results_df.iterrows())}

8. التوصيات
{"-" * 30}
1. مراجعة السجلات الشاذة (التوافق بين Isolation Forest و LOF) للتدقيق اليدوي
2. تعزيز ضوابط الأمن السيبراني للمتغيرات الأكثر تأثيراً
3. استخدام نموذج {best_name} للتنبؤ المبكر بمستويات المخاطرة
4. تحسين جودة البيانات للحقول التي تعاني من قيم مفقودة
5. مراجعة السجلات غير المطابقة لقانون بنفورد للكشف عن التلاعب المحتمل
6. إجراء تحليل دوري للتأكد من استقرار توزيع المخاطر
7. تطبيق مصفوفة التكاليف في تقييم أداء النماذج المستقبلية
"""

write_report(os.path.join(HERE, 'Audit_Report_AR_full.txt'), ar_full)

# ---------- FULL ENGLISH ----------

en_full = f"""AUDIT SECURITY RISK ANALYSIS - FULL REPORT
{"=" * 50}

1. DATA OVERVIEW
{"-" * 30}
Total Records: {n_records}
Total Features: {len(df.columns)}
Target Variable: RiskLevel (Low / Medium / High)
Risk Distribution: Low {n_low} ({n_low / n_records * 100:.1f}%) | Medium {n_med} ({n_med / n_records * 100:.1f}%) | High {n_high} ({n_high / n_records * 100:.1f}%)

Numerical Features ({len(num_cols)}):
{", ".join(num_cols)}

Categorical Features ({len(cat_cols)}):
{", ".join(cat_cols)}

2. EXPLORATORY DATA ANALYSIS
{"-" * 30}
- The target distribution is relatively balanced with a slight skew toward Medium risk
- Cross-tabulation of AuditType vs RiskLevel reveals which audit types are most associated with high risk
- Correlation heatmap identifies collinearity patterns among numerical features
- Distribution plots show how key metrics differ across risk levels

3. STATISTICAL TESTS
{"-" * 30}

Kruskal-Wallis H-test (numerical feature differences by RiskLevel):
{kw_table}

Chi-square Tests (categorical associations):
{chi2_table}

Benford's Law:
- Variable: DataValue
- Chi-square: {chi2_b:.2f}
- p-value: {p_b:.6f}
- Verdict: {benford_verdict_en}

4. ANOMALY DETECTION
{"-" * 30}
- Isolation Forest: {n_if} anomalies ({n_if / n_records * 100:.1f}%)
- Local Outlier Factor: {n_lof} anomalies ({n_lof / n_records * 100:.1f}%)
- Consensus (both methods): {n_both} records ({n_both / n_records * 100:.1f}%)

5. MACHINE LEARNING MODELS
{"-" * 30}

Model Comparison:
{model_table}

Best Model: {best_name}
Confusion Matrix ({best_name}):
{cm_text}

6. TOP RISK FACTORS
{"-" * 30}
{chr(10).join(f'  {i+1}. {f}' for i, f in enumerate(top_features[:10]))}

7. COST ANALYSIS
{"-" * 30}
Total misclassification costs by model:
{chr(10).join(f'  {r["Model"]}: {r["Total Misclassification Cost"]}' for _, r in cost_results_df.iterrows())}

Cost matrix used (rows=actual, cols=predicted):
           Low  Medium  High
  Low        0       1      5
  Medium     3       0      8
  High      10       6      0

8. RECOMMENDATIONS
{"-" * 30}
1. Investigate consensus-anomaly records (flagged by both Isolation Forest and LOF)
2. Strengthen cybersecurity controls for the most influential risk factors
3. Deploy {best_name} for early risk-level prediction and triage
4. Address data quality issues in fields with missing values
5. Review Benford-nonconforming records for potential data manipulation
6. Conduct periodic analyses to monitor risk distribution stability
7. Incorporate the cost matrix into ongoing model evaluation
"""

write_report(os.path.join(HERE, 'Audit_Report_EN_full.txt'), en_full)

# ---------------------------------------------------------------------------
# Final Summary
# ---------------------------------------------------------------------------

print('\n' + '=' * 60)
print('ANALYSIS COMPLETE')
print('=' * 60)
print(f'  Records analyzed:    {n_records}')
print(f'  Charts generated:    11 files')
print(f'  Reports generated:   4 files')
print(f'  Best model:          {best_name}')
print(f'  Accuracy:            {best_acc:.4f}')
print(f'  F1 (macro):          {best_f1_val:.4f}')
print(f'  Anomalies detected:  {n_both} (consensus)')
print(f'  Benford\'s Law:       {benford_verdict_en}')
print(f'  Misclassification cost: {best_cost_val:.0f}')
print('=' * 60)
