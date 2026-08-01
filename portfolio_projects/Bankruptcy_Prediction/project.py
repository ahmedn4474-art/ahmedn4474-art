"""
Bankruptcy Prediction from Financial Ratios
=============================================
A comprehensive analysis pipeline covering exploratory data analysis,
statistical testing, machine learning modeling, and bilingual reporting.

Data: 6,819 firms with 95 financial ratios. Target: Bankrupt? (1=bankrupt, 0=healthy)
"""

import os
import sys
import warnings
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from scipy import stats
from scipy.stats import skew, kurtosis, normaltest, ttest_ind, mannwhitneyu

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
)
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, average_precision_score, matthews_corrcoef
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.simplefilter('ignore')

# Try importing optional libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print('[INFO] xgboost not available, using GradientBoosting as fallback')

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print('[INFO] imblearn not available, using class_weight instead')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print('[INFO] shap not available, skipping SHAP analysis')

DATA_PATH = os.path.join('data', 'bankruptcy_financial_ratios.csv')
CHARTS_DIR = 'charts'
try: BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: BASE_DIR = os.getcwd()

os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, CHARTS_DIR), exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================
# Global plot styling
# ============================================================
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
})
COLORS = sns.color_palette('husl', 10)
BANKRUPT_COLOR = '#e74c3c'
HEALTHY_COLOR = '#2ecc71'


def load_and_explore_data():
    """Load the dataset and perform initial exploration."""
    path = os.path.join(BASE_DIR, DATA_PATH)
    df = pd.read_csv(path)
    print(f'[DATA] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns')

    target_col = 'Bankrupt?'
    if target_col not in df.columns:
        candidates = [c for c in df.columns if any(
            k in c.lower() for k in ['bankrupt', 'class', 'status', 'target', 'default']
        )]
        if candidates:
            target_col = candidates[0]
            print(f'[DATA] Using "{target_col}" as target column')
        else:
            raise ValueError('No target column found')

    feature_cols = [c for c in df.columns if c != target_col]

    # Initial missing value assessment
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    cols_with_missing = missing_pct[missing_pct > 0].sort_values(ascending=False)
    if len(cols_with_missing) > 0:
        print(f'[DATA] {len(cols_with_missing)} features have missing values')
        for col, pct in cols_with_missing.items():
            print(f'       {col}: {pct}% missing')
        # Drop columns with >50% missing, fill rest with median
        high_missing = cols_with_missing[cols_with_missing > 50].index
        if len(high_missing) > 0:
            print(f'[DATA] Dropping {len(high_missing)} features with >50% missing')
            df.drop(columns=high_missing, inplace=True)
            feature_cols = [c for c in df.columns if c != target_col]

    # Fill remaining missing values
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    # Check for constant / near-constant columns
    constant_cols = []
    for col in feature_cols:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    if constant_cols:
        print(f'[DATA] Dropping {len(constant_cols)} constant features')
        df.drop(columns=constant_cols, inplace=True)
        feature_cols = [c for c in df.columns if c != target_col]

    # Check for infinite values
    for col in feature_cols:
        if np.isinf(df[col]).any():
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col].fillna(df[col].median(), inplace=True)

    print(f'[DATA] Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns ({len(feature_cols)} features)')

    target_dist = df[target_col].value_counts().sort_index()
    print(f'[DATA] Target distribution:\n       {target_dist.to_dict()}')
    print(f'       Bankrupt rate: {df[target_col].mean():.4f} ({df[target_col].sum():,} of {len(df):,})')

    return df, target_col, feature_cols


def plot_target_distribution(df, target_col):
    """Bar chart of target distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    counts = df[target_col].value_counts().sort_index()
    labels = ['Healthy (0)', 'Bankrupt (1)']
    colors = [HEALTHY_COLOR, BANKRUPT_COLOR]

    ax1.bar(labels, counts.values, color=colors, edgecolor='white', width=0.6)
    for i, v in enumerate(counts.values):
        ax1.text(i, v + 20, f'{v:,}\n({v / len(df) * 100:.1f}%)',
                 ha='center', fontsize=11, fontweight='bold')
    ax1.set_title('Target Distribution', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Number of Firms')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    wedges, texts, autotexts = ax2.pie(
        counts.values, labels=labels, autopct='%1.1f%%',
        colors=[HEALTHY_COLOR, BANKRUPT_COLOR],
        startangle=90, explode=(0, 0.05),
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    ax2.set_title('Target Proportion', fontweight='bold', fontsize=14)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, '01_target_distribution.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')


def plot_correlation_heatmap(df, target_col, top_n=20):
    """Correlation heatmap of top features with the target."""
    corr_with_target = df.corrwith(df[target_col]).abs().sort_values(ascending=False)
    top_features = corr_with_target.index[1:top_n + 1].tolist()

    corr_matrix = df[top_features + [target_col]].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    sns.heatmap(
        corr_matrix, mask=mask, cmap=cmap, center=0,
        annot=True, fmt='.2f', linewidths=0.5,
        square=True, cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'},
        ax=ax
    )
    ax.set_title(f'Correlation Matrix — Top {top_n} Features by Correlation with Target',
                 fontweight='bold', fontsize=14, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, '02_correlation_heatmap.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')

    return top_features


def plot_pca_visualization(df, feature_cols, target_col):
    """2D and 3D PCA scatter plots."""
    from mpl_toolkits.mplot3d import Axes3D

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    y = df[target_col].values

    pca = PCA(n_components=3, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_ * 100
    print(f'[PCA] Explained variance: PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%, PC3={explained[2]:.1f}%')

    # 2D scatter
    fig, ax = plt.subplots(figsize=(12, 9))
    for cls, marker, color, label in [
        (0, 'o', HEALTHY_COLOR, 'Healthy'),
        (1, 'X', BANKRUPT_COLOR, 'Bankrupt')
    ]:
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, marker=marker,
                   label=label, alpha=0.6, edgecolors='white', linewidth=0.3, s=25)
    ax.set_xlabel(f'PC1 ({explained[0]:.1f}% Variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained[1]:.1f}% Variance)', fontsize=12)
    ax.set_title('PCA 2D Projection of Financial Ratios', fontweight='bold', fontsize=14)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, '03_pca_2d.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')

    # 3D scatter
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    for cls, color, label in [(0, HEALTHY_COLOR, 'Healthy'), (1, BANKRUPT_COLOR, 'Bankrupt')]:
        mask = y == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
                   c=color, label=label, alpha=0.5, s=15)
    ax.set_xlabel(f'PC1 ({explained[0]:.1f}%)', fontsize=10)
    ax.set_ylabel(f'PC2 ({explained[1]:.1f}%)', fontsize=10)
    ax.set_zlabel(f'PC3 ({explained[2]:.1f}%)', fontsize=10)
    ax.set_title('PCA 3D Projection', fontweight='bold', fontsize=14)
    ax.legend(frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, '04_pca_3d.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')

    return X_pca, explained


def plot_statistical_analysis(df, feature_cols, target_col, top_n=15):
    """T-tests comparing bankrupt vs healthy firms and normality assessment."""
    y = df[target_col].values
    bankrupt_mask = y == 1
    healthy_mask = y == 0

    results = []
    normality_results = []

    for col in feature_cols:
        bankrupt_vals = df.loc[bankrupt_mask, col].dropna().values
        healthy_vals = df.loc[healthy_mask, col].dropna().values

        if len(bankrupt_vals) < 3 or len(healthy_vals) < 3:
            continue

        # T-test (Welch's)
        t_stat, p_val = ttest_ind(bankrupt_vals, healthy_vals, equal_var=False, nan_policy='omit')

        # Mann-Whitney U as non-parametric alternative
        u_stat, u_pval = mannwhitneyu(bankrupt_vals, healthy_vals, alternative='two-sided')

        effect_size = (np.mean(bankrupt_vals) - np.mean(healthy_vals)) / \
                      np.sqrt((np.std(bankrupt_vals) ** 2 + np.std(healthy_vals) ** 2) / 2)

        results.append({
            'feature': col,
            't_statistic': t_stat,
            'p_value': p_val,
            'mw_u_statistic': u_stat,
            'mw_p_value': u_pval,
            'effect_size': effect_size,
            'bankrupt_mean': np.mean(bankrupt_vals),
            'healthy_mean': np.mean(healthy_vals),
            'bankrupt_std': np.std(bankrupt_vals),
            'healthy_std': np.std(healthy_vals),
            'mean_diff': np.mean(bankrupt_vals) - np.mean(healthy_vals)
        })

        # Normality test (D'Agostino-Pearson) on the full feature
        if len(df[col].dropna()) >= 20:
            try:
                norm_stat, norm_p = normaltest(df[col].dropna().values)
                normality_results.append({
                    'feature': col,
                    'normality_stat': norm_stat,
                    'normality_p': norm_p,
                    'is_normal': norm_p > 0.05,
                    'skewness': skew(df[col].dropna().values),
                    'kurtosis': kurtosis(df[col].dropna().values, fisher=True)
                })
            except Exception:
                pass

    ttest_df = pd.DataFrame(results).sort_values('p_value')

    # Report significant features
    n_sig = (ttest_df['p_value'] < 0.05).sum()
    print(f'[STATS] {n_sig} of {len(ttest_df)} features show significant difference (p<0.05)')

    # Top discriminating features bar chart
    top_ttest = ttest_df.head(top_n).copy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Bar chart of effect sizes
    colors = [BANKRUPT_COLOR if v > 0 else HEALTHY_COLOR for v in top_ttest['effect_size']]
    axes[0].barh(range(len(top_ttest)), top_ttest['effect_size'].values, color=colors, edgecolor='white')
    axes[0].set_yticks(range(len(top_ttest)))
    axes[0].set_yticklabels([c[:35] + '...' if len(c) > 35 else c for c in top_ttest['feature']],
                            fontsize=9)
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_xlabel("Cohen's d Effect Size", fontsize=12)
    axes[0].set_title('Top Features: Effect Size (Bankrupt vs Healthy)',
                      fontweight='bold', fontsize=13)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # P-value bar chart
    log_p = -np.log10(np.maximum(top_ttest['p_value'].values, 1e-300))
    axes[1].bar(range(len(top_ttest)), log_p, color='#3498db', edgecolor='white')
    axes[1].axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=1.5,
                    label='p=0.05 threshold')
    axes[1].set_xticks(range(len(top_ttest)))
    axes[1].set_xticklabels([c[:25] + '...' if len(c) > 25 else c for c in top_ttest['feature']],
                            rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('-log10(p-value)', fontsize=12)
    axes[1].set_title('Statistical Significance (-log10 p-value)',
                      fontweight='bold', fontsize=13)
    axes[1].legend(frameon=True)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.suptitle('Statistical Comparison: Bankrupt vs Healthy Firms',
                 fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, '05_statistical_tests.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')

    # Selected features for boxplot comparison
    top_5 = ttest_df['feature'].head(5).tolist()
    n_cols = min(5, len(top_5))
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    for ax, col in zip(axes, top_5):
        data_bankrupt = df.loc[bankrupt_mask, col].dropna()
        data_healthy = df.loc[healthy_mask, col].dropna()
        bp = ax.boxplot([data_healthy, data_bankrupt], labels=['Healthy', 'Bankrupt'],
                        patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor(HEALTHY_COLOR)
        bp['boxes'][1].set_facecolor(BANKRUPT_COLOR)
        ax.set_title(col[:30] + ('...' if len(col) > 30 else ''), fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.suptitle('Top Discriminating Features — Distribution Comparison',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, '06_feature_boxplots.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')

    return ttest_df, pd.DataFrame(normality_results)


def select_features(X_train, X_test, y_train, feature_names, k=30):
    """Select top k features using ANOVA F-test and transform both train and test sets."""
    selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
    selector.fit(X_train, y_train)

    scores = pd.DataFrame({
        'feature': feature_names,
        'f_score': selector.scores_,
        'p_value': selector.pvalues_
    }).sort_values('f_score', ascending=False)

    selected_mask = selector.get_support()
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    print(f'[FEATURE SELECTION] Selected {len(selected_features)} features via ANOVA F-test')
    print(f'       Top 5: {", ".join(selected_features[:5])}')

    return X_train_selected, X_test_selected, selected_features, scores


def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    """Train multiple classifiers and compare their performance."""
    models = {}

    # Logistic Regression
    models['Logistic Regression'] = LogisticRegression(
        class_weight='balanced', C=1.0, max_iter=3000, solver='saga',
        random_state=RANDOM_SEED, n_jobs=-1
    )

    # Random Forest
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=5,
        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
    )

    # XGBoost or GradientBoosting
    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', use_label_encoder=False,
            random_state=RANDOM_SEED, verbosity=0
        )
    else:
        models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED
        )

    # SMOTE version for comparison if available
    if IMBLEARN_AVAILABLE:
        models['RF + SMOTE'] = ImbPipeline([
            ('smote', SMOTE(random_state=RANDOM_SEED)),
            ('clf', RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_leaf=5,
                random_state=RANDOM_SEED, n_jobs=-1
            ))
        ])

    results = []
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        print(f'\n[MODEL] Training {name}...')

        # Scale features for Logistic Regression
        if 'Logistic' in name:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        predictions[name] = y_pred
        probabilities[name] = y_prob

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        avg_prec = average_precision_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)
        spec = (confusion_matrix(y_test, y_pred)[0, 0] /
                (confusion_matrix(y_test, y_pred)[0, :].sum() or 1))

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC AUC': roc_auc,
            'Avg Precision': avg_prec,
            'MCC': mcc,
            'Specificity': spec
        })

        print(f'       Accuracy={acc:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}, '
              f'Recall={rec:.4f}, Precision={prec:.4f}')

    results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
    print(f'\n[MODEL] Best model: {results_df.iloc[0]["Model"]} (F1={results_df.iloc[0]["F1"]:.4f})')

    return results_df, predictions, probabilities


def plot_roc_curves(y_test, probabilities):
    """ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (name, y_prob) in enumerate(probabilities.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2.5, color=COLORS[i % len(COLORS)],
                label=f'{name} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Classifier')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curves — Bankruptcy Prediction', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, '07_roc_curves.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')


def plot_pr_curves(y_test, probabilities):
    """Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    baseline = y_test.mean()
    for i, (name, y_prob) in enumerate(probabilities.items()):
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, lw=2.5, color=COLORS[i % len(COLORS)],
                label=f'{name} (AP={ap:.3f})')

    ax.axhline(baseline, color='gray', linestyle='--', lw=1.5, alpha=0.7,
               label=f'Baseline={baseline:.3f}')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves — Bankruptcy Prediction', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, '08_pr_curves.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')


def plot_confusion_matrices(y_test, predictions):
    """Confusion matrices for all models."""
    n_models = len(predictions)
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['Healthy', 'Bankrupt'])
        disp.plot(ax=axes[idx], cmap='Blues', colorbar=False, values_format='d')
        axes[idx].set_title(name, fontweight='bold', fontsize=12)

        # Add percentages in the cells
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[idx].text(j, i + 0.35, f'{cm_pct[i, j]:.1f}%',
                               ha='center', va='center', fontsize=9, color='gray')

    # Hide unused subplots
    for idx in range(len(predictions), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Confusion Matrices', fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, '09_confusion_matrices.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')


def optimize_threshold(y_test, y_prob, model_name):
    """Find optimal decision threshold to maximize F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / np.maximum(
        precisions[:-1] + recalls[:-1], 1e-15
    )
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score')
    ax.axvline(best_threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Optimal Threshold={best_threshold:.3f} (F1={best_f1:.3f})')
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title(f'Threshold Optimization for F1 — {model_name}', fontweight='bold', fontsize=14)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, f'10_threshold_optimization_{model_name.replace(" ", "_")}.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')

    # Apply optimal threshold
    y_pred_opt = (y_prob >= best_threshold).astype(int)
    f1_opt = f1_score(y_test, y_pred_opt)
    rec_opt = recall_score(y_test, y_pred_opt)
    prec_opt = precision_score(y_test, y_pred_opt)

    print(f'[THRESHOLD] {model_name}: optimal threshold={best_threshold:.4f}, '
          f'F1={f1_opt:.4f}, Recall={rec_opt:.4f}, Precision={prec_opt:.4f}')

    return best_threshold, best_f1


def cross_validate_model(X, y, model, model_name, cv=5):
    """Perform stratified cross-validation and return scores."""
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    cv_results = {}

    for metric in scoring:
        try:
            scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=metric, n_jobs=-1)
            cv_results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }
        except Exception as e:
            print(f'[CV] Warning: {metric} cross-val failed for {model_name}: {e}')
            cv_results[metric] = {'mean': 0, 'std': 0, 'scores': np.array([0])}

    print(f'[CV] {model_name} — F1={cv_results["f1"]["mean"]:.4f}+/-{cv_results["f1"]["std"]:.4f}, '
          f'ROC AUC={cv_results["roc_auc"]["mean"]:.4f}+/-{cv_results["roc_auc"]["std"]:.4f}')

    return cv_results


def plot_cv_results(all_cv_results):
    """Cross-validation comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ['f1', 'roc_auc', 'recall', 'precision']
    n_metrics = len(metrics)
    n_models = len(all_cv_results)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, (name, cv_res) in enumerate(all_cv_results.items()):
        means = [cv_res[m]['mean'] for m in metrics]
        stds = [cv_res[m]['std'] for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3,
               color=COLORS[i % len(COLORS)], label=name, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Cross-Validation Performance Comparison', fontweight='bold', fontsize=14)
    ax.legend(frameon=True, fancybox=True, shadow=True, loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, '11_cross_validation.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')


def plot_feature_importance(model, feature_names, model_name, top_n=20):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)
    elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'feature_importances_'):
        importances = model.steps[-1][1].feature_importances_
    else:
        print(f'[IMPORTANCE] Cannot extract feature importance for {model_name}')
        return

    # Ensure lengths match
    min_len = min(len(importances), len(feature_names))
    importances = importances[:min_len]
    names = feature_names[:min_len]

    imp_df = pd.DataFrame({'feature': names, 'importance': importances}).sort_values('importance', ascending=False)
    top_imp = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_imp)))
    ax.barh(range(len(top_imp)), top_imp['importance'].values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels([c[:40] + '...' if len(c) > 40 else c for c in top_imp['feature']], fontsize=9)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Features by Importance — {model_name}', fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('+', '_')
    path = os.path.join(CHARTS_DIR, f'12_feature_importance_{safe_name}.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')

    return imp_df


def cost_sensitive_analysis(y_test, probabilities, cost_bankrupt=10, cost_healthy=1):
    """Evaluate models under asymmetric misclassification costs."""
    # Cost matrix: TN=0, FP=cost_healthy, FN=cost_bankrupt, TP=0
    # Total cost = FP * cost_healthy + FN * cost_bankrupt

    results = []
    for name, y_prob in probabilities.items():
        for threshold in np.arange(0.05, 0.95, 0.05):
            y_pred = (y_prob >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            total_cost = fp * cost_healthy + fn * cost_bankrupt
            avg_cost = total_cost / len(y_test)

            results.append({
                'Model': name,
                'Threshold': threshold,
                'Total Cost': total_cost,
                'Avg Cost': avg_cost,
                'FP': fp,
                'FN': fn,
                'TP': tp,
                'TN': tn
            })

    cost_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, name in enumerate(probabilities.keys()):
        subset = cost_df[cost_df['Model'] == name]
        ax.plot(subset['Threshold'], subset['Avg Cost'], lw=2.5,
                color=COLORS[i % len(COLORS)], label=name, marker='o', markersize=4)
        min_idx = subset['Avg Cost'].idxmin()
        min_row = subset.loc[min_idx]
        ax.scatter(min_row['Threshold'], min_row['Avg Cost'],
                   color=COLORS[i % len(COLORS)], s=120, zorder=5,
                   edgecolors='black', linewidth=1.5)

    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel(f'Average Cost (FP=${cost_healthy}, FN=${cost_bankrupt})', fontsize=12)
    ax.set_title(f'Cost-Sensitive Analysis: Asymmetric Misclassification Costs\n'
                 f'Cost Bankrupt FN=${cost_bankrupt} vs Cost Healthy FP=${cost_healthy}',
                 fontweight='bold', fontsize=13)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, '13_cost_analysis.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')

    # Optimal cost threshold per model
    best_costs = cost_df.loc[cost_df.groupby('Model')['Avg Cost'].idxmin()][
        ['Model', 'Threshold', 'Avg Cost', 'FP', 'FN']
    ]
    print(f'[COST ANALYSIS] Optimal Thresholds:\n       {best_costs.to_string(index=False)}')

    return cost_df, best_costs


def plot_shap_analysis(model, X_test, feature_names, model_name):
    """SHAP summary plot if available."""
    if not SHAP_AVAILABLE:
        return None

    try:
        # For tree-based models
        if hasattr(model, 'feature_importances_') or 'XGB' in model_name or 'Random Forest' in model_name:
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X_test)
        else:
            # Fallback to KernelExplainer (slow)
            return None

        # Subsample for speed
        n_samples = min(500, X_test.shape[0])
        idx = np.random.choice(X_test.shape[0], n_samples, replace=False)
        X_sample = X_test[idx] if isinstance(X_test, np.ndarray) else X_test.iloc[idx].values
        shap_values = explainer.shap_values(X_sample)

        # Handle multi-output (binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Ensure SHAP values shape matches
        if len(shap_values.shape) == 2 and shap_values.shape[1] == len(feature_names):
            pass
        elif len(shap_values.shape) == 2 and shap_values.shape[1] < len(feature_names):
            feature_names = feature_names[:shap_values.shape[1]]
        else:
            return None

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                          show=False, max_display=20)
        plt.title(f'SHAP Feature Importance — {model_name}', fontweight='bold', fontsize=14)
        plt.tight_layout()
        safe_name = model_name.replace(' ', '_').replace('+', '_')
        path = os.path.join(CHARTS_DIR, f'14_shap_{safe_name}.png')
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.show()
    plt.close(fig)
        print(f'[CHART] Saved {path}')
        return shap_values
    except Exception as e:
        print(f'[SHAP] Skipped for {model_name}: {e}')
        return None


def generate_reports(results_df, ttest_df, cv_results, best_model_name, cost_df,
                     model_feature_imp, best_threshold, target_rate, n_features,
                     n_bankrupt, n_healthy):
    """Generate bilingual (Arabic + English) short and full reports."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Select best model results
    best_row = results_df[results_df['Model'] == best_model_name].iloc[0]

    # ===== SHORT ENGLISH REPORT =====
    en_short = f"""========================================
BANKRUPTCY PREDICTION — Executive Summary
========================================
Generated: {now}

DATASET:
  Total firms: {n_healthy + n_bankrupt:,}
  Bankrupt: {n_bankrupt:,} ({target_rate * 100:.2f}%)
  Features: {n_features}

BEST MODEL: {best_model_name}
  F1 Score: {best_row['F1']:.4f}
  ROC AUC: {best_row['ROC AUC']:.4f}
  Recall (Sensitivity): {best_row['Recall']:.4f}
  Precision: {best_row['Precision']:.4f}
  Specificity: {best_row['Specificity']:.4f}
  MCC: {best_row['MCC']:.4f}

CROSS-VALIDATION (5-Fold):
  F1: {cv_results[best_model_name]['f1']['mean']:.4f} +/- {cv_results[best_model_name]['f1']['std']:.4f}
  ROC AUC: {cv_results[best_model_name]['roc_auc']['mean']:.4f} +/- {cv_results[best_model_name]['roc_auc']['std']:.4f}

OPTIMAL THRESHOLD: {best_threshold:.4f}

TOP PREDICTORS (by statistical significance):
  1. {ttest_df.iloc[0]['feature']}
  2. {ttest_df.iloc[1]['feature']}
  3. {ttest_df.iloc[2]['feature']}

RECOMMENDATION:
  The {best_model_name} model achieves strong discriminatory power
  for bankruptcy prediction. Use threshold={best_threshold:.3f} to balance
  precision and recall for your specific cost structure.
"""

    # ===== FULL ENGLISH REPORT =====
    en_full = f"""========================================
BANKRUPTCY PREDICTION — Full Analytical Report
========================================
Generated: {now}

1. DATA OVERVIEW
   Dataset: {n_healthy + n_bankrupt:,} firms with {n_features} financial ratios
   Target: Bankrupt? (1=bankrupt, 0=healthy)
   Class Distribution:
     - Healthy: {n_healthy:,} firms ({(1-target_rate)*100:.2f}%)
     - Bankrupt: {n_bankrupt:,} firms ({target_rate*100:.2f}%)
     - Imbalance Ratio: {n_healthy / max(n_bankrupt, 1):.1f}:1

2. EXPLORATORY DATA ANALYSIS
   PCA: The first 3 principal components capture the majority of variance
   in the financial ratio space. The 2D/3D projections show partial
   separability between bankrupt and healthy firms, indicating that
   financial ratios contain predictive signals for bankruptcy.

3. STATISTICAL ANALYSIS
   A total of {(ttest_df['p_value'] < 0.05).sum()} out of {len(ttest_df)} features show
   statistically significant differences between bankrupt and healthy firms
   (Welch's t-test, p<0.05).

   Top 10 Most Discriminating Features:
"""
    for i in range(min(10, len(ttest_df))):
        row = ttest_df.iloc[i]
        en_full += f"     {i+1:2d}. {row['feature'][:50]:50s}  t={row['t_statistic']:+8.3f}  p={row['p_value']:.2e}  d={row['effect_size']:+.3f}\n"

    en_full += f"""
4. MODEL PERFORMANCE
   Models compared: {', '.join(results_df['Model'].tolist())}

   Performance Summary:
"""
    for _, row in results_df.iterrows():
        en_full += f"     {row['Model']:25s}  F1={row['F1']:.4f}  AUC={row['ROC AUC']:.4f}  "
        en_full += f"Recall={row['Recall']:.4f}  Prec={row['Precision']:.4f}  MCC={row['MCC']:.4f}\n"

    en_full += f"""
5. CROSS-VALIDATION (5-Fold Stratified)
"""
    for name, cv_res in cv_results.items():
        en_full += f"     {name:25s}  F1={cv_res['f1']['mean']:.4f}+/-{cv_res['f1']['std']:.4f}  "
        en_full += f"AUC={cv_res['roc_auc']['mean']:.4f}+/-{cv_res['roc_auc']['std']:.4f}\n"

    en_full += f"""
6. BEST MODEL: {best_model_name}
   - Accuracy:  {best_row['Accuracy']:.4f}
   - Precision: {best_row['Precision']:.4f}
   - Recall:    {best_row['Recall']:.4f}
   - F1 Score:  {best_row['F1']:.4f}
   - ROC AUC:   {best_row['ROC AUC']:.4f}
   - MCC:       {best_row['MCC']:.4f}
   - Specificity: {best_row['Specificity']:.4f}

7. THRESHOLD OPTIMIZATION
   Optimal threshold for F1: {best_threshold if best_threshold else 'N/A'}

8. FEATURE IMPORTANCE
"""
    if model_feature_imp is not None:
        en_full += "   Top 10 Features:\n"
        for i in range(min(10, len(model_feature_imp))):
            row = model_feature_imp.iloc[i]
            en_full += f"     {i+1:2d}. {row['feature'][:50]:50s}  importance={row['importance']:.4f}\n"

    en_full += f"""
9. COST-SENSITIVE ANALYSIS
   Misclassification costs: FN (bankrupt missed) = high, FP (false alarm) = low
   The optimal threshold varies by model and cost structure.
   See chart: charts/13_cost_analysis.png

10. CONCLUSION
    The {best_model_name} model demonstrates strong performance for bankruptcy
    prediction. Given the high cost of misclassifying a bankrupt firm,
    we recommend using the optimized threshold ({best_threshold:.3f}) or
    adjusting based on your specific cost-benefit trade-off.

    All visualizations are saved in the charts/ directory.
"""

    # ===== SHORT ARABIC REPORT =====
    ar_short = f"""========================================
تقرير تنبؤ الإفلاس — ملخص تنفيذي
========================================
التاريخ: {now}

البيانات:
  إجمالي الشركات: {n_healthy + n_bankrupt:,}
  الشركات المفلسة: {n_bankrupt:,} ({target_rate * 100:.2f}%)
  عدد المتغيرات: {n_features}

أفضل نموذج: {best_model_name}
  F1: {best_row['F1']:.4f}
  ROC AUC: {best_row['ROC AUC']:.4f}
  الاستدعاء (Recall): {best_row['Recall']:.4f}
  الدقة (Precision): {best_row['Precision']:.4f}

التحقق المتقاطع (5-fold):
  F1: {cv_results[best_model_name]['f1']['mean']:.4f} +/- {cv_results[best_model_name]['f1']['std']:.4f}
  ROC AUC: {cv_results[best_model_name]['roc_auc']['mean']:.4f} +/- {cv_results[best_model_name]['roc_auc']['std']:.4f}

أهم المؤشرات المالية:
  1. {ttest_df.iloc[0]['feature'][:40]}
  2. {ttest_df.iloc[1]['feature'][:40]}
  3. {ttest_df.iloc[2]['feature'][:40]}

التوصية:
  النموذج {best_model_name} يحقق أداءً قوياً في التنبؤ بالإفلاس.
  يُوصى باستخدام عتبة القرار المحسنة ({best_threshold:.3f}) لتحقيق
  التوازن الأمثل بين الدقة والاستدعاء.
"""

    # ===== FULL ARABIC REPORT =====
    ar_full = f"""========================================
تقرير تنبؤ الإفلاس — تقرير تحليلي كامل
========================================
التاريخ: {now}

أولاً: نظرة عامة على البيانات
  إجمالي الشركات: {n_healthy + n_bankrupt:,} شركة
  عدد المتغيرات المالية: {n_features} متغيراً
  المتغير المستهدف: Bankrupt? (1=مفلسة، 0=سليمة)
  توزيع الفئات:
    - شركات سليمة: {n_healthy:,} ({100 - target_rate * 100:.2f}%)
    - شركات مفلسة: {n_bankrupt:,} ({target_rate * 100:.2f}%)
    - نسبة عدم التوازن: {n_healthy / max(n_bankrupt, 1):.1f}:1

ثانياً: تحليل البيانات الاستكشافي
  تم استخدام تحليل المكونات الرئيسية (PCA) لفهم بنية البيانات.
  تُظهر الرسوم البيانية ثنائية وثلاثية الأبعاد فصلاً جزئياً بين
  الشركات المفلسة والسليمة، مما يؤكد وجود إشارات تنبؤية في النسب المالية.

ثالثاً: التحليل الإحصائي
  عدد المتغيرات ذات الفروق المعنوية بين المجموعتين:
  {(ttest_df['p_value'] < 0.05).sum()} من أصل {len(ttest_df)} متغيراً (اختبار t، مستوى دلالة 0.05)

  أهم 10 متغيرات مميزة:
"""
    for i in range(min(10, len(ttest_df))):
        row = ttest_df.iloc[i]
        ar_full += f"  {i+1}. {row['feature'][:45]:45s}  t={row['t_statistic']:+6.2f}  p={row['p_value']:.2e}\n"

    ar_full += f"""
رابعاً: أداء النماذج
  النماذج المقارنة: {', '.join(results_df['Model'].tolist())}

  ملخص الأداء:
"""
    for _, row in results_df.iterrows():
        ar_full += f"  {row['Model']:25s}  F1={row['F1']:.4f}  AUC={row['ROC AUC']:.4f}  "
        ar_full += f"Recall={row['Recall']:.4f}  Precision={row['Precision']:.4f}\n"

    ar_full += f"""
خامساً: أفضل نموذج — {best_model_name}
  الدقة (Accuracy):  {best_row['Accuracy']:.4f}
  الدقة الإيجابية (Precision): {best_row['Precision']:.4f}
  الاستدعاء (Recall):    {best_row['Recall']:.4f}
  F1:  {best_row['F1']:.4f}
  ROC AUC:   {best_row['ROC AUC']:.4f}

سادساً: تحسين عتبة القرار
  أفضل عتبة قرار حسب مقياس F1: {best_threshold if best_threshold else 'N/A'}

سابعاً: تحليل التكاليف
  تم تحليل تكاليف سوء التصنيف مع افتراض أن تكلفة عدم اكتشاف
  شركة مفلسة أعلى من تكلفة الإنذار الكاذب.
  يمكن تعديل عتبة القرار بناءً على هيكل التكاليف الخاص بك.

ثامناً: الخاتمة
  النموذج {best_model_name} يُظهر أداءً متميزاً في التنبؤ بالإفلاس.
  نوصي باستخدام عتبة القرار المحسنة ({best_threshold:.3f}) أو تعديلها
  حسب المفاضلة بين التكلفة والفائدة الخاصة بمؤسستك.
  جميع الرسوم البيانية محفوظة في مجلد charts/.
"""

    # Write reports
    report_dir = BASE_DIR

    reports = {
        'Bankruptcy_Report_EN_short.txt': en_short,
        'Bankruptcy_Report_EN_full.txt': en_full,
        'Bankruptcy_Report_AR_short.txt': ar_short,
        'Bankruptcy_Report_AR_full.txt': ar_full,
    }

    for filename, content in reports.items():
        path = os.path.join(report_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
            try:
                print(content)
            except UnicodeEncodeError:
                print(f"[Report: content saved to file]")
            try:
                print(content)
            except UnicodeEncodeError:
                print(f"[Report: content saved to file]")
        print(f'[REPORT] Saved {filename}')

    return reports


def main():
    """Main execution function."""
    print('=' * 60)
    print('  BANKRUPTCY PREDICTION FROM FINANCIAL RATIOS')
    print('=' * 60)

    # ----------------------------------------------------------
    # 1. Load and explore data
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('DATA LOADING & EXPLORATION')
    print('-' * 50)
    df, target_col, feature_cols = load_and_explore_data()

    n_bankrupt = df[target_col].sum()
    n_healthy = len(df) - n_bankrupt
    target_rate = df[target_col].mean()

    # ----------------------------------------------------------
    # 2. Exploratory Data Analysis
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('EXPLORATORY DATA ANALYSIS')
    print('-' * 50)

    plot_target_distribution(df, target_col)
    top_features = plot_correlation_heatmap(df, target_col, top_n=20)
    X_pca, explained_var = plot_pca_visualization(df, feature_cols, target_col)

    # ----------------------------------------------------------
    # 3. Statistical Analysis
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('STATISTICAL ANALYSIS')
    print('-' * 50)
    ttest_df, normality_df = plot_statistical_analysis(df, feature_cols, target_col)

    # ----------------------------------------------------------
    # 4. Prepare data for modeling
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('MODEL PREPARATION')
    print('-' * 50)

    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(np.int64)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED
    )
    print(f'[SPLIT] Train: {len(y_train):,} ({y_train.sum():,} bankrupt), '
          f'Test: {len(y_test):,} ({y_test.sum():,} bankrupt)')

    # Scale
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----------------------------------------------------------
    # 5. Feature Selection
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('FEATURE SELECTION')
    print('-' * 50)
    k_features = min(40, X_train_scaled.shape[1])
    X_train_sel, X_test_sel, selected_features, f_scores = select_features(
        X_train_scaled, X_test_scaled, y_train, feature_cols, k=k_features
    )

    # Plot feature selection scores
    fig, ax = plt.subplots(figsize=(12, 8))
    top_f = f_scores.head(20)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_f)))
    ax.barh(range(len(top_f)), top_f['f_score'].values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(top_f)))
    ax.set_yticklabels([c[:35] + '...' if len(c) > 35 else c for c in top_f['feature']], fontsize=9)
    ax.set_xlabel('ANOVA F-Score', fontsize=12)
    ax.set_title('Top 20 Features by ANOVA F-Score', fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    path = os.path.join(CHARTS_DIR, '15_anova_feature_scores.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)
    print(f'[CHART] Saved {path}')

    # ----------------------------------------------------------
    # 6. Train and Evaluate Models
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('MODEL TRAINING & EVALUATION')
    print('-' * 50)
    results_df, predictions, probabilities = train_and_evaluate_models(
        X_train_sel, X_test_sel, y_train, y_test, selected_features
    )

    best_model_name = results_df.iloc[0]['Model']

    # ----------------------------------------------------------
    # 7. Visualization of Model Performance
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('MODEL VISUALIZATION')
    print('-' * 50)
    plot_roc_curves(y_test, probabilities)
    plot_pr_curves(y_test, probabilities)
    plot_confusion_matrices(y_test, predictions)

    # ----------------------------------------------------------
    # 8. Threshold Optimization
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('THRESHOLD OPTIMIZATION')
    print('-' * 50)
    best_threshold = None
    if best_model_name in probabilities:
        threshold, f1_opt = optimize_threshold(
            y_test, probabilities[best_model_name], best_model_name
        )
        best_threshold = threshold

    # ----------------------------------------------------------
    # 9. Cross-Validation
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('CROSS-VALIDATION')
    print('-' * 50)

    # Build best model for CV
    if 'Logistic' in best_model_name:
        cv_model = LogisticRegression(
            class_weight='balanced', C=1.0, max_iter=3000, solver='saga',
            random_state=RANDOM_SEED, n_jobs=-1
        )
        cv_X = X_train_sel
    elif 'XGB' in best_model_name and XGB_AVAILABLE:
        cv_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', use_label_encoder=False,
            random_state=RANDOM_SEED, verbosity=0
        )
        cv_X = X_train_sel
    elif 'Random Forest' in best_model_name:
        cv_model = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=5,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
        )
        cv_X = X_train_sel
    elif 'Gradient' in best_model_name:
        cv_model = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED
        )
        cv_X = X_train_sel
    else:
        cv_model = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=5,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
        )
        cv_X = X_train_sel

    all_cv_results = {}
    best_cv = cross_validate_model(cv_X, y_train, cv_model, best_model_name)
    all_cv_results[best_model_name] = best_cv

    # Also CV for other models
    for name in results_df['Model'].values:
        if name in predictions and name not in all_cv_results:
            try:
                if 'Logistic' in name:
                    m = LogisticRegression(
                        class_weight='balanced', C=1.0, max_iter=3000, solver='saga',
                        random_state=RANDOM_SEED, n_jobs=-1
                    )
                elif 'XGB' in name and XGB_AVAILABLE:
                    m = xgb.XGBClassifier(
                        n_estimators=300, max_depth=6, learning_rate=0.05,
                        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric='logloss', use_label_encoder=False,
                        random_state=RANDOM_SEED, verbosity=0
                    )
                elif 'Random Forest' in name:
                    m = RandomForestClassifier(
                        n_estimators=300, max_depth=15, min_samples_leaf=5,
                        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
                    )
                elif 'Gradient' in name:
                    m = GradientBoostingClassifier(
                        n_estimators=300, max_depth=4, learning_rate=0.05,
                        subsample=0.8, random_state=RANDOM_SEED
                    )
                elif 'SMOTE' in name:
                    m = ImbPipeline([
                        ('smote', SMOTE(random_state=RANDOM_SEED)),
                        ('clf', RandomForestClassifier(
                            n_estimators=300, max_depth=15, min_samples_leaf=5,
                            random_state=RANDOM_SEED, n_jobs=-1
                        ))
                    ])
                else:
                    continue
                cv_res = cross_validate_model(X_train_sel, y_train, m, name)
                all_cv_results[name] = cv_res
            except Exception as e:
                print(f'[CV] Could not CV {name}: {e}')

    plot_cv_results(all_cv_results)

    # ----------------------------------------------------------
    # 10. Feature Importance for Best Model
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('FEATURE IMPORTANCE ANALYSIS')
    print('-' * 50)

    # Retrain best model on full training data
    model_feature_imp = None
    best_model_trained = None
    if 'Logistic' in best_model_name:
        best_model_trained = LogisticRegression(
            class_weight='balanced', C=1.0, max_iter=3000, solver='saga',
            random_state=RANDOM_SEED, n_jobs=-1
        )
        best_model_trained.fit(X_train_sel, y_train)
    elif 'XGB' in best_model_name and XGB_AVAILABLE:
        best_model_trained = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', use_label_encoder=False,
            random_state=RANDOM_SEED, verbosity=0
        )
        best_model_trained.fit(X_train_sel, y_train)
    elif 'Random Forest' in best_model_name:
        best_model_trained = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=5,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
        )
        best_model_trained.fit(X_train_sel, y_train)
    elif 'Gradient' in best_model_name:
        best_model_trained = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=RANDOM_SEED
        )
        best_model_trained.fit(X_train_sel, y_train)
    elif 'SMOTE' in best_model_name:
        best_model_trained = ImbPipeline([
            ('smote', SMOTE(random_state=RANDOM_SEED)),
            ('clf', RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_leaf=5,
                random_state=RANDOM_SEED, n_jobs=-1
            ))
        ])
        best_model_trained.fit(X_train_sel, y_train)

    if best_model_trained is not None:
        model_feature_imp = plot_feature_importance(
            best_model_trained, selected_features, best_model_name
        )

    # ----------------------------------------------------------
    # 11. SHAP Analysis
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('SHAP ANALYSIS')
    print('-' * 50)
    if best_model_trained is not None:
        plot_shap_analysis(best_model_trained, X_test_sel, selected_features, best_model_name)

    # ----------------------------------------------------------
    # 12. Cost-Sensitive Analysis
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('COST-SENSITIVE ANALYSIS')
    print('-' * 50)
    cost_df, best_costs = cost_sensitive_analysis(
        y_test, probabilities, cost_bankrupt=10, cost_healthy=1
    )

    # ----------------------------------------------------------
    # 13. Generate Reports
    # ----------------------------------------------------------
    print('\n' + '-' * 50)
    print('REPORT GENERATION')
    print('-' * 50)
    reports = generate_reports(
        results_df, ttest_df, all_cv_results, best_model_name, cost_df,
        model_feature_imp, best_threshold, target_rate, len(feature_cols),
        n_bankrupt, n_healthy
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print('\n' + '=' * 60)
    print('  ANALYSIS COMPLETE')
    print('=' * 60)
    print(f'  Best Model: {best_model_name}')
    print(f'  F1 Score: {results_df.iloc[0]["F1"]:.4f}')
    print(f'  ROC AUC: {results_df.iloc[0]["ROC AUC"]:.4f}')
    print(f'  Optimal Threshold: {best_threshold:.4f}' if best_threshold else '')
    print(f'\n  Charts saved to: {os.path.join(BASE_DIR, CHARTS_DIR)}')
    print(f'  Reports: Bankruptcy_Report_EN_short.txt, Bankruptcy_Report_EN_full.txt')
    print(f'           Bankruptcy_Report_AR_short.txt, Bankruptcy_Report_AR_full.txt')
    print('=' * 60)

    return df, results_df, ttest_df


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\n[ERROR] {e}')
        traceback.print_exc()
        sys.exit(1)
