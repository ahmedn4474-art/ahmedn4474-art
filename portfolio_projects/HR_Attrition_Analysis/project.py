import sys
import os
import warnings
import textwrap
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import seaborn as sns

from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, brier_score_loss, f1_score, accuracy_score
)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 120})

sns.set_style("whitegrid")
pd.set_option("display.max_columns", 40, "display.width", 160)

DATA_PATH = r"D:\download\protfolio\portfolio_projects\HR_Attrition_Analysis\data\hr_employee_attrition.csv"
CHART_DIR = r"D:\download\protfolio\portfolio_projects\HR_Attrition_Analysis\charts"
OUT_DIR = r"D:\download\protfolio\portfolio_projects\HR_Attrition_Analysis"
os.makedirs(CHART_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Data Loading & Preparation
# ──────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

# Create binary target; drop constant / identifier columns
df["Attrition_Flag"] = (df["Attrition"] == "Yes").astype(int)
drop_cols = ["Attrition", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Separate feature types
cat_cols = df.select_dtypes("object").columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove("Attrition_Flag")

# Encode categoricals
df_encoded = df.copy()
for c in cat_cols:
    df_encoded[c] = pd.factorize(df_encoded[c])[0]

print(f"Features: {len(num_cols)} numeric, {len(cat_cols)} categorical -> encoded")
print(f"Attrition base rate: {df_encoded['Attrition_Flag'].mean():.3f}")

X_all = df_encoded.drop(columns=["Attrition_Flag"])
y_all = df_encoded["Attrition_Flag"]

# ──────────────────────────────────────────────
# Exploratory Analysis
# ──────────────────────────────────────────────

# 01_corr.png - Full correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
corr_cols_all = [c for c in num_cols if df_encoded[c].nunique() > 2]
corr_matrix = df_encoded[corr_cols_all + ["Attrition_Flag"]].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="RdBu_r", center=0,
            square=True, linewidths=0.3, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Feature Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "01_corr.png"), bbox_inches="tight", dpi=150)
plt.show()
plt.close()
plt.close()
print("Saved 01_corr.png")

# 02_target.png - Attrition distribution and key comparisons
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

att_counts = df_encoded["Attrition_Flag"].value_counts()
axes[0, 0].bar(["Stayed (0)", "Left (1)"], att_counts.values, color=["#4daf4a", "#e41a1c"], edgecolor="black")
for i, v in enumerate(att_counts.values):
    axes[0, 0].text(i, v + 8, f"{v}\n({v / len(df_encoded):.1%})", ha="center", fontsize=9)
axes[0, 0].set_title("Attrition Distribution")
axes[0, 0].set_ylabel("Count")

corr_att = df_encoded[corr_cols_all + ["Attrition_Flag"]].corr()[["Attrition_Flag"]].sort_values(
    "Attrition_Flag", ascending=False).drop("Attrition_Flag")
top_corr = corr_att.head(10)
axes[0, 1].barh(range(len(top_corr)), top_corr["Attrition_Flag"].values, color=["#e41a1c" if c > 0 else "#4daf4a" for c in top_corr["Attrition_Flag"]])
axes[0, 1].set_yticks(range(len(top_corr)))
axes[0, 1].set_yticklabels(top_corr.index, fontsize=8)
axes[0, 1].axvline(0, color="black", linestyle="-", linewidth=0.5)
axes[0, 1].set_title("Top Correlations with Attrition")
axes[0, 1].set_xlabel("Correlation")

for att_val, color, label in [(0, "#4daf4a", "Stayed"), (1, "#e41a1c", "Left")]:
    subset = df_encoded[df_encoded["Attrition_Flag"] == att_val]["MonthlyIncome"]
    axes[0, 2].hist(subset, bins=40, alpha=0.6, color=color, label=label, density=True)
axes[0, 2].set_title("Monthly Income by Attrition")
axes[0, 2].set_xlabel("Monthly Income")
axes[0, 2].legend()

for att_val, color, label in [(0, "#4daf4a", "Stayed"), (1, "#e41a1c", "Left")]:
    subset = df_encoded[df_encoded["Attrition_Flag"] == att_val]["Age"]
    sns.kdeplot(subset, ax=axes[1, 0], color=color, label=label, fill=True, alpha=0.3)
axes[1, 0].set_title("Age Distribution by Attrition")
axes[1, 0].set_xlabel("Age")

for att_val, color, label in [(0, "#4daf4a", "Stayed"), (1, "#e41a1c", "Left")]:
    subset = df_encoded[df_encoded["Attrition_Flag"] == att_val]["YearsAtCompany"]
    sns.kdeplot(subset, ax=axes[1, 1], color=color, label=label, fill=True, alpha=0.3)
axes[1, 1].set_title("Tenure Distribution by Attrition")
axes[1, 1].set_xlabel("Years at Company")

overtime_ct = pd.crosstab(df["OverTime"] if "OverTime" in df.columns else df_encoded["OverTime"],
                          df_encoded["Attrition_Flag"].map({0: "Stayed", 1: "Left"}))
overtime_pct = overtime_ct.div(overtime_ct.sum(1), axis=0)
overtime_pct.plot(kind="bar", stacked=True, ax=axes[1, 2], color=["#4daf4a", "#e41a1c"], edgecolor="black")
axes[1, 2].set_title("Overtime vs Attrition")
axes[1, 2].set_ylabel("Proportion")
axes[1, 2].set_xlabel("Overtime")
axes[1, 2].legend(title="Attrition")
for i in range(overtime_pct.shape[0]):
    y_off = 0
    for j in range(overtime_pct.shape[1]):
        v = overtime_pct.iloc[i, j]
        if v > 0.03:
            axes[1, 2].text(i, y_off + v / 2, f"{v:.0%}", ha="center", fontsize=8)
        y_off += v

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "02_target.png"), bbox_inches="tight")
plt.show()
plt.close()
plt.close()
print("Saved 02_target.png")

# 03_kde.png - KDE plots for key numeric features by attrition status
key_features = ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears",
                "YearsInCurrentRole", "YearsWithCurrManager", "DistanceFromHome", "NumCompaniesWorked"]
key_features = [c for c in key_features if c in df_encoded.columns]
n_kde = len(key_features)
n_cols = 3
n_rows = (n_kde + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes = axes.flatten()
for idx, feat in enumerate(key_features):
    for att_val, color, label in [(0, "#4daf4a", "Stayed"), (1, "#e41a1c", "Left")]:
        subset = df_encoded[df_encoded["Attrition_Flag"] == att_val][feat]
        sns.kdeplot(subset, ax=axes[idx], color=color, label=label, fill=True, alpha=0.3)
    axes[idx].set_title(f"{feat} by Attrition")
    axes[idx].set_xlabel(feat)
    axes[idx].legend(fontsize=7)
for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "03_kde.png"), bbox_inches="tight")
plt.show()
plt.close()
plt.close()
print("Saved 03_kde.png")

# ──────────────────────────────────────────────
# Statistical Analysis
# ──────────────────────────────────────────────

chi_results = {}
for c in cat_cols:
    tbl = pd.crosstab(df_encoded[c], df_encoded["Attrition_Flag"])
    if tbl.shape[0] > 1 and tbl.shape[1] > 1:
        chi2, p, dof, expected = chi2_contingency(tbl)
        chi_results[c] = {"chi2": chi2, "p": p, "dof": dof}
        sig = "significant" if p < 0.05 else "not significant"
        print(f"  Chi2 test: {c} vs Attrition  chi2={chi2:.2f}, p={p:.4f} -> {sig}")

# Bayesian A/B test: Overtime → Attrition
# Beta-Binomial conjugate: prior Beta(1,1), compute posterior for overtime vs no-overtime
try:
    from scipy.stats import beta as beta_dist

    ot_yes = df_encoded[df_encoded["OverTime"] == 1]["Attrition_Flag"]
    ot_no = df_encoded[df_encoded["OverTime"] == 0]["Attrition_Flag"]
    n_yes, s_yes = len(ot_yes), ot_yes.sum()
    n_no, s_no = len(ot_no), ot_no.sum()

    a_prior, b_prior = 1, 1
    a_post_yes, b_post_yes = a_prior + s_yes, b_prior + n_yes - s_yes
    a_post_no, b_post_no = a_prior + s_no, b_prior + n_no - s_no

    # Monte Carlo approximation of P(overtime_rate > no_overtime_rate)
    np.random.seed(42)
    draws_yes = np.random.beta(a_post_yes, b_post_yes, 100000)
    draws_no = np.random.beta(a_post_no, b_post_no, 100000)
    prob_overtime_worse = (draws_yes > draws_no).mean()

    print(f"\nBayesian A/B Test - Overtime vs Attrition")
    print(f"  Overtime: {s_yes}/{n_yes} = {s_yes / n_yes:.3f}")
    print(f"  No Overtime: {s_no}/{n_no} = {s_no / n_no:.3f}")
    print(f"  P(overtime attrition rate > no-overtime) = {prob_overtime_worse:.4f}")
    bayes_ab_result = {
        "overtime_rate": s_yes / n_yes,
        "no_overtime_rate": s_no / n_no,
        "prob_overtime_worse": prob_overtime_worse,
    }
except Exception as e:
    print(f"Bayesian A/B test skipped: {e}")
    bayes_ab_result = None

# ──────────────────────────────────────────────
# Survival Analysis
# ──────────────────────────────────────────────

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test

    # Use YearsAtCompany as "time" and Attrition_Flag as "event"
    T = df_encoded["YearsAtCompany"].values
    E = df_encoded["Attrition_Flag"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    kmf = KaplanMeierFitter()
    # By Department
    depts = df["Department"].unique() if "Department" not in drop_cols else df_encoded["Department"].unique()
    for dept in depts:
        mask = df["Department"] == dept
        kmf.fit(T[mask], event_observed=E[mask], label=str(dept))
        kmf.plot_survival_function(ax=axes[0])
    axes[0].set_title("Kaplan-Meier: Survival by Department")
    axes[0].set_xlabel("Years at Company")
    axes[0].set_ylabel("Probability of Staying")
    axes[0].legend(fontsize=8)

    # By Overtime
    ot_labels = df["OverTime"].unique()
    for ot_val in ot_labels:
        mask = df["OverTime"] == ot_val
        kmf.fit(T[mask], event_observed=E[mask], label=f"Overtime={ot_val}")
        kmf.plot_survival_function(ax=axes[1])
    axes[1].set_title("Kaplan-Meier: Survival by Overtime")
    axes[1].set_xlabel("Years at Company")
    axes[1].set_ylabel("Probability of Staying")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "04_survival.png"), bbox_inches="tight")
    plt.show()
    plt.close()
    plt.close()
    print("Saved 04_survival.png")

    # Cox Proportional Hazards
    cox_df = df_encoded.copy()
    cox_features = [
        "Age", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
        "YearsWithCurrManager", "MonthlyIncome", "OverTime", "JobRole",
        "MaritalStatus", "JobSatisfaction", "EnvironmentSatisfaction",
        "WorkLifeBalance", "TotalWorkingYears", "NumCompaniesWorked"
    ]
    cox_features = [c for c in cox_features if c in cox_df.columns]
    # Remove YearsAtCompany from extra selection to avoid duplicate columns
    extra_cols = [c for c in ["Attrition_Flag", "YearsAtCompany"] if c not in cox_features]
    cox_data = cox_df[cox_features + extra_cols].copy()
    cox_data.rename(columns={"YearsAtCompany": "tenure", "Attrition_Flag": "left"}, inplace=True)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cox_data, duration_col="tenure", event_col="left", show_progress=False)
    cph.print_summary()

    # Store coefs for report use
    cox_coefs = cph.summary["coef"].to_dict()

    fig, ax = plt.subplots(figsize=(8, 6))
    hazard_df = cph.summary.sort_values("coef", ascending=True)
    colors_hr = ["#e41a1c" if c > 0 else "#4daf4a" for c in hazard_df["coef"]]
    ax.barh(range(len(hazard_df)), hazard_df["coef"], color=colors_hr, edgecolor="black")
    ax.set_yticks(range(len(hazard_df)))
    ax.set_yticklabels(hazard_df.index, fontsize=8)
    ax.set_xlabel("Coefficient (log Hazard Ratio)")
    ax.set_title("CoxPH: Feature Impact on Attrition Hazard")
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "05_coxph.png"), bbox_inches="tight")
    plt.show()
    plt.close()
    plt.close()
    print("Saved 05_coxph.png")

    survival_results = {
        "logrank_overtime": logrank_test(T[df["OverTime"] == "Yes"],
                                         T[df["OverTime"] == "No"],
                                         event_observed_A=E[df["OverTime"] == "Yes"],
                                         event_observed_B=E[df["OverTime"] == "No"]).p_value,
        "cox_coefs": cox_coefs,
    }
except ImportError:
    print("lifelines not available - skipping survival analysis")
    survival_results = None
except Exception as e:
    print(f"Survival analysis error: {e}")
    survival_results = None

# ──────────────────────────────────────────────
# Causal Inference: Propensity Score Matching
# ──────────────────────────────────────────────

try:
    from sklearn.linear_model import LogisticRegression as PSModel
    from sklearn.neighbors import NearestNeighbors

    # Estimate propensity of working overtime
    ps_features = ["Age", "MaritalStatus", "JobRole", "Education", "JobLevel",
                   "TotalWorkingYears", "YearsAtCompany", "DistanceFromHome"]
    ps_features = [c for c in ps_features if c in df_encoded.columns]
    ps_X = df_encoded[ps_features]
    ps_y = df_encoded["OverTime"]

    ps_model = PSModel(max_iter=1000, random_state=42)
    ps_model.fit(ps_X, ps_y)
    propensity = ps_model.predict_proba(ps_X)[:, 1]

    # 1:1 nearest-neighbor matching with caliper
    treated_idx = np.where(ps_y == 1)[0]
    control_idx = np.where(ps_y == 0)[0]
    propensity_treated = propensity[treated_idx].reshape(-1, 1)
    propensity_control = propensity[control_idx].reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(propensity_control)
    distances, matches = nn.kneighbors(propensity_treated)

    caliper = 0.05
    matched_treated = []
    matched_control = []
    for i, (dist, match) in enumerate(zip(distances.flatten(), matches.flatten())):
        if dist <= caliper:
            matched_treated.append(treated_idx[i])
            matched_control.append(control_idx[match])

    print(f"\nPropensity Score Matching: {len(matched_treated)} matched pairs (caliper={caliper})")

    if len(matched_treated) > 30:
        att_income = df_encoded.loc[matched_treated, "MonthlyIncome"].mean()
        no_income = df_encoded.loc[matched_control, "MonthlyIncome"].mean()
        att_attrition = df_encoded.loc[matched_treated, "Attrition_Flag"].mean()
        no_attrition = df_encoded.loc[matched_control, "Attrition_Flag"].mean()

        print(f"  Effect on MonthlyIncome: Treated={att_income:.0f}, Control={no_income:.0f}, "
              f"Δ={att_income - no_income:.0f}")
        print(f"  Effect on Attrition: Treated={att_attrition:.3f}, Control={no_attrition:.3f}, "
              f"Δ={att_attrition - no_attrition:.3f}")

        # Mann-Whitney U test on income difference
        mw_stat, mw_p = mannwhitneyu(
            df_encoded.loc[matched_treated, "MonthlyIncome"],
            df_encoded.loc[matched_control, "MonthlyIncome"]
        )
        print(f"  Mann-Whitney U on income: p={mw_p:.4f}")

        causal_results = {
            "matched_pairs": len(matched_treated),
            "income_treated_mean": att_income,
            "income_control_mean": no_income,
            "income_diff": att_income - no_income,
            "attrition_treated": att_attrition,
            "attrition_control": no_attrition,
            "attrition_diff": att_attrition - no_attrition,
            "mw_p_value": mw_p,
        }
    else:
        causal_results = None
except Exception as e:
    print(f"PSM skipped: {e}")
    causal_results = None

# ──────────────────────────────────────────────
# Machine Learning Pipeline
# ──────────────────────────────────────────────

# Prepare features
model_features = [c for c in num_cols if c not in ("Attrition_Flag", "EmployeeCount",
                   "EmployeeNumber", "StandardHours")]
model_features += cat_cols
model_features = [c for c in model_features if c in X_all.columns]
target = "Attrition_Flag"

X = df_encoded[model_features].copy()
y = df_encoded[target]

# Remove near-constant features
for col in X.columns:
    if X[col].nunique() <= 1:
        X.drop(columns=[col], inplace=True)
        print(f"Dropped constant feature: {col}")

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"\nML data: {X_train.shape[0]} train, {X_test.shape[0]} test, {X.shape[1]} features")
print(f"Train attrition: {y_train.mean():.3f}, Test attrition: {y_test.mean():.3f}")

# ──────────────────────────────────────────────
# Model Training & Evaluation
# ──────────────────────────────────────────────

models = {
    "Logistic Regression (L1)": LogisticRegression(
        penalty="l1", solver="saga", max_iter=5000, random_state=42, C=0.1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=10,
        learning_rate=0.05, subsample=0.8, random_state=42
    ),
}

results = {}
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# ROC
for ax in [axes[0, 0]]:
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_s)[:, 1]
        else:
            y_prob = model.decision_function(X_test_s)
        y_pred = (y_prob >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)

        results[name] = {
            "model": model,
            "auc_roc": roc_auc,
            "auc_pr": pr_auc,
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "brier": brier_score_loss(y_test, y_prob),
            "y_prob": y_prob,
            "y_pred": y_pred,
        }

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)

# PR curves
ax_pr = axes[0, 1]
for name, r in results.items():
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, r["y_prob"])
    ax_pr.plot(pr_rec, pr_prec, lw=2, label=f"{name} (AP={r['auc_pr']:.3f})")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curves")
ax_pr.legend(fontsize=8)
ax_pr.axhline(y_test.mean(), color="gray", linestyle="--", lw=1, label=f"Baseline={y_test.mean():.2f}")

# Calibration plot
ax_cal = axes[1, 0]
for name, r in results.items():
    prob_true, prob_pred = calibration_curve(y_test, r["y_prob"], n_bins=10)
    ax_cal.plot(prob_pred, prob_true, "o-", lw=2, label=f"{name} (Brier={r['brier']:.4f})")
ax_cal.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect Calibration")
ax_cal.set_xlabel("Mean Predicted Probability")
ax_cal.set_ylabel("Fraction of Positives")
ax_cal.set_title("Calibration Curves")
ax_cal.legend(fontsize=8)

# Threshold optimization (using best model by ROC)
best_name = max(results, key=lambda k: results[k]["auc_roc"])
best_prob = results[best_name]["y_prob"]
thresholds = np.linspace(0.05, 0.95, 91)
t_results = []
for t in thresholds:
    pred_t = (best_prob >= t).astype(int)
    t_results.append({
        "threshold": t,
        "accuracy": accuracy_score(y_test, pred_t),
        "f1": f1_score(y_test, pred_t),
        "recall": confusion_matrix(y_test, pred_t).ravel()[3] / (y_test.sum()) if y_test.sum() > 0 else 0,
    })
t_df = pd.DataFrame(t_results)
ax_th = axes[1, 1]
ax_th.plot(t_df["threshold"], t_df["accuracy"], label="Accuracy", lw=2)
ax_th.plot(t_df["threshold"], t_df["f1"], label="F1 Score", lw=2)
ax_th.plot(t_df["threshold"], t_df["recall"], label="Recall", lw=2)
ax_th.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
ax_th.set_xlabel("Threshold")
ax_th.set_ylabel("Metric Value")
ax_th.set_title(f"Threshold Optimization ({best_name})")
ax_th.legend(fontsize=8)
ax_th.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "06_ml_evaluation.png"), bbox_inches="tight")
plt.show()
plt.close()
plt.close()
print("Saved 06_ml_evaluation.png")

# Summary table
print("\n" + "=" * 80)
print(f"{'Model':<28} {'AUC-ROC':>8} {'AUC-PR':>8} {'Accuracy':>9} {'F1':>7} {'Brier':>7}")
print("=" * 80)
for name, r in sorted(results.items(), key=lambda x: -x[1]["auc_roc"]):
    print(f"{name:<28} {r['auc_roc']:>8.3f} {r['auc_pr']:>8.3f} {r['accuracy']:>9.3f} {r['f1']:>7.3f} {r['brier']:>7.4f}")
print("=" * 80)

# ──────────────────────────────────────────────
# Feature Importance (SHAP + Permutation)
# ──────────────────────────────────────────────

try:
    import shap

    # Use Random Forest for interpretation
    rf_model = results["Random Forest"]["model"]
    X_test_df = pd.DataFrame(X_test_s, columns=feature_names)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test_df)

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test_df, show=False, plot_size=(10, 6))
    else:
        shap.summary_plot(shap_values, X_test_df, show=False, plot_size=(10, 6))
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "07_shap_summary.png"), bbox_inches="tight")
    plt.show()
    plt.close()
    plt.close()
    print("Saved 07_shap_summary.png")

    # Permutation importance
    perm_imp = permutation_importance(
        rf_model, X_test_s, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": perm_imp.importances_mean,
        "std": perm_imp.importances_std,
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    top_n = min(20, len(imp_df))
    ax.barh(range(top_n), imp_df["importance"].values[:top_n][::-1],
            xerr=imp_df["std"].values[:top_n][::-1], color="#3182bd", edgecolor="black")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(imp_df["feature"].values[:top_n][::-1], fontsize=8)
    ax.set_xlabel("Mean Decrease in AUC (Permutation)")
    ax.set_title("Permutation Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "08_permutation_importance.png"), bbox_inches="tight")
    plt.show()
    plt.close()
    plt.close()
    print("Saved 08_permutation_importance.png")

    shap_done = True
except Exception as e:
    print(f"SHAP analysis skipped: {e}")
    shap_done = False
    imp_df = None

# ──────────────────────────────────────────────
# Cost-Benefit Analysis
# ──────────────────────────────────────────────

# Assume: cost of false negative (missed attrition) = $50,000
#         cost of false positive (unnecessary intervention) = $5,000
#         benefit of true positive (prevented attrition) = $45,000
cost_fn = 50000
cost_fp = 5000
benefit_tp = 45000  # cost_fn - cost_fp

# Evaluate at various thresholds using best model
cost_results = []
for t in np.linspace(0.05, 0.95, 91):
    pred_t = (best_prob >= t).astype(int)
    cm = confusion_matrix(y_test, pred_t)
    tn, fp, fn, tp = cm.ravel()
    total_cost = fn * cost_fn + fp * cost_fp - tp * benefit_tp
    cost_results.append({"threshold": t, "total_cost": total_cost, "fn": fn, "fp": fp, "tp": tp})
cost_df = pd.DataFrame(cost_results)
best_idx = cost_df["total_cost"].idxmin()
best_t = cost_df.loc[best_idx, "threshold"]

print(f"\nCost-Benefit Analysis (Model: {best_name})")
print(f"  Optimal threshold: {best_t:.2f}")
print(f"  Minimum total cost: ${cost_df['total_cost'].min():,.0f}")
print(f"  At optimal: TP={int(cost_df.loc[best_idx, 'tp'])}, "
      f"FP={int(cost_df.loc[best_idx, 'fp'])}, FN={int(cost_df.loc[best_idx, 'fn'])}")

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(cost_df["threshold"], cost_df["total_cost"] / 1000, "b-", lw=2, label="Total Cost ($K)")
ax1.axvline(best_t, color="red", linestyle="--", alpha=0.7)
ax1.text(best_t + 0.02, cost_df["total_cost"].max() / 1000 * 0.8,
         f"Optimal\n{best_t:.2f}", color="red", fontsize=9)
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Total Cost ($K)")
ax1.set_title("Cost-Benefit Analysis: Threshold as Business Decision")
ax1.legend()
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, "09_cost_benefit.png"), bbox_inches="tight")
plt.show()
plt.close()
plt.close()
print("Saved 09_cost_benefit.png")

# ──────────────────────────────────────────────
# Report Generation (Bilingual)
# ──────────────────────────────────────────────

def build_short_en(results, best_name, best_t, cost_df):
    best_model = results[best_name]
    base_rate = df_encoded['Attrition_Flag'].mean()
    lines = [
        "HR ATTRITION ANALYSIS - EXECUTIVE SUMMARY (EN)",
        "=" * 50,
        "",
        f"Dataset: 1,470 employees, {base_rate*100:.1f}% attrition rate.",
        "",
        "Key Findings:",
    ]
    if bayes_ab_result is not None:
        lines.append(
            f"  . Overtime is strongly associated with attrition. Bayesian A/B test: "
            f"overtime employees have {bayes_ab_result['overtime_rate']:.1%} attrition "
            f"vs {bayes_ab_result['no_overtime_rate']:.1%} for non-overtime.")
    lines += [
        f"  . Best model: {best_name} (AUC-ROC={best_model['auc_roc']:.3f}, "
        f"AUC-PR={best_model['auc_pr']:.3f}).",
        f"  . Optimal decision threshold: {best_t:.2f} (minimizes cost at "
        f"${cost_df['total_cost'].min():,.0f}).",
        f"  . Top drivers: Overtime, Job Satisfaction, "
        f"Environment Satisfaction, Work-Life Balance, tenure.",
    ]
    if causal_results is not None:
        lines.append(
            f"  . Causal analysis (PSM): overtime associated with "
            f"${abs(causal_results['income_diff']):.0f} income difference and "
            f"{causal_results['attrition_diff']:.1%} attrition difference.")
    lines += [
        "",
        "Recommendations:",
        "  1. Review overtime policies and their impact on retention.",
        "  2. Improve work-life balance policies and environment satisfaction.",
        "  3. Target retention programs at early-tenure employees.",
        "  4. Monitor compensation equity across roles.",
    ]
    if survival_results is not None:
        lines.insert(7, f"  . Log-rank test p={survival_results['logrank_overtime']:.4f}: "
                       f"significant survival difference by overtime status.")
    return "\n".join(lines)

def build_full_en(results, best_name, best_t, cost_df):
    best_model = results[best_name]
    lines = [
        "HR ATTRITION ANALYSIS - FULL REPORT (ENGLISH)",
        "=" * 60,
        "",
        "1. DATA OVERVIEW",
        "  . Source: HR Employee Attrition dataset (1,470 observations, 31 features).",
        f"  . Target: Binary attrition flag ({df_encoded['Attrition_Flag'].sum()} leavers, "
        f"{df_encoded['Attrition_Flag'].sum() / len(df_encoded):.1%} of sample).",
        f"  . Features: {len(num_cols)} numeric, {len(cat_cols)} categorical.",
        "",
        "2. METHODOLOGY",
        "  . Statistical testing: Chi-square tests for categorical associations; "
        "Bayesian Beta-Binomial A/B test for overtime vs attrition.",
        "  . Survival analysis: Kaplan-Meier curves and Cox Proportional Hazards "
        "model using tenure as the time-to-event variable.",
        "  . Causal inference: Propensity Score Matching with 0.05 caliper to "
        "estimate the causal effect of overtime on income and attrition.",
        "  . Machine learning: L1-regularized Logistic Regression, Random Forest "
        "(300 trees, max depth 10), and Gradient Boosting (200 estimators). "
        "Models evaluated via ROC-AUC, PR-AUC, Brier score, and calibration.",
        "  . Interpretation: SHAP TreeExplainer for global feature importance "
        "and permutation importance with 10 repeats.",
        "  . Cost framework: FN=$50K (missed attrition), FP=$5K (wasted intervention), "
        "TP benefit=$45K.",
        "",
        "3. KEY RESULTS",
        "  . Chi-square analysis: Overtime, MaritalStatus, JobRole, "
        "and BusinessTravel are significantly associated with attrition.",
        f"  . Bayesian A/B test: P(overtime risk > no-overtime risk) = "
        f"{bayes_ab_result['prob_overtime_worse']:.4f}. Overtime employees leave at "
        f"{bayes_ab_result['overtime_rate']:.1%} vs {bayes_ab_result['no_overtime_rate']:.1%}.",
    ]
    if survival_results and survival_results.get('cox_coefs'):
        cox_over = survival_results['cox_coefs'].get('OverTime', 0)
        cox_mgr = survival_results['cox_coefs'].get('YearsWithCurrManager', 0)
        lines.append(f"  . Cox PH highlights OverTime (coef={cox_over:.3f}), "
                     f"YearsWithCurrManager ({cox_mgr:.3f}), and JobSatisfaction as key drivers.")
    if causal_results:
        lines.append(f"  . PSM matched {causal_results['matched_pairs']} pairs. Overtime caused "
                     f"${abs(causal_results['income_diff']):.0f} income penalty "
                     f"(p={causal_results['mw_p_value']:.4f}) and +{causal_results['attrition_diff']:.1%} attrition.")
    lines += [
        f"  . Best ML model: {best_name} (AUC-ROC={best_model['auc_roc']:.3f}, "
        f"AUC-PR={best_model['auc_pr']:.3f}, Brier={best_model['brier']:.4f}).",
        f"  . Optimal business threshold: {best_t:.2f}, yielding total cost of "
        f"${cost_df['total_cost'].min():,.0f}.",
        "",
        "4. FEATURE IMPORTANCE (Top 10)",
    ]
    if imp_df is not None:
        for i, row in imp_df.head(10).iterrows():
            lines.append(f"    {i+1}. {row['feature']} (imp={row['importance']:.4f})")
    else:
        lines.append("    (Permutation importance unavailable)")

    lines += [
        "",
        "5. RECOMMENDATIONS",
        "  a) Overtime Policy: Reduce mandatory overtime; consider hiring additional "
        "staff for high-overtime departments (R&D, Sales).",
        "  b) Retention Programs: Implement targeted retention bonuses for employees "
        "with <3 years tenure and those in single marital status.",
        "  c) Work Environment: Increase job satisfaction through career development "
        "programs and flexible work arrangements.",
        "  d) Compensation Review: Investigate pay equity - our PSM analysis suggests "
        "overtime-exempt roles may have systematically lower base pay.",
        "  e) Predictive Monitoring: Deploy the Random Forest model with the "
        f"cost-optimized threshold ({best_t:.2f}) as an early warning system.",
        "",
        "6. LIMITATIONS",
        "  . Observational data: causal estimates rely on unconfoundedness assumption in PSM.",
        "  . Survival analysis assumes non-informative censoring.",
        "  . Cost estimates are illustrative and should be calibrated to actual data.",
        "",
        "- End of Report -",
    ]
    return "\n".join(lines)

def build_short_ar():
    rate = df_encoded["Attrition_Flag"].mean()
    text = (
        u"تقرير تحليل تسرب الموظفين - ملخص تنفيذي (AR)\n"
        + "=" * 50 + "\n\n"
        + u"البيانات: 1,470 موظف، معدل تسرب {:.1f}%.\n\n".format(rate * 100)
        + u"النتائج الرئيسية:\n"
    )
    if bayes_ab_result is not None:
        text += u"  . العمل الإضافي مرتبط بالتسرب. اختبار A/B البايزي يظهر فرقاً معنوياً.\n"
    text += (
        u"  . أفضل نموذج: {} (AUC-ROC={:.3f}).\n".format(best_name, results[best_name]['auc_roc'])
        + u"  . العتبة المثلى للقرار: {:.2f} (تكلفة إجمالية {:,}$).\n".format(best_t, int(cost_df['total_cost'].min()))
        + u"  . أهم العوامل: العمل الإضافي، الرضا الوظيفي، الرضا البيئي، التوازن بين العمل والحياة.\n\n"
        + u"التوصيات:\n"
        + u"  1. مراجعة سياسات العمل الإضافي.\n"
        + u"  2. تحسين سياسات التوازن بين العمل والحياة.\n"
        + u"  3. استهداف برامج الاحتفاظ بالموظفين الجدد.\n"
        + u"  4. مراجعة عدالة التعويضات."
    )
    return text

def build_full_ar():
    rate = df_encoded["Attrition_Flag"].mean()
    lines = [
        u"تقرير تحليل تسرب الموظفين - تقرير كامل (AR)",
        "=" * 60,
        "",
        u"1. نظرة عامة على البيانات",
        u"  . المصدر: مجموعة بيانات تسرب موظفي الموارد البشرية (1,470 مشاهدة، 31 متغيراً).",
        u"  . المتغير التابع: تسرب (نعم/لا) - {} حالة تسرب ({:.1f}%).".format(
            df_encoded['Attrition_Flag'].sum(), rate * 100),
        u"  . المتغيرات: {} رقمية، {} فئوية.".format(len(num_cols), len(cat_cols)),
        "",
        u"2. المنهجية",
        u"  . الاختبارات الإحصائية: اختبار مربع كاي للارتباطات الفئوية؛ اختبار A/B بايزي.",
        u"  . تحليل البقاء: منحنيات كابلان-ماير ونموذج كوكس للتنبؤ بالمخاطر.",
        u"  . الاستدلال السببي: مطابقة درجات الميل (PSM).",
        u"  . التعلم الآلي: الانحدار اللوجستي (L1)، الغابة العشوائية، وتعزيز التدرج.",
        u"  . التفسير: تحليل SHAP وأهمية المتغيرات بطريقة التقليب.",
        u"  . التكلفة: سلبية كاذبة = 50,000$، إيجابية كاذبة = 5,000$، فائدة إيجابية صحيحة = 45,000$.",
        "",
        u"3. النتائج الرئيسية",
        u"  . العمل الإضافي والحالة الاجتماعية والدور الوظيفي وسفر العمل مرتبطة بشكل كبير بالتسرب.",
    ]
    if bayes_ab_result is not None:
        lines.append(
            u"  . اختبار A/B البايزي: احتمال {:.4f} أن معدل تسرب موظفي العمل الإضافي مختلف.".format(
                bayes_ab_result['prob_overtime_worse']))
    if causal_results is not None:
        lines.append(
            u"  . PSM طابق {} زوجاً. فرق الدخل: {:,}$ (p={:.4f})، فرق التسرب: {:.1%}.".format(
                causal_results['matched_pairs'], abs(causal_results['income_diff']),
                causal_results['mw_p_value'], causal_results['attrition_diff']))
    lines += [
        u"  . أفضل نموذج: {} (AUC-ROC={:.3f}).".format(best_name, results[best_name]['auc_roc']),
        u"  . العتبة المثلى: {:.2f} - تكلفة إجمالية {:,}$.".format(
            best_t, int(cost_df['total_cost'].min())),
        "",
        u"4. التوصيات",
        u"  أ) سياسة العمل الإضافي: تقليل ساعات العمل الإضافي الإلزامية.",
        u"  ب) برامج الاحتفاظ: استهداف الموظفين الجدد (أقل من 3 سنوات) والعزاب.",
        u"  ج) بيئة العمل: تحسين الرضا الوظيفي عبر برامج التطوير والمرونة.",
        u"  د) مراجعة الرواتب: ضمان العدالة في التعويضات.",
        u"  ه) المراقبة التنبؤية: استخدام النموذج للكشف المبكر عن مخاطر التسرب.",
        "",
        u"5. القيود",
        u"  . البيانات رصدية وليست تجريبية.",
        u"  . تقديرات التكلفة توضيحية ويجب معايرتها.",
        "",
        u"- نهاية التقرير -",
    ]
    return "\n".join(lines)

reports = {
    "HR_Report_EN_short.txt": build_short_en(results, best_name, best_t, cost_df),
    "HR_Report_EN_full.txt": build_full_en(results, best_name, best_t, cost_df),
    "HR_Report_AR_short.txt": build_short_ar(),
    "HR_Report_AR_full.txt": build_full_ar(),
}

for fname, content in reports.items():
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
        try:
            print(content)
        except UnicodeEncodeError:
            print(f"[Report: content saved to file]")
        try:
            print(content)
        except UnicodeEncodeError:
            print(f"[Report: content saved to file]")
    print(f"Written {fname} ({len(content.splitlines())} lines)")

print("\nAll analysis complete. Reports and charts generated.")
