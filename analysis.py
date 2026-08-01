import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = "D:\\download\\protfolio\\analysis_output"
os.makedirs(OUT, exist_ok=True)

log_path = os.path.join(OUT, "analysis_report.txt")
log = open(log_path, 'w', encoding='utf-8')

def w(text=""):
    log.write(text + "\n")
    try:
        print(text)
    except:
        print("[output written to file]")

def banner(title):
    w("\n" + "="*85)
    w(f"  {title}")
    w("="*85)

def section(title):
    w(f"\n{'─'*70}")
    w(f"  {title}")
    w(f"{'─'*70}")

# =============================================================
# 1. HR EMPLOYEE ATTRITION
# =============================================================
banner("1. HR EMPLOYEE ATTRITION -- DESCRIPTIVE & FREQUENCY STATISTICS")

hr = pd.read_csv("D:\\download\\protfolio\\archive\\WA_Fn-UseC_-HR-Employee-Attrition.csv")
w(f"\n>> Total employees: {len(hr)}")
w(f">> Total variables: {len(hr.columns)}")

section("Frequency Statistics -- Categorical Variables")
for col in ['Attrition', 'Gender', 'Department', 'JobRole', 'EducationField', 
            'MaritalStatus', 'OverTime', 'BusinessTravel', 'JobSatisfaction',
            'WorkLifeBalance', 'PerformanceRating']:
    freq = hr[col].value_counts()
    pct = hr[col].value_counts(normalize=True).mul(100).round(1)
    w(f"\n  {col}:")
    for k in freq.index:
        w(f"    {str(k):30s} -> {freq[k]:5d}  ({pct[k]:5.1f}%)")

section("Frequency Statistics -- Binned Numeric Variables")
age_bins = [18, 25, 35, 45, 55, 65]
hr['AgeGroup'] = pd.cut(hr['Age'], bins=age_bins, labels=['18-25','26-35','36-45','46-55','56-65'])
w(f"\n  Age Groups:")
for k, v in hr['AgeGroup'].value_counts().sort_index().items():
    w(f"    {k}: {v}")

income_bins = [0, 3000, 6000, 10000, 15000, 20000]
hr['IncomeGroup'] = pd.cut(hr['MonthlyIncome'], bins=income_bins, 
                           labels=['<3K','3K-6K','6K-10K','10K-15K','15K+'])
w(f"\n  Monthly Income Groups:")
for k, v in hr['IncomeGroup'].value_counts().sort_index().items():
    w(f"    {k}: {v}")

years_bins = [0, 2, 5, 10, 20, 40]
hr['TenureGroup'] = pd.cut(hr['YearsAtCompany'], bins=years_bins,
                           labels=['<2yr','2-5yr','5-10yr','10-20yr','20+yr'])
w(f"\n  Years at Company:")
for k, v in hr['TenureGroup'].value_counts().sort_index().items():
    w(f"    {k}: {v}")

section("Descriptive Statistics -- Numeric Variables")
num_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate',
            'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears',
            'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion']
desc = hr[num_cols].describe().round(2)
w(f"\n{desc.to_string()}")

# =============================================================
# A/B TESTING
# =============================================================
banner("2. A/B TESTING -- HYPOTHESIS TESTS ON HR ATTRITION")

section("A/B Test 1: OverTime vs Attrition (Chi-square)")
ct = pd.crosstab(hr['OverTime'], hr['Attrition'])
w(f"\n  Contingency Table:\n{ct}\n")
chi2, p, dof, expected = stats.chi2_contingency(ct)
w(f"  Chi-square = {chi2:.4f}, p-value = {p:.6f}, df = {dof}")
w(f"  >> {'SIGNIFICANT (p < 0.05)' if p < 0.05 else 'NOT significant'}")
yes_ot = hr[hr['OverTime']=='Yes']['Attrition'].value_counts(normalize=True)['Yes']*100
no_ot = hr[hr['OverTime']=='No']['Attrition'].value_counts(normalize=True)['Yes']*100
w(f"  Attrition rate with OverTime: {yes_ot:.1f}%")
w(f"  Attrition rate without OverTime: {no_ot:.1f}%")
w(f"  Relative Risk: {yes_ot/no_ot:.2f}x")

section("A/B Test 2: Gender vs Attrition (Chi-square)")
ct2 = pd.crosstab(hr['Gender'], hr['Attrition'])
w(f"\n  Contingency Table:\n{ct2}\n")
chi2_2, p_2 = stats.chi2_contingency(ct2)[:2]
w(f"  Chi-square = {chi2_2:.4f}, p-value = {p_2:.6f}")
w(f"  >> {'SIGNIFICANT difference between genders' if p_2 < 0.05 else 'NO significant difference'}")
m_attr = hr[hr['Gender']=='Male']['Attrition'].value_counts(normalize=True)['Yes']*100
f_attr = hr[hr['Gender']=='Female']['Attrition'].value_counts(normalize=True)['Yes']*100
w(f"  Male attrition rate: {m_attr:.1f}%")
w(f"  Female attrition rate: {f_attr:.1f}%")

section("A/B Test 3: Department vs Attrition (Chi-square)")
ct3 = pd.crosstab(hr['Department'], hr['Attrition'])
w(f"\n  Contingency Table:\n{ct3}\n")
chi2_3, p_3 = stats.chi2_contingency(ct3)[:2]
w(f"  Chi-square = {chi2_3:.4f}, p-value = {p_3:.6f}")
w(f"  >> {'Departments differ significantly' if p_3 < 0.05 else 'No difference'}")
for dept in ct3.index:
    rate = ct3.loc[dept, 'Yes'] / ct3.loc[dept].sum() * 100
    w(f"  {dept}: {rate:.1f}%")

section("A/B Test 4: T-Test -- MonthlyIncome by Attrition")
yes_inc = hr[hr['Attrition']=='Yes']['MonthlyIncome']
no_inc = hr[hr['Attrition']=='No']['MonthlyIncome']
t_stat, p_t = stats.ttest_ind(yes_inc, no_inc)
w(f"\n  Mean Income (Attrition=Yes): {yes_inc.mean():.0f}")
w(f"  Mean Income (Attrition=No): {no_inc.mean():.0f}")
w(f"  t-statistic = {t_stat:.4f}, p-value = {p_t:.6f}")
w(f"  >> {'SIGNIFICANT income difference' if p_t < 0.05 else 'No significant difference'}")

section("A/B Test 5: T-Test -- YearsAtCompany by Attrition")
yes_yr = hr[hr['Attrition']=='Yes']['YearsAtCompany']
no_yr = hr[hr['Attrition']=='No']['YearsAtCompany']
t2, p_t2 = stats.ttest_ind(yes_yr, no_yr)
w(f"\n  Mean Years (Attrition=Yes): {yes_yr.mean():.2f}")
w(f"  Mean Years (Attrition=No): {no_yr.mean():.2f}")
w(f"  t-statistic = {t2:.4f}, p-value = {p_t2:.6f}")
w(f"  >> {'SIGNIFICANT' if p_t2 < 0.05 else 'NOT significant'}")

section("A/B Test 6: ANOVA -- JobSatisfaction by Attrition")
from scipy.stats import f_oneway
sat_yes = hr[hr['Attrition']=='Yes']['JobSatisfaction']
sat_no = hr[hr['Attrition']=='No']['JobSatisfaction']
f_stat, p_f = f_oneway(sat_yes, sat_no)
w(f"\n  Mean Satisfaction (Attrition=Yes): {sat_yes.mean():.2f}")
w(f"  Mean Satisfaction (Attrition=No): {sat_no.mean():.2f}")
w(f"  F-statistic = {f_stat:.4f}, p-value = {p_f:.6f}")
w(f"  >> {'SIGNIFICANT difference in job satisfaction' if p_f < 0.05 else 'NO significant difference'}")

section("A/B Test 7: BusinessTravel vs Attrition (Chi-square)")
ct_bt = pd.crosstab(hr['BusinessTravel'], hr['Attrition'])
w(f"\n{ct_bt}\n")
chi2_bt, p_bt = stats.chi2_contingency(ct_bt)[:2]
w(f"  Chi-square = {chi2_bt:.4f}, p-value = {p_bt:.6f}")
w(f"  >> {'SIGNIFICANT' if p_bt < 0.05 else 'NOT significant'}")
for level in ct_bt.index:
    rate = ct_bt.loc[level, 'Yes'] / ct_bt.loc[level].sum() * 100
    w(f"  {level}: Attrition = {rate:.1f}%")

# =============================================================
# 3. BAYESIAN STATISTICS
# =============================================================
banner("3. BAYESIAN ANALYSIS")

from scipy.stats import beta as beta_dist

a_prior, b_prior = 1, 1

section("Bayesian A/B: Attrition Rate -- OverTime=Yes vs OverTime=No")

n_yes_ot = len(hr[hr['OverTime']=='Yes'])
k_yes_ot = hr[(hr['OverTime']=='Yes') & (hr['Attrition']=='Yes')].shape[0]
a_post_yes = a_prior + k_yes_ot
b_post_yes = b_prior + (n_yes_ot - k_yes_ot)

n_no_ot = len(hr[hr['OverTime']=='No'])
k_no_ot_attr = hr[(hr['OverTime']=='No') & (hr['Attrition']=='Yes')].shape[0]
a_post_no = a_prior + k_no_ot_attr
b_post_no = b_prior + (n_no_ot - k_no_ot_attr)

w(f"\n  OverTime=Yes: {k_yes_ot}/{n_yes_ot} resigned")
w(f"  OverTime=No:  {k_no_ot_attr}/{n_no_ot} resigned")

mean_yes = a_post_yes / (a_post_yes + b_post_yes)
mean_no = a_post_no / (a_post_no + b_post_no)
w(f"\n  Posterior: OverTime=Yes ~ Beta({a_post_yes}, {b_post_yes})")
w(f"  Posterior: OverTime=No  ~ Beta({a_post_no}, {b_post_no})")
w(f"\n  Posterior Mean -- OverTime=Yes: {mean_yes:.4f} ({mean_yes*100:.2f}%)")
w(f"  Posterior Mean -- OverTime=No:  {mean_no:.4f} ({mean_no*100:.2f}%)")

ci_yes = beta_dist.ppf([0.025, 0.975], a_post_yes, b_post_yes)
ci_no = beta_dist.ppf([0.025, 0.975], a_post_no, b_post_no)
w(f"\n  95% Credible Interval -- OverTime=Yes: [{ci_yes[0]:.4f}, {ci_yes[1]:.4f}]")
w(f"  95% Credible Interval -- OverTime=No:  [{ci_no[0]:.4f}, {ci_no[1]:.4f}]")

np.random.seed(42)
samples_yes = beta_dist.rvs(a_post_yes, b_post_yes, size=100000)
samples_no = beta_dist.rvs(a_post_no, b_post_no, size=100000)
prob_yes_higher = (samples_yes > samples_no).mean()
w(f"\n  P(OverTime=Yes > OverTime=No) = {prob_yes_higher:.4f}")
w(f"  >> Probability that overtime increases attrition: {prob_yes_higher*100:.2f}%")

rel_risk = samples_yes / samples_no
w(f"  Posterior Relative Risk -- Mean: {rel_risk.mean():.3f}")
w(f"  95% CI: [{np.percentile(rel_risk, 2.5):.3f}, {np.percentile(rel_risk, 97.5):.3f}]")

section("Bayesian: Attrition Probability by Department")
depts = hr['Department'].unique()
for dept in depts:
    n_dept = len(hr[hr['Department']==dept])
    k_dept = hr[(hr['Department']==dept) & (hr['Attrition']=='Yes')].shape[0]
    a_d = a_prior + k_dept
    b_d = b_prior + (n_dept - k_dept)
    mean_d = a_d / (a_d + b_d)
    ci_d = beta_dist.ppf([0.025, 0.975], a_d, b_d)
    w(f"  {dept:25s}: {k_dept:3d}/{n_dept:4d} -> Posterior mean={mean_d:.3f}, 95% CI=[{ci_d[0]:.3f}, {ci_d[1]:.3f}]")

section("Bayesian: Attrition by MaritalStatus")
for ms in hr['MaritalStatus'].unique():
    n_ms = len(hr[hr['MaritalStatus']==ms])
    k_ms = hr[(hr['MaritalStatus']==ms) & (hr['Attrition']=='Yes')].shape[0]
    a_ms = a_prior + k_ms
    b_ms = b_prior + (n_ms - k_ms)
    mean_ms = a_ms / (a_ms + b_ms)
    ci_ms = beta_dist.ppf([0.025, 0.975], a_ms, b_ms)
    w(f"  {ms:15s}: {k_ms:3d}/{n_ms:4d} -> mean={mean_ms:.3f}, 95% CI=[{ci_ms[0]:.3f}, {ci_ms[1]:.3f}]")

# =============================================================
# 4. FINANCIAL ACCOUNTING
# =============================================================
banner("4. FINANCIAL ACCOUNTING -- DESCRIPTIVE ANALYSIS")

fa = pd.read_csv("D:\\download\\protfolio\\archive (3)\\financial_accounting.csv",
                 parse_dates=['Date'])
w(f"\n>> Total transactions: {len(fa):,}")
w(f">> Period: {fa['Date'].min().date()} -> {fa['Date'].max().date()}")

section("Frequency Statistics -- Accounts & Categories")
for col in ['Account', 'Category', 'Transaction_Type', 'Payment_Method']:
    freq = fa[col].value_counts()
    pct = fa[col].value_counts(normalize=True).mul(100).round(1)
    w(f"\n  {col}:")
    for k in freq.index:
        w(f"    {str(k):20s} -> {freq[k]:8,d}  ({pct[k]:6.1f}%)")

section("Descriptive Statistics -- Amounts")
w(fa[['Debit','Credit']].describe().round(2).to_string())

section("Monthly Analysis")
fa['Month'] = fa['Date'].dt.month
monthly = fa.groupby('Month').agg(
    Transactions=('Debit','count'),
    Total_Debit=('Debit','sum'),
    Total_Credit=('Credit','sum'),
    Avg_Debit=('Debit','mean'))
w(f"\n{monthly.round(2).to_string()}")

# =============================================================
# 5. BANKRUPTCY DATA
# =============================================================
banner("5. BANKRUPTCY PREDICTION -- DESCRIPTIVE ANALYSIS")

bk = pd.read_csv("D:\\download\\protfolio\\archive (4)\\data.csv")
w(f"\n>> Total companies: {len(bk):,}")
w(f">> Total variables: {len(bk.columns)}")

section("Frequency Statistics -- Bankruptcy")
bk_counts = bk['Bankrupt?'].value_counts()
for k in [0, 1]:
    label = 'Non-Bankrupt' if k == 0 else 'Bankrupt'
    w(f"  {label:15s}: {bk_counts[k]:5d}  ({bk_counts[k]/len(bk)*100:.2f}%)")

section("Descriptive Statistics -- Key Financial Ratios")
key_ratios = ['Bankrupt?', ' ROA(C) before interest and depreciation before interest',
              ' Debt ratio %', ' Current Ratio', ' Net Value Per Share (A)',
              ' Operating Gross Margin', ' Total Asset Turnover']
clean_names = ['Bankrupt','ROA','DebtRatio','CurrentRatio','NetValuePerShare',
               'GrossMargin','AssetTurnover']
for orig, clean in zip(key_ratios, clean_names):
    col = orig  # keep original spaces
    vals = pd.to_numeric(bk[col], errors='coerce')
    w(f"\n  {clean}:")
    w(f"    Mean={vals.mean():.4f}, Std={vals.std():.4f}")
    w(f"    Min={vals.min():.4f}, Max={vals.max():.4f}")
    w(f"    Q25={vals.quantile(0.25):.4f}, Median={vals.median():.4f}, Q75={vals.quantile(0.75):.4f}")

section("T-Test: Financial Ratios -- Bankrupt vs Non-Bankrupt")
w(f"\n  {'Ratio':35s} {'Non-Bankrupt':>12s} {'Bankrupt':>12s} {'p-value':>10s}")
w(f"  {'─'*70}")
for orig, clean in zip(key_ratios[1:], clean_names[1:]):
    col = orig
    vals = pd.to_numeric(bk[col], errors='coerce')
    g0 = vals[bk['Bankrupt?']==0].dropna()
    g1 = vals[bk['Bankrupt?']==1].dropna()
    if len(g0) > 1 and len(g1) > 1:
        _, pv = stats.ttest_ind(g0, g1)
        sig = '***' if pv<0.001 else '**' if pv<0.01 else '*' if pv<0.05 else 'ns'
        w(f"  {clean:35s} {g0.mean():>12.4f} {g1.mean():>12.4f} {pv:>10.6f} {sig}")

section("Bayesian: Bankruptcy Probability")
n_bk = len(bk)
k_bk = bk['Bankrupt?'].sum()
a_bk = a_prior + int(k_bk)
b_bk = b_prior + int(n_bk - k_bk)
w(f"\n  Bankrupt: {int(k_bk)}/{n_bk}")
w(f"  Posterior: Beta({a_bk}, {b_bk})")
w(f"  Mean bankruptcy probability: {a_bk/(a_bk+b_bk):.4f}")
ci_bk = beta_dist.ppf([0.025, 0.975], a_bk, b_bk)
w(f"  95% CI: [{ci_bk[0]:.4f}, {ci_bk[1]:.4f}]")

# =============================================================
# 6. FOREIGN EXCHANGE
# =============================================================
banner("6. FOREIGN EXCHANGE -- DESCRIPTIVE ANALYSIS")

fx = pd.read_csv("D:\\download\\protfolio\\archive (5)\\Foreign_Exchange_Rates.csv")
w(f"\n>> Total days: {len(fx):,}")
w(f">> Currency pairs: {len(fx.columns) - 2}")

section("Descriptive Statistics -- Major Pairs")
major_pairs = ['AUSTRALIA - AUSTRALIAN DOLLAR/US$',
               'EURO AREA - EURO/US$',
               'UNITED KINGDOM - UNITED KINGDOM POUND/US$',
               'JAPAN - YEN/US$',
               'CANADA - CANADIAN DOLLAR/US$',
               'SWITZERLAND - FRANC/US$',
               'CHINA - YUAN/US$']
for pair in major_pairs:
    vals = pd.to_numeric(fx[pair], errors='coerce')
    w(f"  {pair.split('-')[-1].strip():30s}: "
      f"Mean={vals.mean():.4f} | Min={vals.min():.4f} | Max={vals.max():.4f} | "
      f"Std={vals.std():.4f}")

section("Bayesian: Probability EUR/USD > 1.20")
eur = pd.to_numeric(fx['EURO AREA - EURO/US$'], errors='coerce').dropna()
n_eur = len(eur)
k_eur = int((eur > 1.20).sum())
a_eur = a_prior + k_eur
b_eur = b_prior + (n_eur - k_eur)
w(f"\n  EUR/USD > 1.20: {k_eur}/{n_eur} = {k_eur/n_eur*100:.1f}%")
w(f"  Posterior: Beta({a_eur}, {b_eur}) -> Mean={a_eur/(a_eur+b_eur):.4f}")
ci_eur = beta_dist.ppf([0.025, 0.975], a_eur, b_eur)
w(f"  95% CI: [{ci_eur[0]:.4f}, {ci_eur[1]:.4f}]")

# =============================================================
# 7. VISUALIZATIONS
# =============================================================
banner("7. VISUALIZATIONS")

# 7.1 HR Attrition
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('HR Attrition -- Professional Analysis Dashboard', fontsize=16, fontweight='bold')

ax = axes[0,0]
hr['Attrition'].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71','#e74c3c'], edgecolor='black')
ax.set_title('Attrition Distribution', fontweight='bold')
ax.set_ylabel('Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

ax = axes[0,1]
pd.crosstab(hr['OverTime'], hr['Attrition'], normalize='index').plot(
    kind='bar', ax=ax, color=['#2ecc71','#e74c3c'], edgecolor='black', stacked=True)
ax.set_title('Attrition Rate by OverTime', fontweight='bold')
ax.set_ylabel('Proportion')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

ax = axes[0,2]
dept_attr = hr.groupby('Department')['Attrition'].apply(lambda x: (x=='Yes').mean()*100)
dept_attr.sort_values().plot(kind='barh', ax=ax, color='#3498db', edgecolor='black')
ax.set_title('Attrition Rate by Department (%)', fontweight='bold')
ax.set_xlabel('Percentage')

ax = axes[1,0]
hr.boxplot(column='MonthlyIncome', by='Attrition', ax=ax)
ax.set_title('Monthly Income by Attrition', fontweight='bold')
ax.set_ylabel('Monthly Income')

ax = axes[1,1]
hr.boxplot(column='YearsAtCompany', by='Attrition', ax=ax)
ax.set_title('Years at Company by Attrition', fontweight='bold')
ax.set_ylabel('Years')

ax = axes[1,2]
x = np.linspace(0, 0.5, 500)
y_yes = beta_dist.pdf(x, a_post_yes, b_post_yes)
y_no = beta_dist.pdf(x, a_post_no, b_post_no)
ax.plot(x, y_yes, 'r-', lw=2.5, label=f'OverTime=Yes (mean={mean_yes:.3f})')
ax.plot(x, y_no, 'g-', lw=2.5, label=f'OverTime=No (mean={mean_no:.3f})')
ax.fill_between(x, y_yes, alpha=0.15, color='red')
ax.fill_between(x, y_no, alpha=0.15, color='green')
ax.set_title('Bayesian Posterior -- Attrition Rate', fontweight='bold')
ax.set_xlabel('Attrition Probability')
ax.set_ylabel('Density')
ax.legend()

plt.tight_layout()
fig.savefig(f"{OUT}\\HR_Analysis.png", dpi=150, bbox_inches='tight')
w(f"  [OK] HR_Analysis.png")

# 7.2 Financial Accounting
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Financial Accounting -- Transactions Analysis', fontsize=16, fontweight='bold')

fa['Account'].value_counts().plot(kind='bar', ax=axes2[0], color=['#1abc9c','#3498db','#9b59b6','#e67e22'])
axes2[0].set_title('Account Distribution', fontweight='bold')
axes2[0].set_xticklabels(axes2[0].get_xticklabels(), rotation=25)

fa['Category'].value_counts().plot(kind='bar', ax=axes2[1], color=['#2ecc71','#e74c3c','#f39c12','#2980b9'])
axes2[1].set_title('Category Distribution', fontweight='bold')
axes2[1].set_xticklabels(axes2[1].get_xticklabels(), rotation=25)

fa['Payment_Method'].value_counts().plot(kind='bar', ax=axes2[2], color=['#34495e','#16a085','#c0392b','#8e44ad'])
axes2[2].set_title('Payment Method Distribution', fontweight='bold')
axes2[2].set_xticklabels(axes2[2].get_xticklabels(), rotation=25)

plt.tight_layout()
fig2.savefig(f"{OUT}\\Financial_Analysis.png", dpi=150, bbox_inches='tight')
w(f"  [OK] Financial_Analysis.png")

# 7.3 Bankruptcy
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle('Bankruptcy Prediction -- Financial Analysis', fontsize=16, fontweight='bold')

bk['Bankrupt?'].value_counts().plot(kind='bar', ax=axes3[0], color=['#2ecc71','#e74c3c'])
axes3[0].set_title('Bankruptcy Distribution', fontweight='bold')
axes3[0].set_xticklabels(['Non-Bankrupt','Bankrupt'], rotation=0)

for i, col in enumerate([' ROA(C) before interest and depreciation before interest',
                          ' Debt ratio %']):
    axes3[i+1].hist(pd.to_numeric(bk[bk['Bankrupt?']==0][col], errors='coerce'), 
                    bins=50, alpha=0.6, label='Non-Bankrupt', color='#2ecc71')
    axes3[i+1].hist(pd.to_numeric(bk[bk['Bankrupt?']==1][col], errors='coerce'), 
                    bins=50, alpha=0.6, label='Bankrupt', color='#e74c3c')
    axes3[i+1].set_title(f'{col} by Bankruptcy', fontweight='bold')
    axes3[i+1].legend()

plt.tight_layout()
fig3.savefig(f"{OUT}\\Bankruptcy_Analysis.png", dpi=150, bbox_inches='tight')
w(f"  [OK] Bankruptcy_Analysis.png")

# 7.4 Forex
fig4, ax4 = plt.subplots(figsize=(14, 6))
dates = pd.to_datetime(fx['Time Serie'])
for pair, color, label in [('EURO AREA - EURO/US$', '#2c3e50', 'EUR/USD'),
                           ('UNITED KINGDOM - UNITED KINGDOM POUND/US$', '#e74c3c', 'GBP/USD'),
                           ('JAPAN - YEN/US$', '#2980b9', 'JPY/USD')]:
    vals = pd.to_numeric(fx[pair], errors='coerce')
    ax4.plot(dates, vals, label=label, color=color, lw=0.8, alpha=0.8)
ax4.set_title('Foreign Exchange Rates -- Time Series (2000-2019)', fontsize=14, fontweight='bold')
ax4.set_ylabel('Exchange Rate (vs USD)')
ax4.legend()
fig4.savefig(f"{OUT}\\Forex_Analysis.png", dpi=150, bbox_inches='tight')
w(f"  [OK] Forex_Analysis.png")

# =============================================================
# 8. SUMMARY
# =============================================================
banner("8. COMPREHENSIVE BAYESIAN SUMMARY")

w("""
  MODEL SPECIFICATION:
    Prior:       Beta(alpha=1, beta=1)   [Uniform, non-informative]
    Likelihood:  Binomial(n, theta)
    Posterior:   Beta(alpha + k, beta + n - k)
""")

w("  >>> BAYESIAN INFERENCE: HR ATTRITION <<<")
w(f"  Overall attrition posterior:  Beta({a_prior + int(hr['Attrition'].value_counts()['Yes'])}, "
  f"{b_prior + len(hr) - int(hr['Attrition'].value_counts()['Yes'])})")
k_all = int(hr['Attrition'].value_counts()['Yes'])
n_all = len(hr)
a_all = a_prior + k_all
b_all = b_prior + (n_all - k_all)
w(f"  Mean attrition rate: {a_all/(a_all+b_all)*100:.2f}%")
w(f"  95% CI: [{beta_dist.ppf(0.025, a_all, b_all)*100:.1f}%, "
  f"{beta_dist.ppf(0.975, a_all, b_all)*100:.1f}%]")

w(f"\n  >>> KEY BAYESIAN A/B RESULT <<<")
w(f"  P(Attrition_OT=Yes > Attrition_OT=No) = {prob_yes_higher*100:.1f}%")
w(f"  Relative Risk posterior mean = {rel_risk.mean():.3f}")
w(f"  >> Employees working overtime have {rel_risk.mean():.2f}x the attrition risk")

w(f"\n  >>> BAYESIAN: DEPARTMENT RISK RANKING <<<")
dept_risks = []
for dept in depts:
    n_dept = len(hr[hr['Department']==dept])
    k_dept = int(hr[(hr['Department']==dept) & (hr['Attrition']=='Yes')].shape[0])
    a_d = a_prior + k_dept
    b_d = b_prior + (n_dept - k_dept)
    mean_d = a_d / (a_d + b_d)
    ci_low = beta_dist.ppf(0.025, a_d, b_d)
    ci_high = beta_dist.ppf(0.975, a_d, b_d)
    dept_risks.append((mean_d, dept, ci_low, ci_high))
dept_risks.sort(reverse=True)
for i, (mean_d, dept, ci_low, ci_high) in enumerate(dept_risks, 1):
    w(f"  #{i} {dept:25s}: {mean_d*100:.1f}% [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")

w(f"\n  >>> FINANCIAL RATIOS: BANKRUPTCY PREDICTION <<<")
w(f"  Base bankruptcy rate: {a_bk/(a_bk+b_bk)*100:.2f}%")
vals_roa = pd.to_numeric(bk[' ROA(C) before interest and depreciation before interest'], errors='coerce')
w(f"  Bankrupt mean ROA: {vals_roa[bk['Bankrupt?']==1].mean():.4f} vs "
  f"Non-Bankrupt: {vals_roa[bk['Bankrupt?']==0].mean():.4f}")

vals_debt = pd.to_numeric(bk[' Debt ratio %'], errors='coerce')
w(f"  Bankrupt mean Debt Ratio: {vals_debt[bk['Bankrupt?']==1].mean():.4f} vs "
  f"Non-Bankrupt: {vals_debt[bk['Bankrupt?']==0].mean():.4f}")

w(f"\n  >>> CURRENCY MARKETS <<<")
eur = pd.to_numeric(fx['EURO AREA - EURO/US$'], errors='coerce')
w(f"  P(EUR/USD > 1.20) = {a_eur/(a_eur+b_eur)*100:.1f}%")
gbp = pd.to_numeric(fx['UNITED KINGDOM - UNITED KINGDOM POUND/US$'], errors='coerce')
w(f"  GBP/USD range: {gbp.min():.4f} - {gbp.max():.4f} (mean={gbp.mean():.4f})")

w(f"\n{'='*85}")
w(f"  ✅ Analysis Complete -- All outputs in: {OUT}")
w(f"  📄 Report: {log_path}")
w(f"  📊 Charts: HR_Analysis.png, Financial_Analysis.png, Bankruptcy_Analysis.png, Forex_Analysis.png")
w(f"{'='*85}")

log.close()
print(f"\nAnalysis complete. Report saved to: {log_path}")
