import pandas as pd, numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

OUT = "D:\\download\\protfolio\\projects\\output\\project5_Bankruptcy"
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT,"report.txt"),'w',encoding='utf-8')
def w(t=""): log.write(t+"\n"); print(t.encode("utf-8","replace").decode("utf-8","replace"))

w("="*80 + "\n  PROJECT 5: BANKRUPTCY PREDICTION\n  ØªØ­Ù„ÙŠÙ„ Ø§Ù„Ø¥ÙÙ„Ø§Ø³ ÙˆØ§Ù„Ù†Ø³Ø¨ Ø§Ù„Ù…Ø§Ù„ÙŠØ©\n" + "="*80)

bk = pd.read_csv("D:\\download\\protfolio\\archive (4)\\data.csv")
w(f"\n  Companies: {len(bk):,}")
w(f"  Financial Ratios: {len(bk.columns) - 1}")

w("\n" + "-"*60 + "\n  1. FREQUENCY STATISTICS\n" + "-"*60)
cnt = bk['Bankrupt?'].value_counts()
w(f"\nBankruptcy Distribution:")
w(f"  Non-Bankrupt (0): {cnt[0]:,} ({cnt[0]/len(bk)*100:.2f}%)")
w(f"  Bankrupt     (1): {cnt[1]:,} ({cnt[1]/len(bk)*100:.2f}%)")

key_ratios = [' ROA(C) before interest and depreciation before interest',
              ' Debt ratio %', ' Current Ratio', ' Net Value Per Share (A)',
              ' Operating Gross Margin', ' Total Asset Turnover',
              ' Cash Flow Per Share', ' Quick Ratio', ' Interest Expense Ratio']
clean_names = ['ROA','DebtRatio','CurrentRatio','NetValuePerShare',
               'GrossMargin','AssetTurnover','CashFlowPS','QuickRatio','InterestExpense']

w("\nDescriptive Stats - Key Ratios:")
for orig, clean in zip(key_ratios, clean_names):
    vals = pd.to_numeric(bk[orig], errors='coerce')
    b0 = vals[bk['Bankrupt?']==0]; b1 = vals[bk['Bankrupt?']==1]
    w(f"  {clean:20s}: All={vals.mean():.4f} | Bk=0={b0.mean():.4f} | Bk=1={b1.mean():.4f}")

w("\n" + "-"*60 + "\n  2. A/B TESTING\n" + "-"*60)

w("\nT-Test: Bankrupt vs Non-Bankrupt for each ratio")
w(f"  {'Ratio':25s} {'t-stat':>8s} {'p-value':>10s}")
w(f"  {'â”€'*45}")
for orig, clean in zip(key_ratios, clean_names):
    vals = pd.to_numeric(bk[orig], errors='coerce')
    g0 = vals[bk['Bankrupt?']==0].dropna()
    g1 = vals[bk['Bankrupt?']==1].dropna()
    t,p = stats.ttest_ind(g0,g1)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    w(f"  {clean:25s} {t:>8.2f} {p:>10.6f} {sig}")

w("\nTop Discriminating Ratios (by absolute t-stat):")
vals_list = [(orig,clean) for orig,clean in zip(key_ratios, clean_names)]
results = []
for orig, clean in vals_list:
    vals = pd.to_numeric(bk[orig], errors='coerce')
    g0 = vals[bk['Bankrupt?']==0].dropna(); g1 = vals[bk['Bankrupt?']==1].dropna()
    t,p = stats.ttest_ind(g0,g1)
    results.append((abs(t), clean, g0.mean(), g1.mean(), p))
results.sort(reverse=True)
for i,(abs_t, clean, m0, m1, p) in enumerate(results,1):
    w(f"  #{i}: {clean:20s} |t|={abs_t:.2f} (Bk=0: {m0:.4f} vs Bk=1: {m1:.4f}) p={p:.6f}")

w("\n" + "-"*60 + "\n  3. BAYESIAN ANALYSIS\n" + "-"*60)
a_prior,b_prior=1,1

k_bk = int(bk['Bankrupt?'].sum()); n_bk = len(bk)
a_post,b_post = a_prior+k_bk, b_prior+(n_bk-k_bk)
lo,hi = beta_dist.ppf(0.025,a_post,b_post), beta_dist.ppf(0.975,a_post,b_post)
w(f"\nBayesian: Base Bankruptcy Rate")
w(f"  Bankrupt: {k_bk}/{n_bk}")
w(f"  Posterior: Beta({a_post},{b_post})")
w(f"  Mean: {a_post/(a_post+b_post)*100:.2f}%")
w(f"  95% CI: [{lo*100:.2f}%, {hi*100:.2f}%]")

w("\nBayesian: ROA Threshold Analysis")
roa = pd.to_numeric(bk[' ROA(C) before interest and depreciation before interest'], errors='coerce')
for thresh in [0.45, 0.50, 0.55]:
    k_below = int((roa < thresh).sum())
    n_roa = len(roa.dropna())
    k_bk_below = int((bk.loc[roa<thresh,'Bankrupt?'].sum())) if k_below>0 else 0
    ar,br = a_prior+k_bk_below, b_prior+(k_below-k_bk_below) if k_below>0 else (a_prior,b_prior)
    mean_r = ar/(ar+br) if (ar+br)>0 else 0
    w(f"  P(Bankrupt | ROA<{thresh}): {k_bk_below}/{k_below} = {mean_r*100:.2f}%")

w("\nBayesian: Debt Ratio Threshold Analysis")
debt = pd.to_numeric(bk[' Debt ratio %'], errors='coerce')
for thresh in [0.10, 0.15, 0.20]:
    k_above = int((debt > thresh).sum())
    k_bk_above = int((bk.loc[debt>thresh,'Bankrupt?'].sum())) if k_above>0 else 0
    ad,bd = a_prior+k_bk_above, b_prior+(k_above-k_bk_above) if k_above>0 else (a_prior,b_prior)
    mean_d = ad/(ad+bd) if (ad+bd)>0 else 0
    w(f"  P(Bankrupt | Debt>{thresh}): {k_bk_above}/{k_above} = {mean_d*100:.2f}%")

w("\n" + "-"*60 + "\n  4. VISUALIZATIONS\n" + "-"*60)
fig, axes = plt.subplots(2,3,figsize=(18,12))
fig.suptitle('Project 5: Bankruptcy Prediction Analysis', fontsize=16, fontweight='bold')

bk['Bankrupt?'].value_counts().plot(kind='bar',ax=axes[0,0],color=['#2ecc71','#e74c3c'],edgecolor='k')
axes[0,0].set_title('Bankruptcy Distribution',fontweight='bold'); axes[0,0].set_xticklabels(['Non-Bankrupt','Bankrupt'],rotation=0)

for i,(orig,clr) in enumerate(zip(key_ratios[:2],['#3498db','#e74c3c'])):
    vals = pd.to_numeric(bk[orig], errors='coerce')
    axes[0,i+1].hist([vals[bk['Bankrupt?']==0], vals[bk['Bankrupt?']==1]], bins=50, alpha=0.6, label=['Non-Bk','Bk'], color=['#2ecc71','#e74c3c'])
    axes[0,i+1].set_title(f'{clean_names[i]} by Bankruptcy',fontweight='bold'); axes[0,i+1].legend()

ax = axes[1,0]
results_sorted = sorted(results, reverse=True)[:5]
ax.barh([r[1] for r in results_sorted], [r[0] for r in results_sorted], color='#e74c3c')
ax.set_title('Top 5 Discriminating Ratios (|t-stat|)',fontweight='bold')

# Logistic regression coefficients (simple approximation)
from sklearn.linear_model import LogisticRegression
X = pd.DataFrame()
for orig in key_ratios:
    X[clean_names[key_ratios.index(orig)]] = pd.to_numeric(bk[orig], errors='coerce')
X = X.fillna(X.mean())
y = bk['Bankrupt?']
lr = LogisticRegression(max_iter=1000)
lr.fit(X, y)
coefs = pd.Series(lr.coef_[0], index=X.columns).sort_values()
axes[1,1].barh(coefs.index, coefs.values, color=['#e74c3c' if c<0 else '#2ecc71' for c in coefs.values])
axes[1,1].set_title('Logistic Regression Coefficients',fontweight='bold')
axes[1,1].axvline(0, color='k', lw=0.5)

axes[1,2].axis('off')

plt.tight_layout()
fig.savefig(f"{OUT}\\Bankruptcy_Project.png",dpi=150,bbox_inches='tight')
w("  [OK] Bankruptcy_Project.png")

log.close()
print(f"\nPROJECT 5 COMPLETE -> {OUT}\\report.txt")

