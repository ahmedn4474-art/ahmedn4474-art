import pandas as pd, numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

OUT = "D:\\download\\protfolio\\projects\\output\\project4_Financial"
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT,"report.txt"),'w',encoding='utf-8')
def w(t=""): log.write(t+"\n"); print(t.encode("utf-8","replace").decode("utf-8","replace"))

w("="*80 + "\n  PROJECT 4: FINANCIAL ACCOUNTING\n  ØªØ­Ù„ÙŠÙ„ Ø§Ù„Ù‚ÙŠÙˆØ¯ Ø§Ù„Ù…Ø­Ø§Ø³Ø¨ÙŠØ©\n" + "="*80)

fa = pd.read_csv("D:\\download\\protfolio\\archive (3)\\financial_accounting.csv", parse_dates=['Date'])
w(f"\n  Total transactions: {len(fa):,}")
w(f"  Period: {fa['Date'].min().date()} to {fa['Date'].max().date()}")

w("\n" + "-"*60 + "\n  1. FREQUENCY STATISTICS\n" + "-"*60)
for col in ['Account','Category','Transaction_Type','Payment_Method']:
    f = fa[col].value_counts(); p = fa[col].value_counts(normalize=True).mul(100).round(1)
    w(f"\n{col}:"); [w(f"  {k:20s}: {f[k]:7,d} ({p[k]:.1f}%)") for k in f.index]

fa['Month'] = fa['Date'].dt.month_name()
fa['Weekday'] = fa['Date'].dt.day_name()
w(f"\nMonthly distribution:")
for k,v in fa['Month'].value_counts().items():
    w(f"  {k:10s}: {v:6,d}")
w(f"\nWeekday distribution:")
for k,v in fa['Weekday'].value_counts().items():
    w(f"  {k:10s}: {v:6,d}")

w("\nDescriptive Stats (Debit/Credit):")
w(fa[['Debit','Credit']].describe().round(2).to_string())

w("\n" + "-"*60 + "\n  2. A/B TESTING\n" + "-"*60)

w("\nT-Test: Debit by Transaction Type (Sale vs Purchase)")
sale = fa[fa.Transaction_Type=='Sale']['Debit']
purch = fa[fa.Transaction_Type=='Purchase']['Debit']
t,pt = stats.ttest_ind(sale,purch)
w(f"  Sale mean: {sale.mean():.2f} | Purchase mean: {purch.mean():.2f}")
w(f"  t={t:.3f}, p={pt:.6f} {'***' if pt<0.001 else 'ns'}")

w("\nANOVA: Debit by Category")
cats = fa.Category.unique()
for c in cats:
    w(f"  {c:10s}: mean={fa[fa.Category==c]['Debit'].mean():.2f}")
f,p = stats.f_oneway(*[fa[fa.Category==c]['Debit'] for c in cats])
w(f"  F={f:.2f}, p={p:.6f} {'***' if p<0.001 else 'ns'}")

w("\nChi-Square: Account vs Payment Method")
ct = pd.crosstab(fa['Account'],fa['Payment_Method'])
chi2,p = stats.chi2_contingency(ct)[:2]
w(f"  chi2={chi2:.2f}, p={p:.6f} {'***' if p<0.001 else 'ns'}")

w("\nChi-Square: Category vs Transaction Type")
ct2 = pd.crosstab(fa['Category'],fa['Transaction_Type'])
chi2,p2 = stats.chi2_contingency(ct2)[:2]
w(f"  chi2={chi2:.2f}, p={p2:.6f} {'***' if p2<0.001 else 'ns'}")

w("\n" + "-"*60 + "\n  3. BAYESIAN ANALYSIS\n" + "-"*60)
a_prior,b_prior=1,1

w("\nBayesian: Average Debit Amount")
mean_debit = fa['Debit'].mean()
std_debit = fa['Debit'].std()
n_debit = len(fa)
w(f"  Mean Debit: {mean_debit:.2f}")
w(f"  Std Debit: {std_debit:.2f}")
w(f"  95% CI (Normal approx): [{mean_debit-1.96*std_debit/np.sqrt(n_debit):.2f}, {mean_debit+1.96*std_debit/np.sqrt(n_debit):.2f}]")

w("\nBayesian: Prop. of Cash Transactions")
k_cash = int((fa.Payment_Method=='Cash').sum()); n_pay = len(fa)
a_cash,b_cash = a_prior+k_cash, b_prior+(n_pay-k_cash)
lo,hi = beta_dist.ppf(0.025,a_cash,b_cash), beta_dist.ppf(0.975,a_cash,b_cash)
w(f"  Cash: {k_cash}/{n_pay}")
w(f"  Posterior: Beta({a_cash},{b_cash}) -> {a_cash/(a_cash+b_cash)*100:.2f}%")
w(f"  95% CI: [{lo*100:.2f}%, {hi*100:.2f}%]")

w("\nBayesian: Category Distribution")
for cat in fa.Category.unique():
    k = int((fa.Category==cat).sum()); n = len(fa)
    a,b = a_prior+k, b_prior+(n-k)
    lo,hi = beta_dist.ppf(0.025,a,b), beta_dist.ppf(0.975,a,b)
    w(f"  {cat:10s}: {k:6d}/{n:6d} -> {a/(a+b)*100:.1f}% [{lo*100:.1f}%-{hi*100:.1f}%]")

w("\n" + "-"*60 + "\n  4. VISUALIZATIONS\n" + "-"*60)
fig, axes = plt.subplots(2,3,figsize=(18,12))
fig.suptitle('Project 4: Financial Accounting Analysis', fontsize=16, fontweight='bold')

fa['Account'].value_counts().plot(kind='bar',ax=axes[0,0],color=['#1abc9c','#3498db','#9b59b6','#e67e22'],edgecolor='k')
axes[0,0].set_title('Account Distribution',fontweight='bold')

fa['Category'].value_counts().plot(kind='bar',ax=axes[0,1],color=['#2ecc71','#e74c3c','#f39c12','#2980b9'],edgecolor='k')
axes[0,1].set_title('Category Distribution',fontweight='bold')

fa['Payment_Method'].value_counts().plot(kind='bar',ax=axes[0,2],color=['#34495e','#16a085','#c0392b','#8e44ad'],edgecolor='k')
axes[0,2].set_title('Payment Method',fontweight='bold')

axes[1,0].hist(fa['Debit'],bins=50,color='#3498db',edgecolor='k',alpha=0.7)
axes[1,0].set_title('Debit Amount Distribution',fontweight='bold')

fa.groupby(fa['Date'].dt.month)['Debit'].sum().plot(kind='line',marker='o',ax=axes[1,1],color='#e74c3c',lw=2)
axes[1,1].set_title('Monthly Total Debits',fontweight='bold')

fa.groupby('Transaction_Type')['Debit'].mean().plot(kind='bar',ax=axes[1,2],color=['#3498db','#e74c3c','#2ecc71','#f39c12'],edgecolor='k')
axes[1,2].set_title('Avg Debit by Transaction Type',fontweight='bold')

plt.tight_layout()
fig.savefig(f"{OUT}\\Financial_Project.png",dpi=150,bbox_inches='tight')
w("  [OK] Financial_Project.png")

# Monthly heatmap
fig2, ax2 = plt.subplots(figsize=(10,6))
monthly_pivot = fa.pivot_table(values='Debit', index=fa['Date'].dt.month, columns=fa['Weekday'], aggfunc='mean')
im = ax2.imshow(monthly_pivot, cmap='YlOrRd', aspect='auto')
ax2.set_xticks(range(len(monthly_pivot.columns))); ax2.set_yticks(range(len(monthly_pivot.index)))
ax2.set_xticklabels(monthly_pivot.columns,rotation=45); ax2.set_yticklabels(monthly_pivot.index)
ax2.set_title('Avg Debit: Month x Weekday Heatmap',fontweight='bold')
fig2.colorbar(im,ax=ax2)
fig2.savefig(f"{OUT}\\Monthly_Heatmap.png",dpi=150,bbox_inches='tight')
w("  [OK] Monthly_Heatmap.png")

log.close()
print(f"\nPROJECT 4 COMPLETE -> {OUT}\\report.txt")

