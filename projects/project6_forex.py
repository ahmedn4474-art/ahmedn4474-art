import pandas as pd, numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

OUT = "D:\\download\\protfolio\\projects\\output\\project6_Forex"
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT,"report.txt"),'w',encoding='utf-8')
def w(t=""): log.write(t+"\n"); print(t.encode("utf-8","replace").decode("utf-8","replace"))

w("="*80 + "\n  PROJECT 6: FOREIGN EXCHANGE RATES\n  ØªØ­Ù„ÙŠÙ„ Ø£Ø³Ø¹Ø§Ø± ØµØ±Ù Ø§Ù„Ø¹Ù…Ù„Ø§Øª\n" + "="*80)

fx = pd.read_csv("D:\\download\\protfolio\\archive (5)\\Foreign_Exchange_Rates.csv")
fx['Date'] = pd.to_datetime(fx['Time Serie'])
w(f"\n  Observations: {len(fx):,}")
w(f"  Period: {fx['Date'].min().date()} to {fx['Date'].max().date()}")
w(f"  Currency pairs: {len([c for c in fx.columns if '/' in c])}")

major_pairs = [
    ('AUSTRALIA - AUSTRALIAN DOLLAR/US$','AUD/USD'),
    ('EURO AREA - EURO/US$','EUR/USD'),
    ('UNITED KINGDOM - UNITED KINGDOM POUND/US$','GBP/USD'),
    ('JAPAN - YEN/US$','JPY/USD'),
    ('CANADA - CANADIAN DOLLAR/US$','CAD/USD'),
    ('SWITZERLAND - FRANC/US$','CHF/USD'),
    ('CHINA - YUAN/US$','CNY/USD'),
]

w("\n" + "-"*60 + "\n  1. FREQUENCY STATISTICS\n" + "-"*60)
w(f"\nDescriptive Statistics - Major Pairs:")
w(f"  {'Pair':10s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Range':>8s}")
w(f"  {'â”€'*50}")
for col, label in major_pairs:
    vals = pd.to_numeric(fx[col], errors='coerce')
    w(f"  {label:10s} {vals.mean():>8.4f} {vals.std():>8.4f} {vals.min():>8.4f} {vals.max():>8.4f} {vals.max()-vals.min():>8.4f}")

# Daily returns
for col, label in major_pairs:
    vals = pd.to_numeric(fx[col], errors='coerce')
    fx[f'ret_{label.replace("/","_")}'] = vals.pct_change()

w("\nDaily Return Statistics:")
w(f"  {'Pair':10s} {'Mean(%)':>8s} {'Std(%)':>8s} {'Min(%)':>8s} {'Max(%)':>8s}")
w(f"  {'â”€'*50}")
for col, label in major_pairs:
    ret = fx[f'ret_{label.replace("/","_")}'] * 100
    w(f"  {label:10s} {ret.mean():>8.4f} {ret.std():>8.4f} {ret.min():>8.4f} {ret.max():>8.4f}")

w("\n" + "-"*60 + "\n  2. A/B TESTING\n" + "-"*60)

# Pre/post 2008 crisis
fx['Period'] = np.where(fx['Date'] < '2008-01-01', 'Pre-2008', 'Post-2008')
w("\nT-Test: EUR/USD before vs after 2008 crisis")
eur = pd.to_numeric(fx['EURO AREA - EURO/US$'], errors='coerce')
g_pre = eur[fx.Period=='Pre-2008'].dropna()
g_post = eur[fx.Period=='Post-2008'].dropna()
t,p = stats.ttest_ind(g_pre, g_post)
w(f"  Pre-2008 mean: {g_pre.mean():.4f} | Post-2008 mean: {g_post.mean():.4f}")
w(f"  t={t:.3f}, p={p:.6f} {'***' if p<0.001 else 'ns'}")

w("\nT-Test: GBP/USD before vs after 2008")
gbp = pd.to_numeric(fx['UNITED KINGDOM - UNITED KINGDOM POUND/US$'], errors='coerce')
g_pre_g = gbp[fx.Period=='Pre-2008'].dropna()
g_post_g = gbp[fx.Period=='Post-2008'].dropna()
t,p = stats.ttest_ind(g_pre_g, g_post_g)
w(f"  Pre-2008: {g_pre_g.mean():.4f} | Post-2008: {g_post_g.mean():.4f}")
w(f"  t={t:.3f}, p={p:.6f} {'***' if p<0.001 else 'ns'}")

w("\nChi-Square: Volatility regime (pre/post 2008)")
for col, label in major_pairs[:3]:
    ret = fx[f'ret_{label.replace("/","_")}'] * 100
    high_vol = ret.abs() > ret.std()
    ct = pd.crosstab(fx['Period'], high_vol)
    chi2,p = stats.chi2_contingency(ct)[:2]
    w(f"  {label}: chi2={chi2:.2f}, p={p:.6f} {'***' if p<0.001 else 'ns'}")

w("\n" + "-"*60 + "\n  3. BAYESIAN ANALYSIS\n" + "-"*60)
a_prior,b_prior=1,1

eur = pd.to_numeric(fx['EURO AREA - EURO/US$'], errors='coerce').dropna()
w("\nBayesian: EUR/USD Trends")
for thresh, label in [(0.80,'EUR<0.80'),(0.90,'EUR<0.90'),(1.00,'EUR>1.00'),(1.10,'EUR>1.10')]:
    k = int((eur > thresh).sum()) if '>' in label else int((eur < thresh).sum())
    n = len(eur)
    a,b = a_prior+k, b_prior+(n-k)
    lo,hi = beta_dist.ppf(0.025,a,b), beta_dist.ppf(0.975,a,b)
    w(f"  P({label}): {k}/{n} -> {a/(a+b)*100:.2f}% [{lo*100:.2f}%-{hi*100:.2f}%]")

w("\nBayesian: GBP/USD Trends")
gbp = pd.to_numeric(fx['UNITED KINGDOM - UNITED KINGDOM POUND/US$'], errors='coerce').dropna()
for thresh, label in [(0.50,'GBP<0.50'),(0.60,'GBP<0.60'),(0.70,'GBP>0.70')]:
    k = int((gbp > thresh).sum()) if '>' in label else int((gbp < thresh).sum())
    n = len(gbp)
    a,b = a_prior+k, b_prior+(n-k)
    lo,hi = beta_dist.ppf(0.025,a,b), beta_dist.ppf(0.975,a,b)
    w(f"  P({label}): {k}/{n} -> {a/(a+b)*100:.2f}% [{lo*100:.2f}%-{hi*100:.2f}%]")

w("\nBayesian: JPY/USD > 110")
jpy = pd.to_numeric(fx['JAPAN - YEN/US$'], errors='coerce').dropna()
k_jpy = int((jpy > 110).sum()); n_jpy = len(jpy)
a_j,b_j = a_prior+k_jpy, b_prior+(n_jpy-k_jpy)
w(f"  P(JPY>110): {k_jpy}/{n_jpy} -> {a_j/(a_j+b_j)*100:.2f}% [{beta_dist.ppf(0.025,a_j,b_j)*100:.2f}%-{beta_dist.ppf(0.975,a_j,b_j)*100:.2f}%]")

# Correlation between pairs
w("\n" + "-"*60 + "\n  4. VISUALIZATIONS\n" + "-"*60)
fig, axes = plt.subplots(2,3,figsize=(18,12))
fig.suptitle('Project 6: Foreign Exchange Analysis (2000-2019)', fontsize=16, fontweight='bold')

dates = fx['Date']
colors = ['#2c3e50','#e74c3c','#2980b9','#27ae60','#f39c12','#8e44ad','#1abc9c']
for i,((col,label),clr) in enumerate(zip(major_pairs,colors)):
    vals = pd.to_numeric(fx[col], errors='coerce')
    ax = axes[0,0] if i<3 else axes[0,1] if i<6 else axes[0,2]
    if i>=3: ax = axes[1,0] if i<4 else axes[1,1] if i<5 else axes[1,2]
    ax.plot(dates, vals, label=label, color=clr, lw=0.8, alpha=0.8)
    
axes[0,0].plot(dates,pd.to_numeric(fx['EURO AREA - EURO/US$'],errors='coerce'),color='#2c3e50',lw=1.5,label='EUR/USD')
axes[0,0].plot(dates,pd.to_numeric(fx['UNITED KINGDOM - UNITED KINGDOM POUND/US$'],errors='coerce'),color='#e74c3c',lw=1.5,label='GBP/USD')
axes[0,0].axvline(x=pd.Timestamp('2008-09-01'),color='k',ls='--',alpha=0.5,label='2008 Crisis')
axes[0,0].set_title('Major EUR/USD & GBP/USD',fontweight='bold'); axes[0,0].legend()

axes[0,1].plot(dates,pd.to_numeric(fx['JAPAN - YEN/US$'],errors='coerce'),color='#2980b9',lw=1.5)
axes[0,1].set_title('USD/JPY',fontweight='bold')

axes[0,2].plot(dates,pd.to_numeric(fx['CHINA - YUAN/US$'],errors='coerce'),color='#27ae60',lw=1.5)
axes[0,2].set_title('USD/CNY',fontweight='bold')

# Returns distribution
returns = fx['ret_EUR_USD'].dropna()*100
axes[1,0].hist(returns, bins=100, color='#3498db', alpha=0.7, edgecolor='k')
axes[1,0].set_title('EUR/USD Daily Returns Distribution',fontweight='bold')
axes[1,0].axvline(returns.mean(), color='r', ls='-', lw=2, label=f'mean={returns.mean():.3f}%')
axes[1,0].axvline([returns.quantile(0.025)], color='orange', ls='--', label='2.5%/97.5%')
axes[1,0].axvline([returns.quantile(0.975)], color='orange', ls='--')
axes[1,0].legend()

# Correlation heatmap
pairs_display = ['EUR/USD','GBP/USD','JPY/USD','CAD/USD','CHF/USD','AUD/USD','CNY/USD']
ret_data = pd.DataFrame()
for col,label in major_pairs:
    ret_data[label] = pd.to_numeric(fx[col], errors='coerce')
corr = ret_data.corr()
ax = axes[1,1]
im = ax.imshow(corr, cmap='RdYlBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(pairs_display))); ax.set_yticks(range(len(pairs_display)))
ax.set_xticklabels(pairs_display, rotation=45, ha='right'); ax.set_yticklabels(pairs_display)
for i in range(len(pairs_display)):
    for j in range(len(pairs_display)):
        ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha='center',va='center',fontsize=8)
ax.set_title('Currency Correlation Matrix',fontweight='bold')

# Rolling vol
ret_data['EUR_ret'] = pd.to_numeric(fx['EURO AREA - EURO/US$'],errors='coerce').pct_change()
ret_data['GBP_ret'] = pd.to_numeric(fx['UNITED KINGDOM - UNITED KINGDOM POUND/US$'],errors='coerce').pct_change()
ret_data['JPY_ret'] = pd.to_numeric(fx['JAPAN - YEN/US$'],errors='coerce').pct_change()
axes[1,2].plot(dates, ret_data['EUR_ret'].rolling(252).std()*np.sqrt(252)*100, label='EUR/USD', color='#2c3e50', lw=1)
axes[1,2].plot(dates, ret_data['GBP_ret'].rolling(252).std()*np.sqrt(252)*100, label='GBP/USD', color='#e74c3c', lw=1)
axes[1,2].plot(dates, ret_data['JPY_ret'].rolling(252).std()*np.sqrt(252)*100, label='JPY/USD', color='#2980b9', lw=1)
axes[1,2].set_title('Rolling 1Y Annualized Volatility',fontweight='bold'); axes[1,2].legend()

plt.tight_layout()
fig.savefig(f"{OUT}\\Forex_Project.png",dpi=150,bbox_inches='tight')
w("  [OK] Forex_Project.png")

log.close()
print(f"\nPROJECT 6 COMPLETE -> {OUT}\\report.txt")

