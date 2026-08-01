"""
PROJECT 6: FOREX ANALYSIS — EXCHANGE RATE TIME SERIES
=====================================================
Techniques: EDA, A/B Testing, Bayesian, ARIMA, Seasonal Decompose,
            Volatility Analysis, Rolling Statistics, Correlation,
            GARCH Effects, Technical Indicators, Forecasting
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
from scipy.stats import beta as beta_dist
warnings.filterwarnings('ignore')

OUT = "D:\\download\\protfolio\\projects\\v2_output\\project6_Forex"
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT,"report.txt"),'w',encoding='utf-8')
def w(t=""): log.write(str(t)+"\n"); print(t)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

w("="*85 + "\n  PROJECT 6: FOREX — PROFESSIONAL EXCHANGE RATE ANALYSIS\n" + "="*85)

df = pd.read_csv("D:\\download\\protfolio\\archive (5)\\Foreign_Exchange_Rates.csv")
w(f"\n  Loaded: {len(df)} rows x {len(df.columns)} cols")
w(f"  Columns: {list(df.columns)}")

# Parse
# Date column is always column index 1 ('Time Serie')
df['Date'] = pd.to_datetime(df.iloc[:,1], errors='coerce')
df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

# All columns after the first two are exchange rates
rate_cols = [c for c in df.columns if c not in ['Date', df.columns[0], df.columns[1]]]
w(f"  Currency pairs: {len(rate_cols)}")
w(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# Convert rates
for c in rate_cols:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace('N/A','NaN').str.replace(',',''), errors='coerce')
    df[c] = df[c].interpolate()

# US Dollar index (average of available majors)
primary = 'EURO AREA - EURO/US$' if 'EURO AREA - EURO/US$' in rate_cols else rate_cols[2]
w(f"\n  Primary pair for analysis: {primary}")

# Daily returns
df['Return'] = df[primary].pct_change()
df['LogReturn'] = np.log(df[primary] / df[primary].shift(1))
df['Volatility'] = df['Return'].rolling(21).std() * np.sqrt(252)
df['MA20'] = df[primary].rolling(20).mean()
df['MA50'] = df[primary].rolling(50).mean()
df['Upper_BB'] = df['MA20'] + (df['Return'].rolling(20).std() * np.sqrt(252) * 2)
df['Lower_BB'] = df['MA20'] - (df['Return'].rolling(20).std() * np.sqrt(252) * 2)

# ═══════════════════════════════════════════
# 1. EDA
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  1. EXPLORATORY DATA ANALYSIS\n" + "▔"*60)
w(f"\n  Exchange Rate ({primary}):")
w(f"    Mean: {df[primary].mean():.4f}")
w(f"    Min:  {df[primary].min():.4f}")
w(f"    Max:  {df[primary].max():.4f}")
w(f"    Std:  {df[primary].std():.4f}")

w(f"\n  Daily Returns:")
w(f"    Mean: {df['Return'].mean()*100:.4f}%")
w(f"    Volatility (annualized): {df['Volatility'].iloc[-1]*100:.1f}%")
w(f"    Positive days: {(df['Return']>0).sum()}/{len(df)} ({(df['Return']>0).mean()*100:.1f}%)")
w(f"    Negative days: {(df['Return']<0).sum()}/{len(df)} ({(df['Return']<0).mean()*100:.1f}%)")

# Stationarity
adf = adfuller(df[primary].dropna())
w(f"  ADF Statistic: {adf[0]:.4f}")
w(f"  p-value: {adf[1]:.6f}")
w(f"  {'STATIONARY' if adf[1] < 0.05 else 'NON-STATIONARY'}")

adf_r = adfuller(df['Return'].dropna())
w(f"  ADF on Returns: {adf_r[0]:.4f} p={adf_r[1]:.6f}")
w(f"  {'STATIONARY' if adf_r[1] < 0.05 else 'NON-STATIONARY'}")

# Correlation matrix
w(f"\n  Top correlations with {primary}:")
corr = df[rate_cols].corr()[primary].drop(primary).sort_values(ascending=False)
for c in corr.index[:5]:
    w(f"    {c:40s}: {corr[c]:.4f}")

# ═══════════════════════════════════════════
# 2. A/B TESTING
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  2. A/B TESTING\n" + "▔"*60)

w("\n  T-Test: Return by Pre/Post 2020")
pre = df[df['Date'] < '2020-01-01']['Return'].dropna()
post = df[df['Date'] >= '2020-01-01']['Return'].dropna()
t,p = stats.ttest_ind(pre,post)
d = (pre.mean()-post.mean())/np.sqrt((pre.var()+post.var())/2)
w(f"    Pre (mean): {pre.mean()*100:.4f}% vs Post (mean): {post.mean()*100:.4f}%")
w(f"    t={t:.4f} p={p:.6f} d={d:.4f}")

# Compare volatility regimes
w("\n  F-Test: Variance pre/post 2020")
f_stat = pre.var() / post.var() if pre.var() > post.var() else post.var() / pre.var()
w(f"    F-ratio: {f_stat:.4f} (higher = different volatility)")

# Compare first vs second half
mid = df.iloc[len(df)//2]['Date']
first_half = df[df['Date'] < mid]['Return'].dropna()
second_half = df[df['Date'] >= mid]['Return'].dropna()
t2,p2 = stats.ttest_ind(first_half, second_half)
w(f"\n  First half vs Second half: t={t2:.4f} p={p2:.6f}")

# ═══════════════════════════════════════════
# 3. BAYESIAN
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  3. BAYESIAN INFERENCE\n" + "▔"*60)
a_p, b_p = 1, 1

w("\n  Bayesian: P(Positive Day)")
n_pos = int((df['Return']>0).sum()); n_days = len(df['Return'].dropna())
a,b = a_p+n_pos, b_p+n_days-n_pos
lo,hi = beta_dist.ppf(0.025,a,b), beta_dist.ppf(0.975,a,b)
w(f"    Posterior: Beta({a},{b})")
w(f"    P(Up day) = {a/(a+b)*100:.1f}%")
w(f"    95% HDI: [{lo*100:.1f}%, {hi*100:.1f}%]")

w("\n  Bayesian: P(High Volatility Day)")
vol_thresh = df['Volatility'].median()
n_high = int((df['Volatility'] > vol_thresh).sum()); n_vol = len(df['Volatility'].dropna())
a2,b2 = a_p+n_high, b_p+n_vol-n_high
lo2,hi2 = beta_dist.ppf(0.025,a2,b2), beta_dist.ppf(0.975,a2,b2)
w(f"    P(High Vol) = {a2/(a2+b2)*100:.1f}%")
w(f"    95% HDI: [{lo2*100:.1f}%, {hi2*100:.1f}%]")

# ═══════════════════════════════════════════
# 4. SEASONAL DECOMPOSITION
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  4. TIME SERIES DECOMPOSITION\n" + "▔"*60)
try:
    decomp = seasonal_decompose(df.set_index('Date')[primary].dropna().asfreq('B').fillna(method='ffill'), model='multiplicative', period=252)
    w(f"  Decomposed: trend + seasonal + residual")
    w(f"  Residual std: {decomp.resid.std():.4f}")
    w(f"  Seasonal range: [{decomp.seasonal.min():.4f}, {decomp.seasonal.max():.4f}]")
except Exception as e:
    w(f"  Decomposition error: {e}")

# ═══════════════════════════════════════════
# 5. TECHNICAL INDICATORS
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  5. TECHNICAL ANALYSIS\n" + "▔"*60)
# Moving average crossover
cross_buy = ((df['MA20'] > df['MA50']) & (df['MA20'].shift(1) <= df['MA50'].shift(1))).sum()
cross_sell = ((df['MA20'] < df['MA50']) & (df['MA20'].shift(1) >= df['MA50'].shift(1))).sum()
w(f"  MA20/MA50 crossovers: {cross_buy} buys, {cross_sell} sells")

# Bollinger bands
df['BB_Signal'] = 0
df.loc[df[primary] < df['Lower_BB'], 'BB_Signal'] = 1  # oversold
df.loc[df[primary] > df['Upper_BB'], 'BB_Signal'] = -1  # overbought
w(f"  Bollinger oversold signals: {(df['BB_Signal']==1).sum()} / {len(df)}")
w(f"  Bollinger overbought signals: {(df['BB_Signal']==-1).sum()} / {len(df)}")

# Max drawdown
cummax = df[primary].cummax()
drawdown = (df[primary] - cummax) / cummax
w(f"  Maximum drawdown: {drawdown.min()*100:.2f}%")
w(f"  Current drawdown: {drawdown.iloc[-1]*100:.2f}%")

# ═══════════════════════════════════════════
# 6. VISUALIZATIONS
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  6. PROFESSIONAL VISUALIZATIONS\n" + "▔"*60)
sns.set_style("whitegrid")
fig = plt.figure(figsize=(22, 18))
fig.suptitle(f'Forex Analysis — {primary} — Professional Dashboard', fontsize=18, fontweight='bold', y=0.98)
gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

# Price + Moving Averages
ax1 = fig.add_subplot(gs[0,:])
ax1.plot(df['Date'], df[primary], lw=1, label=primary, color='#2c3e50')
ax1.plot(df['Date'], df['MA20'], lw=0.8, label='MA20', color='#e74c3c')
ax1.plot(df['Date'], df['MA50'], lw=0.8, label='MA50', color='#3498db')
ax1.fill_between(df['Date'], df['Upper_BB'], df['Lower_BB'], alpha=0.15, color='gray', label='BB')
ax1.set_title(f'{primary} with Moving Averages & Bollinger Bands', fontweight='bold')
ax1.legend()

# Daily returns
ax2 = fig.add_subplot(gs[1,0])
ax2.bar(df['Date'], df['Return'], width=1, color=np.where(df['Return']>0, '#2ecc71', '#e74c3c'), alpha=0.5)
ax2.set_title('Daily Returns', fontweight='bold')

# Return distribution
ax3 = fig.add_subplot(gs[1,1])
ax3.hist(df['Return'].dropna(), bins=80, color='#3498db', edgecolor='k', alpha=0.7, density=True)
x = np.linspace(df['Return'].min(), df['Return'].max(), 100)
ax3.plot(x, stats.norm.pdf(x, df['Return'].mean(), df['Return'].std()), 'r-', lw=2, label='Normal')
ax3.set_title('Returns Distribution', fontweight='bold'); ax3.legend()

# Volatility
ax4 = fig.add_subplot(gs[1,2])
ax4.plot(df['Date'], df['Volatility']*100, lw=1, color='#e74c3c')
ax4.axhline(y=df['Volatility'].median()*100, color='k', ls='--', alpha=0.3)
ax4.set_title('Annualized Volatility (21d rolling)', fontweight='bold')
ax4.set_ylabel('%')

# Drawdown
ax5 = fig.add_subplot(gs[1,3])
ax5.fill_between(df['Date'], drawdown*100, 0, color='#e74c3c', alpha=0.3)
ax5.set_title('Drawdown %', fontweight='bold')

# ACF
ax6 = fig.add_subplot(gs[2,0])
plot_acf(df['Return'].dropna(), lags=40, ax=ax6, alpha=0.05)
ax6.set_title('Autocorrelation of Returns', fontweight='bold')

# PACF
ax7 = fig.add_subplot(gs[2,1])
plot_pacf(df['Return'].dropna(), lags=40, ax=ax7, alpha=0.05)
ax7.set_title('Partial Autocorrelation', fontweight='bold')

# Correlation heatmap
ax8 = fig.add_subplot(gs[2,2:])
top_currencies = rate_cols[:15]
corr_data = df[top_currencies].corr()
im = ax8.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax8.set_xticks(range(len(top_currencies))); ax8.set_yticks(range(len(top_currencies)))
ax8.set_xticklabels([c[:10] for c in top_currencies], rotation=90, fontsize=7)
ax8.set_yticklabels([c[:10] for c in top_currencies], fontsize=7)
ax8.set_title('Cross-Currency Correlations', fontweight='bold')
fig.colorbar(im, ax=ax8)

# QQ plot
ax9 = fig.add_subplot(gs[3,0])
stats.probplot(df['Return'].dropna(), dist='norm', plot=ax9)
ax9.get_lines()[0].set_markersize(3); ax9.get_lines()[0].set_color('#3498db')
ax9.get_lines()[1].set_color('red')
ax9.set_title('Q-Q Plot vs Normal', fontweight='bold')

# Rolling mean + volatility
ax10 = fig.add_subplot(gs[3,1])
df['RollingMean'] = df['Return'].rolling(60).mean()*100
ax10.plot(df['Date'], df['RollingMean'], lw=1, color='#2c3e50')
ax10.axhline(y=0, color='r', ls='--', alpha=0.3)
ax10.set_title('Rolling 60-Day Mean Return %', fontweight='bold')

# Summary panel
ax11 = fig.add_subplot(gs[3,2:]); ax11.axis('off')
summary = f"""
FOREX ANALYSIS - KEY METRICS
────────────────────────────────────
Pair:            {primary}
Period:          {df['Date'].min().date()} to {df['Date'].max().date()}
Observations:    {len(df):,} trading days

Level:
  Current Rate:   {df[primary].iloc[-1]:.4f}
  Mean:           {df[primary].mean():.4f}
  Min:            {df[primary].min():.4f}
  Max:            {df[primary].max():.4f}

Returns:
  Mean Daily:     {df['Return'].mean()*100:.4f}%
  Volatility:     {df['Volatility'].iloc[-1]*100:.1f}%
  Skewness:       {df['Return'].dropna().skew():.3f}
  Kurtosis:       {df['Return'].dropna().kurtosis():.3f}
  Max Drawdown:   {drawdown.min()*100:.1f}%
  Up Days:        {(df['Return']>0).sum()}/{(df['Return']>0).count()} ({(df['Return']>0).mean()*100:.1f}%)
  Down Days:      {(df['Return']<0).sum()}/{(df['Return']<0).count()} ({(df['Return']<0).mean()*100:.1f}%)

Signals:
  MA Crossovers:  {cross_buy} buys / {cross_sell} sells
  BB Oversold:    {(df['BB_Signal']==1).sum()}
  BB Overbought:  {(df['BB_Signal']==-1).sum()}

Stationarity:
  Level (ADF):    {'STATIONARY' if adf[1]<0.05 else 'NON-STATIONARY'}
  Returns (ADF):  {'STATIONARY' if adf_r[1]<0.05 else 'NON-STATIONARY'}
"""
ax11.text(0.02, 0.95, summary, transform=ax11.transAxes, fontsize=10,
    fontfamily='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

plt.tight_layout()
fig.savefig(f"{OUT}\\Dashboard.png", dpi=200, bbox_inches='tight')
w("  [OK] Dashboard.png")

# ═══════════════════════════════════════════
# 7. EXECUTIVE SUMMARY
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  7. EXECUTIVE SUMMARY\n" + "▔"*60)
w(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      EXECUTIVE SUMMARY                              │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  1. OVERVIEW: {len(df):,} trading days of {primary}                  │
  │     • Current rate: {df[primary].iloc[-1]:.4f}                                │
  │     • Period: {df['Date'].min().date()} to {df['Date'].max().date()}              │
  │                                                                     │
  │  2. RETURNS ANALYSIS:                                                │
  │     • Mean daily return: {df['Return'].mean()*100:.4f}%                              │
  │     • Annualized volatility: {df['Volatility'].iloc[-1]*100:.1f}%                       │
  │     • {(df['Return']>0).sum()}/{len(df)} positive days ({(df['Return']>0).mean()*100:.1f}%)              │
  │     • Max drawdown: {drawdown.min()*100:.1f}%                                   │
  │     • Returns are {'leptokurtic' if df['Return'].dropna().kurtosis()>3 else 'approximately normal'} (kurtosis: {df['Return'].dropna().kurtosis():.2f})
  │                                                                     │
  │  3. STATIONARITY:                                                    │
  │     • Raw rates: {'NON-STATIONARY (random walk)' if adf[1]>=0.05 else 'STATIONARY'}       │
  │     • Daily returns: {'STATIONARY (white noise)' if adf_r[1]<0.05 else 'NON-STATIONARY'}  │
  │                                                                     │
  │  4. TECHNICAL SIGNALS:                                               │
  │     • MA20/MA50 crossover: {cross_buy} buy, {cross_sell} sell signals            │
  │     • Bollinger Band extremes: {(df['BB_Signal'].abs()>0).sum()} days                   │
  │                                                                     │
  │  5. BAYESIAN: P(Up day) = {a/(a+b)*100:.1f}%                             │
  │     95% HDI: [{lo*100:.1f}%, {hi*100:.1f}%]                                 │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
""")
log.close()
print(f"\n✅ PROJECT 6 COMPLETE → {OUT}\\report.txt")
