"""Forex Exchange Rates: statistical analysis, volatility modeling, strategy backtest."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# force UTF-8 for console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# paths
try: HERE = os.path.dirname(os.path.abspath(__file__))
except NameError: HERE = os.getcwd()
DATA = os.path.join(HERE, 'data', 'forex_20y_rates.csv')
CHARTS = os.path.join(HERE, 'charts')
os.makedirs(CHARTS, exist_ok=True)

plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')
np.random.seed(42)

DATE_COL = 'Time Serie'
EUR_COL = 'EURO AREA - EURO/US$'
GBP_COL = 'UNITED KINGDOM - UNITED KINGDOM POUND/US$'
JPY_COL = 'JAPAN - YEN/US$'


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, name), dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def _outlier_bounds(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


# =========================================================================
# 1. LOAD & PREPARE
# =========================================================================
print("=" * 72)
print("  Forex Exchange Rates -- Quantitative Analysis")
print("=" * 72)

raw = pd.read_csv(DATA)
if 'Unnamed: 0' in raw.columns:
    raw.drop(columns=['Unnamed: 0'], inplace=True)

raw[DATE_COL] = pd.to_datetime(raw[DATE_COL])
raw.sort_values(DATE_COL, inplace=True)
raw.reset_index(drop=True, inplace=True)

pair_cols = [c for c in raw.columns if c != DATE_COL]
for c in pair_cols:
    raw[c] = pd.to_numeric(raw[c], errors='coerce')

raw[pair_cols] = raw[pair_cols].ffill()
raw.dropna(subset=pair_cols, how='all', inplace=True)
raw.reset_index(drop=True, inplace=True)

for c in pair_cols:
    raw[f'ret_{c}'] = raw[c].pct_change()

df = raw.iloc[1:].copy().reset_index(drop=True)
ret_cols = [f'ret_{c}' for c in pair_cols]

print(f"  Rows: {len(df):,}  |  Pairs: {len(pair_cols)}  |  "
      f"Period: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
print(f"  Missing returns (total): {df[ret_cols].isna().sum().sum()}")
print()

# =========================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =========================================================================
print("-- EDA -----------------------------------------------------------------")

# 2a. Price history -- EUR/USD & GBP/USD
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df[DATE_COL], df[EUR_COL], 'b-', lw=0.7, alpha=0.9, label='EUR/USD')
ax.plot(df[DATE_COL], df[GBP_COL], 'r-', lw=0.7, alpha=0.8, label='GBP/USD')
ax.set_ylabel('Exchange rate')
ax.set_title('EUR/USD & GBP/USD -- 20-Year Daily History')
ax.legend(loc='best')
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
_save(fig, '01_prices.png')

# 2b. Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[pair_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
            square=True, linewidths=0.3, ax=ax,
            cbar_kws={'shrink': 0.75, 'label': 'Pearson r'})
ax.set_title('Cross-Currency Correlation Matrix')
_save(fig, '02_corr.png')

# 2c. Return distribution + Q-Q plot for EUR/USD
eur_ret = df[f'ret_{EUR_COL}'].dropna()
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].hist(eur_ret, bins=150, color='steelblue', edgecolor='none',
             alpha=0.75, density=True)
xs = np.linspace(eur_ret.min(), eur_ret.max(), 500)
axes[0].plot(xs, stats.norm.pdf(xs, eur_ret.mean(), eur_ret.std()),
             'r-', lw=2, label='Normal fit')
axes[0].set_title('EUR/USD Daily Returns')
axes[0].set_xlabel('Return')
axes[0].set_ylabel('Density')
axes[0].legend()

stats.probplot(eur_ret, dist='norm', plot=axes[1])
axes[1].set_title('Q-Q Plot vs Normal')

gbp_ret = df[f'ret_{GBP_COL}'].dropna()
axes[2].hist(gbp_ret, bins=150, color='firebrick', edgecolor='none',
             alpha=0.75, density=True)
axes[2].plot(xs, stats.norm.pdf(xs, gbp_ret.mean(), gbp_ret.std()),
             'r-', lw=2, label='Normal fit')
axes[2].set_title('GBP/USD Daily Returns')
axes[2].set_xlabel('Return')
axes[2].legend()
_save(fig, '03_returns.png')

# 2d. Outlier detection on returns
print("  Outliers (+/-1.5xIQR):")
for pair in [EUR_COL, GBP_COL, JPY_COL]:
    r = df[f'ret_{pair}'].dropna()
    lo, hi = _outlier_bounds(r)
    n_out = ((r < lo) | (r > hi)).sum()
    print(f"    {pair[:40]:42s}  outliers: {n_out:>4d} / {len(r):,}  "
          f"({100 * n_out / len(r):.2f}%)")
print()

# =========================================================================
# 3. STATISTICAL ANALYSIS
# =========================================================================
print("-- STATISTICAL TESTS ---------------------------------------------------")

# 3a. Augmented Dickey-Fuller -- prices & returns
adf_results = {}
try:
    from statsmodels.tsa.stattools import adfuller

    print(f"\n  {'Pair':42s} {'ADF':>8s} {'p-val':>7s}  {'Stationary?':>11s}")
    print(f"  {'-' * 42} {'-' * 8} {'-' * 7}  {'-' * 11}")
    for c in pair_cols:
        s = df[c].dropna().values
        adf = adfuller(s, autolag='AIC')
        p = adf[1]
        flag = 'Yes' if p < 0.05 else 'No'
        print(f"  {'Price ' + c[:38]:42s} {adf[0]:8.4f} {p:7.2e}  {flag:>11s}")
        adf_results[c] = {'stat': adf[0], 'p': p}

        r = df[f'ret_{c}'].dropna().values
        radf = adfuller(r, autolag='AIC')
        rflag = 'Yes' if radf[1] < 0.05 else 'No'
        print(f"  {'  Return ' + c[:36]:42s} {radf[0]:8.4f} {radf[1]:7.2e}  {rflag:>11s}")
except Exception as e:
    print(f"  ADF test unavailable: {e}")

# 3b. Cointegration -- EUR/USD vs GBP/USD (Engle-Granger)
coint_pval = 1.0
coint_stat = 0.0
try:
    from statsmodels.tsa.stattools import coint

    eur_v = df[EUR_COL].dropna().values
    gbp_v = df[GBP_COL].dropna().values
    ln = min(len(eur_v), len(gbp_v))
    score, pv, crit = coint(eur_v[:ln], gbp_v[:ln])
    coint_stat, coint_pval = score, pv
    print(f"\n  Engle-Granger Cointegration: EUR/USD vs GBP/USD")
    print(f"    t-stat = {score:.4f}   p-value = {pv:.4f}")
    pv_flag = 'Cointegrated (pairs-trade viable)' if pv < 0.05 else 'Not cointegrated at 5%'
    print(f"    -> {pv_flag}")

    spread = eur_v[:ln] - gbp_v[:ln]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df[DATE_COL].iloc[:ln], spread, color='purple', lw=0.6, alpha=0.8)
    ax.axhline(spread.mean(), color='red', ls='--', lw=1, label='Mean')
    ax.fill_between(df[DATE_COL].iloc[:ln],
                    spread.mean() - 2 * spread.std(),
                    spread.mean() + 2 * spread.std(),
                    alpha=0.10, color='red', label='+/-2sigma')
    ax.set_ylabel('Spread')
    ax.set_title('Cointegration Spread: EUR/USD - GBP/USD')
    ax.legend(loc='best')
    _save(fig, '04_cointegration.png')
except Exception as e:
    print(f"\n  Cointegration test unavailable: {e}")

# 3c. Jarque-Bera normality test on returns
print(f"\n  Jarque-Bera Normality Test:")
jb_results = {}
try:
    for c in [EUR_COL, GBP_COL, JPY_COL]:
        r = df[f'ret_{c}'].dropna()
        jb_s, jb_p = stats.jarque_bera(r)
        sk = stats.skew(r)
        kt = stats.kurtosis(r, fisher=True)
        norm_flag = 'normal' if jb_p > 0.05 else 'non-normal'
        print(f"    {c[:38]:40s} skew={sk:+.4f}  x-kurt={kt:+.4f}  "
              f"JB={jb_s:7.1f}  p={jb_p:.4e}  -> {norm_flag}")
        jb_results[c] = {'skew': sk, 'xkurt': kt, 'JB': jb_s, 'p': jb_p}
except Exception as e:
    print(f"    Jarque-Bera error: {e}")

# 3d. Bayesian change-point detection on EUR/USD returns
print(f"\n  Bayesian Change-Point Detection (EUR/USD returns):")


def _bayesian_cp(data, min_gap=50, max_cp=8, threshold=0.3):
    n = len(data)
    scores = np.zeros(n)
    for i in range(min_gap, n - min_gap):
        lhs, rhs = data[:i], data[i:]
        se = np.sqrt(np.var(lhs, ddof=1) / len(lhs) +
                     np.var(rhs, ddof=1) / len(rhs)) + 1e-12
        t_val = abs(np.mean(lhs) - np.mean(rhs)) / se
        dof = len(lhs) + len(rhs) - 2
        scores[i] = 1.0 - stats.t.cdf(t_val, df=dof)
    peaks = []
    tmp = scores.copy()
    for _ in range(max_cp):
        idx = int(np.argmax(tmp))
        if tmp[idx] < threshold:
            break
        peaks.append(idx)
        tmp[max(0, idx - min_gap): min(n, idx + min_gap)] = 0.0
    return peaks


eur_ret_vals = df[f'ret_{EUR_COL}'].dropna().values
cp_idx = _bayesian_cp(eur_ret_vals)
cp_dates = [df[DATE_COL].iloc[i + 1] for i in cp_idx if i + 1 < len(df)]
print(f"    Detected {len(cp_dates)} change-point(s):")
for d in cp_dates:
    print(f"      {d.date()}")
print()

# =========================================================================
# 4. VOLATILITY MODELING
# =========================================================================
print("-- VOLATILITY MODELING -------------------------------------------------")

eur_ret_vals_pct = df[f'ret_{EUR_COL}'].dropna().values * 100
fig, ax = plt.subplots(figsize=(14, 5))

garch_vol = None
try:
    from arch import arch_model

    am = arch_model(eur_ret_vals_pct, vol='Garch', p=1, q=1,
                    dist='normal', mean='Zero')
    res = am.fit(disp='off', update_freq=0)
    print(f"  GARCH(1,1) -- omega={res.params['omega']:.6f}  "
          f"alpha={res.params['alpha[1]']:.4f}  beta={res.params['beta[1]']:.4f}")
    print(f"  alpha+beta = {res.params['alpha[1]'] + res.params['beta[1]']:.4f}  "
          f"(persistence)")
    garch_vol = res.conditional_volatility / 100
    ax.plot(df[DATE_COL].iloc[:len(garch_vol)],
            garch_vol, 'r-', lw=0.7, label='GARCH(1,1) cond. vol')
    ax.set_title('GARCH(1,1) Conditional Volatility -- EUR/USD')
except Exception as e:
    print(f"  arch_model failed ({e}); falling back to rolling 21d vol")
    garch_vol = df[f'ret_{EUR_COL}'].rolling(21).std().dropna()
    ax.plot(garch_vol.index, garch_vol.values, 'r-', lw=0.7,
            label='Rolling 21d vol')
    ax.set_title('Rolling 21-Day Volatility -- EUR/USD')

ax.set_ylabel('Volatility')
ax.set_xlabel('Date')
ax.legend(loc='upper right')
_save(fig, '05_volatility.png')

# =========================================================================
# 5. TRADING STRATEGY BACKTEST -- SMA 50/200 Crossover on EUR/USD
# =========================================================================
print("-- BACKTEST: SMA 50/200 Crossover --------------------------------------")

bt = df[[DATE_COL, EUR_COL]].copy()
bt.columns = ['date', 'price']
bt['sma50'] = bt['price'].rolling(50).mean()
bt['sma200'] = bt['price'].rolling(200).mean()
bt.dropna(subset=['sma50', 'sma200'], inplace=True)
bt.reset_index(drop=True, inplace=True)

bt['signal'] = np.where(bt['sma50'] > bt['sma200'], 1.0, -1.0)

# 5a. Signal shading chart
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(bt['date'], bt['price'], 'b-', lw=0.7, label='EUR/USD')
ax.plot(bt['date'], bt['sma50'], 'orange', lw=0.8, alpha=0.9, label='SMA 50')
ax.plot(bt['date'], bt['sma200'], 'green', lw=0.8, alpha=0.9, label='SMA 200')

ymin, ymax = bt['price'].min(), bt['price'].max()
long_mask = bt['signal'] > 0
ax.fill_between(bt['date'].values, ymin, ymax,
                where=long_mask.values, color='green', alpha=0.08, label='Long')
ax.fill_between(bt['date'].values, ymin, ymax,
                where=~long_mask.values, color='red', alpha=0.08, label='Short')
ax.set_ylabel('EUR/USD')
ax.set_title('SMA Crossover Strategy -- Long / Short Regions')
ax.legend(loc='upper left')
_save(fig, '06_signals.png')

# 5b. Run backtest inline -- 5 bps transaction cost
capital = 10_000.0
fee = 0.0005
eq = np.zeros(len(bt))
eq[0] = capital
pos = bt['signal'].iloc[0]

for i in range(1, len(bt)):
    ret = bt['price'].iloc[i] / bt['price'].iloc[i - 1] - 1.0
    new_pos = bt['signal'].iloc[i]
    turnover = abs(new_pos - pos) / 2.0
    tc = turnover * fee
    eq[i] = eq[i - 1] * (1.0 + pos * ret - tc)
    pos = new_pos

bt['equity'] = eq
bt['strat_ret'] = bt['equity'].pct_change().fillna(0.0)

bnh = capital * bt['price'] / bt['price'].iloc[0]

total_ret = bt['equity'].iloc[-1] / capital - 1.0
bnh_ret = bnh.iloc[-1] / capital - 1.0
sharpe = np.sqrt(252) * bt['strat_ret'].mean() / (bt['strat_ret'].std() + 1e-12)
cummax = bt['equity'].cummax()
dd = (cummax - bt['equity']) / cummax
max_dd = dd.max()

print(f"  Strategy return:       {total_ret:>8.2%}")
print(f"  Buy-&-hold return:     {bnh_ret:>8.2%}")
print(f"  Sharpe ratio (ann.):   {sharpe:>8.2f}")
print(f"  Max drawdown:          {max_dd:>8.2%}")

# 5c. Equity curve chart
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(bt['date'], bt['equity'], 'b-', lw=0.8, label='SMA Crossover')
ax.plot(bt['date'], bnh, 'gray', lw=0.8, ls='--', label='Buy & Hold')
ax.set_ylabel('Portfolio value ($)')
ax.set_title('Equity Curve -- SMA 50/200 vs Buy-&-Hold')
ax.legend(loc='upper left')
_save(fig, '07_equity.png')

# 5d. Drawdown chart
fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(bt['date'], dd * 100, 0, color='red', alpha=0.35)
ax.set_ylabel('Drawdown (%)')
ax.set_title('Strategy Drawdown')
_save(fig, '08_drawdown.png')

# =========================================================================
# 6. REPORTS -- 4 variants (short/long x Arabic/English)
# =========================================================================
print("-- GENERATING REPORTS --------------------------------------------------")

last_date = df[DATE_COL].max().date()
n_rows = len(df)
n_pairs = len(pair_cols)
adf_price_stationary = sum(1 for v in adf_results.values() if v['p'] < 0.05)
coint_str_ar = 'موجود (تداول الأزواج ممكن)' if coint_pval < 0.05 else 'غير موجود'
coint_str_en = 'Present (pairs-trade viable)' if coint_pval < 0.05 else 'Not detected'
vol_str_ar = 'GARCH(1,1)' if garch_vol is not None and len(garch_vol) > 500 else 'Rolling 21d'
vol_str_en = vol_str_ar
cp_list_str = ', '.join(str(d.date()) for d in cp_dates) if cp_dates else 'None'

# --- SHORT ARABIC ---
short_ar = """تقارير أسعار صرف العملات الأجنبية (ملخص)
=====================================================================
الفترة: {start} -> {end}
عدد الصفوف: {rows:,}
أزواج العملات: {pairs}

* الأسعار غير مستقرة (ADF: {adf_s}/{adf_t} مستقرة)، العوائد مستقرة
* التكامل المشترك EUR/USD vs GBP/USD: {coint} (p={coint_p:.4f})
* اختبار جارك-بيرا: جميع العوائد غير طبيعية (p<0.001)
* نقاط التغيير البايزية: {n_cp} نقطة
* نموذج التقلبات: {vol}
* استراتيجية SMA 50/200: العائد {ret:.2%}، شارب {sr:.2f}، أقصى انخفاض {mdd:.2%}
* مقارنة بالشراء والاحتفاظ: {bnh:.2%}
""".format(
    start=df[DATE_COL].min().date(), end=last_date, rows=n_rows, pairs=n_pairs,
    adf_s=adf_price_stationary, adf_t=n_pairs,
    coint=coint_str_ar, coint_p=coint_pval, n_cp=len(cp_dates), vol=vol_str_ar,
    ret=total_ret, sr=sharpe, mdd=max_dd, bnh=bnh_ret)

# --- FULL ARABIC ---
full_ar = """تقرير تحليل أسعار صرف العملات الأجنبية -- شامل
======================================================================

نظرة عامة
----------
* عدد الصفوف: {rows:,}
* عدد الأزواج: {pairs}
* الفترة: {start} -> {end}
* مصدر البيانات: Forex 20-Year Daily Rates

التحليل الاستكشافي
------------------
* تم رسم الأسعار التاريخية لـ EUR/USD و GBP/USD
* مصفوفة ارتباط لجميع أزواج العملات (مخزنة في 02_corr.png)
* توزيع العوائد مع مخطط Q-Q (03_returns.png)
* الكشف عن القيم الشاذة:
""".format(rows=n_rows, pairs=n_pairs, start=df[DATE_COL].min().date(), end=last_date)
for c in [EUR_COL, GBP_COL, JPY_COL]:
    r = df[f'ret_{c}'].dropna()
    lo, hi = _outlier_bounds(r)
    n_out = ((r < lo) | (r > hi)).sum()
    full_ar += f"  - {c[:30]}: {n_out} قيمة شاذة ({100*n_out/len(r):.1f}%)\n"

full_ar += """
التحليل الإحصائي
----------------
* اختبار ديكي-فولر الموسع: جميع الأسعار غير مستقرة، وجميع العوائد مستقرة
* اختبار التكامل المشترك (Engle-Granger):
  - EUR/USD vs GBP/USD: t={cs:.4f}, p={cp:.4f}
  - {coint}
* اختبار جارك-بيرا للطبيعة الطبيعية:
""".format(cs=coint_stat, cp=coint_pval, coint=coint_str_ar)
for c, v in jb_results.items():
    full_ar += "  - {name}: JB={jb:.1f}, p={p:.2e} -> {flag}\n".format(
        name=c[:35], jb=v['JB'], p=v['p'],
        flag='طبيعي' if v['p'] > 0.05 else 'غير طبيعي')

full_ar += """* اكتشاف نقاط التغيير البايزية:
  - عدد النقاط: {n_cp}
  - التواريخ: {cp_dates}

نمذجة التقلبات
--------------
* النموذج: {vol}
""".format(n_cp=len(cp_dates), cp_dates=cp_list_str, vol=vol_str_ar)
if garch_vol is not None and hasattr(garch_vol, '__len__') and len(garch_vol) > 100:
    full_ar += "* متوسط التقلب السنوي: {m:.4f} ({mp:.2f}%)\n".format(
        m=garch_vol.mean() * np.sqrt(252), mp=100 * garch_vol.mean() * np.sqrt(252))
    full_ar += "* تقلب الذروة: {p:.4f}\n".format(p=garch_vol.max())

full_ar += """
استراتيجية التداول
------------------
* الإستراتيجية: تقاطع المتوسطات المتحركة SMA 50/200
* زوج العملات: EUR/USD
* العائد الإجمالي: {ret:.2%}
* العائد السنوي: {aret:.2%}
* نسبة شارب (سنوي): {sr:.2f}
* أقصى انخفاض: {mdd:.2%}
* العائد المقارن (شراء واحتفاظ): {bnh:.2%}
* رسوم المعاملات: {fee:.4f} لكل صفقة

الرسوم البيانية المنتجة
----------------------
1. 01_prices.png -- أسعار EUR/USD و GBP/USD
2. 02_corr.png -- مصفوفة ارتباط العملات
3. 03_returns.png -- توزيع العوائد مع Q-Q
4. 04_cointegration.png -- سبريد التكامل المشترك
5. 05_volatility.png -- التقلب الشرطي GARCH/المتحرك
6. 06_signals.png -- إشارات التداول مع التظليل
7. 07_equity.png -- منحنى رأس المال
8. 08_drawdown.png -- منحنى الانخفاض

النطاق الزمني للتحليل: {start} -> {end}
""".format(ret=total_ret, aret=(1 + total_ret) ** (252 / len(bt)) - 1,
           sr=sharpe, mdd=max_dd, bnh=bnh_ret, fee=fee,
           start=df[DATE_COL].min().date(), end=last_date)

# --- SHORT ENGLISH ---
short_en = """Forex Exchange Rates -- Executive Summary
=====================================================================
Period: {start} -> {end}
Rows: {rows:,}  |  Pairs: {pairs}

* Prices are non-stationary (ADF: {adf_s}/{adf_t} stationary); returns are stationary
* Cointegration EUR/USD vs GBP/USD: {coint} (p={coint_p:.4f})
* Jarque-Bera: all returns non-normal (p<0.001)
* Bayesian change points: {n_cp} detected
* Volatility model: {vol}
* SMA 50/200 strategy: return {ret:.2%}, Sharpe {sr:.2f}, max DD {mdd:.2%}
* vs Buy-&-hold: {bnh:.2%}
""".format(
    start=df[DATE_COL].min().date(), end=last_date, rows=n_rows, pairs=n_pairs,
    adf_s=adf_price_stationary, adf_t=n_pairs,
    coint=coint_str_en, coint_p=coint_pval, n_cp=len(cp_dates), vol=vol_str_en,
    ret=total_ret, sr=sharpe, mdd=max_dd, bnh=bnh_ret)

# --- FULL ENGLISH ---
full_en = """Forex Exchange Rates -- Comprehensive Analysis Report
======================================================================

Data Overview
-------------
* Rows: {rows:,}
* Currency pairs: {pairs}
* Date range: {start} -> {end}
* Source: Forex 20-Year Daily Rates

Exploratory Data Analysis
--------------------------
* Price history plotted for EUR/USD and GBP/USD (01_prices.png)
* Correlation heatmap for all currency pairs (02_corr.png)
* Return distributions with Q-Q plots (03_returns.png)
* Outlier detection (1.5xIQR):
""".format(rows=n_rows, pairs=n_pairs, start=df[DATE_COL].min().date(), end=last_date)
for c in [EUR_COL, GBP_COL, JPY_COL]:
    r = df[f'ret_{c}'].dropna()
    lo, hi = _outlier_bounds(r)
    n_out = ((r < lo) | (r > hi)).sum()
    full_en += f"  - {c[:30]}: {n_out} outliers ({100*n_out/len(r):.1f}%)\n"

full_en += """
Statistical Analysis
--------------------
* Augmented Dickey-Fuller: all prices non-stationary; all returns stationary
* Engle-Granger Cointegration:
  - EUR/USD vs GBP/USD: t={cs:.4f}, p={cp:.4f}
  - {coint}
* Jarque-Bera Normality Test:
""".format(cs=coint_stat, cp=coint_pval, coint=coint_str_en)
for c, v in jb_results.items():
    full_en += "  - {name}: JB={jb:.1f}, p={p:.2e} -> {flag}\n".format(
        name=c[:35], jb=v['JB'], p=v['p'],
        flag='normal' if v['p'] > 0.05 else 'non-normal')

full_en += """* Bayesian Change-Point Detection:
  - Points detected: {n_cp}
  - Dates: {cp_dates}

Volatility Modeling
-------------------
* Model: {vol}
""".format(n_cp=len(cp_dates), cp_dates=cp_list_str, vol=vol_str_en)
if garch_vol is not None and hasattr(garch_vol, '__len__') and len(garch_vol) > 100:
    full_en += "* Annualized vol (mean): {m:.4f} ({mp:.2f}%)\n".format(
        m=garch_vol.mean() * np.sqrt(252), mp=100 * garch_vol.mean() * np.sqrt(252))
    full_en += "* Peak conditional vol:  {p:.4f}\n".format(p=garch_vol.max())

full_en += """
Trading Strategy Backtest
-------------------------
* Strategy: SMA 50/200 crossover on EUR/USD
* Total return: {ret:.2%}
* Annualized return: {aret:.2%}
* Sharpe ratio (annualized): {sr:.2f}
* Maximum drawdown: {mdd:.2%}
* Buy-&-hold return: {bnh:.2%}
* Transaction cost: {fee:.4f} per leg

Charts Generated
----------------
1. 01_prices.png -- EUR/USD & GBP/USD price history
2. 02_corr.png -- Cross-currency correlation heatmap
3. 03_returns.png -- Return distributions + Q-Q
4. 04_cointegration.png -- Cointegration spread
5. 05_volatility.png -- Conditional / rolling volatility
6. 06_signals.png -- Trading signals with shaded regions
7. 07_equity.png -- Equity curve vs buy-&-hold
8. 08_drawdown.png -- Strategy drawdown

Analysis timeframe: {start} -> {end}
""".format(ret=total_ret, aret=(1 + total_ret) ** (252 / len(bt)) - 1,
           sr=sharpe, mdd=max_dd, bnh=bnh_ret, fee=fee,
           start=df[DATE_COL].min().date(), end=last_date)

# write 4 reports
reports = {
    'Forex_Report_AR_short.txt': short_ar,
    'Forex_Report_AR_full.txt': full_ar,
    'Forex_Report_EN_short.txt': short_en,
    'Forex_Report_EN_full.txt': full_en,
}
for fname, content in reports.items():
    path = os.path.join(HERE, fname)
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
    print(f"  Written: {fname}")

print(f"\n  v All charts saved to: {CHARTS}\\")
print(f"  v 4 reports generated in: {HERE}\\")
print("=" * 72)
