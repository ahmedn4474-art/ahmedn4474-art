"""Financial Accounting: Transaction Analysis & Reporting"""

import os, sys, warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.stats import zscore

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    _HAS_DECOMPOSE = True
except ImportError:
    _HAS_DECOMPOSE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_HW = True
except ImportError:
    _HAS_HW = False

try: BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError: BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, 'data', 'financial_transactions_100k.csv')
CHARTS_DIR = os.path.join(BASE_DIR, 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150,
    'font.size': 9, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B',
           '#44BBA4', '#E94F37', '#393E41', '#EDC79B', '#6A8D92']


def save_fig(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close(fig)
    print(f'  Saved: charts/{name}')


# ===================================================================
# 1. LOAD DATA & ENGINEER FEATURES
# ===================================================================
print('=' * 60)
print('FINANCIAL ACCOUNTING ANALYSIS')
print('=' * 60)

df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
n_total = len(df)
print(f'\nLoaded {n_total:,} transactions')
print(f'Date range: {df["Date"].min().date()} to {df["Date"].max().date()}')

missing = df.isnull().sum()
if missing.sum() > 0:
    print(f'Missing values:\n{missing[missing > 0]}')
else:
    print('No missing values')

# Core derived fields
df['Amount'] = df['Debit']
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayName'] = df['Date'].dt.day_name()
iso_weeks = df['Date'].dt.isocalendar()
df['WeekOfYear'] = iso_weeks['week'].astype(int)

# Accounting convention: Revenue/Liability = Inflow, Expense/Asset = Outflow
direction_map = {'Revenue': 'Inflow', 'Expense': 'Outflow',
                 'Asset': 'Outflow', 'Liability': 'Inflow'}
df['Direction'] = df['Category'].map(direction_map)

print(f'Accounts: {df["Account"].nunique()} | Categories: {df["Category"].nunique()}')
print(f'Types: {df["Transaction_Type"].nunique()} | Payment methods: {df["Payment_Method"].nunique()}')
print(f'Customers/Vendors: {df["Customer_Vendor"].nunique()}')
print(f'Amount range: ${df["Amount"].min():.2f} to ${df["Amount"].max():.2f}')
print(f'Average transaction: ${df["Amount"].mean():.2f}')


# ===================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ===================================================================
print('\n--- EDA ---')

# Combined overview: category volume, payment mix, transaction types
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

cat_vol = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
colors_cat = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]
cat_vol.plot(kind='bar', ax=axes[0], color=colors_cat, edgecolor='white', width=0.7)
axes[0].set_title('Transaction Volume by Category')
axes[0].set_ylabel('Total Amount ($)')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(cat_vol.values):
    axes[0].text(i, v + 5000, f'${v/1e6:.2f}M', ha='center', fontsize=8, fontweight='bold')

pm_counts = df['Payment_Method'].value_counts()
axes[1].pie(pm_counts.values, labels=pm_counts.index, autopct='%1.1f%%',
            colors=[PALETTE[0], PALETTE[4], PALETTE[2], PALETTE[1]],
            startangle=90, explode=[0.03]*4)
axes[1].set_title('Payment Method Distribution')

tt_vol = df.groupby('Transaction_Type')['Amount'].sum().sort_values(ascending=False)
tt_vol.plot(kind='bar', ax=axes[2], color=[PALETTE[5], PALETTE[6], PALETTE[7], PALETTE[8]],
            edgecolor='white', width=0.7)
axes[2].set_title('Volume by Transaction Type')
axes[2].set_ylabel('Total Amount ($)')
axes[2].tick_params(axis='x', rotation=45)
for i, v in enumerate(tt_vol.values):
    axes[2].text(i, v + 5000, f'${v/1e6:.2f}M', ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
save_fig(fig, '01_eda_overview.png')

# Monthly cash flow: credits vs debits
monthly = df.groupby('Month').agg({'Credit': 'sum', 'Debit': 'sum'})
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly.index, monthly['Credit'] / 1e6, 'o-', color=PALETTE[0],
        linewidth=2, markersize=6, label='Credits (Inflow)')
ax.plot(monthly.index, monthly['Debit'] / 1e6, 's-', color=PALETTE[3],
        linewidth=2, markersize=6, label='Debits (Outflow)')
ax.set_xlabel('Month')
ax.set_ylabel('Volume ($ Millions)')
ax.set_title('Monthly Cash Flow: Credits vs Debits')
ax.set_xticks(range(1, 13))
ax.legend(frameon=True)
ax.grid(axis='y', alpha=0.3)
save_fig(fig, '02_monthly_cash_flow.png')

# Top customers/vendors
top_n = 20
cv_vol = df.groupby('Customer_Vendor')['Amount'].sum().sort_values(ascending=False).head(top_n)
fig, ax = plt.subplots(figsize=(14, 6))
ax.barh(range(len(cv_vol)), cv_vol.values / 1e6, color=[PALETTE[i % len(PALETTE)] for i in range(top_n)],
        edgecolor='white')
ax.set_yticks(range(len(cv_vol)))
ax.set_yticklabels(cv_vol.index, fontsize=7)
ax.set_xlabel('Total Volume ($ Millions)')
ax.set_title(f'Top {top_n} Customers / Vendors by Transaction Volume')
ax.invert_yaxis()
plt.tight_layout()
save_fig(fig, '03_top_customers.png')


# ===================================================================
# 3. STATISTICAL ANALYSIS
# ===================================================================
print('\n--- Statistical Analysis ---')

# Month-over-month growth in total volume
monthly_vol = df.groupby('Month')['Amount'].sum()
mom_growth = monthly_vol.pct_change() * 100
print('Month-over-Month Volume Growth:')
for m in range(2, 13):
    print(f'  Month {m-1} -> {m}: {mom_growth[m]:+.2f}%')

# Day-of-week activity patterns
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_stats = df.groupby('DayName')['Amount'].agg(['sum', 'mean', 'count']).reindex(dow_order)
dow_stats.columns = ['Total_Volume', 'Average_Amount', 'Transaction_Count']
print('\nDay-of-Week Transaction Profile:')
print(dow_stats.round(0).to_string())

# ANOVA: do amounts differ significantly across days?
dow_groups = [g['Amount'].values for _, g in df.groupby('DayOfWeek')]
f_stat, p_anova = stats.f_oneway(*dow_groups)
print(f'\nANOVA (Amount ~ DayOfWeek): F={f_stat:.2f}, p={p_anova:.6f}')

# Category concentration
cat_share = df.groupby('Category')['Amount'].sum() / df['Amount'].sum() * 100
cat_share = cat_share.sort_values(ascending=False)
print('\nCategory Concentration:')
for cat, pct in cat_share.items():
    print(f'  {cat}: {pct:.2f}%')

top2_share = cat_share.head(2).sum()
print(f'  Top 2 categories: {top2_share:.1f}% of total')

# Outlier detection via IQR
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)]
print(f'\nIQR Outlier Detection:')
print(f'  Q1=${Q1:.2f}, Q3=${Q3:.2f}, IQR=${IQR:.2f}')
print(f'  Bounds: [${lower_bound:.2f}, ${upper_bound:.2f}]')
print(f'  Outliers: {len(outliers_iqr):,} ({len(outliers_iqr)/n_total*100:.2f}%)')

# Z-score outliers
z_vals = np.abs(zscore(df['Amount']))
outliers_z = df[z_vals > 3]
print(f'  Z-score outliers (|z|>3): {len(outliers_z):,} ({len(outliers_z)/n_total*100:.2f}%)')


# ===================================================================
# 4. TIME SERIES ANALYSIS
# ===================================================================
print('\n--- Time Series Analysis ---')

# Daily net cash flow = Inflow categories - Outflow categories
daily_inflow = df[df['Direction'] == 'Inflow'].groupby('Date')['Amount'].sum()
daily_outflow = df[df['Direction'] == 'Outflow'].groupby('Date')['Amount'].sum()
daily_vol = df.groupby('Date')['Amount'].sum()

# Reindex to full date range
date_range = pd.date_range(df['Date'].min(), df['Date'].max(), freq='D')
daily_inflow = daily_inflow.reindex(date_range, fill_value=0)
daily_outflow = daily_outflow.reindex(date_range, fill_value=0)
daily_net = daily_inflow - daily_outflow
daily_vol = daily_vol.reindex(date_range, fill_value=0)

print(f'Daily net cash flow: mean=${daily_net.mean():.0f}, std=${daily_net.std():.0f}')
print(f'Days with net inflow: {(daily_net > 0).sum()}')
print(f'Days with net outflow: {(daily_net < 0).sum()}')

# Daily net cash flow plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(daily_net.index, daily_net.values / 1e6, 0,
                where=(daily_net.values >= 0), color=PALETTE[0], alpha=0.5, label='Net Inflow')
ax.fill_between(daily_net.index, daily_net.values / 1e6, 0,
                where=(daily_net.values < 0), color=PALETTE[3], alpha=0.5, label='Net Outflow')
ax.plot(daily_net.index, daily_net.values / 1e6, color='#393E41', linewidth=0.5, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Date')
ax.set_ylabel('Net Cash Flow ($ Millions)')
ax.set_title('Daily Net Cash Flow: Inflow minus Outflow by Category')
ax.legend()
plt.tight_layout()
save_fig(fig, '04_daily_net_cash_flow.png')

# Seasonal decomposition (7-day weekly period)
if _HAS_DECOMPOSE:
    try:
        vol_series = daily_vol.copy()
        decomp = seasonal_decompose(vol_series, model='additive', period=7, extrapolate_trend='freq')

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        decomp.observed.plot(ax=axes[0], color=PALETTE[0], linewidth=0.7)
        axes[0].set_title('Daily Transaction Volume (Observed)')
        axes[0].set_ylabel('Amount ($)')

        decomp.trend.plot(ax=axes[1], color=PALETTE[3], linewidth=1.5)
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Amount ($)')

        decomp.seasonal.plot(ax=axes[2], color=PALETTE[5], linewidth=1)
        axes[2].set_title('Seasonal Component (7-day period)')
        axes[2].set_ylabel('Amount ($)')

        decomp.resid.plot(ax=axes[3], color=PALETTE[6], linewidth=0.5, marker='o', markersize=1)
        axes[3].set_title('Residual / Noise')
        axes[3].set_ylabel('Amount ($)')
        axes[3].set_xlabel('Date')

        plt.tight_layout()
        save_fig(fig, '05_decomposition.png')

        total_var = np.var(vol_series.values)
        trend_var = np.var(decomp.trend.dropna().values)
        seasonal_var = np.var(decomp.seasonal.dropna().values)
        resid_var = np.var(decomp.resid.dropna().values)
        print(f'\nDecomposition variance explained:')
        print(f'  Trend: {trend_var/total_var*100:.1f}%')
        print(f'  Seasonal: {seasonal_var/total_var*100:.1f}%')
        print(f'  Residual: {resid_var/total_var*100:.1f}%')
    except Exception as e:
        print(f'  Decomposition error: {e}')

# Structural break detection via CUSUM and rolling statistics
rolling_mean = daily_vol.rolling(window=30).mean()
rolling_std = daily_vol.rolling(window=30).std()

target = daily_vol.mean()
k_val = 0.5 * daily_vol.std()
n_days = len(daily_vol)
cusum_pos = np.zeros(n_days)
cusum_neg = np.zeros(n_days)
for i in range(1, n_days):
    cusum_pos[i] = max(0, cusum_pos[i-1] + daily_vol.iloc[i] - target - k_val)
    cusum_neg[i] = max(0, cusum_neg[i-1] + target - daily_vol.iloc[i] - k_val)
cusum = cusum_pos - cusum_neg
cusum_threshold = 4 * daily_vol.std()
change_points = np.where(np.abs(cusum) > cusum_threshold)[0]
n_cp = len(change_points)

print(f'\nCUSUM analysis: {n_cp} potential structural breaks detected')

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
axes[0].plot(daily_vol.index, daily_vol.values / 1e6, alpha=0.4, color=PALETTE[0],
             linewidth=0.5, label='Daily Volume')
axes[0].plot(rolling_mean.index, rolling_mean.values / 1e6, color=PALETTE[3],
             linewidth=2, label='30-Day Rolling Mean')
axes[0].fill_between(rolling_mean.index,
                     (rolling_mean - 2*rolling_std).values / 1e6,
                     (rolling_mean + 2*rolling_std).values / 1e6,
                     alpha=0.1, color=PALETTE[3], label='\u00b12\u03c3 Band')
axes[0].set_ylabel('Volume ($ Millions)')
axes[0].set_title('Daily Transaction Volume with Rolling Statistics')
axes[0].legend()

axes[1].plot(daily_vol.index, cusum / 1e6, color='#393E41', linewidth=1, label='CUSUM')
axes[1].axhline(cusum_threshold / 1e6, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[1].axhline(-cusum_threshold / 1e6, color='red', linestyle='--', linewidth=1, alpha=0.7)
for cp in change_points[:30]:
    if cp < n_days:
        axes[1].axvline(daily_vol.index[cp], color='orange', linewidth=0.5, alpha=0.4)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('CUSUM ($ Millions)')
axes[1].set_title(f'CUSUM Change Detection ({n_cp} breakpoints)')
axes[1].legend()
plt.tight_layout()
save_fig(fig, '06_cusum.png')

# 30-day forecast
print('\n--- Forecasting ---')
if _HAS_HW:
    try:
        hw_model = ExponentialSmoothing(daily_vol, seasonal_periods=7, trend='add',
                                        seasonal='add', initialization_method='estimated')
        hw_fit = hw_model.fit()
        forecast = hw_fit.forecast(30)
        resid = hw_fit.resid
        mse = np.mean(resid ** 2)
        forecast_ci = 1.96 * np.sqrt(mse)
        forecast_avg = forecast.mean()
    except Exception as e:
        print(f'  Holt-Winters failed: {e}')
        _HAS_HW = False

if not _HAS_HW:
    last_7 = daily_vol.iloc[-7:].mean()
    forecast = pd.Series([last_7] * 30,
                         index=pd.date_range(daily_vol.index[-1] + timedelta(1), periods=30))
    forecast_ci = 0
    forecast_avg = last_7

print(f'Forecast next 30 days: avg ${forecast_avg:,.0f}/day')
print(f'Projected monthly volume: ${forecast_avg * 30:,.0f}')

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily_vol.index[-90:], daily_vol.values[-90:] / 1e6,
        color=PALETTE[0], linewidth=1, label='Historical (last 90 days)')
ax.plot(forecast.index, forecast.values / 1e6,
        color=PALETTE[3], linewidth=2, linestyle='--', label='30-Day Forecast')
if forecast_ci > 0:
    ax.fill_between(forecast.index,
                    (forecast - forecast_ci).values / 1e6,
                    (forecast + forecast_ci).values / 1e6,
                    alpha=0.15, color=PALETTE[3], label='95% Confidence Interval')
ax.set_xlabel('Date')
ax.set_ylabel('Volume ($ Millions)')
ax.set_title(f'30-Day Forecast: Expected ${forecast_avg:,.0f}/day')
ax.legend()
plt.tight_layout()
save_fig(fig, '07_forecast.png')


# ===================================================================
# 5. PATTERN ANALYSIS
# ===================================================================
print('\n--- Pattern Analysis ---')

# Pareto: customer-level concentration
cv_total = df.groupby('Customer_Vendor')['Amount'].sum().sort_values(ascending=False)
cv_cum_pct = cv_total.cumsum() / cv_total.sum() * 100
top20pct_count = max(1, int(len(cv_total) * 0.2))
pareto_pct = cv_cum_pct.iloc[top20pct_count - 1]
print(f'Pareto: top 20% of customers ({top20pct_count}) account for {pareto_pct:.1f}% of volume')

# Category concentration
cat_total = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
cat_pct_vals = cat_total / cat_total.sum() * 100
cat_cum = cat_pct_vals.cumsum()
top1_cat_name = cat_total.index[0]
top1_cat_pct = cat_pct_vals.iloc[0]

# Pareto chart: customer + category
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Customer Pareto
ax1 = axes[0]
x_vals = range(1, min(31, len(cv_total) + 1))
ax1.bar(x_vals, cv_total.values[:30] / 1e6, color=PALETTE[0], alpha=0.7, edgecolor='white')
ax1_twin = ax1.twinx()
ax1_twin.plot(x_vals, cv_cum_pct.values[:30], 'o-', color=PALETTE[3], linewidth=2, markersize=4)
ax1_twin.axhline(80, color='gray', linestyle='--', alpha=0.5)
ax1_twin.set_ylabel('Cumulative %')
ax1.set_xlabel('Customers (ranked)')
ax1.set_ylabel('Volume ($ Millions)')
ax1.set_title(f'Customer Pareto: Top 20% = {pareto_pct:.1f}% of Volume')

# Category concentration
ax2 = axes[1]
colors_cc = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]
ax2.bar(range(len(cat_pct_vals)), cat_pct_vals.values, color=colors_cc, edgecolor='white', alpha=0.8)
ax2_twin = ax2.twinx()
ax2_twin.plot(range(len(cat_cum)), cat_cum.values, 'o-', color=PALETTE[3], linewidth=2, markersize=6)
ax2_twin.axhline(80, color='gray', linestyle='--', alpha=0.5)
ax2.set_xticks(range(len(cat_pct_vals)))
ax2.set_xticklabels(cat_pct_vals.index, rotation=45)
ax2.set_xlabel('Category')
ax2.set_ylabel('% of Total Volume')
ax2_twin.set_ylabel('Cumulative %')
ax2.set_title(f'Category Concentration: Top = {top1_cat_name} ({top1_cat_pct:.1f}%)')
plt.tight_layout()
save_fig(fig, '08_pareto.png')

# Payment method preference by category
pm_cat = pd.crosstab(df['Category'], df['Payment_Method'],
                     values=df['Amount'], aggfunc='sum', normalize='index')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pm_cat, annot=True, fmt='.1%', cmap='YlOrRd', ax=ax,
            linewidths=0.5, cbar_kws={'label': 'Proportion of Category Volume'})
ax.set_title('Payment Method Preference by Category')
ax.set_xlabel('Payment Method')
ax.set_ylabel('Category')
plt.tight_layout()
save_fig(fig, '09_payment_by_category.png')

print('\nPayment Method Mix by Category:')
print(pm_cat.round(3) * 100)

# Recurring vs one-time detection
# Use rounded amounts to group near-identical transactions
print('\nRecurring vs One-Time Analysis:')
df['Amount_Rnd'] = df['Amount'].round(-1)  # round to nearest $10
rc_groups = df.groupby(['Amount_Rnd', 'Category', 'Customer_Vendor']).agg(
    count=('Amount', 'count'),
    avg_amount=('Amount', 'mean')
).reset_index()
rc_groups['is_recurring'] = rc_groups['count'] >= 3
rec_patterns = rc_groups['is_recurring'].sum()
onetime_patterns = (~rc_groups['is_recurring']).sum()
rec_txns = rc_groups.loc[rc_groups['is_recurring'], 'count'].sum()
onetime_txns = rc_groups.loc[~rc_groups['is_recurring'], 'count'].sum()
# If still no recurring due to data granularity, try coarser rounding
if rec_txns == 0:
    df['Amount_Rnd2'] = (df['Amount'] / 50).round() * 50
    rc_groups2 = df.groupby(['Amount_Rnd2', 'Category', 'Customer_Vendor']).agg(
        count=('Amount', 'count'),
        avg_amount=('Amount', 'mean')
    ).reset_index()
    rc_groups2['is_recurring'] = rc_groups2['count'] >= 3
    rec_patterns = rc_groups2['is_recurring'].sum()
    rec_txns = rc_groups2.loc[rc_groups2['is_recurring'], 'count'].sum()
    onetime_txns = n_total - rec_txns
print(f'  Recurring patterns: {rec_patterns} groups, {rec_txns:,} transactions ({rec_txns/n_total*100:.1f}%)')
print(f'  One-time: {onetime_txns:,} transactions ({onetime_txns/n_total*100:.1f}%)')

# Recurring vs one-time chart
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie([rec_txns, onetime_txns], labels=['Recurring', 'One-Time'],
       autopct='%1.1f%%', colors=[PALETTE[0], PALETTE[1]],
       startangle=90, explode=[0.03, 0.03])
ax.set_title('Recurring vs One-Time Transactions')
save_fig(fig, '12_recurring_vs_onetime.png')


# ===================================================================
# 6. ANOMALY DETECTION
# ===================================================================
print('\n--- Anomaly Detection ---')

# IQR-based flag
df['Is_IQR_Outlier'] = (df['Amount'] < lower_bound) | (df['Amount'] > upper_bound)

# Day-of-week conditional z-score
dow_mean = df.groupby('DayOfWeek')['Amount'].transform('mean')
dow_std = df.groupby('DayOfWeek')['Amount'].transform('std').replace(0, np.nan)
df['DOW_Z'] = (df['Amount'] - dow_mean) / dow_std
df['Is_DOW_Anomaly'] = df['DOW_Z'].abs() > 3

# Combined flag
df['Is_Flagged'] = df['Is_IQR_Outlier'] | df['Is_DOW_Anomaly']
n_flagged = df['Is_Flagged'].sum()
n_high_val = (np.abs(zscore(df['Amount'])) > 3).sum()

print(f'IQR outliers: {df["Is_IQR_Outlier"].sum():,}')
print(f'High-value (|z|>3): {n_high_val:,}')
print(f'Day-of-week anomalies: {df["Is_DOW_Anomaly"].sum():,}')
print(f'Total flagged: {n_flagged:,} ({n_flagged/n_total*100:.2f}%)')

# Breakdown by category
cat_outliers = df[df['Is_Flagged']].groupby('Category').agg(
    count=('Amount', 'count'),
    total_value=('Amount', 'sum'),
    avg_amount=('Amount', 'mean'),
    max_amount=('Amount', 'max')
).round(2)
print('\nFlagged Transactions by Category:')
print(cat_outliers.to_string())

# Anomaly visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_cat_map = {'Revenue': PALETTE[0], 'Expense': PALETTE[3],
                  'Asset': PALETTE[2], 'Liability': PALETTE[1]}
df_plot = df.copy()
df_plot['Outlier_Label'] = df_plot['Is_Flagged'].map({True: 'Flagged', False: 'Normal'})

sns.boxplot(data=df_plot, x='Category', y='Amount', ax=axes[0],
            palette=colors_cat_map, width=0.6)
axes[0].set_title('Transaction Amount Distribution by Category')

daily_out = df_plot.groupby('Date').agg(
    total_vol=('Amount', 'sum'),
    n_outliers=('Is_Flagged', 'sum')
)
sc = axes[1].scatter(daily_out.index, daily_out['total_vol'] / 1e6,
                     c=daily_out['n_outliers'], cmap='YlOrRd', s=30,
                     alpha=0.7, edgecolors='gray', linewidth=0.5)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Daily Volume ($ Millions)')
axes[1].set_title('Daily Volume Colored by Outlier Count')
cbar = plt.colorbar(sc, ax=axes[1])
cbar.set_label('Outliers')
plt.tight_layout()
save_fig(fig, '10_outliers.png')


# ===================================================================
# 7. DAY-OF-WEEK PATTERNS (dedicated chart)
# ===================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dow_vol_sum = df.groupby('DayName')['Amount'].sum().reindex(dow_order)
dow_vol_cnt = df.groupby('DayName')['Amount'].count().reindex(dow_order)
dow_vol_avg = df.groupby('DayName')['Amount'].mean().reindex(dow_order)

colors_dow = [PALETTE[0]] * 5 + [PALETTE[1]] * 2
dow_vol_sum.plot(kind='bar', ax=axes[0], color=colors_dow, edgecolor='white', width=0.7)
axes[0].set_title('Total Transaction Volume by Day of Week')
axes[0].set_ylabel('Total Volume ($)')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(dow_vol_sum.values):
    axes[0].text(i, v + 5000, f'${v/1e6:.2f}M', ha='center', fontsize=7)

dow_vol_avg.plot(kind='bar', ax=axes[1], color=colors_dow, edgecolor='white', width=0.7)
axes[1].set_title('Average Transaction Amount by Day of Week')
axes[1].set_ylabel('Average Amount ($)')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(dow_vol_avg.values):
    axes[1].text(i, v + 5, f'${v:.0f}', ha='center', fontsize=7)

plt.tight_layout()
save_fig(fig, '11_day_of_week.png')


# ===================================================================
# 8. REPORTS
# ===================================================================
print('\n--- Generating Reports ---')

total_volume = df['Amount'].sum()
avg_amount = df['Amount'].mean()
median_amount = df['Amount'].median()
std_amount = df['Amount'].std()
busiest_month = int(monthly_vol.idxmax())
busiest_month_vol = monthly_vol.max()
top_cat_name = cat_vol.index[0]
top_cat_vol_val = cat_vol.iloc[0]
top_payment_name = pm_counts.index[0]
top_ttype_name = tt_vol.index[0]
weekday_avg = df[df['DayOfWeek'] < 5]['Amount'].mean()
weekend_avg = df[df['DayOfWeek'] >= 5]['Amount'].mean()

# ---- English short ----
en_short = f"""FINANCIAL ACCOUNTING REPORT \u2014 SHORT SUMMARY
============================================
Period:         {df['Date'].min().date()} to {df['Date'].max().date()}
Transactions:   {n_total:,}
Total Volume:   ${total_volume:,.0f}
Avg Amount:     ${avg_amount:.2f}
Median Amount:  ${median_amount:.2f}
Std Deviation:  ${std_amount:.2f}

Top Category:      {top_cat_name} (${top_cat_vol_val:,.0f})
Top Payment:       {top_payment_name} ({pm_counts.iloc[0]:,} txns)
Top Type:          {top_ttype_name}
Busiest Month:     Month {busiest_month} (${busiest_month_vol:,.0f})

Weekday Avg:   ${weekday_avg:.2f}
Weekend Avg:   ${weekend_avg:.2f}
Ratio:         {weekend_avg/weekday_avg:.2f}

Anomalies Flagged:   {n_flagged:,} ({n_flagged/n_total*100:.1f}%)
Top 2 Categories:    {top2_share:.1f}% of total volume
Recurring Txns:      {rec_txns:,} ({rec_txns/n_total*100:.1f}%)
"""

# ---- English full ----
en_full = f"""FINANCIAL ACCOUNTING REPORT \u2014 FULL ANALYSIS
============================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

1. DATA OVERVIEW
   Dataset: {n_total:,} transactions
   Period: {df['Date'].min().date()} to {df['Date'].max().date()}
   Accounts: {df['Account'].nunique()} unique
   Categories: {df['Category'].nunique()} unique
   Transaction Types: {df['Transaction_Type'].nunique()}
   Payment Methods: {df['Payment_Method'].nunique()}
   Customers/Vendors: {df['Customer_Vendor'].nunique()}
   Missing Values: {df.isnull().sum().sum()}

2. TRANSACTION STATISTICS
   Total Volume: ${total_volume:,.0f}
   Average Amount: ${avg_amount:.2f}
   Median Amount: ${median_amount:.2f}
   Standard Deviation: ${std_amount:.2f}
   Minimum: ${df['Amount'].min():.2f}
   Maximum: ${df['Amount'].max():.2f}
   Q1: ${Q1:.2f}  |  Q3: ${Q3:.2f}  |  IQR: ${IQR:.2f}

   Distribution by Category:
"""

for cat_name, pct_val in cat_share.items():
    en_full += f"     {cat_name}: ${cat_vol[cat_name]:,.0f} ({pct_val:.1f}%)\n"

en_full += f"""
3. MONTHLY ANALYSIS
   Month-over-Month Growth:
"""
for m in range(2, 13):
    en_full += f"     Month {m-1} \u2192 {m}: {mom_growth[m]:+.2f}%\n"

en_full += f"""
   Busiest Month: Month {busiest_month} (${busiest_month_vol:,.0f})

4. DAY-OF-WEEK ANALYSIS
"""
for day_name in dow_order:
    en_full += (f"     {day_name}: {dow_vol_cnt[day_name]:,} txns, "
                f"${dow_vol_sum[day_name]:,.0f} total, ${dow_vol_avg[day_name]:.2f} avg\n")

en_full += f"""
   ANOVA (Amount ~ DayOfWeek): F={f_stat:.2f}, p={p_anova:.6f}
   Interpretation: {'Significant differences across days' if p_anova < 0.05 else 'No significant differences'}

5. CASH FLOW & TIME SERIES
   Daily Net Cash Flow: Mean=${daily_net.mean():,.0f}, Std=${daily_net.std():,.0f}
   Days Net Inflow: {(daily_net > 0).sum()}  |  Days Net Outflow: {(daily_net < 0).sum()}

   CUSUM Breakpoints: {n_cp} detected
"""

if _HAS_DECOMPOSE:
    en_full += f"""
   Seasonal Decomposition (7-day period):
     Trend component:  {trend_var/total_var*100:.1f}% of variance
     Seasonal:         {seasonal_var/total_var*100:.1f}%
     Residual:         {resid_var/total_var*100:.1f}%
"""

en_full += f"""
   30-Day Forecast: ${forecast_avg:,.0f} average daily volume
   Projected Monthly Volume: ${forecast_avg * 30:,.0f}

6. PATTERN ANALYSIS
   Pareto (Customers): Top 20% = {pareto_pct:.1f}% of total volume
   Category Concentration: Top 1 = {top1_cat_name} ({top1_cat_pct:.1f}%)

   Payment Method Preference by Category:
"""
for cat_name in pm_cat.index:
    entry = ' | '.join([f"{col}: {pm_cat.loc[cat_name, col]*100:.1f}%" for col in pm_cat.columns])
    en_full += f"     {cat_name}: {entry}\n"

en_full += f"""
   Transaction Patterns:
     Recurring: {rec_txns:,} ({rec_txns/n_total*100:.1f}%)
     One-Time:  {onetime_txns:,} ({onetime_txns/n_total*100:.1f}%)

7. ANOMALY DETECTION
   IQR Method:      {len(outliers_iqr):,} ({len(outliers_iqr)/n_total*100:.2f}%)
   Z-Score (|z|>3): {n_high_val:,} ({n_high_val/n_total*100:.2f}%)
   Day-of-Week:     {df['Is_DOW_Anomaly'].sum():,} transactions
   Total Flagged:   {n_flagged:,} ({n_flagged/n_total*100:.2f}%)

   Outliers by Category:
"""
if len(cat_outliers) > 0:
    for cat_name, row in cat_outliers.iterrows():
        en_full += f"     {cat_name}: {int(row['count'])} txns, ${row['total_value']:,.0f} total, avg ${row['avg_amount']:.0f}\n"
else:
    en_full += "     No significant outliers detected\n"

en_full += f"""
8. CHARTS GENERATED
"""
for cf in sorted([f for f in os.listdir(CHARTS_DIR) if f.endswith('.png')]):
    en_full += f"     charts/{cf}\n"


# ---- Arabic short ----
ar_short = f"""\u062a\u0642\u0631\u064a\u0631 \u0627\u0644\u0645\u062d\u0627\u0633\u0628\u0629 \u0627\u0644\u0645\u0627\u0644\u064a\u0629 \u2014 \u0645\u0644\u062e\u0635 \u0633\u0631\u064a\u0639
============================================
\u0627\u0644\u0641\u062a\u0631\u0629: {df['Date'].min().date()} \u0625\u0644\u0649 {df['Date'].max().date()}
\u0639\u062f\u062f \u0627\u0644\u0645\u0639\u0627\u0645\u0644\u0627\u062a: {n_total:,}
\u0625\u062c\u0645\u0627\u0644\u064a \u0627\u0644\u062d\u062c\u0645: ${total_volume:,.0f}
\u0645\u062a\u0648\u0633\u0637 \u0627\u0644\u0645\u0628\u0644\u063a: ${avg_amount:.2f}
\u0627\u0644\u0648\u0633\u064a\u0637: ${median_amount:.2f}

\u0623\u0643\u062b\u0631 \u0641\u0626\u0629: {top_cat_name} (${top_cat_vol_val:,.0f})
\u0623\u0643\u062b\u0631 \u0648\u0633\u064a\u0644\u0629 \u062f\u0641\u0639: {top_payment_name} ({pm_counts.iloc[0]:,} \u0645\u0639\u0627\u0645\u0644\u0629)
\u0623\u0643\u062b\u0631 \x0634\u0647\u0631 \u0627\u0632\u062f\u062d\u0627\u0645\u0627\u064b: \x0634\u0647\u0631 {busiest_month} (${busiest_month_vol:,.0f})

\u0645\u062a\u0648\u0633\u0637 \u0623\u064a\u0627\u0645 \u0627\u0644\u0623\u0633\u0628\u0648\u0639: ${weekday_avg:.2f}
\u0645\u062a\u0648\u0633\u0637 \u0639\u0637\u0644\u0629 \u0646\u0647\u0627\u064a\u0629 \u0627\u0644\u0623\u0633\u0628\u0648\u0639: ${weekend_avg:.2f}
\u0627\u0644\u0646\u0633\u0628\u0629: {weekend_avg/weekday_avg:.2f}

\u0627\u0644\u062d\u0627\u0644\u0627\u062a \u0627\u0644\u0634\u0627\u0630\u0629: {n_flagged:,} ({n_flagged/n_total*100:.1f}%)
\u0623\u0639\u0644\u0649 \x0641\u0626\u062a\u064a\u0646: {top2_share:.1f}% \u0645\u0646 \u0625\u062c\u0645\u0627\u0644\u064a \u0627\u0644\u062d\u062c\u0645
\u0627\u0644\u0645\u0639\u0627\u0645\u0644\u0627\u062a \u0627\u0644\u0645\u062a\u0643\u0631\u0631\u0629: {rec_txns:,} ({rec_txns/n_total*100:.1f}%)
"""

# ---- Arabic full ----
ar_full = f"""\u062a\u0642\u0631\u064a\u0631 \u0627\u0644\u0645\u062d\u0627\u0633\u0628\u0629 \u0627\u0644\u0645\u0627\u0644\u064a\u0629 \u2014 \u062a\u062d\u0644\u064a\u0644 \u0643\u0627\u0645\u0644
============================================
\u062a\u0627\u0631\u064a\u062e \u0627\u0644\u062a\u0648\u0644\u064a\u062f: {datetime.now().strftime('%Y-%m-%d %H:%M')}

1. \u0646\u0638\u0631\u0629 \u0639\u0627\u0645\u0629
   \u0639\u062f\u062f \u0627\u0644\u0645\u0639\u0627\u0645\u0644\u0627\u062a: {n_total:,}
   \u0627\u0644\u0641\u062a\u0631\u0629: {df['Date'].min().date()} \u0625\u0644\u0649 {df['Date'].max().date()}
   \u0627\u0644\u062d\u0633\u0627\u0628\u0627\u062a: {df['Account'].nunique()} \u0641\u0631\u064a\u062f\u0629
   \u0627\u0644\u0641\u0626\u0627\u062a: {df['Category'].nunique()} \u0641\u0631\u064a\u062f\u0629
   \u0623\u0646\u0648\u0627\u0639 \u0627\u0644\u0645\u0639\u0627\u0645\u0644\u0627\u062a: {df['Transaction_Type'].nunique()}
   \u0648\u0633\u0627\u0626\u0644 \u0627\u0644\u062f\u0641\u0639: {df['Payment_Method'].nunique()}
   \u0627\u0644\u0639\u0645\u0644\u0627\u0621/\u0627\u0644\u0645\u0648\u0631\u062f\u064a\u0646: {df['Customer_Vendor'].nunique()}
   \u0627\u0644\u0642\u064a\u0645 \u0627\u0644\u0645\u0641\u0642\u0648\u062f\u0629: {df.isnull().sum().sum()}

2. \u0625\u062d\u0635\u0627\u0626\u064a\u0627\u062a \u0627\u0644\u0645\u0639\u0627\u0645\u0644\u0627\u062a
   \u0625\u062c\u0645\u0627\u0644\u064a \u0627\u0644\u062d\u062c\u0645: ${total_volume:,.0f}
   \u0645\u062a\u0648\u0633\u0637 \u0627\u0644\u0645\u0628\u0644\u063a: ${avg_amount:.2f}
   \u0627\u0644\u0648\u0633\u064a\u0637: ${median_amount:.2f}
   \u0627\u0646\u062d\u0631\u0627\u0641 \u0645\u0639\u064a\u0627\u0631\u064a: ${std_amount:.2f}
   \u0627\u0644\u062d\u062f \u0627\u0644\u0623\u062f\u0646\u0649: ${df['Amount'].min():.2f}
   \u0627\u0644\u062d\u062f \u0627\u0644\u0623\u0642\u0635\u0649: ${df['Amount'].max():.2f}

   \u0627\u0644\u062a\u0648\u0632\u064a\u0639 \u062d\u0633\u0628 \u0627\u0644\u0641\u0626\u0629:
"""
for cat_name, pct_val in cat_share.items():
    ar_full += f"     {cat_name}: ${cat_vol[cat_name]:,.0f} ({pct_val:.1f}%)\n"

ar_full += f"""
3. \u0627\u0644\u062a\u062d\u0644\u064a\u0644 \u0627\u0644\u0634\u0647\u0631\u064a
   \u0646\u0645\u0648 \x0634\u0647\u0631 \u0639\u0644\u0649 \x0634\u0647\u0631:
"""
for m in range(2, 13):
    ar_full += f"     \x0634\u0647\u0631 {m-1} \u2192 {m}: {mom_growth[m]:+.2f}%\n"

ar_full += f"""
   \u0623\u0643\u062b\u0631 \x0634\u0647\u0631 \u0627\u0632\u062f\u062d\u0627\u0645\u0627\u064b: \x0634\u0647\u0631 {busiest_month} (${busiest_month_vol:,.0f})

4. \u062a\u062d\u0644\u064a\u0644 \u0623\u064a\u0627\u0645 \u0627\u0644\u0623\u0633\u0628\u0648\u0639
"""
for day_name in dow_order:
    ar_full += (f"     {day_name}: {dow_vol_cnt[day_name]:,} \u0645\u0639\u0627\u0645\u0644\u0629, "
                f"${dow_vol_sum[day_name]:,.0f} \u0625\u062c\u0645\u0627\u0644\u064a, ${dow_vol_avg[day_name]:.2f} \u0645\u062a\u0648\u0633\u0637\n")

ar_full += f"""
   ANOVA: F={f_stat:.2f}, p={p_anova:.6f}
   \u0627\u0644\u062a\u0641\u0633\u064a\u0631: {'\u0641\u0631\u0648\u0642 \u0645\u0639\u0646\u0648\u064a\u0629 \u0628\u064a\u0646 \u0627\u0644\u0623\u064a\u0627\u0645' if p_anova < 0.05 else '\u0644\u0627 \u062a\u0648\u062c\u062f \u0641\u0631\u0648\u0642 \u0645\u0639\u0646\u0648\u064a\u0629'}

5. \u062a\u062d\u0644\u064a\u0644 \u0627\u0644\u062a\u062f\u0641\u0642 \u0627\u0644\u0646\u0642\u062f\u064a
   \u0645\u062a\u0648\u0633\u0637 \u0635\u0627\u0641\u064a \u0627\u0644\u062a\u062f\u0641\u0642 \u0627\u0644\u064a\u0648\u0645\u064a: ${daily_net.mean():,.0f}
   \u0623\u064a\u0627\u0645 \u0627\u0644\u062a\u062f\u0641\u0642 \u0627\u0644\u0625\u064a\u062c\u0627\u0628\u064a: {(daily_net > 0).sum()}
   \u0623\u064a\u0627\u0645 \u0627\u0644\u062a\u062f\u0641\u0642 \u0627\u0644\u0633\u0644\u0628\u064a: {(daily_net < 0).sum()}

   \u0646\u0642\u0627\u0637 \u0627\u0644\u062a\u063a\u064a\u064a\u0631 (CUSUM): {n_cp} \u0646\u0642\u0637\u0629
"""

if _HAS_DECOMPOSE:
    ar_full += f"""
   \u0627\u0644\u062a\u062d\u0644\u064a\u0644 \u0627\u0644\u0645\u0648\u0633\u0645\u064a:
     \u0627\u0644\u0627\u062a\u062c\u0627\u0647: {trend_var/total_var*100:.1f}% \u0645\u0646 \u0627\u0644\u062a\u0628\u0627\u064a\u0646
     \u0627\u0644\u0645\u0648\u0633\u0645\u064a\u0629: {seasonal_var/total_var*100:.1f}% \u0645\u0646 \u0627\u0644\u062a\u0628\u0627\u064a\u0646
     \u0627\u0644\u0645\u062a\u0628\u0642\u064a: {resid_var/total_var*100:.1f}% \u0645\u0646 \u0627\u0644\u062a\u0628\u0627\u064a\u0646
"""

ar_full += f"""
   \u0627\u0644\u062a\u0646\u0628\u0624 (\u0663\u0660 \u064a\u0648\u0645\u0627\u064b):
     \u0645\u062a\u0648\u0633\u0637 \u0627\u0644\u062d\u062c\u0645 \u0627\u0644\u064a\u0648\u0645\u064a \u0627\u0644\u0645\u062a\u0648\u0642\u0639: ${forecast_avg:,.0f}
     \u0627\u0644\u062d\u062c\u0645 \u0627\u0644\u0634\u0647\u0631\u064a \u0627\u0644\u0645\u062a\u0648\u0642\u0639: ${forecast_avg * 30:,.0f}

6. \u062a\u062d\u0644\u064a\u0644 \u0627\u0644\u0623\u0646\u0645\u0627\u0637
   \u0628\u0627\u0631\u064a\u062a\u0648 (\u0627\u0644\u0639\u0645\u0644\u0627\u0621): \u0623\u0639\u0644\u0649 20% = {pareto_pct:.1f}% \u0645\u0646 \u0627\u0644\u062d\u062c\u0645
   \u062a\u0631\u0643\u064a\u0632 \u0627\u0644\u0641\u0626\u0627\u062a: \u0623\u0639\u0644\u0649 \u0641\u0626\u0629 = {top1_cat_name} ({top1_cat_pct:.1f}%)

   \u062a\u0641\u0636\u064a\u0644\u0627\u062a \u0648\u0633\u0627\u0626\u0644 \u0627\u0644\u062f\u0641\u0639 \u062d\u0633\u0628 \u0627\u0644\u0641\u0626\u0629:
"""
for cat_name in pm_cat.index:
    entry = ' | '.join([f"{col}: {pm_cat.loc[cat_name, col]*100:.1f}%" for col in pm_cat.columns])
    ar_full += f"     {cat_name}: {entry}\n"

ar_full += f"""
   \u0623\u0646\u0645\u0627\u0637 \u0627\u0644\u0645\u0639\u0627\u0645\u0644\u0627\u062a:
     \u0645\u062a\u0643\u0631\u0631\u0629: {rec_txns:,} ({rec_txns/n_total*100:.1f}%)
     \u0644\u0645\u0631\u0629 \u0648\u0627\u062d\u062f\u0629: {onetime_txns:,} ({onetime_txns/n_total*100:.1f}%)

7. \u0643\u0634\u0641 \u0627\u0644\u062d\u0627\u0644\u0627\u062a \u0627\u0644\u0634\u0627\u0630\u0629
   \u0637\u0631\u064a\u0642\u0629 IQR: {len(outliers_iqr):,} ({len(outliers_iqr)/n_total*100:.2f}%)
   Z-Score (|z|>3): {n_high_val:,} ({n_high_val/n_total*100:.2f}%)
   \u0634\u0630\u0648\u0630 \u0623\u064a\u0627\u0645 \u0627\u0644\u0623\u0633\u0628\u0648\u0639: {df['Is_DOW_Anomaly'].sum():,}
   \u0625\u062c\u0645\u0627\u0644\u064a \u0627\u0644\u062d\u0627\u0644\u0627\u062a \u0627\u0644\u0645\u0645\u064a\u0632\u0629: {n_flagged:,} ({n_flagged/n_total*100:.2f}%)

   \u0627\u0644\u062d\u0627\u0644\u0627\u062a \u0627\u0644\u0634\u0627\u0630\u0629 \u062d\u0633\u0628 \u0627\u0644\u0641\u0626\u0629:
"""
if len(cat_outliers) > 0:
    for cat_name, row in cat_outliers.iterrows():
        ar_full += f"     {cat_name}: {int(row['count'])} \u0645\u0639\u0627\u0645\u0644\u0629, ${row['total_value']:,.0f} \u0625\u062c\u0645\u0627\u0644\u064a\n"
else:
    ar_full += "     \u0644\u0645 \u064a\u062a\u0645 \u0627\u0643\u062a\u0634\u0627\u0641 \u062d\u0627\u0644\u0627\u062a \u0634\u0627\u0630\u0629 \u0643\u0628\u064a\u0631\u0629\n"

ar_full += f"""
8. \u0627\u0644\u0631\u0633\u0648\u0645 \u0627\u0644\u0628\u064a\u0627\u0646\u064a\u0629
"""
for cf in sorted([f for f in os.listdir(CHARTS_DIR) if f.endswith('.png')]):
    ar_full += f"     charts/{cf}\n"


# Write all four reports
report_dir = BASE_DIR
with open(os.path.join(report_dir, 'Financial_Report_EN_short.txt'), 'w', encoding='utf-8') as f:
    f.write(en_short.strip() + '\n')
    try:
        print(en_short)
    except UnicodeEncodeError:
        print(f"[Report: en_short saved to file]")
print('  Wrote: Financial_Report_EN_short.txt')

with open(os.path.join(report_dir, 'Financial_Report_EN_full.txt'), 'w', encoding='utf-8') as f:
    f.write(en_full.strip() + '\n')
    try:
        print(en_full)
    except UnicodeEncodeError:
        print(f"[Report: en_full saved to file]")
print('  Wrote: Financial_Report_EN_full.txt')

with open(os.path.join(report_dir, 'Financial_Report_AR_short.txt'), 'w', encoding='utf-8') as f:
    f.write(ar_short.strip() + '\n')
    try:
        print(ar_short)
    except UnicodeEncodeError:
        print(f"[Report: ar_short saved to file]")
print('  Wrote: Financial_Report_AR_short.txt')

with open(os.path.join(report_dir, 'Financial_Report_AR_full.txt'), 'w', encoding='utf-8') as f:
    f.write(ar_full.strip() + '\n')
    try:
        print(ar_full)
    except UnicodeEncodeError:
        print(f"[Report: ar_full saved to file]")
print('  Wrote: Financial_Report_AR_full.txt')

plt.close('all')

print(f'\n{"=" * 60}')
print('ANALYSIS COMPLETE')
print(f'Charts: {CHARTS_DIR}')
print(f'Reports: {BASE_DIR}')
print(f'{"=" * 60}')
