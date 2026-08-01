"""
Data Analysis and Machine Learning Pipeline
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
from scipy.stats import beta as beta_dist
warnings.filterwarnings('ignore')

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.ensemble import IsolationForest
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

OUT = r"D:\download\protfolio\projects\v3_output\project4_Financial"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("D:\\download\\protfolio\\archive (3)\\financial_accounting.csv", parse_dates=['Date'])
logger.info(f"\n  Transactions: {len(df):,}")
logger.info(f"  Period: {df['Date'].min().date()} to {df['Date'].max().date()}")

# Feature engineering
df['Month'] = df['Date'].dt.month; df['MonthName'] = df['Date'].dt.month_name()
df['Weekday'] = df['Date'].dt.day_name(); df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['IsWeekend'] = df['Weekday'].isin(['Saturday','Sunday']).astype(int)
df['Amount'] = (df['Debit'] + df['Credit']) / 2

daily = df.groupby(df['Date'].dt.date).agg(Count=('Amount','count'), Total=('Amount','sum'), Avg=('Amount','mean')).reset_index()
daily['Date'] = pd.to_datetime(daily['Date'])

# ═══════════════════════════════════════════
# 1. ADVANCED ANOMALY DETECTION
# ═══════════════════════════════════════════
logger.info("▔"*60)
iso = IsolationForest(contamination=0.03, random_state=42)
daily['Anomaly'] = iso.fit_predict(daily[['Count','Total','Avg']])
logger.info(f"  Anomalous days flagged: {(daily.Anomaly==-1).sum()} / {len(daily)}")

# ═══════════════════════════════════════════
# 2. TIME SERIES FORECASTING
# ═══════════════════════════════════════════
forecast_df = None
if STATS_AVAILABLE:
    try:
        ts_data = daily.set_index('Date')['Total'].resample('W').sum()
        model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=52)
        fit_model = model.fit()
        forecast = fit_model.forecast(12)  # 12 weeks
        
        forecast_df = pd.DataFrame({
            'Date': forecast.index,
            'Forecast': forecast.values
        })
        logger.info("  [OK] Successfully forecasted 12 weeks ahead using Holt-Winters Exponential Smoothing.")
    except Exception as e:
        logger.info(f"  Forecast Error: {e}")
        STATS_AVAILABLE = False

# ═══════════════════════════════════════════
# 3. INTERACTIVE PLOTLY DASHBOARD
# ═══════════════════════════════════════════

fig_html = make_subplots(rows=2, cols=1, subplot_titles=(
    "Daily Transaction Volume with Anomalies", "Weekly Transaction Forecast (Holt-Winters)"
), vertical_spacing=0.15)

# 1. Daily Volume + Anomalies
fig_html.add_trace(go.Scatter(x=daily['Date'], y=daily['Total'], mode='lines', name='Daily Total', line=dict(color='#3498db')), row=1, col=1)
anom = daily[daily.Anomaly == -1]
fig_html.add_trace(go.Scatter(x=anom['Date'], y=anom['Total'], mode='markers', name='Anomaly', 
                              marker=dict(color='#e74c3c', size=10, symbol='x')), row=1, col=1)

# 2. Forecast
if STATS_AVAILABLE and forecast_df is not None:
    fig_html.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines', name='Historical (Weekly)', line=dict(color='#2ecc71')), row=2, col=1)
    fig_html.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines', name='Forecast (12 Weeks)', 
                                  line=dict(color='#f39c12', dash='dash')), row=2, col=1)

fig_html.update_layout(height=800, title_text="Financial Transactions Interactive Dashboard", template='plotly_dark')
pio.write_html(fig_html, file=os.path.join(OUT, 'Interactive_Dashboard.html'), auto_open=False)
logger.info("  [OK] Interactive_Dashboard.html generated.")

# ═══════════════════════════════════════════
# 4. PROFESSIONAL STATIC DASHBOARD
# ═══════════════════════════════════════════
sns.set_theme(style="darkgrid", context="talk", palette="deep")
fig = plt.figure(figsize=(24, 18))
fig.suptitle('Financial Accounting Analytics — Wow Edition', fontsize=26, fontweight='black', y=0.98, color='#2c3e50')
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

# 1. Time Series + Forecast
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(ts_data.index, ts_data.values, color='#3498db', lw=2, label='Historical')
if STATS_AVAILABLE and forecast_df is not None:
    ax1.plot(forecast_df['Date'], forecast_df['Forecast'], color='#e67e22', lw=3, linestyle='--', label='Forecast')
ax1.set_title('Weekly Transaction Volume & Forecast', fontweight='bold')
ax1.legend()

# 2. Daily Anomalies
ax2 = fig.add_subplot(gs[0, 2:])
sns.scatterplot(data=daily, x='Date', y='Total', hue='Anomaly', palette={1:'#2ecc71', -1:'#e74c3c'}, alpha=0.8, s=60, ax=ax2)
ax2.plot(daily['Date'], daily['Total'], color='gray', alpha=0.3, lw=1)
ax2.set_title('Daily Volume Anomalies (Isolation Forest)', fontweight='bold')

# 3. Monthly Category Heatmap
ax3 = fig.add_subplot(gs[1, 0:2])
monthly_cat = df.pivot_table(values='Amount', index='MonthName', columns='Category', aggfunc='sum').fillna(0)
monthly_cat = monthly_cat.reindex(['January','February','March','April','May','June','July','August','September','October','November','December'])
sns.heatmap(monthly_cat, cmap='YlGnBu', ax=ax3, annot=True, fmt='.0f', cbar_kws={'shrink':0.8})
ax3.set_title('Monthly Volume by Category', fontweight='bold')

# 4. Bayesian: Payment Method
ax4 = fig.add_subplot(gs[1, 2:])
xx = np.linspace(0, 0.4, 500)
for pm in df['Payment_Method'].unique():
    k = int((df['Payment_Method']==pm).sum())
    n = len(df)
    ax4.plot(xx, beta_dist.pdf(xx, 1+k, 1+n-k), lw=3, label=pm)
ax4.set_title('Bayesian Posterior: Payment Method Probability', fontweight='bold'); ax4.legend()

# 5. Account Balances
ax5 = fig.add_subplot(gs[2, 0])
sns.barplot(x=df['Account'].value_counts().index, y=df['Account'].value_counts().values, palette='Set2', ax=ax5, edgecolor='k')
ax5.set_title('Account Entry Count', fontweight='bold')
ax5.tick_params(axis='x', rotation=45)

# 6. Amount Distribution by Payment Method (Violin)
ax6 = fig.add_subplot(gs[2, 1:3])
sns.violinplot(data=df, x='Payment_Method', y='Amount', palette='muted', inner='quartile', ax=ax6)
ax6.set_title('Transaction Value Distribution by Method', fontweight='bold')

# 7. Summary
ax7 = fig.add_subplot(gs[2, 3]); ax7.axis('off')
summary = f"""
KEY FINANCIAL INSIGHTS:
───────────────────────
- Total Volume: ${df['Amount'].sum():,.0f}
- Avg Trans: ${df['Amount'].mean():.2f}
- Anomalies: {(daily.Anomaly==-1).sum()} Days
- Forecast 12W: 
  Trend captured.
  (See Dashboard)
"""
ax7.text(0.1, 0.9, summary, transform=ax7.transAxes, fontsize=16, fontfamily='monospace',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "Advanced_Dashboard.png"), dpi=300, bbox_inches='tight')
logger.info("  [OK] Advanced_Dashboard.png generated.")

# Write report.txt
report_path = os.path.join(OUT, "report.txt")
with open(report_path, 'w', encoding='utf-8') as rf:
    rf.write("PROJECT 4: FINANCIAL ACCOUNTING ANALYSIS REPORT\n")
    rf.write("===============================================\n")
    rf.write(f"Total Transactions: {len(df):,}\n")
    rf.write(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")
    rf.write(f"Total Volume: ${df['Amount'].sum():,.2f}\n")
    rf.write(f"Average Transaction Value: ${df['Amount'].mean():.2f}\n")
    rf.write(f"Anomalous Days Flagged: {(daily.Anomaly==-1).sum()} out of {len(daily)} days\n")
    if forecast_df is not None:
        rf.write("\n12-Week Forecast Summary (Weekly Volumes):\n")
        for idx, row in forecast_df.iterrows():
            rf.write(f"  Week {idx+1} ({row['Date'].date()}): ${row['Forecast']:,.2f}\n")

logger.info(f"  [OK] report.txt generated at {report_path}")
logger.info("\n✅ PROJECT 4 COMPLETE.")
