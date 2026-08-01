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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

OUT = r"D:\download\protfolio\projects\v3_output\project6_Forex"
os.makedirs(OUT, exist_ok=True)

logger.info("="*85)

df = pd.read_csv("D:\\download\\protfolio\\archive (5)\\Foreign_Exchange_Rates.csv")
df['Date'] = pd.to_datetime(df.iloc[:,1], errors='coerce')
df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)

rate_cols = [c for c in df.columns if c not in ['Date', df.columns[0], df.columns[1]]]
logger.info(f"\n  Loaded {len(df)} days for {len(rate_cols)} currency pairs.")

for c in rate_cols:
    df[c] = pd.to_numeric(df[c].astype(str).str.replace('ND','NaN').str.replace('N/A','NaN').str.replace(',',''), errors='coerce')
    df[c] = df[c].interpolate()

primary = 'EURO AREA - EURO/US$' if 'EURO AREA - EURO/US$' in rate_cols else rate_cols[2]
logger.info(f"  Primary pair for deep analysis: {primary}")

# ═══════════════════════════════════════════
# 1. ADVANCED FEATURE ENGINEERING (MACD, RSI, BB)
# ═══════════════════════════════════════════
logger.info("▔"*60)
df['Return'] = df[primary].pct_change()
df['LogReturn'] = np.log(df[primary] / df[primary].shift(1))
df['Volatility'] = df['Return'].rolling(21).std() * np.sqrt(252)
df['MA20'] = df[primary].rolling(20).mean()
df['MA50'] = df[primary].rolling(50).mean()
df['Upper_BB'] = df['MA20'] + (df['Return'].rolling(20).std() * np.sqrt(252) * 2)
df['Lower_BB'] = df['MA20'] - (df['Return'].rolling(20).std() * np.sqrt(252) * 2)

# Calculate RSI
delta = df[primary].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Calculate MACD
ema12 = df[primary].ewm(span=12, adjust=False).mean()
ema26 = df[primary].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

logger.info("  [OK] Successfully calculated RSI, MACD, and Bollinger Bands.")

# ═══════════════════════════════════════════
# 2. TIME SERIES FORECASTING (ARIMA)
# ═══════════════════════════════════════════
try:
    train_data = df[primary].dropna().values
    model = ARIMA(train_data, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    logger.info("  [OK] ARIMA(5,1,0) Model successfully fit. 30-day forecast generated.")
except Exception as e:
    logger.info(f"  ARIMA Error: {e}")
    forecast = []

# ═══════════════════════════════════════════
# 3. INTERACTIVE PLOTLY FINANCIAL DASHBOARD
# ═══════════════════════════════════════════
fig_html = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                         vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
                         subplot_titles=(f"{primary} Price with Bollinger Bands", "MACD", "RSI"))

# Price and BB
fig_html.add_trace(go.Scatter(x=df['Date'], y=df[primary], mode='lines', name='Price', line=dict(color='#3498db')), row=1, col=1)
fig_html.add_trace(go.Scatter(x=df['Date'], y=df['Upper_BB'], mode='lines', name='Upper BB', line=dict(color='rgba(255,0,0,0.2)'), showlegend=False), row=1, col=1)
fig_html.add_trace(go.Scatter(x=df['Date'], y=df['Lower_BB'], fill='tonexty', mode='lines', name='Bollinger Band', line=dict(color='rgba(255,0,0,0.2)'), fillcolor='rgba(231,76,60,0.1)'), row=1, col=1)

# Add Forecast
if len(forecast) > 0:
    future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30)
    fig_html.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='ARIMA 30-day Forecast', line=dict(color='#f39c12', dash='dash')), row=1, col=1)

# MACD
fig_html.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD', line=dict(color='#e74c3c')), row=2, col=1)
fig_html.add_trace(go.Scatter(x=df['Date'], y=df['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='#2ecc71')), row=2, col=1)

# RSI
fig_html.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI', line=dict(color='#9b59b6')), row=3, col=1)
fig_html.add_hline(y=70, line_dash="dot", row=3, col=1, line_color="red")
fig_html.add_hline(y=30, line_dash="dot", row=3, col=1, line_color="green")

fig_html.update_layout(height=900, title_text=f"Professional Forex Terminal: {primary}", template='plotly_dark')
pio.write_html(fig_html, file=os.path.join(OUT, 'Interactive_Dashboard.html'), auto_open=False)
logger.info("  [OK] Interactive_Dashboard.html generated.")

# ═══════════════════════════════════════════
# 4. PROFESSIONAL STATIC MASTER DASHBOARD
# ═══════════════════════════════════════════
sns.set_theme(style="darkgrid", context="talk", palette="deep")
fig = plt.figure(figsize=(24, 18))
fig.suptitle(f'Forex Advanced Analytics — {primary}', fontsize=26, fontweight='black', y=0.98, color='#2c3e50')
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

# 1. Price + Forecast
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df['Date'], df[primary], lw=2, color='#2c3e50', label='Historical Price')
if len(forecast) > 0:
    ax1.plot(future_dates, forecast, color='#e67e22', lw=3, linestyle='--', label='ARIMA 30D Forecast')
ax1.set_title(f'{primary} Exchange Rate Forecasting', fontweight='bold')
ax1.legend()

# 2. Volatility Regimes
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(df['Date'], df['Volatility']*100, color='#e74c3c', lw=1.5)
ax2.axhline(df['Volatility'].median()*100, color='k', linestyle='--', alpha=0.5)
ax2.set_title('Annualized Volatility (21D Rolling)', fontweight='bold')
ax2.set_ylabel('%')

# 3. RSI Distribution
ax3 = fig.add_subplot(gs[1, 1])
sns.histplot(df['RSI'].dropna(), bins=40, color='#9b59b6', kde=True, ax=ax3)
ax3.axvline(30, color='g', linestyle='--', lw=2, label='Oversold')
ax3.axvline(70, color='r', linestyle='--', lw=2, label='Overbought')
ax3.set_title('RSI Distribution', fontweight='bold')
ax3.legend()

# 4. Cross-Currency Correlation
ax4 = fig.add_subplot(gs[2, 0])
top_pairs = rate_cols[:8]
corr = df[top_pairs].corr()
sns.heatmap(corr, cmap='RdYlGn', annot=True, fmt='.2f', cbar=False, ax=ax4, annot_kws={"size": 12})
ax4.set_xticklabels([c[:12] for c in top_pairs], rotation=45, ha='right')
ax4.set_yticklabels([c[:12] for c in top_pairs], rotation=0)
ax4.set_title('Currency Correlation Matrix', fontweight='bold')

# 5. Summary Text
ax5 = fig.add_subplot(gs[2, 1]); ax5.axis('off')
summary = f"""
KEY FOREX INSIGHTS:
───────────────────
- Latest Rate: {df[primary].iloc[-1]:.4f}
- ARIMA Forecast End (30D): {forecast[-1] if len(forecast)>0 else 'N/A':.4f}
- Current RSI: {df['RSI'].iloc[-1]:.2f} 
  (>70 Overbought, <30 Oversold)
- MACD Signal: {'BUY' if df['MACD'].iloc[-1] > df['Signal_Line'].iloc[-1] else 'SELL'}
- Current Volatility: {df['Volatility'].iloc[-1]*100:.1f}%

The advanced dashboard captures technical regimes
and provides a 30-day statistical projection.
"""
ax5.text(0.1, 0.9, summary, transform=ax5.transAxes, fontsize=16, fontfamily='monospace',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUT, "Advanced_Dashboard.png"), dpi=300, bbox_inches='tight')
logger.info("  [OK] Advanced_Dashboard.png generated.")

# Write report.txt
report_path = os.path.join(OUT, "report.txt")
with open(report_path, 'w', encoding='utf-8') as rf:
    rf.write("PROJECT 6: FOREIGN EXCHANGE RATE FORECASTING REPORT\n")
    rf.write("===================================================\n")
    rf.write(f"Analyzed Currency Pair: {primary}\n")
    rf.write(f"Total historical days: {len(df):,}\n")
    rf.write(f"Date Range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")
    rf.write(f"Latest Rate: {df[primary].iloc[-1]:.4f}\n")
    rf.write(f"Historical Mean: {df[primary].mean():.4f} | Std: {df[primary].std():.4f}\n")
    rf.write(f"Historical Min: {df[primary].min():.4f} | Max: {df[primary].max():.4f}\n")
    rf.write(f"Current Volatility (rolling 21D annualized): {df['Volatility'].iloc[-1]*100:.2f}%\n")
    rf.write(f"Current RSI: {df['RSI'].iloc[-1]:.2f} (Neutral: 30-70)\n")
    
    if len(forecast) > 0:
        rf.write(f"\nARIMA(5,1,0) 30-Day Forecast:\n")
        rf.write(f"  Start Date: {future_dates[0].date()} -> Rate: {forecast[0]:.4f}\n")
        rf.write(f"  End Date:   {future_dates[-1].date()} -> Rate: {forecast[-1]:.4f}\n")
        rf.write("  Full Forecast Values:\n")
        for d, val in zip(future_dates, forecast):
            rf.write(f"    {d.date()}: {val:.4f}\n")

logger.info(f"  [OK] report.txt generated at {report_path}")
logger.info("\n✅ PROJECT 6 COMPLETE.")
