import pandas as pd, numpy as np
from scipy import stats
from scipy.stats import beta as beta_dist
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

OUT = "D:\\download\\protfolio\\projects\\output\\project2_Sentiment"
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT,"report.txt"),'w',encoding='utf-8')
def w(t=""): log.write(t+"\n"); print(t.encode("utf-8","replace").decode("utf-8","replace"))

w("="*80 + "\n  PROJECT 2: SENTIMENT ANALYSIS - TWITTER\n  ØªØ­Ù„ÙŠÙ„ Ø§Ù„Ù…Ø´Ø§Ø¹Ø± ÙÙŠ Ø§Ù„ØªØºØ±ÙŠØ¯Ø§Øª\n" + "="*80)

cols = ['sentiment','id','date','query','user','text']
w("\nLoading full dataset (sampling 200K balanced)...")
df = pd.read_csv("D:\\download\\protfolio\\archive (1)\\training.1600000.processed.noemoticon.csv",
                 names=cols, encoding='latin1')
df_neg = df[df.sentiment==0].sample(n=100000, random_state=42)
df_pos = df[df.sentiment==4].sample(n=100000, random_state=42)
df = pd.concat([df_neg, df_pos], ignore_index=True)
w(f"  Loaded {len(df):,} tweets (100K negative + 100K positive)")

# ----- 1. Frequency Statistics -----
w("\n" + "-"*60 + "\n  1. FREQUENCY STATISTICS / Ø§Ù„Ø¥Ø­ØµØ§Ø¡ Ø§Ù„ØªÙƒØ±Ø§Ø±ÙŠ\n" + "-"*60)
w(f"\nSentiment distribution:")
freq = df['sentiment'].value_counts()
pct = df['sentiment'].value_counts(normalize=True).mul(100).round(2)
for k in freq.index:
    label = 'Positive (4)' if k==4 else 'Negative (0)'
    w(f"  {label:15s}: {freq[k]:6d} ({pct[k]:.2f}%)")

df['year'] = pd.to_datetime(df['date'], format='mixed').dt.year
df['hour'] = pd.to_datetime(df['date'], format='mixed').dt.hour
w(f"\nYear range: {df['year'].min()} - {df['year'].max()}")
w(f"Unique users: {df['user'].nunique():,}")

df['has_url'] = df['text'].str.contains('http', na=False)
df['is_mention'] = df['text'].str.startswith('@')
df['tweet_len'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

w("\nTweet length stats:")
w(df['tweet_len'].describe().round(1).to_string())
w(f"\nWord count stats:")
w(df['word_count'].describe().round(1).to_string())

w(f"\nHas URL: {df['has_url'].mean()*100:.1f}%")
w(f"Is @mention: {df['is_mention'].mean()*100:.1f}%")

# ----- 2. A/B Testing -----
w("\n" + "-"*60 + "\n  2. A/B HYPOTHESIS TESTING\n" + "-"*60)

w("\nT-Test: Tweet length by Sentiment")
g0 = df[df.sentiment==0]['tweet_len'].dropna()
g4 = df[df.sentiment==4]['tweet_len'].dropna()
t,p = stats.ttest_ind(g0,g4)
w(f"  Negative: mean={g0.mean():.1f} | Positive: mean={g4.mean():.1f}")
w(f"  t={t:.3f}, p={p:.6f} {'*** SIGNIFICANT' if p<0.001 else 'ns'}")

w("\nT-Test: Word count by Sentiment")
g0w = df[df.sentiment==0]['word_count'].dropna()
g4w = df[df.sentiment==4]['word_count'].dropna()
t,p = stats.ttest_ind(g0w,g4w)
w(f"  Negative: mean={g0w.mean():.1f} | Positive: mean={g4w.mean():.1f}")
w(f"  t={t:.3f}, p={p:.6f} {'***' if p<0.001 else 'ns'}")

w("\nChi-Square: Has URL by Sentiment")
ct = pd.crosstab(df['has_url'],df['sentiment'])
chi2,p = stats.chi2_contingency(ct)[:2]
w(f"\n{ct}\n  chi2={chi2:.2f}, p={p:.6f} {'***' if p<0.001 else 'ns'}")

w("\nChi-Square: Is Mention by Sentiment")
ct = pd.crosstab(df['is_mention'],df['sentiment'])
chi2,p = stats.chi2_contingency(ct)[:2]
w(f"\n{ct}\n  chi2={chi2:.2f}, p={p:.6f} {'***' if p<0.001 else 'ns'}")

w("\nANOVA: Year Negative vs Positive")
g0y = df[df.sentiment==0]['year'].dropna()
g4y = df[df.sentiment==4]['year'].dropna()
f,p = stats.f_oneway(g0y,g4y)
w(f"  Negative year mean: {g0y.mean():.1f} | Positive: {g4y.mean():.1f}")
w(f"  F={f:.2f}, p={p:.6f} {'***' if p<0.001 else 'ns'}")

# ----- 3. Bayesian Analysis -----
w("\n" + "-"*60 + "\n  3. BAYESIAN ANALYSIS\n" + "-"*60)
a_prior,b_prior=1,1

# Bayesian: proportion of positive tweets
k_pos = int((df.sentiment==4).sum())
n_tot = len(df)
a_pos,b_pos = a_prior+k_pos, b_prior+(n_tot-k_pos)
w(f"\nBayesian: Proportion of Positive Tweets")
w(f"  Positive: {k_pos}/{n_tot}")
w(f"  Posterior: Beta({a_pos},{b_pos})")
w(f"  Mean: {a_pos/(a_pos+b_pos)*100:.2f}%")
w(f"  95% CI: [{beta_dist.ppf(0.025,a_pos,b_pos)*100:.2f}%, {beta_dist.ppf(0.975,a_pos,b_pos)*100:.2f}%]")

# Bayesian A/B: URL presence impact
w("\nBayesian A/B: URL presence -> Positive sentiment")
with_url = df[df.has_url]; wo_url = df[~df.has_url]
k_w = int((with_url.sentiment==4).sum()); n_w = len(with_url)
k_wo = int((wo_url.sentiment==4).sum()); n_wo = len(wo_url)
aw,bw = a_prior+k_w, b_prior+(n_w-k_w)
awo,bwo = a_prior+k_wo, b_prior+(n_wo-k_wo)
w(f"  With URL:   {k_w}/{n_w} = {k_w/n_w*100:.1f}%")
w(f"  Without URL:{k_wo}/{n_wo} = {k_wo/n_wo*100:.1f}%")
w(f"  Posterior: URL=Yes Beta({aw},{bw}) mean={aw/(aw+bw)*100:.2f}%")
w(f"  Posterior: URL=No  Beta({awo},{bwo}) mean={awo/(awo+bwo)*100:.2f}%")
sw = beta_dist.rvs(aw,bw,100000); swo = beta_dist.rvs(awo,bwo,100000)
w(f"  P(URL > No URL) = {(sw>swo).mean()*100:.2f}%")

# ----- 4. Visualizations -----
w("\n" + "-"*60 + "\n  4. VISUALIZATIONS\n" + "-"*60)
fig, axes = plt.subplots(2,3,figsize=(18,12))
fig.suptitle('Project 2: Sentiment Analysis - Twitter', fontsize=16, fontweight='bold')

df['sentiment'].map({0:'Negative',4:'Positive'}).value_counts().plot(kind='bar',ax=axes[0,0],color=['#e74c3c','#2ecc71'],edgecolor='k')
axes[0,0].set_title('Sentiment Distribution',fontweight='bold')

axes[0,1].hist([df[df.sentiment==0]['tweet_len'],df[df.sentiment==4]['tweet_len']],bins=50,label=['Negative','Positive'],alpha=0.6,color=['#e74c3c','#2ecc71'])
axes[0,1].set_title('Tweet Length by Sentiment',fontweight='bold'); axes[0,1].legend()

df.groupby('year')['sentiment'].mean().plot(marker='o',ax=axes[0,2],color='#3498db')
axes[0,2].set_title('Avg Sentiment by Year',fontweight='bold'); axes[0,2].set_ylabel('Proportion Positive')

pd.crosstab(df['has_url'],df['sentiment'],normalize='index').plot(kind='bar',stacked=True,ax=axes[1,0],color=['#e74c3c','#2ecc71'],edgecolor='k')
axes[1,0].set_title('Sentiment by URL Presence',fontweight='bold')

pd.crosstab(df['is_mention'],df['sentiment'],normalize='index').plot(kind='bar',stacked=True,ax=axes[1,1],color=['#e74c3c','#2ecc71'],edgecolor='k')
axes[1,1].set_title('Sentiment by @Mention',fontweight='bold')

x=np.linspace(0.4,0.6,500)
axes[1,2].plot(x,beta_dist.pdf(x,aw,bw),'b-',lw=2,label='With URL')
axes[1,2].plot(x,beta_dist.pdf(x,awo,bwo),'orange',lw=2,label='W/o URL')
axes[1,2].fill_between(x,beta_dist.pdf(x,aw,bw),alpha=0.1,color='blue')
axes[1,2].fill_between(x,beta_dist.pdf(x,awo,bwo),alpha=0.1,color='orange')
axes[1,2].set_title('Bayesian: P(Positive) by URL',fontweight='bold'); axes[1,2].legend()

plt.tight_layout()
fig.savefig(f"{OUT}\\Sentiment_Project.png",dpi=150,bbox_inches='tight')
w("  [OK] Sentiment_Project.png")

log.close()
print(f"\nPROJECT 2 COMPLETE -> {OUT}\\report.txt")

