"""
PROJECT 2: SENTIMENT ANALYSIS — TWITTER
========================================
Techniques: EDA, A/B Testing, Bayesian, TF-IDF + Logistic Regression,
            Word Clouds, N-grams, Emoji Analysis, Confusion Matrix
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
from scipy.stats import beta as beta_dist
warnings.filterwarnings('ignore')

OUT = "D:\\download\\protfolio\\projects\\v2_output\\project2_Sentiment"
os.makedirs(OUT, exist_ok=True)
log = open(os.path.join(OUT,"report.txt"),'w',encoding='utf-8')
def w(t=""): log.write(str(t)+"\n"); print(t)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from collections import Counter
import re

w("="*85 + "\n  PROJECT 2: TWITTER SENTIMENT ANALYSIS — PROFESSIONAL\n" + "="*85)

# ── Load Data ──
cols=['sentiment','id','date','query','user','text']
w("\nLoading full dataset & sampling 200K balanced...")
df = pd.read_csv("D:\\download\\protfolio\\archive (1)\\training.1600000.processed.noemoticon.csv",
                 names=cols, encoding='latin1')
neg = df[df.sentiment==0].sample(100000, random_state=42)
pos = df[df.sentiment==4].sample(100000, random_state=42)
df = pd.concat([neg, pos], ignore_index=True)
df['sentiment'] = (df['sentiment']==4).astype(int)
w(f"  Loaded {len(df):,} (100K neg + 100K pos)")

# ── Feature Engineering ──
df['text_len'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['has_url'] = df['text'].str.contains('http', na=False).astype(int)
df['is_mention'] = df['text'].str.startswith('@').astype(int)
df['has_qmark'] = df['text'].str.contains(r'\?', na=False).astype(int)
df['has_excl'] = df['text'].str.contains(r'!', na=False).astype(int)
df['caps_ratio'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper())/max(len(x),1))
df['has_emoji'] = df['text'].str.contains(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', na=False).astype(int)

# ═══════════════════════════════════════════
# 1. EDA
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  1. EXPLORATORY DATA ANALYSIS\n" + "▔"*60)
w(f"\n  Tweet length stats:\n{df['text_len'].describe().round(1).to_string()}")
w(f"\n  Features prevalence:")
for col in ['has_url','is_mention','has_qmark','has_excl','has_emoji']:
    w(f"    {col:15s} {df[col].mean()*100:.2f}%")

# ═══════════════════════════════════════════
# 2. A/B TESTING
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  2. A/B TESTING\n" + "▔"*60)
for col in ['has_url','is_mention','has_qmark','has_excl','has_emoji']:
    g0 = df[df.sentiment==0][col]; g1 = df[df.sentiment==1][col]
    ct = pd.crosstab(df[col], df['sentiment'])
    chi2, p = stats.chi2_contingency(ct)[:2]
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    w(f"  {col:15s}: Neg={g0.mean()*100:.1f}% Pos={g1.mean()*100:.1f}% chi2={chi2:.2f} p={p:.6f} {sig}")

w(f"\n  T-Test: Text Length by Sentiment")
g0l = df[df.sentiment==0]['text_len']; g1l = df[df.sentiment==1]['text_len']
t,p = stats.ttest_ind(g0l, g1l); d = (g1l.mean()-g0l.mean())/np.sqrt((g0l.var()+g1l.var())/2)
w(f"    Neg={g0l.mean():.1f} Pos={g1l.mean():.1f} t={t:.3f} p={p:.6f} Cohen's d={d:.4f}")

w(f"\n  T-Test: Word Count by Sentiment")
g0w = df[df.sentiment==0]['word_count']; g1w = df[df.sentiment==1]['word_count']
t,p = stats.ttest_ind(g0w, g1w)
w(f"    Neg={g0w.mean():.1f} Pos={g1w.mean():.1f} t={t:.3f} p={p:.6f}")

# ═══════════════════════════════════════════
# 3. BAYESIAN
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  3. BAYESIAN INFERENCE\n" + "▔"*60)
a_p,b_p=1,1
k_pos = int(df.sentiment.sum()); n_tot = len(df)
w(f"\n  Overall P(Positive): Beta({a_p+k_pos},{b_p+n_tot-k_pos})")
w(f"    Mean: {(a_p+k_pos)/(a_p+b_p+n_tot)*100:.2f}%")
w(f"    95% HDI: [{beta_dist.ppf(0.025,a_p+k_pos,b_p+n_tot-k_pos)*100:.2f}%, {beta_dist.ppf(0.975,a_p+k_pos,b_p+n_tot-k_pos)*100:.2f}%]")

w(f"\n  Bayesian A/B: URL presence")
for lab, cond in [('Has URL', df.has_url==1), ('No URL', df.has_url==0)]:
    sub = df[cond]; k=int(sub.sentiment.sum()); n=len(sub)
    a,b=a_p+k, b_p+n-k; lo,hi=beta_dist.ppf(0.025,a,b),beta_dist.ppf(0.975,a,b)
    w(f"    {lab:10s}: {k}/{n} -> Beta({a},{b}) = {a/(a+b)*100:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%]")
ku = int(df[df.has_url==1].sentiment.sum()); nu = int(df.has_url.sum())
knu = int(df[df.has_url==0].sentiment.sum()); nnu = int((df.has_url==0).sum())
su = beta_dist.rvs(a_p+ku,b_p+(nu-ku),100000); snu = beta_dist.rvs(a_p+knu,b_p+(nnu-knu),100000)
w(f"    P(URL > No URL) = {(su>snu).mean()*100:.2f}%")

# ═══════════════════════════════════════════
# 4. TF-IDF + TEXT CLASSIFICATION
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  4. TEXT CLASSIFICATION — TF-IDF + LOGISTIC REGRESSION\n" + "▔"*60)
w("\n  Vectorizing text (TF-IDF, max 5000 features)...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english', sublinear_tf=True)
X_txt = tfidf.fit_transform(df['text'])
y_txt = df['sentiment'].values
X_tr, X_te, y_tr, y_te = train_test_split(X_txt, y_txt, test_size=0.2, random_state=42)

lr_txt = LogisticRegression(max_iter=500, C=1.0, penalty='l2', solver='liblinear')
lr_txt.fit(X_tr, y_tr); y_pred = lr_txt.predict(X_te); y_prob = lr_txt.predict_proba(X_te)[:,1]
w(f"\n  Logistic Regression (TF-IDF):")
w(f"    Accuracy:  {lr_txt.score(X_te, y_te):.4f}")
w(f"    AUC-ROC:   {roc_auc_score(y_te, y_prob):.4f}")
w(f"\n  Classification Report:")
w(f"\n{classification_report(y_te, y_pred, target_names=['Negative','Positive'])}")

# Top words
feat_names = tfidf.get_feature_names_out()
coefs = lr_txt.coef_[0]
top_pos_idx = coefs.argsort()[-20:][::-1]
top_neg_idx = coefs.argsort()[:20]
w("\n  Top 20 Words for Positive Sentiment:")
for i in top_pos_idx: w(f"    {feat_names[i]:20s} coef={coefs[i]:+.4f}")
w("\n  Top 20 Words for Negative Sentiment:")
for i in top_neg_idx: w(f"    {feat_names[i]:20s} coef={coefs[i]:+.4f}")

# ═══════════════════════════════════════════
# 5. VISUALIZATIONS
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  5. PROFESSIONAL VISUALIZATIONS\n" + "▔"*60)
sns.set_style("whitegrid")

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Twitter Sentiment Analysis — Professional Dashboard', fontsize=18, fontweight='bold', y=0.98)
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# [1] Sentiment pie
ax1 = fig.add_subplot(gs[0,0])
df['sentiment'].map({0:'Negative',1:'Positive'}).value_counts().plot(kind='pie',
    ax=ax1, autopct='%1.1f%%', colors=['#e74c3c','#2ecc71'], explode=(0.02,0.02),
    textprops={'fontweight':'bold'}, startangle=90)
ax1.set_title('Sentiment Distribution', fontweight='bold'); ax1.set_ylabel('')

# [2] Tweet length hist
ax2 = fig.add_subplot(gs[0,1])
ax2.hist([df[df.sentiment==0]['text_len'], df[df.sentiment==1]['text_len']],
    bins=50, alpha=0.6, label=['Negative','Positive'], color=['#e74c3c','#2ecc71'], density=True)
ax2.set_title('Tweet Length Distribution', fontweight='bold'); ax2.legend(); ax2.set_xlabel('Length')

# [3] Has URL bar
ax3 = fig.add_subplot(gs[0,2])
pd.crosstab(df['has_url'], df['sentiment'], normalize='index').plot(kind='bar',
    stacked=True, ax=ax3, color=['#e74c3c','#2ecc71'], edgecolor='k', legend=False)
ax3.set_title('Sentiment by URL Presence', fontweight='bold')
ax3.set_xticklabels(['No URL','Has URL'], rotation=0)

# [4] Emoji bar
ax4 = fig.add_subplot(gs[0,3])
pd.crosstab(df['has_emoji'], df['sentiment'], normalize='index').plot(kind='bar',
    stacked=True, ax=ax4, color=['#e74c3c','#2ecc71'], edgecolor='k', legend=False)
ax4.set_title('Sentiment by Emoji Presence', fontweight='bold')
ax4.set_xticklabels(['No Emoji','Has Emoji'], rotation=0)

# [5-6] Word Clouds
for idx, (sentiment, title, color) in enumerate([(0,'Negative Words','#e74c3c'),(1,'Positive Words','#2ecc71')]):
    ax = fig.add_subplot(gs[1,idx])
    text = ' '.join(df[df.sentiment==sentiment]['text'].head(10000).values)
    wc = WordCloud(width=400, height=300, background_color='white', max_words=100,
                   colormap='Reds' if sentiment==0 else 'Greens', collocations=False).generate(text)
    ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); ax.set_title(title, fontweight='bold')

# [7] Confusion Matrix
ax7 = fig.add_subplot(gs[1,2])
cm = confusion_matrix(y_te, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax7, cbar=False,
            xticklabels=['Negative','Positive'], yticklabels=['Negative','Positive'])
ax7.set_title('Confusion Matrix (TF-IDF + LogReg)', fontweight='bold')
ax7.set_ylabel('Actual'); ax7.set_xlabel('Predicted')

# [8] ROC
ax8 = fig.add_subplot(gs[1,3])
fpr, tpr, _ = roc_curve(y_te, y_prob); roc_auc = auc(fpr, tpr)
ax8.plot(fpr, tpr, 'b-', lw=2.5, label=f'AUC={roc_auc:.4f}')
ax8.plot([0,1],[0,1],'k--',alpha=0.3); ax8.set_xlabel('FPR'); ax8.set_ylabel('TPR')
ax8.set_title('ROC Curve', fontweight='bold'); ax8.legend()

# [9-10] Top words
for idx, (coef_mult, title, color) in enumerate([(1,'Top Positive Words','#2ecc71'),(-1,'Top Negative Words','#e74c3c')]):
    ax = fig.add_subplot(gs[2,idx])
    top_idx = coefs.argsort()[::coef_mult][:15]
    top_words = [feat_names[i] for i in top_idx][::-1]
    top_vals = [coefs[i] for i in top_idx][::-1]
    colors = [color if coef_mult==1 else color]*len(top_words)
    ax.barh(range(len(top_words)), top_vals, color=color, edgecolor='k')
    ax.set_yticks(range(len(top_words))); ax.set_yticklabels(top_words)
    ax.invert_yaxis(); ax.set_title(title, fontweight='bold')

# [11] Feature prevalence
ax11 = fig.add_subplot(gs[2,2])
feat_prevalence = df[['has_url','is_mention','has_qmark','has_excl','has_emoji']].mean()*100
feat_prevalence.plot(kind='bar', ax=ax11, color=['#3498db','#e74c3c','#2ecc71','#f39c12','#9b59b6'], edgecolor='k')
ax11.set_title('Feature Prevalence (%)', fontweight='bold'); ax11.set_xticklabels(ax11.get_xticklabels(), rotation=20)

# [12] Bayesian posteriors
ax12 = fig.add_subplot(gs[2,3])
xx = np.linspace(0.45, 0.55, 500)
for col, lab, clr in [('has_url','Has URL','blue'),('is_mention','@Mention','orange')]:
    k = int(df[df[col]==1].sentiment.sum()); n = int(df[col].sum())
    ax12.plot(xx, beta_dist.pdf(xx, a_p+k, b_p+n-k), lw=2, label=lab, color=clr)
ax12.set_title('Bayesian Posteriors', fontweight='bold'); ax12.legend()

# [13] Bigram frequencies
ax13 = fig.add_subplot(gs[3,:2]); ax13.axis('off')
bigram_text = "Top Positive Bigrams:\n"
pos_texts = ' '.join(df[df.sentiment==1]['text'].head(5000))
neg_texts = ' '.join(df[df.sentiment==0]['text'].head(5000))
vec_bigram = TfidfVectorizer(ngram_range=(2,2), max_features=10, stop_words='english')
for sentiment, label in [(1,'Positive'),(0,'Negative')]:
    bigram_data = vec_bigram.fit_transform([pos_texts if sentiment==1 else neg_texts])
    bgs = vec_bigram.get_feature_names_out()
    bg_sum = np.array(bigram_data.sum(axis=0)).flatten()
    top_bg_idx = bg_sum.argsort()[::-1][:10]
    bigram_text += f"\n  {label}:\n"
    for i in top_bg_idx: bigram_text += f"    {bgs[i]}: {bg_sum[i]:.0f}\n"
ax13.text(0.05, 0.95, bigram_text, transform=ax13.transAxes, fontsize=9,
    fontfamily='monospace', verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
ax13.set_title('Key N-grams', fontweight='bold')

# [14] caps ratio
ax14 = fig.add_subplot(gs[3,2:])
ax14.hist([df[df.sentiment==0]['caps_ratio'], df[df.sentiment==1]['caps_ratio']],
    bins=30, alpha=0.6, label=['Negative','Positive'], color=['#e74c3c','#2ecc71'], density=True)
ax14.set_title('Capitalization Ratio', fontweight='bold'); ax14.legend(); ax14.set_xlabel('Ratio of Caps')

plt.tight_layout()
fig.savefig(f"{OUT}\\Dashboard.png", dpi=200, bbox_inches='tight')
w("  [OK] Dashboard.png")

# ═══════════════════════════════════════════
# 6. EXECUTIVE SUMMARY
# ═══════════════════════════════════════════
w("\n" + "▔"*60 + "\n  6. EXECUTIVE SUMMARY\n" + "▔"*60)
w(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │                      EXECUTIVE SUMMARY                              │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  1. DATASET: 200,000 tweets (50% negative, 50% positive)            │
  │                                                                     │
  │  2. TEXT FEATURES THAT DIFFER SIGNIFICANTLY:                        │
  │     • URLs:       {df[df.has_url==1].sentiment.mean()*100:.1f}% positive vs baseline 50%     │
  │     • @mentions:  {df[df.is_mention==1].sentiment.mean()*100:.1f}% positive                  │
  │     • Emojis:     {df[df.has_emoji==1].sentiment.mean()*100:.1f}% positive                    │
  │     • Exclamation: {df[df.has_excl==1].sentiment.mean()*100:.1f}% positive                    │
  │                                                                     │
  │  3. CLASSIFICATION PERFORMANCE:                                     │
  │     • Model: Logistic Regression + TF-IDF (5K features, 1-2grams)  │
  │     • Accuracy:  {lr_txt.score(X_te, y_te):.3f}                              │
  │     • AUC-ROC:   {roc_auc_score(y_te, y_prob):.3f}                              │
  │     • Precision: {classification_report(y_te, y_pred, output_dict=True)['weighted avg']['precision']:.3f}          │
  │                                                                     │
  │  4. TOP DISCRIMINATIVE WORDS:                                       │
  │     Positive: {', '.join([feat_names[i] for i in top_pos_idx[:5]])}                    │
  │     Negative: {', '.join([feat_names[i] for i in top_neg_idx[:5]])}                    │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
""")
log.close()
print(f"\n✅ PROJECT 2 COMPLETE → {OUT}\\report.txt")
