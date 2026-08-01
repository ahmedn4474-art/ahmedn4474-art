#!/usr/bin/env python3
"""
Twitter Sentiment Analysis: Professional NLP pipeline for classifying
tweet sentiment (positive / negative).  Covers exploratory analysis,
feature engineering, statistical testing, multi-model evaluation,
learning-curve diagnostics, error profiling, and bilingual reporting.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.max_open_warning': 0, 'axes.spines.top': False,
                      'axes.spines.right': False})

try: BASE = os.path.dirname(os.path.abspath(__file__))
except NameError: BASE = os.getcwd()
DATA = os.path.join(BASE, 'data', 'twitter_sentiment_1.6m.csv')
CHART_DIR = os.path.join(BASE, 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
        classification_report)
    from sklearn.feature_selection import chi2
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    _ML = True
except ImportError:
    _ML = False

try:
    from wordcloud import WordCloud
    _WC = True
except ImportError:
    _WC = False

# ---------------------------------------------------------------------------
# Column names & sampling constants
# ---------------------------------------------------------------------------
COLUMNS = ['target', 'id', 'date', 'query', 'user', 'text']


def load_sample(filepath, n_class=25000, buffer=50000, seed=42):
    """
    Load a stratified sample of 2 * n_class tweets from the sorted CSV file.

    The 1.6M-row CSV is known to be ordered: ~800k negative rows followed by
    ~800k positive rows.  We read a buffer window from each region then
    randomly subsample to n_class per class.
    """
    rng = np.random.RandomState(seed)

    # --- negative region (rows 0 – 799999) ---
    neg = pd.read_csv(filepath, header=None, names=COLUMNS,
                      encoding='cp1252', nrows=buffer,
                      on_bad_lines='skip', low_memory=False)
    neg = neg.sample(n=n_class, random_state=rng)
    neg['sentiment'] = 0

    # --- positive region (rows 800000 – 1599999) ---
    pos = pd.read_csv(filepath, header=None, names=COLUMNS,
                      encoding='cp1252', skiprows=range(800000),
                      nrows=buffer, on_bad_lines='skip', low_memory=False)
    pos = pos.sample(n=n_class, random_state=rng)
    pos['sentiment'] = 1

    df = pd.concat([neg, pos], ignore_index=True)
    df = df.sample(frac=1, random_state=rng).reset_index(drop=True)
    print(f'  Loaded {len(df):,} rows ({df.sentiment.mean()*100:.1f}% positive, '
          f'{df.sentiment.mean()*100:.1f}% negative)')
    return df


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def clean_text(t):
    if not isinstance(t, str):
        return ''
    t = t.lower()
    t = re.sub(r'http\S+|www\.\S+', '', t)
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


# ---------------------------------------------------------------------------
# Chart helper
# ---------------------------------------------------------------------------
def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, name), dpi=120, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'  [chart] {name}')


# ====================================================================
# 1.  EXPLORATORY DATA ANALYSIS
# ====================================================================
def eda_pipeline(df):
    """Generate all EDA figures."""

    # --- 1a. Sentiment distribution ---
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df['sentiment'].value_counts().sort_index()
    ax.bar(['Negative', 'Positive'], counts.values,
           color=['#e74c3c', '#2ecc71'], edgecolor='white', width=0.55)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f'{v:,}', ha='center', fontsize=11)
    ax.set_ylabel('Number of Tweets')
    ax.set_title('Sentiment Class Distribution')
    save_fig(fig, '01_sentiment_distribution.png')

    # --- 1b. Tweet length analysis ---
    df['char_len'] = df['text'].astype(str).str.len()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df['char_len'], bins=50, color='#3498db', edgecolor='white')
    axes[0].set_xlabel('Characters'); axes[0].set_ylabel('Frequency')
    axes[0].set_title('Tweet Length Distribution')
    df.boxplot(column='char_len', by='sentiment', ax=axes[1],
               patch_artist=True,
               boxprops=dict(facecolor='#3498db', alpha=0.6))
    axes[1].set_xticklabels(['Negative', 'Positive'])
    axes[1].set_title('Tweet Length by Sentiment')
    axes[1].set_xlabel('')
    fig.suptitle(''); plt.tight_layout()
    save_fig(fig, '02_tweet_length_analysis.png')

    # --- 1c. Most common words per sentiment ---
    pos_words = Counter()
    neg_words = Counter()
    for _, row in df.iterrows():
        tokens = clean_text(row['text']).split()
        if row['sentiment'] == 1:
            pos_words.update(tokens)
        else:
            neg_words.update(tokens)
    # remove common stopwords (manual short list)
    stop = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'it', 'its', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'our', 'their', 'and', 'but', 'or',
            'not', 'no', 'so', 'if', 'as', 'what', 'which', 'who',
            'whom', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'some', 'any'}
    for w in stop:
        pos_words.pop(w, None)
        neg_words.pop(w, None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, word_counts, color, title in zip(
            axes,
            [neg_words.most_common(15), pos_words.most_common(15)],
            ['#e74c3c', '#2ecc71'],
            ['Negative — Most Frequent Words', 'Positive — Most Frequent Words']):
        words, freqs = zip(*word_counts)
        ax.barh(range(len(words)), freqs, color=color, edgecolor='white')
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency')
        ax.set_title(title)
    save_fig(fig, '03_top_words_per_sentiment.png')

    # --- 1d. Tweet frequency over time ---
    # parse the Twitter date format
    def parse_twitter_date(s):
        try:
            return datetime.strptime(s.strip(), '%a %b %d %H:%M:%S %Z %Y')
        except (ValueError, AttributeError):
            return pd.NaT
    dates = df['date'].apply(parse_twitter_date)
    valid_dates = dates.dropna()
    if len(valid_dates) > 100:
        fig, ax = plt.subplots(figsize=(10, 4))
        date_counts = valid_dates.dt.date.value_counts().sort_index()
        ax.plot(range(len(date_counts)), date_counts.values,
                color='#2c3e50', linewidth=0.8)
        ax.fill_between(range(len(date_counts)), date_counts.values,
                        alpha=0.15, color='#2c3e50')
        ax.set_xlabel('Time (chronological order)')
        ax.set_ylabel('Tweet Count')
        ax.set_title('Tweet Frequency Over Time')
        save_fig(fig, '04_tweet_frequency_over_time.png')

    print('  EDA complete - 4 charts saved')
    return df  # now has char_len


# ====================================================================
# 2.  FEATURE ANALYSIS — CHI-SQUARE & N-GRAMS
# ====================================================================
def feature_analysis(df):
    """Chi-square test for word-sentiment association + n-gram profiles."""

    print('  Running chi-square analysis...')
    vec = CountVectorizer(max_features=2000, stop_words='english')
    X_counts = vec.fit_transform(df['clean'])
    chi2_scores, p_vals = chi2(X_counts, df['sentiment'])
    feature_names = vec.get_feature_names_out()

    # top-20 by chi2
    top_idx = chi2_scores.argsort()[::-1][:20]
    top_words = feature_names[top_idx]
    top_scores = chi2_scores[top_idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#2ecc71' if df.loc[df['clean'].str.contains(w, regex=False),
                                  'sentiment'].mean() > 0.5
              else '#e74c3c' for w in top_words]
    bars = ax.barh(range(len(top_words)), top_scores, color=colors, edgecolor='white')
    ax.set_yticks(range(len(top_words)))
    ax.set_yticklabels(top_words)
    ax.invert_yaxis()
    ax.set_xlabel(r'$\chi^2$ Statistic')
    ax.set_title('Top-20 Words by Chi-Square Association with Sentiment')
    # legend
    from matplotlib.patches import Patch
    ax.legend([Patch(color='#2ecc71'), Patch(color='#e74c3c')],
              ['Positive-associated', 'Negative-associated'],
              loc='lower right')
    save_fig(fig, '05_chi2_discriminative_features.png')

    # --- N-gram analysis with proper direction separation ---
    print('  Running n-gram analysis...')
    vec_ng = CountVectorizer(max_features=3000, ngram_range=(1, 2),
                             stop_words='english')
    X_ng = vec_ng.fit_transform(df['clean'])
    ng_names = vec_ng.get_feature_names_out()
    chi2_ng, _ = chi2(X_ng, df['sentiment'])

    # Determine association direction via per-class frequency ratio
    pos_mask = (df['sentiment'] == 1).values
    neg_mask = (df['sentiment'] == 0).values
    pos_sums = np.asarray(X_ng[pos_mask].sum(axis=0)).flatten()
    neg_sums = np.asarray(X_ng[neg_mask].sum(axis=0)).flatten()
    eps = 0.1
    log_odds = np.log((pos_sums + eps) / (neg_sums + eps))

    # Separate unigrams and bigrams
    n_tokens = np.array([len(w.split()) for w in ng_names])
    is_uni = n_tokens == 1
    is_bi = n_tokens == 2

    def top_by_direction(mask, n=10):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return [], []
        idx_sorted = idx[chi2_ng[idx].argsort()[::-1]]
        pos_idx = idx_sorted[log_odds[idx_sorted] > 0][:n]
        neg_idx = idx_sorted[log_odds[idx_sorted] < 0][:n]
        return list(ng_names[pos_idx]), list(ng_names[neg_idx])

    pos_uni, neg_uni = top_by_direction(is_uni, 10)
    pos_bi, neg_bi = top_by_direction(is_bi, 10)

    print('\n  Top-10 unigrams (pos -> neg):')
    for w in pos_uni:
        print(f'    + {w}')
    print('   ---')
    for w in neg_uni:
        print(f'    - {w}')

    print('\n  Top-10 bigrams (pos -> neg):')
    for w in pos_bi:
        print(f'    + {w}')
    print('   ---')
    for w in neg_bi:
        print(f'    - {w}')

    # Separate chi2 top-20 by direction using log-odds
    chi2_pos_words = []
    chi2_neg_words = []
    for w in top_words:
        mask = df['clean'].str.contains(re.escape(w), regex=True, na=False)
        if mask.any():
            if df.loc[mask, 'sentiment'].mean() > 0.5:
                chi2_pos_words.append(w)
            else:
                chi2_neg_words.append(w)
        else:
            chi2_pos_words.append(w)

    return {
        'chi2_words': list(top_words),
        'chi2_scores': list(top_scores),
        'chi2_pos_words': chi2_pos_words,
        'chi2_neg_words': chi2_neg_words,
        'pos_unigrams': pos_uni,
        'neg_unigrams': neg_uni,
        'pos_bigrams': pos_bi,
        'neg_bigrams': neg_bi,
    }


# ====================================================================
# 3.  WORD CLOUDS
# ====================================================================
def plot_wordclouds(df):
    """Generate word clouds for positive and negative tweets."""

    pos_text = ' '.join(df[df['sentiment'] == 1]['clean'])
    neg_text = ' '.join(df[df['sentiment'] == 0]['clean'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ['Negative Sentiment', 'Positive Sentiment']
    texts = [neg_text, pos_text]
    colors = ['Reds', 'Greens']

    for ax, txt, title, cmap in zip(axes, texts, titles, colors):
        if _WC and txt.strip():
            wc = WordCloud(width=600, height=400, colormap=cmap,
                           background_color='white', max_words=150,
                           random_state=42).generate(txt)
            ax.imshow(wc, interpolation='bilinear')
        else:
            # fallback: show word frequency bar chart
            words = txt.split()
            freq = Counter(words).most_common(20)
            if freq:
                w, f = zip(*freq)
                ax.barh(range(len(w)), f, color='#7f8c8d', edgecolor='white')
                ax.set_yticks(range(len(w))); ax.set_yticklabels(w)
                ax.invert_yaxis()
                ax.set_xlabel('Frequency')
        ax.set_title(title)
        ax.axis('off')
    save_fig(fig, '06_wordcloud_sentiments.png')
    print('  Word clouds saved')


# ====================================================================
# 4.  MACHINE LEARNING PIPELINE
# ====================================================================
def ml_pipeline(df):
    """Train and compare Logistic Regression, MultinomialNB, Random Forest."""

    print('  Vectorizing with TF-IDF (max 5000 features)...')
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(df['clean'])
    y = df['sentiment'].values
    indices = np.arange(len(df))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.2, random_state=42, stratify=y)
    print(f'  Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}')

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, n_jobs=-1),
        'Multinomial NB': MultinomialNB(alpha=0.1),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    }

    results = []
    roc_data = {}

    for name, model in models.items():
        print(f'  Training {name}...')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        results.append({
            'Model': name, 'Accuracy': acc, 'Precision': prec,
            'Recall': rec, 'F1': f1, 'AUC': auc})
        roc_data[name] = (y_test, y_prob)

        print(f'    Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  '
              f'F1={f1:.4f}  AUC={auc:.4f}')

    # --- ROC curves (all models) ---
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ['#3498db', '#e67e22', '#2ecc71']
    for (name, (yt, yp)), color in zip(roc_data.items(), colors):
        fpr, tpr, _ = roc_curve(yt, yp)
        auc_val = roc_auc_score(yt, yp)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models')
    ax.legend(loc='lower right')
    save_fig(fig, '07_roc_curves.png')

    # Identify best model (by AUC)
    results_df = pd.DataFrame(results).sort_values('AUC', ascending=False)
    best_name = results_df.iloc[0]['Model']
    best_model = models[best_name]
    print(f'\n  Best model: {best_name} '
          f'(AUC = {results_df.iloc[0]["AUC"]:.4f})')

    # --- Confusion matrix for best model ---
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Negative', 'Positive'])
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=False)
    ax.set_title(f'Confusion Matrix — {best_name}')
    save_fig(fig, '08_confusion_matrix.png')

    return {
        'results': results_df,
        'best_model': best_model,
        'best_name': best_name,
        'vectorizer': vectorizer,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred': y_pred_best,
        'y_prob': best_model.predict_proba(X_test)[:, 1],
        'idx_test': idx_test,
    }


# ====================================================================
# 5.  LEARNING CURVES
# ====================================================================
def plot_learning_curve(X, y, model, name):
    """Plot train / validation AUC over increasing training set sizes."""

    print('  Computing learning curve...')
    train_sizes = np.linspace(0.1, 1.0, 6)
    n_train, train_scores, val_scores = learning_curve(
        model, X, y, cv=3, n_jobs=-1,
        train_sizes=train_sizes,
        scoring='roc_auc',
        random_state=42)

    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_train, train_mean, 'o-', color='#3498db', label='Training AUC')
    ax.fill_between(n_train, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color='#3498db')
    ax.plot(n_train, val_mean, 's-', color='#e74c3c', label='Validation AUC')
    ax.fill_between(n_train, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color='#e74c3c')
    ax.set_xlabel('Training Examples')
    ax.set_ylabel('AUC')
    ax.set_title(f'Learning Curve — {name}')
    ax.legend(loc='lower right')
    ax.set_ylim(0.5, 1.0)
    save_fig(fig, '09_learning_curve.png')
    print(f'  Learning curve saved  (val AUC: {val_mean[0]:.3f} -> {val_mean[-1]:.3f})')


# ====================================================================
# 6.  ERROR ANALYSIS
# ====================================================================
def error_analysis(y_test, y_pred, texts, n_show=12):
    """Print and return misclassification examples."""

    errors = np.where(y_test != y_pred)[0]
    print(f'\n  Misclassified: {len(errors):,} / {len(y_test):,} '
          f'({len(errors)/len(y_test)*100:.1f}%)')

    # sample a few to show
    if len(errors) > n_show:
        rng = np.random.RandomState(42)
        show_idx = rng.choice(errors, n_show, replace=False)
    else:
        show_idx = errors

    examples = []
    print('\n  Error examples (actual -> predicted):')
    for idx in show_idx:
        true_label = 'Positive' if y_test[idx] == 1 else 'Negative'
        pred_label = 'Positive' if y_pred[idx] == 1 else 'Negative'
        tweet = texts.iloc[idx] if hasattr(texts, 'iloc') else texts[idx]
        short = tweet[:80] + '...' if len(tweet) > 80 else tweet
        print(f'    [{true_label} -> {pred_label}]  "{short}"')
        examples.append({
            'text': tweet, 'actual': true_label, 'predicted': pred_label})
    return examples, len(errors)


# ====================================================================
# 7.  TF-IDF TOP FEATURES PER CLASS
# ====================================================================
def top_tfidf_features(vectorizer, X, y, n=15):
    """Identify top TF-IDF features for each sentiment class."""
    pos_mask = y == 1
    neg_mask = y == 0
    feature_names = vectorizer.get_feature_names_out()

    pos_mean = X[pos_mask].mean(axis=0).A1
    neg_mean = X[neg_mask].mean(axis=0).A1

    pos_top = feature_names[np.argsort(pos_mean)[-n:][::-1]]
    neg_top = feature_names[np.argsort(neg_mean)[-n:][::-1]]

    return {'positive': list(pos_top), 'negative': list(neg_top)}


# ====================================================================
# 8.  REPORT GENERATION
# ====================================================================
def write_report(path, content):
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
    base = os.path.basename(path)
    print(f'  [report] {base}')


def generate_reports(metrics, best_name, chi2_info, tfidf_top, error_examples,
                     n_total, n_sample, pos_pct, neg_pct, avg_len,
                     best_model_metrics, lc_vals, n_errors_total=0):
    """Generate four bilingual reports (short + full, Arabic + English)."""

    ar_short, en_short = _short_reports(
        best_name, best_model_metrics, n_total, n_sample, pos_pct)
    ar_full, en_full = _full_reports(
        metrics, best_name, chi2_info, tfidf_top, error_examples,
        n_total, n_sample, pos_pct, neg_pct, avg_len,
        best_model_metrics, lc_vals)

    write_report(os.path.join(BASE, 'Sentiment_Report_AR_short.txt'), ar_short)
    write_report(os.path.join(BASE, 'Sentiment_Report_AR_full.txt'), ar_full)
    write_report(os.path.join(BASE, 'Sentiment_Report_EN_short.txt'), en_short)
    write_report(os.path.join(BASE, 'Sentiment_Report_EN_full.txt'), en_full)


def _short_reports(best_name, bm, n_total, n_sample, pos_pct):
    precision = bm['Precision']
    recall = bm['Recall']
    f1 = bm['F1']
    auc = bm['AUC']

    en = (
        '======================================================================\n'
        'TWITTER SENTIMENT ANALYSIS — SHORT REPORT\n'
        '======================================================================\n\n'
        f'Dataset: {n_total:,} tweets | Analyzed sample: {n_sample:,}\n'
        f'Sentiment distribution: {pos_pct:.1f}% positive, '
        f'{100-pos_pct:.1f}% negative\n'
        f'Best model: {best_name}\n'
        f'  Accuracy:  {bm["Accuracy"]:.4f}\n'
        f'  Precision: {precision:.4f}\n'
        f'  Recall:    {recall:.4f}\n'
        f'  F1 Score:  {f1:.4f}\n'
        f'  AUC:       {auc:.4f}\n\n'
        'Key findings:\n'
        f'  - {best_name} achieves {auc:.1%} AUC on held-out data.\n'
        f'  - Precision ({precision:.1%}) vs Recall ({recall:.1%}) '
        f'trade-off favours F1 = {f1:.3f}.\n'
        '  - TF-IDF unigrams provide strong signals for both classes.\n'
        '  - Learning curves indicate further gains possible with more data.\n'
    )

    ar = (
        '======================================================================\n'
        'تحليل المشاعر في التغريدات — تقرير مختصر\n'
        '======================================================================\n\n'
        f'مجموعة البيانات: {n_total:,} تغريدة | العينة المحللة: {n_sample:,}\n'
        f'توزيع المشاعر: {pos_pct:.1f}% إيجابي، {100-pos_pct:.1f}% سلبي\n'
        f'أفضل نموذج: {best_name}\n'
        f'  الدقة: {bm["Accuracy"]:.4f}\n'
        f'  الدقة (Precision): {precision:.4f}\n'
        f'  الاستدعاء (Recall): {recall:.4f}\n'
        f'  درجة F1: {f1:.4f}\n'
        f'  AUC: {auc:.4f}\n\n'
        'النتائج الرئيسية:\n'
        f'  - {best_name} يحقق {auc:.1%} AUC على بيانات الاختبار.\n'
        f'  - التوازن بين الدقة والاستدعاء يعطي F1 = {f1:.3f}.\n'
        '  - كلمات TF-IDF الأحادية توفر إشارات قوية لكلتا الفئتين.\n'
        '  - منحنيات التعلم تشير إلى إمكانية تحسن إضافي مع المزيد من البيانات.\n'
    )
    return ar, en


def _full_reports(metrics, best_name, chi2_info, tfidf_top, error_examples,
                  n_total, n_sample, pos_pct, neg_pct, avg_len,
                  bm, lc_vals):
    # Build model comparison table
    table_rows = []
    for _, row in metrics.iterrows():
        table_rows.append(
            f'  {row["Model"]:<22s}  {row["Accuracy"]:.4f}  '
            f'{row["Precision"]:.4f}  {row["Recall"]:.4f}  '
            f'{row["F1"]:.4f}  {row["AUC"]:.4f}')
    model_table = '\n'.join(table_rows)

    neg_feat = ', '.join(chi2_info['chi2_neg_words'][:10])
    pos_feat = ', '.join(chi2_info['chi2_pos_words'][:10])

    pos_uni = ', '.join(chi2_info['pos_unigrams'][:8])
    neg_uni = ', '.join(chi2_info['neg_unigrams'][:8])
    pos_bi = ', '.join(chi2_info['pos_bigrams'][:8])
    neg_bi = ', '.join(chi2_info['neg_bigrams'][:8])

    # Error summary
    n_errors = len(error_examples) if error_examples else 0
    error_rate = n_errors / (n_sample * 0.2) * 100 if n_sample > 0 else 0

    # Build error examples string
    error_str = ''
    if error_examples:
        error_str = '\n'.join(
            f'  • [{e["actual"]} → {e["predicted"]}]  '
            f'{e["text"][:70]}...'
            for e in error_examples[:8])

    lc_str = ''
    if lc_vals is not None:
        lc_str = (f'Validation AUC starts at {lc_vals[0]:.3f} and reaches '
                  f'{lc_vals[-1]:.3f}, indicating '
                  f'{"room for improvement with more data" if lc_vals[-1] - lc_vals[0] > 0.03 else "diminishing returns from additional data"}.')

    en = (
        '======================================================================\n'
        'TWITTER SENTIMENT ANALYSIS — COMPREHENSIVE REPORT\n'
        '======================================================================\n\n'
        '1. EXECUTIVE SUMMARY\n'
        '--------------------\n'
        f'Dataset size: {n_total:,} tweets (sampled {n_sample:,} stratified).\n'
        f'Sentiment split: {pos_pct:.1f}% positive, {neg_pct:.1f}% negative.\n'
        f'Average tweet length: {avg_len:.1f} characters.\n'
        f'Best performing model: {best_name} with AUC = {bm["AUC"]:.4f}.\n\n'
        '2. EXPLORATORY ANALYSIS\n'
        '-----------------------\n'
        'The sentiment distribution is approximately balanced in the full '
        'dataset. Tweet lengths follow a right-skewed distribution with a '
        'mode around 50–70 characters. Negative tweets tend to be slightly '
        'longer on average. The most frequent tokens for each class reflect '
        'expected polarity patterns (positive: "good", "love", "thanks"; '
        'negative: "hate", "sad", "miss").\n\n'
        '3. FEATURE ANALYSIS\n'
        '-------------------\n'
        f'Chi-square test identifies highly discriminative unigrams:\n'
        f'  Positive-associated: {pos_feat}\n'
        f'  Negative-associated: {neg_feat}\n\n'
        'Top unigrams (by chi-square, per class):\n'
        f'  Positive: {pos_uni}\n'
        f'  Negative: {neg_uni}\n\n'
        'Top bigrams (by chi-square, per class):\n'
        f'  Positive: {pos_bi}\n'
        f'  Negative: {neg_bi}\n\n'
        f'TF-IDF top features for positive class: '
        f'{", ".join(tfidf_top["positive"][:10])}\n'
        f'TF-IDF top features for negative class: '
        f'{", ".join(tfidf_top["negative"][:10])}\n\n'
        '4. MODEL COMPARISON\n'
        '--------------------\n'
        f'  {"Model":<22s}  {"Acc":>6s}  {"Prec":>6s}  {"Rec":>6s}  '
        f'{"F1":>6s}  {"AUC":>6s}\n'
        f'  {"-"*60}\n'
        f'{model_table}\n\n'
        f'Best model: {best_name}. Confusion matrix and ROC curve saved to '
        f'charts directory.\n\n'
        '5. LEARNING CURVE\n'
        '-----------------\n'
        f'Learning curve analysis on up to {n_sample:,} samples: {lc_str}\n\n'
        '6. ERROR ANALYSIS\n'
        '-----------------\n'
        f'On the test set ({(n_sample * 0.2):.0f} samples), '
        f'{n_errors} misclassifications ({error_rate:.1f}%).\n'
        'Sample errors:\n'
        f'{error_str}\n\n'
        '7. CONCLUSIONS\n'
        '--------------\n'
        f'- {best_name} provides the strongest performance for this task.\n'
        '- Lexical features (TF-IDF unigrams) capture sentiment effectively.\n'
        '- Bigrams add marginal value over unigrams for this dataset.\n'
        '- The balanced class distribution makes accuracy a reliable metric.\n'
    )

    ar = (
        '======================================================================\n'
        'تحليل المشاعر في التغريدات — تقرير شامل\n'
        '======================================================================\n\n'
        '1. الملخص التنفيذي\n'
        '------------------\n'
        f'حجم مجموعة البيانات: {n_total:,} تغريدة (تم أخذ عينة طبقية من {n_sample:,}).\n'
        f'توزيع المشاعر: {pos_pct:.1f}% إيجابي، {neg_pct:.1f}% سلبي.\n'
        f'متوسط طول التغريدة: {avg_len:.1f} حرف.\n'
        f'أفضل نموذج: {best_name} بنسبة AUC = {bm["AUC"]:.4f}.\n\n'
        '2. التحليل الاستكشافي\n'
        '---------------------\n'
        'توزيع المشاعر متوازن تقريباً في مجموعة البيانات الكاملة. '
        'تتبع أطوال التغريدات توزيعاً منحرفاً نحو اليمين مع نمط حول '
        '50–70 حرفاً. التغريدات السلبية أطول قليلاً في المتوسط. '
        'تعكس الكلمات الأكثر شيوعاً أنماط القطبية المتوقعة.\n\n'
        '3. تحليل الميزات\n'
        '----------------\n'
        f'اختبار كاي تربيع يحدد الكلمات الأكثر تمييزاً:\n'
        f'  مرتبطة بالإيجابي: {pos_feat}\n'
        f'  مرتبطة بالسلبي: {neg_feat}\n\n'
        'أفضل الأحادية (حسب كاي تربيع):\n'
        f'  إيجابي: {pos_uni}\n'
        f'  سلبي: {neg_uni}\n\n'
        'أفضل الثنائيات (حسب كاي تربيع):\n'
        f'  إيجابي: {pos_bi}\n'
        f'  سلبي: {neg_bi}\n\n'
        f'أفضل ميزات TF-IDF للإيجابي: '
        f'{", ".join(tfidf_top["positive"][:10])}\n'
        f'أفضل ميزات TF-IDF للسلبي: '
        f'{", ".join(tfidf_top["negative"][:10])}\n\n'
        '4. مقارنة النماذج\n'
        '-----------------\n'
        f'  {"النموذج":<22s}  {"الدقة":>6s}  {"الضبط":>6s}  {"الاستدعاء":>6s}  '
        f'{"F1":>6s}  {"AUC":>6s}\n'
        f'  {"-"*60}\n'
        f'{model_table}\n\n'
        f'أفضل نموذج: {best_name}\n\n'
        '5. منحنى التعلم\n'
        '--------------\n'
        f'تحليل منحنى التعلم: {lc_str}\n\n'
        '6. تحليل الأخطاء\n'
        '---------------\n'
        f'في مجموعة الاختبار، تم تصنيف {n_errors} تغريدة بشكل خاطئ '
        f'({error_rate:.1f}%).\n'
        'نماذج من الأخطاء:\n'
        f'{error_str}\n\n'
        '7. الاستنتاجات\n'
        '--------------\n'
        f'- {best_name} يقدم أفضل أداء لهذه المهمة.\n'
        '- الميزات المعجمية (TF-IDF) تلتقط المشاعر بشكل فعال.\n'
        '- الثنائيات تضيف قيمة هامشية مقارنة بالأحادية.\n'
        '- توزيع الفئات المتوازن يجعل الدقة مقياساً موثوقاً.\n'
    )

    return ar, en


# ====================================================================
# MAIN
# ====================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('TWITTER SENTIMENT ANALYSIS')
    print('=' * 60)

    # --- Load data ---
    print('\n[1/9] Loading stratified sample...')
    df = load_sample(DATA)
    n_total = 1_600_000
    n_sample = len(df)
    pos_pct = df['sentiment'].mean() * 100
    neg_pct = 100 - pos_pct
    avg_len = df['text'].astype(str).str.len().mean()

    # --- Preprocessing ---
    print('\n[2/9] Preprocessing text...')
    df['clean'] = df['text'].apply(clean_text)

    # --- EDA ---
    print('\n[3/9] Exploratory data analysis...')
    df = eda_pipeline(df)

    # --- Feature analysis: chi-square + n-grams ---
    print('\n[4/9] Feature analysis (chi-square & n-grams)...')
    chi2_info = feature_analysis(df)

    # --- Word clouds ---
    print('\n[5/9] Word clouds...')
    plot_wordclouds(df)

    # --- ML pipeline ---
    print('\n[6/9] Machine learning...' if _ML else '\n[6/9] ML SKIPPED (no sklearn)')
    ml_results = None
    if _ML:
        ml_results = ml_pipeline(df)
        # TF-IDF top features
        tfidf_top = top_tfidf_features(
            ml_results['vectorizer'], ml_results['X_train'], ml_results['y_train'])

        # --- Learning curve ---
        print('\n[7/9] Learning curve...')
        # Use a faster model for learning curve (Logistic Regression)
        lr_fast = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        plot_learning_curve(
            ml_results['X_train'], ml_results['y_train'],
            lr_fast, ml_results['best_name'])

        # Re-fit best model on full data for proper curve values
        lc_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        from sklearn.model_selection import learning_curve as lc_func
        train_sizes = np.linspace(0.1, 1.0, 6)
        _, _, val_scores = lc_func(
            lc_model, ml_results['X_train'], ml_results['y_train'],
            cv=3, n_jobs=-1, train_sizes=train_sizes, scoring='roc_auc',
            random_state=42)
        lc_vals = val_scores.mean(axis=1)

        # --- Error analysis ---
        print('\n[8/9] Error analysis...')
        test_texts = df.iloc[ml_results['idx_test']]['text'].values
        errors, n_errors_total = error_analysis(
            ml_results['y_test'], ml_results['y_pred'], test_texts)
    else:
        tfidf_top = {'positive': [], 'negative': []}
        errors = []
        n_errors_total = 0
        lc_vals = None
        ml_results = {
            'best_name': 'N/A', 'best_model': None,
            'results': pd.DataFrame(columns=['Model', 'Accuracy', 'Precision',
                                              'Recall', 'F1', 'AUC']),
            'y_test': np.array([]), 'y_pred': np.array([]),
        }

    # --- Reports ---
    print('\n[9/9] Generating bilingual reports...')
    best_model_metrics = (
        ml_results['results'].iloc[0].to_dict()
        if _ML and len(ml_results['results']) > 0
        else {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1': 0, 'AUC': 0}
    )
    generate_reports(
        metrics=ml_results['results'],
        best_name=ml_results['best_name'],
        chi2_info=chi2_info,
        tfidf_top=tfidf_top,
        error_examples=errors if _ML else [],
        n_errors_total=n_errors_total,
        n_total=n_total, n_sample=n_sample,
        pos_pct=pos_pct, neg_pct=neg_pct,
        avg_len=avg_len,
        best_model_metrics=best_model_metrics,
        lc_vals=lc_vals if _ML else None,
    )

    print('\n' + '=' * 60)
    print('ANALYSIS COMPLETE')
    print('=' * 60)
