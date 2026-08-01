"""
Twitter Sentiment Prediction Pipeline
"""
import os
import logging
import argparse
import pandas as pd
import numpy as np
import warnings
from scipy.stats import beta as beta_dist

warnings.filterwarnings('ignore')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Sentiment_Pipeline')


class SentimentAnalysisPipeline:
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.model = None
        self.tfidf = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        logger.info(f"Loading and sampling balanced dataset from {self.data_path}")
        cols = ['sentiment', 'id', 'date', 'query', 'user', 'text']
        df_raw = pd.read_csv(self.data_path, names=cols, encoding='latin1')
        
        neg = df_raw[df_raw.sentiment == 0].sample(100000, random_state=42)
        pos = df_raw[df_raw.sentiment == 4].sample(100000, random_state=42)
        
        self.df = pd.concat([neg, pos], ignore_index=True)
        self.df['sentiment'] = (self.df['sentiment'] == 4).astype(int)
        logger.info(f"Dataset shape after sampling: {self.df.shape}")

    def feature_engineering(self):
        logger.info("Extracting meta-features from text...")
        self.df['text_len'] = self.df['text'].str.len()
        self.df['word_count'] = self.df['text'].str.split().str.len()
        self.df['has_url'] = self.df['text'].str.contains('http', na=False).astype(int)
        self.df['has_qmark'] = self.df['text'].str.contains(r'\?', na=False).astype(int)
        self.df['has_excl'] = self.df['text'].str.contains(r'!', na=False).astype(int)
        self.df['has_emoji'] = self.df['text'].str.contains(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]', na=False).astype(int)

    def generate_eda_dashboard(self):
        logger.info("Generating interactive EDA dashboard...")
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "Sentiment Distribution", "Tweet Length Distribution", 
            "URL Presence by Sentiment", "Exclamation Presence by Sentiment"
        ), specs=[[{"type": "domain"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]])

        # Sentiment Distribution
        fig.add_trace(go.Pie(labels=['Negative', 'Positive'], values=self.df['sentiment'].value_counts(), hole=0.4, 
                             marker_colors=['#e74c3c', '#2ecc71']), row=1, col=1)

        # Text Length Distribution (sampled)
        df_sample = self.df.sample(10000, random_state=42)
        for i, label in enumerate(['Negative', 'Positive']):
            fig.add_trace(go.Histogram(x=df_sample[df_sample['sentiment']==i]['text_len'], name=label, 
                                       marker_color=['#e74c3c', '#2ecc71'][i], opacity=0.7), row=1, col=2)

        # Bar charts for URLs and Exclamations
        for j, feat in enumerate(['has_url', 'has_excl']):
            crosstab = pd.crosstab(self.df[feat], self.df['sentiment'], normalize='index').reset_index()
            fig.add_trace(go.Bar(name='Negative', x=crosstab[feat].astype(str), y=crosstab[0], marker_color='#e74c3c'), row=2, col=j+1)
            fig.add_trace(go.Bar(name='Positive', x=crosstab[feat].astype(str), y=crosstab[1], marker_color='#2ecc71'), row=2, col=j+1)

        fig.update_layout(title_text="Twitter Sentiment EDA Dashboard", height=800, barmode='stack', template='plotly_white')
        pio.write_html(fig, file=os.path.join(self.output_dir, 'eda_dashboard.html'), auto_open=False)
        logger.info("EDA dashboard generated.")

    def preprocess_text_and_train(self):
        logger.info("Vectorizing text using TF-IDF (max 5000 features)...")
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english', sublinear_tf=True)
        X_txt = self.tfidf.fit_transform(self.df['text'])
        y_txt = self.df['sentiment'].values
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_txt, y_txt, test_size=0.2, random_state=42)
        self.y_test = y_te
        
        if LGBM_AVAILABLE:
            logger.info("Training LightGBM Classifier...")
            self.model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, max_depth=7, random_state=42, n_jobs=-1)
            self.model.fit(X_tr.astype('float32'), y_tr)
            X_te_input = X_te.astype('float32')
        else:
            logger.info("Training Logistic Regression...")
            self.model = LogisticRegression(max_iter=500, C=1.0)
            self.model.fit(X_tr, y_tr)
            X_te_input = X_te

        logger.info("Evaluating model...")
        self.y_pred = self.model.predict(X_te_input)
        self.y_prob = self.model.predict_proba(X_te_input)[:, 1]
        
        self.auc_score = roc_auc_score(self.y_test, self.y_prob)
        self.ap_score = average_precision_score(self.y_test, self.y_prob)
        
        logger.info(f"AUC-ROC: {self.auc_score:.4f}")
        logger.info(f"Average Precision: {self.ap_score:.4f}")

    def generate_static_dashboard(self):
        logger.info("Generating static evaluation metrics dashboard...")
        sns.set_theme(style="whitegrid", context="paper")
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

        # ROC & PR Curves
        ax1 = fig.add_subplot(gs[0, 0:2])
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        ax1.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC (AUC = {self.auc_score:.3f})')
        prec, rec, _ = precision_recall_curve(self.y_test, self.y_prob)
        ax1.plot(rec, prec, color='#8e44ad', lw=2, label=f'PR (AP = {self.ap_score:.3f})')
        ax1.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
        ax1.set_title('ROC & Precision-Recall Curves')
        ax1.legend(loc='lower center', ncol=2)

        # Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 2:])
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2, cbar=False)
        ax2.set_xticklabels(['Negative', 'Positive']); ax2.set_yticklabels(['Negative', 'Positive'])
        ax2.set_title('Confusion Matrix'); ax2.set_ylabel('Actual'); ax2.set_xlabel('Predicted')

        # Top Features
        ax3 = fig.add_subplot(gs[1, 0:2])
        feat_names = self.tfidf.get_feature_names_out()
        if LGBM_AVAILABLE:
            importances = self.model.feature_importances_
            top_idx = importances.argsort()[-15:][::-1]
            sns.barplot(x=importances[top_idx], y=[feat_names[i] for i in top_idx], palette='viridis', ax=ax3)
            ax3.set_title('Top 15 Most Important Words (LightGBM)')
        else:
            coefs = self.model.coef_[0]
            top_idx = np.abs(coefs).argsort()[-15:][::-1]
            sns.barplot(x=np.abs(coefs)[top_idx], y=[feat_names[i] for i in top_idx], palette='viridis', ax=ax3)
            ax3.set_title('Top 15 Most Important Words (LogReg)')

        # Word Clouds
        ax4 = fig.add_subplot(gs[1, 2])
        text_neg = ' '.join(self.df[self.df.sentiment==0]['text'].sample(10000, replace=True).values)
        wc_neg = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(text_neg)
        ax4.imshow(wc_neg, interpolation='bilinear'); ax4.axis('off'); ax4.set_title('Negative Cloud')

        ax5 = fig.add_subplot(gs[1, 3])
        text_pos = ' '.join(self.df[self.df.sentiment==1]['text'].sample(10000, replace=True).values)
        wc_pos = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(text_pos)
        ax5.imshow(wc_pos, interpolation='bilinear'); ax5.axis('off'); ax5.set_title('Positive Cloud')

        # Density plots
        ax6 = fig.add_subplot(gs[2, 0:2])
        sns.kdeplot(data=self.df, x='word_count', hue='sentiment', fill=True, alpha=0.5, ax=ax6)
        ax6.set_title('Word Count Distribution by Sentiment')

        # Bayesian Posteriors
        ax7 = fig.add_subplot(gs[2, 2:])
        xx = np.linspace(0.48, 0.52, 500)
        for col, lab, clr in [('has_url','Has URL','#3498db'),('has_emoji','Has Emoji','#f1c40f')]:
            k = int(self.df[self.df[col]==1].sentiment.sum()); n = int(self.df[col].sum())
            if n > 0:
                ax7.plot(xx, beta_dist.pdf(xx, 1+k, 1+n-k), lw=2, label=lab, color=clr)
                ax7.fill_between(xx, beta_dist.pdf(xx, 1+k, 1+n-k), alpha=0.2, color=clr)
        ax7.set_title('Bayesian Posterior P(Positive Sentiment | Feature)')
        ax7.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "evaluation_metrics.png"), dpi=300)
        logger.info("Evaluation dashboard saved successfully.")

    def write_report(self):
        logger.info("Saving performance report to report.txt...")
        report_path = os.path.join(self.output_dir, "report.txt")
        with open(report_path, 'w', encoding='utf-8') as rf:
            rf.write("PROJECT 2: TWITTER SENTIMENT ANALYSIS REPORT\n")
            rf.write("============================================\n")
            rf.write(f"Total sampled Tweets: {len(self.df):,}\n")
            rf.write(f"Class Distribution: Negative={len(self.df[self.df.sentiment==0])} | Positive={len(self.df[self.df.sentiment==1])}\n")
            rf.write(f"\nModel Performance Metrics:\n")
            rf.write(f"  AUC-ROC: {self.auc_score:.4f}\n")
            rf.write(f"  Average Precision: {self.ap_score:.4f}\n")
            rf.write(f"\nClassification Report:\n")
            rf.write(classification_report(self.y_test, self.y_pred, target_names=['Negative', 'Positive']))
            
            # Feature words
            feat_names = self.tfidf.get_feature_names_out()
            if LGBM_AVAILABLE:
                importances = self.model.feature_importances_
                top_idx = importances.argsort()[-15:][::-1]
                rf.write(f"\nTop 15 Most Important Words (LightGBM Importance):\n")
                for rank, idx in enumerate(top_idx, 1):
                    rf.write(f"  {rank}. '{feat_names[idx]}': {importances[idx]:.4f}\n")
            else:
                coefs = self.model.coef_[0]
                top_idx = np.abs(coefs).argsort()[-15:][::-1]
                rf.write(f"\nTop 15 Most Influential Words (Logistic Regression Coefficients):\n")
                for rank, idx in enumerate(top_idx, 1):
                    rf.write(f"  {rank}. '{feat_names[idx]}': {coefs[idx]:.4f}\n")

    def run(self):
        self.load_data()
        self.feature_engineering()
        self.generate_eda_dashboard()
        self.preprocess_text_and_train()
        self.generate_static_dashboard()
        self.write_report()
        logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Twitter Sentiment Prediction Pipeline")
    parser.add_argument("--data_path", type=str, default=r"D:\download\protfolio\archive (1)\training.1600000.processed.noemoticon.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--output_dir", type=str, default=r"D:\download\protfolio\projects\v3_output\project2_Sentiment",
                        help="Directory to save pipeline outputs")
    
    args = parser.parse_args()
    
    pipeline = SentimentAnalysisPipeline(data_path=args.data_path, output_dir=args.output_dir)
    pipeline.run()
