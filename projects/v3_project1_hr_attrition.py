"""
HR Employee Attrition Prediction Pipeline
"""
import os
import logging
import argparse
import pandas as pd
import numpy as np
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings('ignore')

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                             roc_auc_score, precision_recall_curve, average_precision_score)
import shap

try:
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ADVANCED_LIBS = True
except ImportError:
    ADVANCED_LIBS = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HR_Attrition_Pipeline')

class HRAttritionPipeline:
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.model = None
        self.feature_cols = []
        self.target = 'Attrition'
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_and_explore_data(self):
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        self.df[self.target] = (self.df[self.target] == 'Yes').astype(int)
        
        logger.info(f"Dataset shape: {self.df.shape}")
        logger.info(f"Missing values: {self.df.isnull().sum().sum()}")
        logger.info(f"Baseline attrition rate: {self.df[self.target].mean():.2%}")
        
    def generate_eda_dashboard(self):
        logger.info("Generating interactive EDA dashboard...")
        fig = make_subplots(rows=2, cols=2, subplot_titles=(
            "Attrition Rate", "Income Distribution by Attrition", 
            "Attrition by Department", "Attrition by OverTime"
        ), specs=[[{"type": "domain"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]])

        attr_counts = self.df[self.target].value_counts()
        fig.add_trace(go.Pie(labels=['Stay', 'Leave'], values=attr_counts, hole=0.4, 
                             marker_colors=['#2ecc71', '#e74c3c']), row=1, col=1)

        for i, label in enumerate(['Stay', 'Leave']):
            fig.add_trace(go.Box(y=self.df[self.df[self.target]==i]['MonthlyIncome'], name=label, 
                                 marker_color=['#2ecc71', '#e74c3c'][i]), row=1, col=2)

        dept_attr = self.df.groupby('Department')[self.target].mean().reset_index()
        fig.add_trace(go.Bar(x=dept_attr['Department'], y=dept_attr[self.target], marker_color='#3498db'), row=2, col=1)

        ot_attr = self.df.groupby('OverTime')[self.target].mean().reset_index()
        fig.add_trace(go.Bar(x=ot_attr['OverTime'], y=ot_attr[self.target], marker_color='#9b59b6'), row=2, col=2)

        fig.update_layout(title_text="HR Attrition Dashboard", height=800, showlegend=False, template='plotly_white')
        pio.write_html(fig, file=os.path.join(self.output_dir, 'eda_dashboard.html'), auto_open=False)
        logger.info("EDA dashboard generated successfully.")
        
    def preprocess_data(self):
        logger.info("Preprocessing data for machine learning...")
        df_ml = self.df.copy()
        
        # Label Encoding for categorical variables
        for col in df_ml.select_dtypes('object').columns:
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col])
            
        self.feature_cols = [c for c in df_ml.columns if c != self.target]
        X = df_ml[self.feature_cols]
        y = df_ml[self.target]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scaling
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        
    def train_model(self):
        if ADVANCED_LIBS:
            logger.info("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(self.X_train_scaled, self.y_train)
            
            logger.info("Running hyperparameter optimization with Optuna...")
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                    'use_label_encoder': False,
                    'eval_metric': 'logloss',
                    'random_state': 42
                }
                model = XGBClassifier(**params)
                model.fit(X_train_res, y_train_res)
                preds = model.predict_proba(self.X_test_scaled)[:, 1]
                return roc_auc_score(self.y_test, preds)

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=10)
            
            best_params = study.best_params
            best_params['use_label_encoder'] = False
            best_params['eval_metric'] = 'logloss'
            best_params['random_state'] = 42
            
            logger.info(f"Best AUC: {study.best_value:.4f}")
            logger.info("Training final XGBoost model with best parameters...")
            self.model = XGBClassifier(**best_params)
            self.model.fit(X_train_res, y_train_res)
        else:
            logger.warning("Advanced libraries not found. Falling back to RandomForest.")
            self.model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
            self.model.fit(self.X_train_scaled, self.y_train)
            
    def evaluate_model(self):
        logger.info("Evaluating model performance...")
        self.y_pred = self.model.predict(self.X_test_scaled)
        self.y_prob = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        self.auc_score = roc_auc_score(self.y_test, self.y_prob)
        self.ap_score = average_precision_score(self.y_test, self.y_prob)
        
        logger.info(f"AUC-ROC: {self.auc_score:.4f}")
        logger.info(f"Average Precision: {self.ap_score:.4f}")
        
    def generate_shap_explanations(self):
        logger.info("Generating SHAP explanations...")
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(self.X_test_scaled)
            
            plt.figure(figsize=(10, 8))
            shap_v = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap.summary_plot(shap_v, self.X_test_scaled, feature_names=self.feature_cols, show=False)
            plt.title("SHAP Feature Importance & Impact", fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "shap_summary.png"), dpi=300)
            logger.info("SHAP explanations saved.")
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations: {str(e)}")

    def generate_evaluation_dashboard(self):
        logger.info("Generating static evaluation metrics dashboard...")
        sns.set_theme(style="whitegrid", context="paper")
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # ROC Curve
        ax1 = fig.add_subplot(gs[0, 0])
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)
        ax1.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'AUC = {self.auc_score:.3f}')
        ax1.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
        ax1.set_title('ROC Curve')
        ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate'); ax1.legend()

        # Feature Importance
        ax2 = fig.add_subplot(gs[0, 1:])
        if hasattr(self.model, 'feature_importances_'):
            fi = pd.DataFrame({'Feature': self.feature_cols, 'Importance': self.model.feature_importances_})
            fi = fi.sort_values('Importance', ascending=False).head(10)
            sns.barplot(x='Importance', y='Feature', data=fi, palette='viridis', ax=ax2)
            ax2.set_title('Top 10 Feature Importances')

        # Confusion Matrix
        ax3 = fig.add_subplot(gs[1, 0])
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
        ax3.set_xticklabels(['Stay', 'Leave']); ax3.set_yticklabels(['Stay', 'Leave'])
        ax3.set_title('Confusion Matrix'); ax3.set_ylabel('Actual'); ax3.set_xlabel('Predicted')

        # Precision-Recall Curve
        ax4 = fig.add_subplot(gs[1, 1])
        prec, rec, _ = precision_recall_curve(self.y_test, self.y_prob)
        ax4.plot(rec, prec, color='#8e44ad', lw=2, label=f'AP = {self.ap_score:.3f}')
        ax4.set_title('Precision-Recall Curve')
        ax4.set_xlabel('Recall'); ax4.set_ylabel('Precision'); ax4.legend()

        # Distribution Density
        ax5 = fig.add_subplot(gs[1, 2])
        sns.kdeplot(data=self.df, x='MonthlyIncome', hue=self.target, fill=True, alpha=0.5, ax=ax5)
        ax5.set_title('Income Distribution by Attrition')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "evaluation_metrics.png"), dpi=300)
        logger.info("Evaluation dashboard saved successfully.")

    def write_report(self):
        logger.info("Saving performance report to report.txt...")
        report_path = os.path.join(self.output_dir, "report.txt")
        with open(report_path, 'w', encoding='utf-8') as rf:
            rf.write("PROJECT 1: HR EMPLOYEE ATTRITION PREDICTION REPORT\n")
            rf.write("==================================================\n")
            rf.write(f"Total Employees: {len(self.df):,}\n")
            rf.write(f"Baseline Attrition Rate: {self.df[self.target].mean():.2%}\n")
            rf.write(f"Class Distribution: Stay={len(self.df)-self.df[self.target].sum()} | Leave={self.df[self.target].sum()}\n")
            rf.write(f"\nModel Performance Metrics:\n")
            rf.write(f"  AUC-ROC: {self.auc_score:.4f}\n")
            rf.write(f"  Average Precision: {self.ap_score:.4f}\n")
            rf.write(f"\nClassification Report:\n")
            rf.write(classification_report(self.y_test, self.y_pred, target_names=['Stay', 'Leave']))
            
            if hasattr(self.model, 'feature_importances_'):
                rf.write(f"\nTop 10 Feature Importances:\n")
                fi = pd.DataFrame({'Feature': self.feature_cols, 'Importance': self.model.feature_importances_})
                fi = fi.sort_values('Importance', ascending=False).head(10)
                for idx, (index_val, row) in enumerate(fi.iterrows()):
                    rf.write(f"  {idx+1}. {row['Feature']}: {row['Importance']:.4f}\n")

    def run(self):
        self.load_and_explore_data()
        self.generate_eda_dashboard()
        self.preprocess_data()
        self.train_model()
        self.evaluate_model()
        self.generate_shap_explanations()
        self.generate_evaluation_dashboard()
        self.write_report()
        logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HR Attrition Prediction Pipeline")
    parser.add_argument("--data_path", type=str, default=r"D:\download\protfolio\archive\WA_Fn-UseC_-HR-Employee-Attrition.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--output_dir", type=str, default=r"D:\download\protfolio\projects\v3_output\project1_HR",
                        help="Directory to save pipeline outputs")
    
    args = parser.parse_args()
    
    pipeline = HRAttritionPipeline(data_path=args.data_path, output_dir=args.output_dir)
    pipeline.run()
