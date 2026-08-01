import streamlit as st
import os
import re
import subprocess
import time
import pandas as pd
import numpy as np

# Set page layout
st.set_page_config(
    page_title="Data Science Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark glassmorphism styling
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background-color: #0f111a;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #171923;
        border-right: 1px solid #2d3748;
    }
    
    /* Glowing Title header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3182ce, #319795, #805ad5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
        text-shadow: 0 0 30px rgba(49, 130, 206, 0.2);
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    /* Card design */
    .kpi-card {
        background: rgba(26, 32, 44, 0.6);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s, border-color 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: #4a5568;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #319795;
        margin-top: 0.5rem;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Code area terminal styling */
    .stCodeBlock, code {
        background-color: #1a202c !important;
        border: 1px solid #2d3748 !important;
        border-radius: 8px !important;
    }
    
    /* Buttons styling */
    .stButton>button {
        background: linear-gradient(135deg, #3182ce, #2b6cb0) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(49, 130, 206, 0.3) !important;
        transition: all 0.2s !important;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 20px rgba(49, 130, 206, 0.5) !important;
    }
    
    /* Section dividers */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #2d3748, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Project Config
PROJECTS = {
    "1": {
        "title": "1. HR Employee Attrition Prediction",
        "script": "projects/v3_project1_hr_attrition.py",
        "out_dir": "projects/v3_output/project1_HR",
        "description": "Predicts employee attrition using machine learning classifiers and handles class imbalance using SMOTE. Includes hyperparameter tuning with Optuna and model interpretability with SHAP.",
        "techniques": ["SMOTE Over-sampling", "XGBoost Classifier", "Optuna Hyperparameter Tuning", "SHAP Feature Interpretability"],
        "has_shap": True,
        "interactive_html": "eda_dashboard.html",
        "static_png": "evaluation_metrics.png",
        "shap_png": "shap_summary.png"
    },
    "2": {
        "title": "2. Twitter Sentiment Analysis",
        "script": "projects/v3_project2_sentiment.py",
        "out_dir": "projects/v3_output/project2_Sentiment",
        "description": "Classifies sentiment of 1.6 million Twitter messages using natural language processing (TF-IDF vectorizer) and classification algorithms. Analyzes key words and checks correlations with URLs/emojis.",
        "techniques": ["TF-IDF Text Vectorization", "LightGBM / Logistic Regression", "NLP Preprocessing", "Bayesian Sentiment Probabilities"],
        "has_shap": False,
        "interactive_html": "eda_dashboard.html",
        "static_png": "evaluation_metrics.png"
    },
    "3": {
        "title": "3. Audit & Security Risk Analysis",
        "script": "projects/v3_project3_audit.py",
        "out_dir": "projects/v3_output/project3_Audit",
        "description": "Performs risk assessment on financial audits. Integrates Excel sheets data, applies dimensionality reduction (PCA) on security controls, clusters profiles with K-Means, and catches anomalies.",
        "techniques": ["Principal Component Analysis (PCA)", "K-Means Clustering", "Isolation Forest Anomaly Detection", "Bayesian Risk Inference"],
        "has_shap": False,
        "interactive_html": "Interactive_Dashboard.html",
        "static_png": "Advanced_Dashboard.png"
    },
    "4": {
        "title": "4. Financial Accounting Analysis",
        "script": "projects/v3_project4_financial.py",
        "out_dir": "projects/v3_output/project4_Financial",
        "description": "Analyzes debit/credit transactions, flags anomalous transaction days, and generates a 12-week forecast of cash volumes using Holt-Winters Exponential Smoothing.",
        "techniques": ["Holt-Winters Exponential Smoothing", "Isolation Forest Anomaly Detection", "Bayesian Probability", "Time Series Forecasting"],
        "has_shap": False,
        "interactive_html": "Interactive_Dashboard.html",
        "static_png": "Advanced_Dashboard.png"
    },
    "5": {
        "title": "5. Corporate Bankruptcy Prediction",
        "script": "projects/v3_project5_bankruptcy.py",
        "out_dir": "projects/v3_output/project5_Bankruptcy",
        "description": "Analyzes financial ratios of companies to predict bankruptcy risk. Utilizes XGBoost on SMOTE-balanced data, optimized via Optuna, and explains risk contributions with SHAP values.",
        "techniques": ["SMOTE Over-sampling", "XGBoost Classifier", "SHAP Medical/Corporate Explainability", "Bayesian Bankruptcy Inference"],
        "has_shap": False,
        "interactive_html": "Interactive_Dashboard.html",
        "static_png": "Advanced_Dashboard.png"
    },
    "6": {
        "title": "6. Foreign Exchange rates Forecasting",
        "script": "projects/v3_project6_forex.py",
        "out_dir": "projects/v3_output/project6_Forex",
        "description": "Forecasts major currency pairs against USD. Calculates financial indicators (RSI, Bollinger Bands, MACD) and generates a 30-day forecast with ARIMA models.",
        "techniques": ["ARIMA Time Series Forecasting", "Bollinger Bands, RSI, MACD calculation", "Correlation Analysis", "Bayesian Threshold Probability"],
        "has_shap": False,
        "interactive_html": "Interactive_Dashboard.html",
        "static_png": "Advanced_Dashboard.png"
    },
    "7": {
        "title": "7. Customer Churn Prediction",
        "script": "projects/v3_project7_churn.py",
        "out_dir": "projects/v3_output/project7_Churn",
        "description": "Predicts customer churn for telecom accounts. Uses SMOTE imbalance treatment, trains XGBoost, evaluates with AUPRC metrics, and highlights drivers with SHAP explainability.",
        "techniques": ["SMOTE Over-sampling", "XGBoost Classifier", "Precision-Recall AUC (AUPRC)", "SHAP Feature Interpretability"],
        "has_shap": True,
        "interactive_html": "Interactive_Dashboard.html",
        "static_png": "Advanced_Dashboard.png",
        "shap_png": "SHAP_Summary.png"
    },
    "8": {
        "title": "8. Credit Card Fraud Detection",
        "script": "projects/v3_project8_fraud.py",
        "out_dir": "projects/v3_output/project8_Fraud",
        "description": "Handles extreme class imbalance (0.5% fraud rate) to detect fraudulent credit card charges using unsupervised anomaly isolation and cost-sensitive supervised random forests.",
        "techniques": ["Isolation Forest Anomaly Detection", "Cost-Sensitive Random Forest", "Precision-Recall Curve Optimization", "Unsupervised & Supervised Hybrid model"],
        "has_shap": False,
        "interactive_html": "Interactive_Dashboard.html",
        "static_png": "Advanced_Dashboard.png"
    },
    "9": {
        "title": "9. Healthcare Analytics (Breast Cancer Diagnosis)",
        "script": "projects/v3_project9_healthcare.py",
        "out_dir": "projects/v3_output/project9_Healthcare",
        "description": "Classifies breast cancer malignancies from clinical biopsy parameters. Features PCA dimensionality profiling, LightGBM classification, SHAP medical explains, and Bayesian risk estimations.",
        "techniques": ["Principal Component Analysis (PCA)", "LightGBM Classifier", "SHAP Medical Explainability", "Bayesian Malignancy Probability"],
        "has_shap": True,
        "interactive_html": "Interactive_Dashboard.html",
        "static_png": "Advanced_Dashboard.png",
        "shap_png": "SHAP_Summary.png"
    }
}

# Python path configuration (using Anaconda env)
PYTHON_EXE = r"D:\anaconda\python.exe"

def parse_metrics(report_path):
    metrics = {"AUC-ROC": "N/A", "Avg Precision": "N/A", "Samples": "N/A"}
    if not os.path.exists(report_path):
        return metrics
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Samples
        samples_match = re.search(r'(Total Employees|Total sampled Tweets|Total audits|Total Transactions|Total Companies analyzed|Total Customers|Total Patients):\s*([\d,]+)', content, re.IGNORECASE)
        if samples_match:
            metrics["Samples"] = samples_match.group(2)
            
        # AUC-ROC
        auc_match = re.search(r'(AUC-ROC|Clinical AUC-ROC|Diagnostic AUC-ROC|Diagnostic AUC|AUC):\s*([\d\.]+)', content, re.IGNORECASE)
        if not auc_match:
            auc_match = re.search(r'AUC\s*=\s*([\d\.]+)', content, re.IGNORECASE)
        if auc_match:
            metrics["AUC-ROC"] = f"{float(auc_match.group(2)):.3f}"
            
        # AP
        ap_match = re.search(r'(Average Precision|Avg Precision|Clinical Avg Precision|Diagnostic Average Precision|AP):\s*([\d\.]+)', content, re.IGNORECASE)
        if not ap_match:
            ap_match = re.search(r'AP\s*=\s*([\d\.]+)', content, re.IGNORECASE)
        if ap_match:
            metrics["Avg Precision"] = f"{float(ap_match.group(2)):.3f}"
    except Exception as e:
        pass
    return metrics

# Dashboard Title
st.markdown('<div class="main-header">محفظة مشاريع تحليل البيانات وتعلم الآلة</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Data Science, Analytics & Machine Learning Pipeline Dashboard</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 📊 Project Browser")
project_id = st.sidebar.selectbox(
    "Select a Pipeline Project:",
    options=list(PROJECTS.keys()),
    format_func=lambda x: PROJECTS[x]["title"]
)

project = PROJECTS[project_id]

# Sidebar Metadata
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Selected Script:**\n`{project['script']}`")
st.sidebar.markdown(f"**Output Directory:**\n`{project['out_dir']}`")

# Main Page Layout
col_main, col_sidebar_info = st.columns([3, 1])

with col_main:
    st.markdown(f"## {project['title']}")
    st.write(project["description"])
    
    # Techniques tags
    st.markdown("**Key Data Science Techniques Applied:**")
    tech_tags = " ".join([f'<span style="background-color: #2d3748; padding: 0.3rem 0.6rem; margin-right: 0.5rem; border-radius: 4px; font-size: 0.85rem; border: 1px solid #4a5568;">{t}</span>' for t in project["techniques"]])
    st.markdown(tech_tags, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

with col_sidebar_info:
    # Execution panel
    st.markdown("### ⚙️ Pipeline Control")
    run_btn = st.button("🚀 Run Pipeline")
    
    # Check status
    report_file_path = os.path.join(project["out_dir"], "report.txt")
    status_color = "green" if os.path.exists(report_file_path) else "orange"
    status_label = "Outputs Ready" if os.path.exists(report_file_path) else "Not Executed"
    st.markdown(f"**Status:** <span style='color: {status_color}; font-weight: bold;'>● {status_label}</span>", unsafe_allow_html=True)

# Live runner console
if run_btn:
    st.markdown("### 🖥️ Real-time Execution Console")
    terminal_placeholder = st.empty()
    terminal_text = ""
    
    # Run process
    process = subprocess.Popen(
        [PYTHON_EXE, project["script"]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream output
    for line in process.stdout:
        terminal_text += line
        terminal_placeholder.code(terminal_text, language="bash")
        time.sleep(0.01)
        
    process.wait()
    if process.returncode == 0:
        st.success("✅ Pipeline executed successfully! Refreshing dashboard...")
        time.sleep(1)
        st.rerun()
    else:
        st.error(f"❌ Execution failed with return code {process.returncode}")

# Parse and display metrics
metrics = parse_metrics(report_file_path)

st.markdown("### 📈 Key Performance Indicators")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Samples Analyzed</div>
        <div class="kpi-value">{metrics['Samples']}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">ROC AUC Score</div>
        <div class="kpi-value">{metrics['AUC-ROC']}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Average Precision</div>
        <div class="kpi-value">{metrics['Avg Precision']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Visualization and reports tabs
st.markdown("### 📂 Analysis Artifacts")
tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 Interactive Plotly Dashboard", 
    "🖼️ Static Analytics Dashboard", 
    "🧠 Model Interpretability (SHAP)", 
    "📄 Executive Report (report.txt)"
])

with tab1:
    html_path = os.path.join(project["out_dir"], project["interactive_html"])
    if os.path.exists(html_path):
        st.markdown(f"**Embedding interactive Plotly dashboard:** `{project['interactive_html']}`")
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=800, scrolling=True)
        except Exception as e:
            st.error(f"Failed to read HTML dashboard: {e}")
    else:
        st.warning("⚠️ Interactive dashboard file not found. Click 'Run Pipeline' to generate it.")

with tab2:
    png_path = os.path.join(project["out_dir"], project["static_png"])
    if os.path.exists(png_path):
        st.markdown(f"**Matplotlib / Seaborn Master Dashboard:** `{project['static_png']}`")
        st.image(png_path, use_container_width=True)
    else:
        st.warning("⚠️ Static dashboard file not found. Click 'Run Pipeline' to generate it.")

with tab3:
    if project.get("has_shap", False):
        shap_path = os.path.join(project["out_dir"], project["shap_png"])
        if os.path.exists(shap_path):
            st.markdown(f"**SHAP Global Feature Impact Plot:** `{project['shap_png']}`")
            st.image(shap_path, use_container_width=True)
        else:
            st.warning("⚠️ SHAP plot file not found. Run the pipeline to generate it.")
    else:
        st.info("ℹ️ SHAP interpretability is not configured for this project, or its features are integrated inside the static master dashboard.")

with tab4:
    if os.path.exists(report_file_path):
        st.markdown(f"**Text Report Summary:** `{report_file_path}`")
        try:
            with open(report_file_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            st.code(report_content, language="text")
        except Exception as e:
            st.error(f"Failed to read report: {e}")
    else:
        st.warning("⚠️ report.txt not found. Click 'Run Pipeline' to execute the script and write the report.")
