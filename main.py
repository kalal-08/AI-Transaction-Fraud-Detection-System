"""
AI-Based Transaction Fraud Detection System
A comprehensive fraud detection application with analytics dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import urllib.parse
import re
import os
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

PROJECT_DIR = Path(__file__).parent
DATASET_PATHS = [
    PROJECT_DIR / "creditcard.csv",
    PROJECT_DIR / "data.csv",
    PROJECT_DIR / "transactions.csv",
]
MODEL_PATHS = {
    'rf': PROJECT_DIR / "trained_credit_modelRF.sav",
    'lr': PROJECT_DIR / "trained_credit_card_model.sav",
    'svm': PROJECT_DIR / "trained_credit_modelSVM.sav",
    'knn': PROJECT_DIR / "trained_credit_modelKNN.sav",
}
FEEDBACK_FILE = PROJECT_DIR / "feedback.text"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_dataset():
    """Load the credit card dataset if it exists."""
    def _download_file(url, target_path):
        url = url.strip()
        # fix duplicated protocol mistakes like https://https://
        if url.startswith('https://https://'):
            url = url.replace('https://https://', 'https://', 1)
        if url.startswith('http://http://'):
            url = url.replace('http://http://', 'http://', 1)

        # support Google Drive share links
        if 'drive.google.com' in url:
            parsed = urllib.parse.urlparse(url)
            if '/file/d/' in parsed.path:
                file_id = parsed.path.split('/file/d/')[1].split('/')[0]
                url = f'https://drive.google.com/uc?export=download&id={file_id}'
            else:
                query_params = urllib.parse.parse_qs(parsed.query)
                if 'id' in query_params:
                    file_id = query_params['id'][0]
                    url = f'https://drive.google.com/uc?export=download&id={file_id}'

        session = requests.Session()
        resp = session.get(url, stream=True, timeout=60)

        def _is_html_response(response):
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                return True
            if 'application/json' in content_type:
                return True
            return False

        if 'drive.google.com' in url and _is_html_response(resp):
            text = resp.text
            action_match = re.search(r'<form[^>]+action="([^"]+)"', text)
            confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', text)
            id_match = re.search(r'name="id"\s+value="([^"]+)"', text)
            export_match = re.search(r'name="export"\s+value="([^"]+)"', text)
            if action_match and confirm_match and id_match and export_match:
                action_url = action_match.group(1)
                confirm_token = confirm_match.group(1)
                file_id = id_match.group(1)
                export_value = export_match.group(1)
                if action_url.startswith('/'):
                    action_url = urllib.parse.urljoin('https://drive.google.com', action_url)
                url = f'{action_url}?id={file_id}&export={export_value}&confirm={confirm_token}'
                resp = session.get(url, stream=True, timeout=60)

        resp.raise_for_status()

        with open(target_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    for path in DATASET_PATHS:
        if path.exists():
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                st.warning(f"Error loading dataset from {path}: {e}")
                return None

    # Not found locally — attempt to download from environment-provided URL
    # 1) Check environment variables (preferred for CI / Streamlit settings)
    data_url = os.environ.get('DATASET_URL') or os.environ.get('CREDITCARD_URL')

    # 2) If not set, check Streamlit `secrets` (TOML) which the UI shows.
    #    Users often paste values into the Secrets panel like:
    #      DATASET_URL = "https://..."
    #    or inside a section:
    #      [dataset]
    #      url = "..."
    try:
        secrets = getattr(st, 'secrets', {}) or {}
        if not data_url:
            # Top-level keys
            data_url = secrets.get('DATASET_URL') or secrets.get('CREDITCARD_URL')

        if not data_url:
            # Search nested sections for common keys
            for v in secrets.values():
                if isinstance(v, dict):
                    if 'DATASET_URL' in v:
                        data_url = v['DATASET_URL']
                        break
                    if 'CREDITCARD_URL' in v:
                        data_url = v['CREDITCARD_URL']
                        break
                    # common alt key names
                    if 'url' in v and isinstance(v['url'], str) and 'credit' in v.get('name', 'creditcard'):
                        data_url = v['url']
                        break
    except Exception:
        # If secrets are not accessible in this runtime, ignore and rely on env vars
        pass
    if data_url:
        target = PROJECT_DIR / 'creditcard.csv'
        try:
            print(f"DATASET_URL used: {data_url}")
            st.info('Dataset not found locally — downloading from provided URL...')
            _download_file(data_url, target)
            df = pd.read_csv(target)
            st.success('Dataset downloaded successfully.')
            return df
        except Exception as e:
            st.warning(f"Failed to download dataset from {data_url}: {e}")
            return None

    return None


def patch_monotonic_cst(estimator):
    """Apply compatibility patch for sklearn estimators loaded across sklearn versions."""
    if estimator is None:
        return
    if not hasattr(estimator, 'monotonic_cst'):
        setattr(estimator, 'monotonic_cst', None)
    if not hasattr(estimator, '_monotonic_cst'):
        setattr(estimator, '_monotonic_cst', None)


def fix_model_compatibility(model):
    """Fix cross-version sklearn model attributes after unpickling."""
    if model is None:
        return None

    patch_monotonic_cst(model)
    if hasattr(model, 'estimators_'):
        for estimator in model.estimators_:
            patch_monotonic_cst(estimator)

    return model


def is_model_proba_valid(model, dataset=None, feature_cols=None, n_checks=5):
    """Quick sanity check for model.predict_proba outputs.

    Returns True if multiple test inputs produce finite probabilities in [0,1].
    """
    if model is None or not hasattr(model, 'predict_proba'):
        return False

    # determine input dimension
    if feature_cols is not None:
        dim = len(feature_cols)
    else:
        dim = getattr(model, 'n_features_in_', None) or 30

    checks = []
    try:
        if dataset is not None and feature_cols is not None and len(dataset) > 0:
            num_rows = min(n_checks, len(dataset))
            for idx in range(num_rows):
                checks.append(dataset[feature_cols].iloc[idx].values.astype(float))
        else:
            checks.append(np.zeros(dim))
            checks.append(np.random.normal(size=(dim,)))

        for arr in checks[:n_checks]:
            arr = np.array(arr).reshape(1, -1)
            proba = model.predict_proba(arr)[0, 1]
            if not np.isfinite(proba) or proba < 0.0 or proba > 1.0:
                return False

        return True
    except Exception:
        return False


def load_model(dataset=None):
    """Load the trained model (Random Forest preferred, fallback to others)."""
    feature_cols = None
    if dataset is not None and 'Class' in dataset.columns:
        feature_cols = [col for col in dataset.columns if col != 'Class']
    elif dataset is not None:
        feature_cols = list(dataset.columns)

    candidates = [
        ('Random Forest', MODEL_PATHS['rf']),
        ('Logistic Regression', MODEL_PATHS['lr']),
        ('KNN', MODEL_PATHS['knn']),
        ('SVM', MODEL_PATHS['svm']),
    ]

    for name, path in candidates:
        if not path.exists():
            continue

        try:
            model = pickle.load(open(path, 'rb'))
            model = fix_model_compatibility(model)

            if feature_cols is not None and hasattr(model, 'predict_proba'):
                if not is_model_proba_valid(model, dataset, feature_cols):
                    st.warning(f"Model '{name}' returned invalid probabilities; trying next fallback.")
                    continue

            return model, name
        except Exception as e:
            st.warning(f"Error loading {name} model: {e}")

    return None, None


def get_risk_level(probability):
    """Determine risk level based on fraud probability."""
    if probability < 0.30:
        return "🟢 Low Risk", probability
    elif probability < 0.70:
        return "🟡 Medium Risk", probability
    else:
        return "🔴 High Risk", probability


def make_prediction(model, features):
    """Make a fraud prediction using the model."""
    try:
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        
        # Try to get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_array)[0]
            probability = proba[1]  # Fraud probability
        
        # Validate probability value (protect against corrupted/unexpected outputs)
        if probability is not None:
            try:
                probability = float(probability)
                if not (0.0 <= probability <= 1.0):
                    st.warning("Model returned invalid probability value; hiding probability.")
                    probability = None
            except Exception:
                probability = None

        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None


# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def show_home():
    """Home page with project overview."""
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🛡️ Transaction Fraud Detection System</div>
        <div class="hero-subtitle">AI-Powered Real-Time Fraud Detection with Advanced Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview section
    st.markdown("### Project Overview")
    st.write(
        "This advanced system uses machine learning algorithms to detect suspicious credit card "
        "transactions in real-time. It analyzes transaction patterns and uses PCA-transformed "
        "features to identify potential fraud with exceptional accuracy."
    )
    
    # System status - 3 column metrics
    dataset = load_dataset()
    model, loaded_from = load_model(dataset)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="kpi-label">Model Status</div>
                <div class="kpi-value">{'✅' if model else '❌'}</div>
                <div class="kpi-subtext">{'Loaded' if model else 'Not Found'}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="kpi-label">Dataset Status</div>
                <div class="kpi-value">{'✅' if dataset is not None else '❌'}</div>
                <div class="kpi-subtext">{'Available' if dataset is not None else 'Not Found'}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with col3:
        dataset_count = len(dataset) if dataset is not None else 0
        st.markdown(
            f"""<div class="metric-card">
                <div class="kpi-label">Transactions</div>
                <div class="kpi-value">{dataset_count:,}</div>
                <div class="kpi-subtext">Total records</div>
            </div>""",
            unsafe_allow_html=True
        )
    
    st.divider()
    
    # Key capabilities
    st.markdown("### 🚀 Core Capabilities")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <strong>📊 Analytics Dashboard</strong><br>
            Comprehensive transaction statistics, fraud patterns, and risk analysis
        </div>
        <div class="feature-card">
            <strong>🧪 Sample Prediction</strong><br>
            Test the model on real dataset samples with instant fraud probability
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <strong>✍️ Manual Prediction</strong><br>
            Input custom transaction features and get real-time detection results
        </div>
        <div class="feature-card">
            <strong>📈 Model Performance</strong><br>
            Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC analysis
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Technical details
    st.markdown("### 🔧 Technical Architecture")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Feature Engineering**
        - **V1-V28**: PCA-transformed anonymized features (privacy-protected)
        - **Time**: Seconds since first transaction in dataset
        - **Amount**: Transaction value in currency units
        - **Class**: Target label (0 = Genuine, 1 = Fraud)
        """)
    
    with col2:
        st.markdown("""
        **Model Architecture**
        - Multi-algorithm ensemble approach
        - Trained on real credit card fraud dataset (284K+ transactions)
        - Optimized for high recall (catching fraud is priority)
        - Automatic model selection with probability validation
        """)
    
    st.divider()
    
    # Model details
    st.markdown("### 📋 System Information")
    if model:
        st.success(f"✅ Active Model: {loaded_from}")
        st.info(f"💾 Total Dataset Size: {len(dataset):,} transactions loaded" if dataset is not None else "Dataset not available")
    else:
        st.error("❌ Model not loaded. Please check model files.")


def show_analytics_dashboard():
    """Analytics dashboard with transaction insights."""
    st.markdown("""
    <div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="hero-title">📊 Analytics Dashboard</div>
        <div class="hero-subtitle">Transaction Insights & Fraud Pattern Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    dataset = load_dataset()
    
    if dataset is None:
        st.warning("⚠️ Dataset CSV not found. Please add creditcard.csv to the project folder.")
        st.info("Expected columns: Time, V1-V28, Amount, Class")
        return
    
    # Calculate KPIs
    total_transactions = len(dataset)
    fraud_transactions = (dataset['Class'] == 1).sum() if 'Class' in dataset.columns else 0
    genuine_transactions = total_transactions - fraud_transactions
    fraud_rate = (fraud_transactions / total_transactions * 100) if total_transactions > 0 else 0
    avg_amount = dataset['Amount'].mean() if 'Amount' in dataset.columns else 0
    max_amount = dataset['Amount'].max() if 'Amount' in dataset.columns else 0
    
    # Display KPI cards with professional styling
    st.markdown("### 📈 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Transactions", f"{total_transactions:,}", help="Total number of transactions")
    with col2:
        st.metric("Fraud Cases", f"{fraud_transactions:,}", f"{fraud_rate:.2f}%", help="Fraudulent transactions detected")
    with col3:
        st.metric("Genuine", f"{genuine_transactions:,}", f"{100-fraud_rate:.2f}%", help="Legitimate transactions")
    with col4:
        st.metric("Avg Amount", f"${avg_amount:.2f}", help="Average transaction amount")
    with col5:
        st.metric("Max Amount", f"${max_amount:.2f}", help="Highest transaction value")
    
    st.divider()
    
    # Charts section
    st.markdown("### 📊 Transaction Analysis")
    
    if 'Class' in dataset.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Fraud vs Genuine Distribution**")
            class_counts = dataset['Class'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#2ecc71', '#e74c3c']
            bars = ax.bar(['Genuine', 'Fraud'], 
                   [class_counts.get(0, 0), class_counts.get(1, 0)],
                   color=colors, edgecolor='#333', linewidth=1.5)
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("**Fraud Rate Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sizes = [genuine_transactions, fraud_transactions]
            labels = [f'Genuine\n{genuine_transactions:,}', f'Fraud\n{fraud_transactions:,}']
            colors = ['#2ecc71', '#e74c3c']
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', 
                            colors=colors, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
            ax.axis('equal')
            plt.tight_layout()
            st.pyplot(fig)
    
    st.markdown("### 💰 Amount Analysis")
    
    if 'Amount' in dataset.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Overall Amount Distribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(dataset['Amount'], bins=50, color='#3498db', edgecolor='#333', alpha=0.8, linewidth=0.5)
            ax.set_xlabel('Amount ($)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            if 'Class' in dataset.columns:
                st.markdown("**Fraud vs Genuine Amount Distribution**")
                fig, ax = plt.subplots(figsize=(6, 4))
                genuine_amounts = dataset[dataset['Class'] == 0]['Amount']
                fraud_amounts = dataset[dataset['Class'] == 1]['Amount']
                ax.hist([genuine_amounts, fraud_amounts], 
                       bins=30, label=['Genuine', 'Fraud'], 
                       color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='#333', linewidth=0.5)
                ax.set_xlabel('Amount ($)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                plt.tight_layout()
                st.pyplot(fig)
    
    st.markdown("### ⏰ Temporal Analysis")
    
    if 'Time' in dataset.columns:
        st.markdown("**Transaction Count by Hour of Day**")
        dataset_copy = dataset.copy()
        dataset_copy['Hour'] = (dataset_copy['Time'] / 3600).astype(int) % 24
        hourly_counts = dataset_copy['Hour'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(14, 4))
        bars = ax.bar(hourly_counts.index, hourly_counts.values, color='#667eea', edgecolor='#333', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
        ax.set_ylabel('Transaction Count', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticks(range(0, 24, 2))
        plt.tight_layout()
        st.pyplot(fig)


def show_sample_prediction():
    """Sample transaction prediction from dataset."""
    st.markdown("""
    <div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="hero-title">🧪 Sample Transaction Prediction</div>
        <div class="hero-subtitle">Analyze Real Transactions from Dataset</div>
    </div>
    """, unsafe_allow_html=True)
    
    dataset = load_dataset()
    model, loaded_from = load_model(dataset)
    
    if dataset is None:
        st.error("❌ Dataset not found. Cannot load sample transactions.")
        st.info("Please ensure creditcard.csv is in the project directory.")
        return
    
    if model is None:
        st.error("❌ Model not found. Cannot make predictions.")
        st.info(f"Active Model: {loaded_from}")
        return
    
    # Feature columns (exclude Class if present)
    feature_cols = [col for col in dataset.columns if col != 'Class']
    
    st.markdown("### 📋 Transaction Browser")
    st.info("Use the slider to inspect any transaction and analyze it with the active model.")
    st.markdown("**Highly imbalanced dataset means fraud examples are rare — this preview helps you find interesting cases quickly.**")

    max_index = len(dataset) - 1
    sample_idx = st.slider("Transaction Index", 0, max_index, 0)

    transaction = dataset.iloc[sample_idx]

    # Display transaction details in professional cards
    st.markdown("### 📊 Transaction Details")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'Time' in transaction.index:
            st.metric("⏱️ Time (sec)", f"{transaction['Time']:.0f}")
    with col2:
        if 'Amount' in transaction.index:
            st.metric("💰 Amount", f"${transaction['Amount']:.2f}")
    with col3:
        if 'Class' in transaction.index:
            actual_class = "Genuine" if transaction['Class'] == 0 else "Fraud"
            actual_icon = "✅" if transaction['Class'] == 0 else "⚠️"
            st.metric(f"{actual_icon} Actual", actual_class)
    with col4:
        st.metric("📍 Index", f"{sample_idx:,}")

    with st.expander("View raw transaction features"):
        feature_table = transaction[feature_cols].to_frame().T
        st.dataframe(feature_table, use_container_width=True)

    st.divider()

    force_proba = st.checkbox("🔓 Show probabilities anyway (unsafe)", value=False, key="force_proba_sample", help="Override model validation if needed")

    # Make prediction
    if st.button("🔍 Analyze Transaction", use_container_width=True, key="btn_sample_pred"):
        with st.spinner("Analyzing transaction..."):
            features = transaction[feature_cols].values.astype(float)
            prediction, probability = make_prediction(model, features)

            proba_ok = is_model_proba_valid(model, dataset, feature_cols)
            show_proba = proba_ok or force_proba
            if force_proba and not proba_ok:
                st.warning("⚠️ Forcing display of probabilities despite model/version mismatch; results may be invalid.")

            if prediction is not None:
                st.divider()
                st.markdown("### 🎯 Prediction Results")
                col1, col2, col3 = st.columns(3)

                with col1:
                    pred_label = "Fraud" if prediction == 1 else "Genuine"
                    if prediction == 1:
                        st.error(f"🔴 **Prediction: {pred_label}**")
                    else:
                        st.success(f"🟢 **Prediction: {pred_label}**")

                with col2:
                    if probability is not None and show_proba:
                        st.metric("Fraud Probability", f"{probability:.2%}")
                    else:
                        st.warning("Probability unavailable")

                with col3:
                    if probability is not None and show_proba:
                        risk_level, _ = get_risk_level(probability)
                        st.info(f"**{risk_level}**")
                    else:
                        st.info("Only class prediction available")

                if prediction == 0:
                    st.balloons()


def show_manual_prediction():
    """Manual transaction input for fraud prediction."""
    st.markdown("""
    <div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="hero-title">✍️ Advanced Manual Input</div>
        <div class="hero-subtitle">Enter Transaction Features & Get Fraud Assessment</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **ℹ️ About This Tool:**
    - V1–V28 are anonymized PCA features for privacy protection
    - This section is for technical demonstration
    - Use **"Auto-fill with random sample"** to load real transaction data
    """)
    
    dataset = load_dataset()
    model, loaded_from = load_model(dataset)
    
    if model is None:
        st.error("❌ Model not found. Cannot make predictions.")
        return
    
    st.markdown("### 📝 Enter Transaction Details")
    
    # Auto-fill button first
    if dataset is not None:
        if st.button("📊 Auto-fill with Random Sample", use_container_width=True, key="autofill_btn"):
            sample = dataset.sample(1).iloc[0]
            st.session_state['time_seconds'] = float(sample['Time']) if 'Time' in sample.index else 0.0
            for i in range(1, 29):
                key = f"V{i}"
                if key in sample.index:
                    st.session_state[key] = float(sample[key])
            st.session_state['amount'] = float(sample['Amount']) if 'Amount' in sample.index else 0.0
            st.success("✅ Loaded random sample! Scroll down to review and predict.")
            st.rerun()
    
    st.divider()
    
    # Time and Amount in two columns
    col1, col2 = st.columns(2)
    with col1:
        time_seconds = st.number_input("⏱️ Time (seconds since first transaction)", 
                                       min_value=0.0, value=st.session_state.get('time_seconds', 0.0), step=1.0, key='time_seconds')
    with col2:
        amount = st.number_input("💰 Transaction Amount ($)", 
                                min_value=0.0, value=st.session_state.get('amount', 0.0), step=0.01, key='amount')
    
    st.divider()
    
    # V1-V28 features in a grid with better layout
    st.markdown("### 🔢 PCA Features (V1-V28)")
    st.markdown("*Use realistic values from the dataset if possible. Leave unchanged to keep defaults.*")
    
    v_features = []
    cols_per_row = 7
    for i in range(1, 29):
        col_idx = (i - 1) % cols_per_row
        if col_idx == 0:
            cols = st.columns(cols_per_row)
        key = f"V{i}"
        v_features.append(cols[col_idx].number_input(f"V{i}", value=st.session_state.get(key, 0.0), step=0.01, key=key, label_visibility="collapsed"))
    
    st.divider()
    
    # Checkbox outside button
    force_proba = st.checkbox("🔓 Show probabilities anyway (unsafe)", value=False, key="force_proba_manual", help="Override model validation if needed")
    
    # Make prediction
    if st.button("🚀 Predict Fraud Risk", use_container_width=True, key="btn_manual_pred"):
        with st.spinner("🔄 Analyzing transaction..."):
            features = [time_seconds] + v_features + [amount]
            prediction, probability = make_prediction(model, features)
            
            proba_ok = is_model_proba_valid(model, load_dataset(), [c for c in load_dataset().columns if c!='Class'])
            show_proba = proba_ok or force_proba
            if force_proba and not proba_ok:
                st.warning("⚠️ Forcing display of probabilities despite model/version mismatch; results may be invalid.")

            if prediction is not None:
                st.divider()
                st.markdown("### 🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)

                with col1:
                    pred_label = "FRAUD" if prediction == 1 else "GENUINE"
                    if prediction == 1:
                        st.error(f"🔴 **{pred_label}**", icon="🚨")
                    else:
                        st.success(f"🟢 **{pred_label}**", icon="✅")

                with col2:
                    if probability is not None and show_proba:
                        st.metric("Fraud Probability", f"{probability:.2%}")
                    else:
                        st.warning("Probability unavailable")

                with col3:
                    if probability is not None and show_proba:
                        risk_level, _ = get_risk_level(probability)
                        st.info(f"**{risk_level}**")
                    else:
                        st.info("Validation skipped")

                if prediction == 0:
                    st.balloons()


def show_model_performance():
    """Model performance metrics and evaluation."""
    st.markdown("""
    <div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="hero-title">📈 Model Performance Metrics</div>
        <div class="hero-subtitle">Comprehensive Evaluation on Full Dataset</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📊 Understanding Fraud Detection Metrics
    
    For fraud detection, different metrics have different importance:
    
    | Metric | Importance | What It Means |
    |--------|-----------|--------------|
    | **Recall** ⭐⭐⭐ | CRITICAL | Of all fraud cases, how many did we catch? (minimize missed fraud) |
    | **Precision** ⭐⭐ | High | Of cases we flagged as fraud, how many were actually fraud? (reduce false alarms) |
    | **F1-Score** ⭐⭐ | High | Balance between precision and recall |
    | **ROC-AUC** ⭐⭐⭐ | CRITICAL | Model's ability to distinguish between classes |
    | **Accuracy** ⭐ | Low | Overall correctness (less important due to class imbalance) |
    """)
    
    st.divider()
    
    dataset = load_dataset()
    model, loaded_from = load_model(dataset)
    
    if dataset is None or model is None:
        if dataset is None:
            st.warning("⚠️ Dataset not found.")
        if model is None:
            st.warning("⚠️ Model not found.")
        st.info("Cannot evaluate model without dataset and trained model.")
        return
    
    if 'Class' not in dataset.columns:
        st.warning("⚠️ Dataset does not contain 'Class' column for evaluation.")
        return
    
    try:
        # Prepare features and labels
        feature_cols = [col for col in dataset.columns if col != 'Class']
        X = dataset[feature_cols].values.astype(float)
        y = dataset['Class'].values
        
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        # Display metrics in a professional grid
        st.markdown("### 📊 Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("🔍 Precision", f"{precision:.2%}")
        with col3:
            st.metric("⚡ Recall", f"{recall:.2%}", delta="Most Important")
        with col4:
            st.metric("⚖️ F1-Score", f"{f1:.2%}")
        with col5:
            if y_pred_proba is not None:
                roc_auc = roc_auc_score(y, y_pred_proba)
                st.metric("📈 ROC-AUC", f"{roc_auc:.2%}", delta="Critical Metric")
        
        
        st.divider()
        
        # Confusion Matrix
        st.markdown("### 🎯 Confusion Matrix")
        col1, col2 = st.columns([1, 1])
        with col1:
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Genuine', 'Fraud'],
                       yticklabels=['Genuine', 'Fraud'],
                       ax=ax, cbar=True, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
            ax.set_ylabel('Actual', fontweight='bold')
            ax.set_xlabel('Predicted', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            cm = confusion_matrix(y, y_pred)
            tn, fp, fn, tp = cm.ravel()
            st.markdown("**Matrix Breakdown:**")
            st.metric("✅ True Negatives", f"{tn:,}", "Correct genuine")
            st.metric("❌ False Positives", f"{fp:,}", "Genuine flagged as fraud")
            st.metric("❌ False Negatives", f"{fn:,}", "Fraud missed")
            st.metric("✅ True Positives", f"{tp:,}", "Correct fraud detection")
        
        st.divider()
        
        # Classification Report
        st.markdown("### 📋 Classification Report")
        report = classification_report(y, y_pred, 
                                      target_names=['Genuine', 'Fraud'],
                                      output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error evaluating model: {e}")
        st.info("This may happen if features don't match between training and dataset.")


def show_about_dataset():
    """Information about the dataset and features."""
    st.markdown("""
    <div class="hero-section" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="hero-title">📚 About the Dataset</div>
        <div class="hero-subtitle">Understanding Credit Card Fraud Detection Data</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 📊 Credit Card Fraud Detection Dataset
    
    This application uses the **Credit Card Fraud Detection dataset** from Kaggle, originally 
    compiled by the **ULB (Université Libre de Bruxelles)**.
    
    ### 📋 Dataset Characteristics
    
    **Size & Scope:**
    - 💳 Contains credit card transactions from USA cardholders
    - 📅 Time period: September 2013 (28 days of data)
    - 📊 Total transactions: 284,807
    - 🔴 Fraudulent transactions: 492 (0.17%)
    - 🟢 Genuine transactions: 284,315 (99.83%)
    
    ### 🔢 Features Overview
    
    **V1 to V28** (Principal Component Analysis Features)
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **What are they?** PCA-transformed features from raw transaction data
        - **Why anonymized?** Protects customer privacy
        - **Values?** Standardized (centered around zero)
        """)
    with col2:
        st.markdown("""
        - **Purpose**: Dimensionality reduction for ML models
        - **Benefits**: Privacy + faster training
        - **Interpretation**: Cannot reverse-engineer merchant/location data
        """)
    
    st.markdown("""
    **Time**
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("⏱️ Seconds elapsed since first transaction in dataset")
    with col2:
        st.markdown("📈 Range: 0 to ~2.4 million seconds (28 days)")
    
    st.markdown("""
    **Amount**
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("💰 Transaction value in currency units (USD)")
    with col2:
        st.markdown("📊 Range: $0 to $25,691")
    
    st.markdown("""
    **Class** (Target Variable)
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("🟢 **0 = Genuine** transaction (99.83%)")
    with col2:
        st.markdown("🔴 **1 = Fraud** transaction (0.17%)")
    
    st.divider()
    
    st.markdown("""
    ### ⚙️ Dataset Limitations & Considerations
    
    **⚠️ Key Points:**
    - **Highly imbalanced**: ~99.8% genuine, ~0.2% fraud (extreme class imbalance)
    - **Privacy by Design**: PCA transformation prevents reverse-engineering of merchant/location data
    - **Feature Interpretation**: Individual V1-V28 features have no direct business meaning
    - **Time Zone Agnostic**: Dataset doesn't distinguish between time zones
    - **Geographic Blind**: No country/region information due to anonymization
    
    ### 💡 Why These Characteristics Matter
    
    1. **Privacy Protection** 🔒: Organizations can share transaction data safely
    2. **Model Generalization** 📈: Prevents overfitting to specific merchants/patterns
    3. **Statistical Challenge** ⚖️: Models must handle severe class imbalance
    4. **Interpretation Trade-off** 🎯: High privacy at cost of feature interpretability
    
    ### 🔬 Model Training Implications
    
    Due to dataset characteristics:
    - Standard accuracy metrics are misleading (99%+ accuracy by predicting all genuine)
    - **Recall** is prioritized to catch fraud (even if false positives increase)
    - **ROC-AUC** is key metric for model evaluation
    - Sampling techniques or class weights are used during training
    
    ---
    
    **📚 Source**: Kaggle - Credit Card Fraud Detection  
    **📖 Original Citation**: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. 
    "Calibrating Probability with Undersampling for Unbalanced Classification." 
    2015 IEEE Symposium Series on Computational Intelligence.
    """)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Fraud Detection System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom professional styling
    st.markdown("""
    <style>
    body {
        background-color: #090b16;
        color: #e6ecff;
        font-family: Inter, "Segoe UI", sans-serif;
    }
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    .stApp,
    .stAppViewContainer,
    .main,
    .block-container,
    .element-container {
        background-color: #090b16 !important;
        color: #e6ecff !important;
    }
    header.stAppHeader,
    .stAppHeader {
        background: linear-gradient(90deg, #111827 0%, #131c2d 100%) !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }
    .stSidebar {
        background: #0b1222 !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }
    .stAlert,
    .stAlertContainer {
        background: rgba(18, 28, 51, 0.94) !important;
        color: #e6ecff !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #5467ff 0%, #6c84ff 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 14px 30px rgba(84, 103, 255, 0.22) !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #6179ff 0%, #7c8dff 100%) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f4f7ff;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
        font-weight: 500;
        color: #bdd3ff;
    }
    .stMarkdown,
    .stMarkdown div,
    .stMarkdown p,
    .stMarkdown span {
        color: #e6ecff !important;
    }
    .stDivider {
        border-color: rgba(255,255,255,0.12) !important;
    }
    .css-1d391kg,
    .css-1lcbmhc,
    .css-1y4p8pa,
    .css-10trblm {
        background-color: #090b16 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🛡️ Navigation")
    st.sidebar.write("AI-Based Transaction Fraud Detection System")
    st.sidebar.divider()
    
    page = st.sidebar.radio(
        "Select a page:",
        ["Home", "Analytics Dashboard", "Sample Prediction", 
         "Manual Prediction", "Model Performance", "About Dataset"],
        index=0
    )
    
    st.sidebar.divider()
    st.sidebar.info("""
    **System Status**
    - Model files: Loaded from project directory
    - Dataset: Auto-detected if present
    - Contact: For issues or feedback, use the feedback section
    """)
    
    # Route to pages
    if page == "Home":
        show_home()
    elif page == "Analytics Dashboard":
        show_analytics_dashboard()
    elif page == "Sample Prediction":
        show_sample_prediction()
    elif page == "Manual Prediction":
        show_manual_prediction()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "About Dataset":
        show_about_dataset()
    
    # Footer with feedback
    st.divider()
    st.markdown("---")
    with st.expander("📝 Feedback"):
        feedback_text = st.text_area("Share your feedback here:")
        if st.button("Submit Feedback"):
            try:
                with open(FEEDBACK_FILE, "a") as f:
                    f.write(f"{feedback_text}\n")
                st.success("Thank you for your feedback!")
            except Exception as e:
                st.error(f"Error saving feedback: {e}")


if __name__ == "__main__":
    main()
