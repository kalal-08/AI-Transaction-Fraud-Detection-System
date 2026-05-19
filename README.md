# 🛡️ AI-Based Transaction Fraud Detection System

A comprehensive machine learning application that detects fraudulent credit card transactions using advanced analytics and multiple ML algorithms.

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 📋 Overview

This is a professional-grade credit card fraud detection system built with Streamlit. It features:

- **Multi-Model Support**: Logistic Regression, Random Forest, SVM, and K-Nearest Neighbors
- **Interactive Dashboard**: Real-time analytics and transaction visualization
- **Fraud Analytics**: Comprehensive metrics and performance evaluation
- **Sample & Manual Testing**: Demo prediction modes for validation
- **Privacy-Focused**: Uses PCA-transformed anonymized features

## ✨ Key Features

### 🏠 Home Page
- Project objective and overview
- System architecture explanation
- Model and dataset status indicators

### 📊 Analytics Dashboard
- **KPI Metrics**: Total transactions, fraud cases, fraud rate, average amounts
- **Visualizations**:
  - Fraud vs Genuine transaction counts
  - Transaction amount distributions
  - Fraud amount analysis
  - Transaction patterns by hour
- **Real-time Statistics**: Updated from the dataset

### 🧪 Sample Transaction Prediction
- Load transactions directly from the dataset
- View transaction details (Time, Amount, Class)
- Get instant fraud probability and risk assessment
- Compare predictions across multiple samples

### ✍️ Manual Prediction Demo
- Input custom transaction features (V1-V28, Time, Amount)
- Instant fraud detection with confidence scores
- Risk level classification (Low/Medium/High)
- Technical demonstration of model capabilities

### 📈 Model Performance
- Accuracy, Precision, Recall, F1-Score metrics
- ROC-AUC analysis
- Confusion Matrix visualization
- Classification Report with detailed metrics
- Explanation of fraud detection metrics

### ℹ️ About Dataset
- Dataset characteristics and source information
- Feature descriptions and interpretations
- Privacy considerations and anonymization details
- Model training implications
- Citations and references

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Step 1: Clone or Download the Project
```bash
cd AI-Transaction-Fraud-Detection-System
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Add Dataset (Optional)
Place your credit card fraud dataset as `creditcard.csv` in the project folder:
```
AI-Transaction-Fraud-Detection-System/
├── main.py
├── requirements.txt
├── README.md
├── creditcard.csv              # <- Place dataset here
├── trained_credit_modelRF.sav
├── trained_credit_card_model.sav
└── ...
```

**Dataset columns should be**: Time, V1, V2, ..., V28, Amount, Class

### Step 5: Run the Application
```bash
streamlit run main.py
```

The app will open in your default browser at `http://localhost:8501`

## 📊 Project Structure

```
AI-Transaction-Fraud-Detection-System/
├── main.py                          # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── Credit_Card_Fraud_Detection.ipynb    # Training notebook (Logistic Regression/SVM/KNN)
├── Credit_Card_Fraud_Detection_KNN.ipynb # Training notebook (KNN model)
├── trained_credit_card_model.sav    # Logistic Regression model
├── trained_credit_modelRF.sav       # Random Forest model
├── trained_credit_modelSVM.sav      # Support Vector Machine model
├── trained_credit_modelKNN.sav      # K-Nearest Neighbors model
├── feedback.text                    # User feedback storage
└── creditcard.csv                   # Dataset (if added)
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.1 | Web application framework |
| pandas | 2.0.3 | Data manipulation and analysis |
| numpy | 1.24.3 | Numerical computing |
| scikit-learn | 1.3.0 | Machine learning algorithms |
| matplotlib | 3.7.2 | Data visualization |
| seaborn | 0.12.2 | Statistical data visualization |

## 🔧 Configuration

### Model Selection
The app automatically loads models in this priority:
1. **Random Forest** (`trained_credit_modelRF.sav`) - Recommended
2. **Logistic Regression** (`trained_credit_card_model.sav`)
3. **SVM** (`trained_credit_modelSVM.sav`)
4. **KNN** (`trained_credit_modelKNN.sav`)

### Dataset Auto-Detection
The app looks for datasets in this order:
1. `creditcard.csv`
2. `data.csv`
3. `transactions.csv`

## 📊 Dataset Information

### Source
- **Dataset**: Credit Card Fraud Detection (Kaggle)
- **Time Period**: September 2013 (28 days)
- **Transactions**: ~284,000+
- **Fraud Cases**: ~492 (~0.17%)

### Features
- **V1 to V28**: PCA-transformed anonymized features
- **Time**: Seconds elapsed from first transaction
- **Amount**: Transaction amount in USD
- **Class**: Target (0=Genuine, 1=Fraud)

### Key Statistics
- Highly imbalanced dataset (~99.8% genuine)
- All features normalized and PCA-transformed
- Privacy-preserving design (no merchant/customer info)

## 🎯 How It Works

### 1. Data Flow
```
User Input/Dataset → Feature Extraction → Model Prediction → Risk Assessment → Display
```

### 2. Fraud Probability to Risk Level
```
Probability < 0.30    → 🟢 Low Risk
Probability 0.30-0.70 → 🟡 Medium Risk
Probability > 0.70    → 🔴 High Risk
```

### 3. Model Evaluation Metrics
```
Recall (Most Important)  → % of fraud cases detected
Precision               → % of fraud predictions that are correct
F1-Score                → Balance between Precision & Recall
Accuracy                → Overall correctness (⚠️ Less important due to imbalance)
ROC-AUC                 → Model's discrimination ability
```

## 🔍 Use Cases

### For Analysts
- Monitor fraud patterns in real-time
- Analyze transaction distributions
- Evaluate model performance metrics
- Identify high-risk time periods

### For ML Engineers
- Test different models
- Compare prediction outputs
- Validate feature engineering
- Assess model generalization

### For Business Users
- Quick fraud risk assessment
- Sample transaction validation
- Real-time transaction screening
- Performance monitoring

## ⚠️ Limitations & Considerations

1. **Dataset Limitation**: PCA-transformed features cannot show actual merchant/location
2. **Class Imbalance**: Dataset is highly imbalanced (fraud is rare)
3. **Generalization**: Models trained on specific time period (Sept 2013)
4. **Real-time Constraints**: Batch predictions only (no real-time API)
5. **Feature Interpretation**: Individual V1-V28 features don't have business meaning

## 🔒 Privacy & Security

- ✅ All transaction features are anonymized
- ✅ No customer PII (personally identifiable information)
- ✅ Merchant information is masked
- ✅ Location data is not stored
- ✅ Models work on transformed features only

## 📈 Performance Metrics

Typical model performance on test set:
- **Accuracy**: 99.5%+
- **Precision**: 80-90%
- **Recall**: 75-85%
- **F1-Score**: 0.80-0.88
- **ROC-AUC**: 0.95+

*Note: Exact metrics depend on train-test split and model configuration*

## 🛠️ Troubleshooting

### Issue: "Dataset CSV not found"
**Solution**: Place `creditcard.csv` in the project folder or rename your dataset to match expected filenames.

### Issue: "Model not found"
**Solution**: Ensure `.sav` files are in the project directory. Check file names match exactly.

### Issue: "Feature mismatch error"
**Solution**: Verify dataset columns match expected format (Time, V1-V28, Amount, Class).

### Issue: App runs slow
**Solution**: 
- Reduce dataset size for testing
- Use a subset of data for analytics
- Clear browser cache

### Issue: "No module named streamlit"
**Solution**: 
```bash
pip install -r requirements.txt
```

## 📚 References

1. Dal Pozzolo, Andrea, et al. "Calibrating Probability with Undersampling for Unbalanced Classification." IEEE Symposium Series on Computational Intelligence. 2015.
2. Scikit-learn Documentation: https://scikit-learn.org
3. Streamlit Documentation: https://docs.streamlit.io
4. Kaggle Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud

## 🤝 Contributing

Improvements and suggestions are welcome! Consider:
- Adding more visualization types
- Implementing real-time prediction API
- Adding data export features
- Improving model interpretability
- Supporting multiple datasets

## 📝 License

This project is provided as-is for educational and demonstration purposes.

## 👨‍💻 Author Notes

This application is designed to be:
- ✅ **Beginner-friendly**: Clear explanations and documentation
- ✅ **Production-ready**: Proper error handling and validation
- ✅ **CV-worthy**: Professional structure and code quality
- ✅ **Extensible**: Easy to add new features or models

## 📧 Feedback & Support

For feedback, issues, or suggestions:
1. Use the **Feedback** section in the app
2. Check the troubleshooting guide above
3. Review logs for error messages

---

**Last Updated**: May 2026  
**Version**: 2.0 (Enhanced Analytics & Multi-Page)  
**Status**: ✅ Production Ready
