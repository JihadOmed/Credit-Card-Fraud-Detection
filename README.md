
# 💳 Credit Card Fraud Detection with Ensemble Learning & SHAP Explainability

This project aims to detect fraudulent credit card transactions using an ensemble of powerful machine learning models, advanced preprocessing techniques, and model explainability through SHAP values.

## 📌 Overview

Fraud detection is a critical problem with high class imbalance, where fraudulent transactions are extremely rare. This project tackles this challenge using:

- **SMOTE oversampling** to balance classes
- **Stratified K-Fold Cross Validation** for robust evaluation
- **Ensemble modeling** using XGBoost, LightGBM, and CatBoost
- **Stacking with Logistic Regression** as a meta-model
- **Threshold tuning** based on F1-score from precision-recall curve
- **Model explainability** using SHAP

## 📂 Dataset

- The dataset used is a standard credit card fraud detection dataset (CSV format).
- `Class` column indicates whether a transaction is fraudulent (`1`) or legitimate (`0`).
- `Amount` and `Time` columns are scaled for model input.

## 🔍 Key Techniques

| Component              | Details |
|------------------------|---------|
| Preprocessing          | Standard Scaling, Feature Engineering (`scaled_amount`, `scaled_time`) |
| Resampling             | SMOTE (Synthetic Minority Oversampling Technique) |
| Models Used            | XGBoost, LightGBM, CatBoost |
| Model Strategy         | 5-Fold Stratified Cross Validation with Ensemble Voting |
| Final Meta-model       | Logistic Regression |
| Explainability         | SHAP values with LGBM model |
| Evaluation Metrics     | Accuracy, Precision, Recall, F1-score, ROC AUC |
| Output Files           | `ensemble_models.pkl`, `meta_model.pkl`, `test_predictions.csv` |

## 📈 Model Performance

Average performance across 5 folds:

- **Accuracy**: ~0.99
- **Precision**: High (due to careful threshold tuning)
- **Recall**: Balanced to reduce false negatives
- **F1 Score**: Optimized via PR-curve
- **AUC**: Excellent separability between fraud and non-fraud

Final test set evaluation:

- All core metrics are printed after running the script.

## 📊 Dataset

The dataset used in this project is publicly available on Kaggle:

**[Credit Card Fraud Detection Dataset – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**

It contains anonymized credit card transactions labeled as fraudulent or legitimate. The dataset is highly imbalanced, which is handled during preprocessing.

## ▶️ How to Run

1. **Install Dependencies**  
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost shap tqdm joblib
