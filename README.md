# 📉 Customer Churn Prediction

> End-to-end machine learning classification project predicting customer churn for a telecommunications company using real-world CRM data.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)


---

## 🧠 Overview

Retaining an existing customer is significantly cheaper than acquiring a new one. This project takes the role of a **Data Analyst at a telecom company** facing the problem of customer churn — clients leaving for competitors.

Using a raw CRM export containing customer demographics, contract details, and service usage, the goal is to answer one fundamental business question:

> **Will a given customer cancel their subscription in the near future?**

The project covers full data preprocessing, exploratory analysis, multi-model classification, threshold optimization, and actionable business recommendations.

---

## 📊 Models Implemented

| Model | Configuration |
|---|---|
| Logistic Regression (baseline) | Default parameters |
| Lasso (L1) | `penalty='l1'`, `solver='liblinear'` |
| Logistic Regression (Balanced) | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| Gradient Boosting | `n_estimators=100` |
| Extra Trees | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` adjusted for class imbalance |
| LightGBM | `class_weight='balanced'` |
| SVM | `kernel='rbf'`, `class_weight='balanced'` |

---

## 🔬 Methodology

**Data Preprocessing**
- Identification and repair of hidden missing values in `TotalCharges` (whitespace entries)
- Missing value imputation using median — robust to outliers in financial data
- Binary encoding for Yes/No columns and service-related features (`No internet service` → 0)
- Ordinal Encoding for `Contract` and `InternetService` (order matters)
- One-Hot Encoding for `PaymentMethod` and `gender`
- Feature removal: `customerID` (non-informative), `InternetService` (correlation 0.91 with `MonthlyCharges`), `TotalCharges` (correlation 0.83 with `tenure`)

**Class Imbalance Handling**
- Dataset distribution: **73% No Churn / 27% Churn**
- Applied `class_weight='balanced'` and `scale_pos_weight` across models
- Custom threshold optimization with FP ratio constraint

**Evaluation**
- 5-Fold Cross-Validation (`cross_validate`)
- Custom decision threshold scorer (threshold = 0.30)

**Key Metric: Recall**
> Missing a customer who is about to leave (False Negative) is far more costly than a false alarm (False Positive). The company can offer a retention discount to a misclassified loyal customer, but cannot recover a customer who has already left.

---

## 📈 Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall correct classification rate |
| **Recall** | % of churning customers correctly identified ← primary metric |
| Precision | % of churn predictions that are correct |
| F1 Score | Harmonic mean of Precision and Recall |

---

## 📁 Project Structure

```
customer-churn-prediction/
├── main.ipynb       # Main analysis notebook
└── README.md        # Project documentation
```

---

## 🛠️ Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install xgboost lightgbm
```

---

## 🚀 Getting Started

1. Clone the repository
2. Install the required dependencies (see above)
3. Open `main.ipynb` in Jupyter Notebook or VS Code
4. Run the cells sequentially to reproduce the analysis

Data source:
```python
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
```

---

## 📉 Visualizations

The project includes:
- Class distribution (Churn vs No Churn)
- Correlation heatmap with values
- Confusion Matrix with TP / TN / FP / FN labels
- Feature importance (Logistic Regression coefficients)
- Model comparison table with highlighted best scores

---

## 💡 Key Business Findings

- Customers on **Month-to-month contracts** show the highest churn risk
- **Fiber optic** internet service users churn more than DSL users
- High **MonthlyCharges** combined with short **tenure** strongly predicts churn

---

## 🎯 Business Recommendations

- [ ] Target Month-to-month customers with long-term contract offers
- [ ] Monitor new customers closely — short tenure = highest churn risk
- [ ] Deploy model with threshold 0.30–0.51 depending on retention budget
- [ ] Proactively offer loyalty discounts before customers decide to leave

---

## 🧰 Tech Stack

`pandas` `numpy` `scikit-learn` `xgboost` `lightgbm` `matplotlib` `seaborn`

---

*End-to-end churn prediction project — from raw CRM data to actionable business insights.*
