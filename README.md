# UPI Fraud Detection

## V4.2 Machine Learning Fraud Risk Assessment

An end-to-end machine learning system for detecting potentially fraudulent UPI transactions using behavioral, transaction, merchant, IP risk, velocity, and temporal signals.

The project covers the complete workflow:

**Data → Feature Engineering → Imbalanced Classification → Temporal Validation → Threshold Optimization → Model Evaluation → Flask API → Interactive Web UI**

---

## 🚀 Project Overview

Fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a small fraction of total transactions.

This project develops a fraud detection pipeline designed around:

- Temporal train/validation/test separation
- Leakage prevention
- Class-imbalance handling
- Behavioral feature engineering
- Precision-Recall based model evaluation
- Business-oriented threshold selection
- External temporal testing
- Model calibration diagnostics
- Feature importance analysis
- REST API deployment
- Interactive browser-based fraud assessment

The final V4.2 pipeline selects the best model based on **validation PR-AUC**, rather than relying on accuracy alone.

---

## 📊 Dataset

The project uses transaction-level data containing:

- Transaction information
- Account characteristics
- KYC information
- Merchant risk
- IP risk
- Transaction velocity
- Failed transaction behavior
- Geographic distance
- Spending behavior
- Payment channel
- Device type
- Fraud labels

### Dataset Splits

| Dataset | Time Period | Records |
|---|---|---:|
| Training | 2023-01-01 → 2023-09-30 | 300,113 |
| External Test | 2023-10-01 → 2023-12-30 | 99,887 |

Fraud prevalence:

| Dataset | Fraud Rate |
|---|---:|
| Training | 1.62% |
| External Test | 1.99% |

The classes are highly imbalanced, making **PR-AUC, precision, recall, and F1** more informative than accuracy alone.

---

# 🧠 Machine Learning Pipeline

## 1. Data Preparation

The training dataset is separated into:

- Training data
- Validation data

The validation set is used for model selection and threshold optimization.

The later external test period is kept separate to evaluate temporal generalization.

---

## 2. Leakage Prevention

Several fields are intentionally excluded from model training.

### Post-Authorization Information

`post_auth_risk_score` is excluded because it represents information generated after authorization and would not be available when making a real-time transaction decision.

### Raw Identifiers

The following identifiers are excluded:

- `transaction_id`
- `customer_id`
- `merchant_id`

These identifiers can cause memorization or unrealistic generalization.

### Target

`is_fraud` is excluded from the prediction features.

### Raw Timestamp

`transaction_time` is transformed into temporal features rather than being supplied directly to the model.

---

# ⚙️ Feature Engineering

The V4.2 pipeline generates temporal, transaction, behavioral, velocity, geographic, and risk-derived features.

### Temporal Features

- Transaction hour
- Day of week
- Day
- Month
- Weekend indicator
- Business-hours indicator
- Late-night indicator

### Transaction Features

- Log transaction amount
- Amount-to-monthly-spend ratio
- Amount minus monthly spend
- Absolute amount deviation

### Risk Features

- Merchant × IP risk
- Maximum risk score
- Mean risk score
- International IP risk
- International merchant risk
- Behavioral risk composite

### Velocity Features

- Transactions in the last hour
- Transactions in the last 24 hours
- Transaction velocity ratio
- Transaction count difference
- International velocity
- Failed transaction ratio

### Geographic Features

- Log geographic distance from previous transaction

The final V4.2 pipeline uses **37 model features**.

---

# 🤖 Models Evaluated

Three supervised models are evaluated:

### Logistic Regression

A balanced logistic regression model provides a strong and interpretable baseline.

### Random Forest

A class-balanced Random Forest is used to capture nonlinear relationships and feature interactions.

### HistGradientBoosting

Histogram-based gradient boosting is trained using balanced sample weights to account for class imbalance.

---

# 🏆 Model Selection

Models are compared using validation **PR-AUC**.

| Model | Validation PR-AUC | Validation ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.3277 | 0.8437 |
| Random Forest | 0.2646 | 0.8197 |
| HistGradientBoosting | 0.3132 | 0.8398 |

### Selected Model

**Logistic Regression — Balanced**

Validation PR-AUC:

**0.3277**

The model is selected based on PR-AUC because fraud detection is highly imbalanced and the precision-recall trade-off is more informative than accuracy alone.

---

# 📈 External Temporal Evaluation

The selected model is evaluated on the later time period:

**2023-10-01 → 2023-12-30**

At the validation-selected 80%-recall operating threshold:

| Metric | External Test |
|---|---:|
| Precision | 0.0522 |
| Recall | 0.7978 |
| F1 | 0.0980 |
| ROC-AUC | 0.8413 |
| PR-AUC | 0.3411 |

This temporal evaluation provides a more realistic estimate of how the model behaves on transactions occurring after the training period.

---

# 🎯 Threshold Optimization

Rather than automatically using `0.5` as the classification threshold, the project evaluates thresholds using the validation data.

The selected operating point targets approximately **80% fraud recall**.

### Selected Operating Threshold

```text
0.475376
