# UPI Fraud Detection

A leakage-aware machine learning pipeline for detecting fraudulent UPI transactions using behavioral, transaction, device, geographic, and risk-related signals.

The project is designed as an end-to-end fraud detection workflow with **chronological validation, temporal external testing, class-imbalance handling, threshold optimization, business-oriented risk tiers, top-K fraud review analysis, feature ablation, permutation importance, and calibration diagnostics**.

---

## Project Overview

UPI fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a small fraction of total transactions.

This project focuses not only on predicting fraud, but also on evaluating how the model could support different operational objectives:

* Prioritizing high-risk transactions for investigation
* Achieving a target fraud recall
* Selecting operating thresholds based on precision/recall trade-offs
* Assigning transactions to low, medium, and high-risk tiers
* Supporting limited-capacity manual fraud review
* Understanding which behavioral signals contribute most to predictions

A key focus of the project is **preventing data leakage** and evaluating the model on a temporally separated external test set.

---

## Key Results

The final V4.2 pipeline selected **balanced Logistic Regression** based on validation PR-AUC.

### External Temporal Test

| Metric    |     Result |
| --------- | ---------: |
| PR-AUC    | **0.3411** |
| ROC-AUC   | **0.8413** |
| Precision | **0.0522** |
| Recall    | **0.7978** |
| F1-score  | **0.0980** |

The external test uses transactions from a later time period than the model-training data, providing a temporal evaluation of model performance.

### F1-Optimal Operating Point

At the threshold selected using the validation set:

| Metric    | External Test |
| --------- | ------------: |
| Precision |    **0.4504** |
| Recall    |    **0.3364** |
| F1-score  |    **0.3851** |

This demonstrates the trade-off between a high-recall fraud-screening strategy and a higher-precision investigation strategy.

---

## Dataset

The dataset contains transaction-level UPI information including:

* Account age
* Credit score band
* KYC level
* Average monthly spending
* Merchant risk score
* Transaction amount
* Payment channel
* Device type
* International transaction indicator
* IP risk score
* Transaction velocity
* Failed transaction counts
* Geographic distance
* Amount deviation from the user's historical mean
* Fraud label

### Dataset Size

| Dataset               | Transactions |
| --------------------- | -----------: |
| Training dataset      |      300,113 |
| External test dataset |       99,887 |

### Fraud Distribution

**Training data**

* Legitimate: **295,242**
* Fraudulent: **4,871**
* Fraud rate: **1.6231%**

**External temporal test**

* Legitimate: **97,904**
* Fraudulent: **1,983**
* Fraud rate: **1.9852%**

---

## Temporal Validation Strategy

Instead of randomly splitting transactions, the project uses a chronological evaluation strategy.

### Timeline

```text
2023-01-01 ───────────────────── 2023-09-30
                  │
                  ├── Model Training
                  │
                  └── Validation

2023-10-01 ───────────────────── 2023-12-30
                  │
                  └── External Temporal Test
```

The training dataset is divided chronologically into:

* Model training: **247,293 transactions**
* Validation: **52,820 transactions**

The separate external test contains **99,887 transactions** from a later time period.

This setup is intended to better represent a production scenario where a model is trained on historical transactions and evaluated on future transactions.

---

## Data Leakage Prevention

Fraud detection models can easily produce misleadingly high scores if information unavailable at transaction-decision time is accidentally included.

The pipeline explicitly excludes:

* `is_fraud` — target variable
* `transaction_id` — raw transaction identifier
* `customer_id` — raw customer identifier
* `merchant_id` — raw merchant identifier
* `post_auth_risk_score` — post-authorization information

The `post_auth_risk_score` feature was specifically excluded because it represents information generated after authorization and would not be appropriate for a real-time fraud decision.

The project therefore evaluates the model using features intended to be available during transaction processing.

---

## Feature Engineering

The final pipeline uses **37 model features**.

### Transaction & Account Features

* `account_age_days`
* `avg_monthly_spend`
* `transaction_amount`
* `credit_score_band`
* `kyc_level`

### Risk Features

* `merchant_risk_score`
* `ip_risk_score`
* `combined_merchant_ip_risk`
* `max_risk_score`
* `mean_risk_score`
* `international_ip_risk`
* `international_merchant_risk`

### Behavioral Features

* `txn_count_1h`
* `txn_count_24h`
* `failed_txn_count_24h`
* `txn_velocity_ratio`
* `txn_count_difference`
* `international_velocity`
* `failed_txn_ratio`
* `behavioral_risk_composite`

### Amount Features

* `log_transaction_amount`
* `amount_to_monthly_spend`
* `amount_minus_monthly_spend`
* `abs_amount_deviation`
* `amount_deviation_from_user_mean`

### Time Features

* `transaction_hour`
* `transaction_dayofweek`
* `transaction_day`
* `transaction_month`
* `is_weekend`
* `is_business_hours`
* `is_late_night`

### Geographic Feature

* `geo_distance_from_last_txn`
* `log_geo_distance`

### Categorical Features

* `payment_channel`
* `device_type`

---

## Models Evaluated

Three classification approaches were evaluated:

1. Logistic Regression with balanced class weights
2. Random Forest with balanced class weights
3. HistGradientBoosting with balanced sample weights

Because fraud is highly imbalanced, class weighting/sample weighting was used to make the learning process more sensitive to fraudulent transactions.

### Validation Model Comparison

| Model                   |     PR-AUC |    ROC-AUC |  Precision |     Recall |         F1 |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression** | **0.3277** | **0.8437** | **0.0533** | **0.8002** | **0.0999** |
| HistGradientBoosting    |     0.3055 |     0.8413 |     0.0508 |     0.8002 |     0.0955 |
| Random Forest           |     0.2377 |     0.8035 |     0.0363 |     0.8002 |     0.0695 |

**Logistic Regression was selected because it achieved the highest validation PR-AUC.**

---

## Why PR-AUC?

Fraud is a rare class in this dataset, with approximately 1.6–2.0% of transactions being fraudulent.

For highly imbalanced classification problems, accuracy can be misleading.

A model that predicts every transaction as legitimate would achieve very high accuracy while completely failing to detect fraud.

Therefore, this project emphasizes:

* Precision
* Recall
* F1-score
* Precision-Recall AUC

ROC-AUC is also reported, but PR-AUC is used as the primary model-selection metric.

---

## Threshold Optimization

The model produces fraud scores rather than simply predicting fraud/non-fraud using the default threshold.

Different thresholds support different operational objectives.

### 80% Recall Operating Point

The validation set was used to identify a threshold targeting approximately 80% recall.

**Threshold: `0.475376`**

External temporal-test performance:

| Metric    |     Result |
| --------- | ---------: |
| Precision | **0.0522** |
| Recall    | **0.7978** |
| F1-score  | **0.0980** |
| PR-AUC    | **0.3411** |
| ROC-AUC   | **0.8413** |

This operating point prioritizes finding fraudulent transactions, but results in a relatively high number of false positives.

### F1-Optimal Threshold

**Threshold: `0.875090`**

External temporal-test performance:

| Metric    |     Result |
| --------- | ---------: |
| Precision | **0.4504** |
| Recall    | **0.3364** |
| F1-score  | **0.3851** |

This provides substantially higher precision at the cost of lower recall.

---

## Business Risk Tiers

The validation precision-recall relationship was also used to establish data-driven risk thresholds.

```text
LOW
Probability < 0.757239

MEDIUM
0.757239 <= Probability < 0.859161

HIGH
Probability >= 0.859161
```

### External Test Risk Distribution

| Risk Tier | Transactions | Fraud | Fraud Rate |
| --------- | -----------: | ----: | ---------: |
| Low       |       93,798 |   829 |    0.8838% |
| Medium    |        4,152 |   395 |    9.5135% |
| High      |        1,937 |   759 |   39.1843% |

The high-risk tier contains a substantially higher concentration of fraudulent transactions than the overall external-test fraud rate.

---

## Top-K Fraud Review Analysis

In many fraud detection systems, investigators cannot manually review every transaction.

The model can instead rank transactions by fraud score and send the highest-risk transactions for review.

### External Temporal Test

| Review Capacity |  Precision | Recall |
| --------------- | ---------: | -----: |
| Top 1%          | **54.45%** | 27.43% |
| Top 2%          |     38.34% | 38.63% |
| Top 5%          |     21.94% | 55.27% |
| Top 10%         |     12.97% | 65.36% |
| Top 20%         |      7.39% | 74.43% |

For example, reviewing only the highest-risk **1%** of transactions captured approximately **27.4% of all fraudulent transactions**, with a precision of **54.45%** on the external temporal test.

---

## Feature Importance

Permutation importance was calculated on the validation data to identify features that most influenced model performance.

### Top Features

| Feature                           | Importance |
| --------------------------------- | ---------: |
| `abs_amount_deviation`            |     0.2787 |
| `txn_count_24h`                   |     0.2413 |
| `txn_count_difference`            |     0.1876 |
| `transaction_amount`              |     0.1654 |
| `amount_deviation_from_user_mean` |     0.1623 |
| `failed_txn_count_24h`            |     0.1325 |
| `txn_velocity_ratio`              |     0.0913 |

The results indicate that **transaction behavior, amount deviation, and transaction velocity** are particularly important signals for fraud detection in this dataset.

---

## Feature Ablation

Feature groups were removed and the model was retrained to evaluate their contribution.

| Experiment                     | Features Remaining |     PR-AUC |
| ------------------------------ | -----------------: | ---------: |
| All features                   |                 37 | **0.3277** |
| Without `txn_count_24h`        |                 36 |     0.3277 |
| Without risk features          |                 29 |     0.3241 |
| Without `failed_txn_count_24h` |                 36 |     0.2848 |
| Without velocity features      |                 32 | **0.1359** |

Removing the velocity-related features produced the largest performance degradation.

This supports the importance of transaction-frequency and behavioral velocity signals in the final model.

---

## Calibration Analysis

The pipeline also evaluates probability calibration.

The validation analysis produced a Brier score of:

**`0.171047`**

The predicted scores were not well calibrated to observed fraud frequencies.

Therefore, the model's output should primarily be interpreted as a **risk-ranking score** rather than a directly calibrated probability of fraud.

A future improvement would be to apply post-training calibration using a temporally appropriate calibration set.

---

## Confusion Matrix — External Test

At the validation-selected 80%-recall threshold:

```text
[[69169 28735]
 [  401  1582]]
```

This corresponds to:

* True Negatives: **69,169**
* False Positives: **28,735**
* False Negatives: **401**
* True Positives: **1,582**

The large number of false positives demonstrates the practical trade-off involved in targeting high fraud recall in a severely imbalanced environment.

---

## Project Structure

```text
upi-fraud-detection/
│
├── data/
│   └── raw/
│
├── outputs/
│   ├── models/
│   ├── reports/
│   └── plots/
│
├── src/
│   └── upi_fraud_detection.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Raw transaction data and local virtual-environment files are intentionally excluded from version control.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/RISHIKARTHIGAYAN/upi-fraud-detection.git
cd upi-fraud-detection
```

Create a virtual environment:

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Place the required dataset files inside:

```text
data/raw/
```

Expected files:

```text
data/raw/transactions_train.csv
data/raw/transactions_test.csv
```

The raw datasets are not included in this repository.

---

## Running the Pipeline

From the project root:

```bash
python src/upi_fraud_detection.py
```

The pipeline performs:

1. Data loading
2. Data auditing
3. Chronological train/validation split
4. Feature engineering
5. Class-imbalance handling
6. Model training
7. PR-AUC/ROC-AUC evaluation
8. Threshold optimization
9. Cost-sensitive analysis
10. Risk-tier analysis
11. Top-K fraud review analysis
12. Calibration analysis
13. Permutation importance
14. Feature ablation
15. Report and plot generation
16. Model serialization

---

## Generated Outputs

The pipeline generates artifacts under:

```text
outputs/
├── models/
├── reports/
└── plots/
```

The trained model is saved as:

```text
outputs/models/upi_fraud_model_v4_2.joblib
```

Reports contain model metrics, threshold analysis, risk-tier analysis, calibration results, feature importance, and ablation results.

---

## Limitations

This project is an experimental fraud detection pipeline and should not be interpreted as a production-ready banking fraud prevention system.

Important limitations include:

* The dataset is synthetic/experimental and may not represent real-world UPI transaction distributions.
* Model performance can change under real-world concept drift.
* The high-recall operating point produces a substantial number of false positives.
* Predicted scores are not well calibrated.
* The model has not been evaluated on live production traffic.
* Real-world fraud systems require additional controls such as monitoring, retraining, model governance, privacy protection, and human investigation workflows.

---

## Future Improvements

Potential next steps include:

* Probability calibration using a dedicated temporal calibration set
* Temporal cross-validation
* Hyperparameter optimization
* Gradient-boosting model experimentation
* Cost-based model selection using realistic fraud investigation costs
* Drift monitoring
* Model explainability for individual transactions
* Online/streaming fraud scoring
* Real-time feature stores
* Production API deployment
* Model monitoring and automated retraining
* Evaluation on a larger and more realistic transaction dataset

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib
* Joblib

---

## Model Selection Summary

The final pipeline prioritizes **PR-AUC and temporal generalization** rather than accuracy alone.

The selected model is:

**Balanced Logistic Regression**

```text
Validation PR-AUC : 0.3277
External PR-AUC   : 0.3411
External ROC-AUC  : 0.8413
```

The results demonstrate that the model can meaningfully rank suspicious transactions while also highlighting the precision/recall trade-offs inherent in highly imbalanced fraud detection.

---

## Disclaimer

This project is intended for educational, research, and portfolio purposes. It is not intended for direct use in financial decision-making or production fraud prevention without additional validation, governance, security controls, and domain-specific testing.

---

## Author

**Rishi Karthigayan S**

Machine Learning / Data Science Project
