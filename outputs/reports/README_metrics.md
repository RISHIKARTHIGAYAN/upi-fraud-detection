# UPI Fraud Detection - V4.2

## Model

LogisticRegression_Balanced

The model was selected using validation PR-AUC.

## Dataset

- Model training transactions: 247,293
- Validation transactions: 52,820
- External temporal test transactions: 99,887

## Class Imbalance

- Validation fraud rate: 1.9046%
- External test fraud rate: 1.9852%

## Validation Performance

Using the threshold selected for approximately 80% recall:

- PR-AUC: 0.3277
- ROC-AUC: 0.8437
- Precision: 0.0533
- Recall: 0.8002
- F1: 0.0999

## External Temporal Test

The threshold was selected using validation data and then frozen for the external test.

- PR-AUC: 0.3411
- ROC-AUC: 0.8413
- Precision: 0.0522
- Recall: 0.7978
- F1: 0.0980

## F1-Optimal Operating Point

Validation threshold:

0.875090

Validation:

- Precision: 0.4549
- Recall: 0.3260
- F1: 0.3798

External:

- Precision: 0.4504
- Recall: 0.3364
- F1: 0.3851

## Business Risk Thresholds

Medium-risk threshold:

0.757239

High-risk threshold:

0.859161

These thresholds were selected using validation data.

## Calibration

Validation Brier score:

0.171047

Raw model scores should not automatically be interpreted as perfectly calibrated probabilities.

## Leakage Prevention

The following fields were excluded:

- transaction_id
- customer_id
- merchant_id
- is_fraud
- transaction_time
- post_auth_risk_score

The post-authorization risk score was specifically excluded because it would not be available at the point where a real-time fraud decision is made.

## Evaluation Design

The dataset was split chronologically.

Earlier transactions were used for model training.

Later transactions from the training period were used for validation.

A completely later external test period was retained for final evaluation.

Decision thresholds were selected on validation data and not tuned using the external test set.

## Main Behavioral Signals

The feature engineering pipeline includes:

- transaction velocity
- failed transaction behavior
- transaction amount deviation
- transaction amount relative to monthly spending
- geographic distance
- merchant risk
- IP risk
- international transaction interactions
- time-of-day behavior