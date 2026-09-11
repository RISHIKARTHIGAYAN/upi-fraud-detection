\# UPI Fraud Detection



A leakage-aware machine learning pipeline for detecting fraudulent UPI transactions using behavioral, transaction, device, geographic, and risk-related signals.



The project is designed as an end-to-end fraud detection workflow with \*\*chronological validation, temporal external testing, class-imbalance handling, threshold optimization, business-oriented risk tiers, top-K fraud review analysis, feature ablation, permutation importance, and calibration diagnostics\*\*.



\---



\## Project Overview



UPI fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a small fraction of total transactions.



This project focuses not only on predicting fraud, but also on evaluating how the model could support different operational objectives:



\* Prioritizing high-risk transactions for investigation

\* Achieving a target fraud recall

\* Selecting operating thresholds based on precision/recall trade-offs

\* Assigning transactions to low, medium, and high-risk tiers

\* Supporting limited-capacity manual fraud review

\* Understanding which behavioral signals contribute most to predictions



A key focus of the project is \*\*preventing data leakage\*\* and evaluating the model on a temporally separated external test set.



\---



\## Key Results



The final V4.2 pipeline selected \*\*balanced Logistic Regression\*\* based on validation PR-AUC.



\### External Temporal Test



| Metric    |     Result |

| --------- | ---------: |

| PR-AUC    | \*\*0.3411\*\* |

| ROC-AUC   | \*\*0.8413\*\* |

| Precision | \*\*0.0522\*\* |

| Recall    | \*\*0.7978\*\* |

| F1-score  | \*\*0.0980\*\* |



The external test uses transactions from a later time period than the model-training data, providing a temporal evaluation of model performance.



\### F1-Optimal Operating Point



At the threshold selected using the validation set:



| Metric    | External Test |

| --------- | ------------: |

| Precision |    \*\*0.4504\*\* |

| Recall    |    \*\*0.3364\*\* |

| F1-score  |    \*\*0.3851\*\* |



This demonstrates the trade-off between a high-recall fraud-screening strategy and a higher-precision investigation strategy.



\---



\## Dataset



The dataset contains transaction-level UPI information including:



\* Account age

\* Credit score band

\* KYC level

\* Average monthly spending

\* Merchant risk score

\* Transaction amount

\* Payment channel

\* Device type

\* International transaction indicator

\* IP risk score

\* Transaction velocity

\* Failed transaction counts

\* Geographic distance

\* Amount deviation from the user's historical mean

\* Fraud label



\### Dataset Size



| Dataset               | Transactions |

| --------------------- | -----------: |

| Training dataset      |      300,113 |

| External test dataset |       99,887 |



\### Fraud Distribution



Training data:



\* Legitimate: \*\*295,242\*\*

\* Fraudulent: \*\*4,871\*\*

\* Fraud rate: \*\*1.6231%\*\*



External temporal test:



\* Legitimate: \*\*97,904\*\*

\* Fraudulent: \*\*1,983\*\*

\* Fraud rate: \*\*1.9852%\*\*



\---



\## Temporal Validation Strategy



Instead of randomly splitting transactions, the project uses a chronological evaluation strategy.



\### Timeline



```text

2023-01-01 ─────────────── 2023-09-30

&#x20;             │

&#x20;             ├── Model Training

&#x20;             └── Validation



2023-10-01 ───────────────────────── 2023-12-30

&#x20;                   │

&#x20;                   └── External Temporal Test

```



The training dataset is divided chronologically into:



