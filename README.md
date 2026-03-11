# Online Payments Fraud Detection

An XGBoost-based fraud detection system trained on 6.3M+ transactions with SMOTE oversampling to handle extreme class imbalance (0.13% fraud rate). Achieves 99.77% accuracy with 99% recall on fraudulent transactions.

## Problem Statement

Online payment fraud is rare but costly — in this dataset, only 0.13% of transactions are fraudulent. Standard classifiers fail on such imbalanced data, predicting everything as non-fraud and still scoring high accuracy. This project tackles the imbalance head-on using SMOTE and evaluates performance with metrics that matter: recall and precision on the minority (fraud) class.

## Dataset

- **Source:** Kaggle — Synthetic Financial Datasets for Fraud Detection (PaySim)
- **Records:** 6,362,620 transactions
- **Features (11):** `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`
- **Transaction Types:** CASH_OUT (2.2M), PAYMENT (2.2M), CASH_IN (1.4M), TRANSFER (533K), DEBIT (41K)
- **Fraud Rate:** 8,213 fraudulent out of 6.36M (0.13%)

## Approach

1. **EDA** — Analyzed class imbalance, fraud distribution by transaction type, and feature correlations with target
2. **Preprocessing** — Dropped identifiers (`nameOrig`, `nameDest`), label-encoded `type`, 80/20 train-test split
3. **SMOTE Oversampling** — Balanced training set from 6,554 fraud → 5,083,503 fraud samples (1:1 ratio)
4. **Model Training** — XGBoost classifier with log-loss evaluation metric
5. **Evaluation** — Confusion matrix, classification report with per-class precision/recall/F1

## Key Results

| Metric | Non-Fraud | Fraud |
|---|---|---|
| Precision | 1.00 | 0.35 |
| Recall | 1.00 | **0.99** |
| F1-Score | 1.00 | 0.52 |

- **Overall Accuracy:** 99.77%
- **Fraud Recall:** 99% — catches nearly all fraudulent transactions
- **Trade-off:** Lower fraud precision (0.35) means some false positives, acceptable in fraud detection where missing fraud is far costlier than flagging legitimate transactions

## Tech Stack

- **Language:** Python
- **Libraries:** XGBoost, Scikit-learn, imbalanced-learn (SMOTE), Pandas, NumPy, Seaborn, Matplotlib
- **Techniques:** SMOTE oversampling, XGBoost classification, Label Encoding, class imbalance handling

## Project Structure

```
├── Fraud_Detection.ipynb   # Full analysis and modeling notebook
├── online-payments.png     # Visualization
└── README.md
```

## How to Run

```bash
pip install xgboost imbalanced-learn
jupyter notebook Fraud_Detection.ipynb
```

## Author

**Mahi Sharma**
B.Tech CSE (Data Science) — Manipal University Jaipur
[GitHub](https://github.com/mahi-sharmas)
