## Online Payments Fraud Detection

An XGBoost-based fraud detection system trained on 6.3 million synthetic financial transactions, using SMOTE oversampling to handle extreme class imbalance and achieve 99% fraud recall.

### Highlights

- Processed 6.3M+ transactions from the PaySim dataset — demonstrating real-world scale data handling
- Achieved 99% recall on fraudulent transactions, catching 1,603 of 1,620 fraud cases in the test set
- Handled extreme class imbalance (0.13% fraud) using SMOTE, scaling from 6,554 to 5,083,503 synthetic fraud samples
- Overall model accuracy of 99.77% with a deliberate precision-recall trade-off favoring fraud detection

### Problem Statement

Online payment fraud causes billions in losses annually, yet fraudulent transactions make up a tiny fraction of all activity — only 0.13% in this dataset. Standard classifiers trained on such imbalanced data simply predict everything as non-fraud and still achieve high accuracy. This project tackles the imbalance head-on using SMOTE oversampling and XGBoost, optimizing for recall on the fraud class since missing a fraudulent transaction is far more costly than flagging a legitimate one for review.

### Dataset

- **Source:** [PaySim — Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle)
- **Size:** 6,362,620 transactions × 11 features
- **Target:** `isFraud` (binary — 0 = legitimate, 1 = fraudulent)
- **Class distribution:** 99.87% non-fraud (6,354,407) vs. 0.13% fraud (8,213)
- **Transaction types:** CASH_OUT (2.24M), PAYMENT (2.15M), CASH_IN (1.40M), TRANSFER (533K), DEBIT (41K)
- **Key features:** step, type, amount (mean $179,862), oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
- **Top correlations with fraud:** amount (0.077), isFlaggedFraud (0.044), step (0.032)

### Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-purple)
![Pandas](https://img.shields.io/badge/Pandas-Data-green)
![Seaborn](https://img.shields.io/badge/Seaborn-Viz-teal)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Plots-yellow)

### Methodology

1. **Data Loading & EDA** — Loaded 6.3M transactions, confirmed zero missing values, analyzed class imbalance (0.13% fraud rate), computed feature correlations with the target, and visualized fraud distribution across transaction types
2. **Preprocessing** — Dropped non-predictive identifier columns (nameOrig, nameDest), applied LabelEncoder to the categorical `type` feature
3. **Train-Test Split** — 80/20 split with `random_state=42` (5.08M train, 1.27M test samples)
4. **SMOTE Oversampling** — Applied Synthetic Minority Over-sampling Technique to balance the training set from 6,554 fraud samples to 5,083,503 (1:1 ratio with non-fraud), keeping the test set at its original real-world distribution
5. **Model Training** — Trained XGBoost classifier with `eval_metric='logloss'` on the SMOTE-resampled training data
6. **Evaluation** — Generated classification report with per-class precision/recall/F1 and confusion matrix heatmap

### Key Results

| Metric | Non-Fraud | Fraud |
|---|---|---|
| Precision | 1.00 | 0.35 |
| Recall | 1.00 | **0.99** |
| F1-Score | 1.00 | 0.52 |
| Support | 1,270,904 | 1,620 |

| Overall Metric | Value |
|---|---|
| **Accuracy** | **99.77%** |
| **Fraud Recall** | **99%** — caught 1,603 of 1,620 fraud cases |
| **Macro Avg F1** | **0.76** |

The model correctly identifies 99% of fraudulent transactions. The lower fraud precision (0.35) is an intentional and acceptable trade-off — in fraud detection, the cost of missing fraud far exceeds the cost of flagging a legitimate transaction for manual review.

### How to Run

```bash
git clone https://github.com/mahi-sharmas/Online-Payments-Fraud-Detection.git
cd Online-Payments-Fraud-Detection
pip install -r requirements.txt
jupyter notebook Fraud_Detection.ipynb
```

**Note:** The PaySim dataset (~470MB) must be downloaded separately from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and placed in the project directory.

### Project Structure

```
Online-Payments-Fraud-Detection/
├── Fraud_Detection.ipynb    # Complete pipeline — EDA, SMOTE, XGBoost training, evaluation
├── online-payments.png      # Project visualization
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

### Future Improvements

- Experiment with threshold tuning and cost-sensitive learning to improve fraud precision without sacrificing recall
- Add feature engineering (transaction velocity, balance change ratios, time-windowed aggregates) to capture more fraud patterns
- Benchmark against LightGBM and CatBoost to compare gradient boosting implementations on this scale

### Author

**Mahi Sharma** — B.Tech CSE (Data Science), Manipal University Jaipur (2023–2027)

GitHub: [github.com/mahi-sharmas](https://github.com/mahi-sharmas) | Email: mahi.sh4rma7@gmail.com

*Project completed: Apr 2024*
