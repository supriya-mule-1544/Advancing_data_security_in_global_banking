# Advancing_data_security_in_global_banking
# Advancing Data Security in Global Banking: Innovative Big Data Management Techniques

**Student:** Supriya | SRN: 23070872
**Programme:** M.Sc. Data Science — University of Hertfordshire
**Module:** 7PAM2002
**Supervisor:** Stephen Kane
**Date Submitted:** 22 April 2026

---

## Project Overview

This project applies machine learning to predict credit card default using a real-world dataset of 30,000 clients from a Taiwanese commercial bank. Three classification models — Logistic Regression, Random Forest, and XGBoost — are trained, tuned, and compared to identify the best-performing approach for credit risk prediction in banking.

The core research question is:

> *Which machine learning algorithm performs best among Logistic Regression, Random Forest, and XGBoost for predicting credit card default, in terms of Accuracy, Precision, Recall, and F1 Score?*

---

## Dataset

- **Name:** Default of Credit Card Clients
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Format:** .xls (structured tabular data)
- **Size:** 30,000 rows × 25 columns
- **Target Variable:** `default_payment_next_month` (0 = No Default, 1 = Default)
- **Class Distribution:** ~78% Non-default | ~22% Default (imbalanced)

### Key Features

| Feature | Description |
|---|---|
| `LIMIT_BAL` | Credit limit in New Taiwan Dollars |
| `SEX`, `EDUCATION`, `MARRIAGE`, `AGE` | Client demographics |
| `PAY_0` – `PAY_6` | Repayment status for the past 6 months |
| `BILL_AMT1` – `BILL_AMT6` | Monthly bill statement amounts |
| `PAY_AMT1` – `PAY_AMT6` | Monthly payment amounts |
| `default_payment_next_month` | Target variable (binary) |

---

## Project Structure

```
├── default_of_credit_card_clients.xls   # Raw dataset
├── Untitled10.ipynb                      # Main Jupyter notebook (full ML pipeline)
├── EV_ML_Presentation.pptx              # Project presentation slides
└── README.md                            # This file
```

---

## Methodology

The project follows a structured end-to-end machine learning pipeline:

1. **Exploratory Data Analysis (EDA)** — target distribution, correlation heatmap, categorical inspection
2. **Data Preprocessing** — dropped `ID` column, fixed irregular `EDUCATION` and `MARRIAGE` values, standardised column names, applied `StandardScaler` for Logistic Regression
3. **Feature Engineering** — created 9 new features from billing and payment history:
   - `avg_bill_amt`, `avg_pay_amt`, `max_bill_amt`
   - `max_pay_delay`, `avg_pay_delay`
   - `total_bill_amt`, `total_pay_amt`
   - `pay_to_bill_ratio`, `utilization_ratio`
4. **Train-Test Split** — 80/20 stratified split (`random_state=42`)
5. **Model Training** — Logistic Regression, Random Forest, XGBoost
6. **Hyperparameter Tuning** — GridSearchCV with Stratified K-Fold (3 folds), optimising for F1 Score
7. **Evaluation** — Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix

---

## Models

| Model | Key Settings |
|---|---|
| Logistic Regression | `class_weight='balanced'`, `max_iter=2000` |
| Random Forest | `class_weight='balanced'`, `n_estimators=100` |
| XGBoost | `scale_pos_weight` set to negative/positive ratio |

---

## Results

### Baseline (Before Tuning)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.808 | 0.688 | 0.241 | 0.357 |
| Random Forest | 0.815 | 0.642 | 0.366 | 0.466 |
| XGBoost | 0.809 | 0.616 | 0.362 | 0.456 |

### After Hyperparameter Tuning

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.808 | 0.691 | 0.241 | 0.358 |
| Random Forest | 0.814 | 0.639 | 0.367 | **0.466** ✅ |
| XGBoost | 0.817 | 0.657 | 0.357 | 0.463 |

**Best Model: Random Forest** — highest F1 Score (0.466) in both baseline and tuned conditions, demonstrating the best balance between Precision and Recall.

> F1 Score is used as the primary metric due to class imbalance. A model that only maximises accuracy would trivially predict "No Default" ~78% of the time without identifying any real risk.

---

## Key Findings

- **Repayment status (PAY_0)** is the strongest single predictor of credit default — clients with recent payment delays are at significantly higher risk.
- **Random Forest** outperforms Logistic Regression and XGBoost on F1 Score, thanks to its ensemble averaging approach which handles class imbalance and non-linear patterns effectively.
- **Logistic Regression** achieves high precision but very low recall (0.241), meaning it misses ~75% of actual default cases — unsuitable for practical deployment.
- **XGBoost** achieves the highest accuracy after tuning (81.7%) but falls marginally behind Random Forest on F1 Score.
- **Feature engineering** expanded the feature space from 24 to 33 variables, capturing aggregate financial behaviour such as credit utilisation and repayment coverage ratios.
- **Hyperparameter tuning** produced modest but consistent improvements; the gains were most meaningful for Random Forest and XGBoost.

---

## How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl
```

### Steps

1. Clone or download this repository
2. Place the dataset file (`default_of_credit_card_clients.xls`) in the same directory
3. Open `Untitled10.ipynb` in Jupyter Notebook or Google Colab
4. Run all cells in order from top to bottom

The notebook covers the full pipeline: data loading → EDA → preprocessing → feature engineering → model training → hyperparameter tuning → evaluation and comparison.

> **Google Colab link:** https://colab.research.google.com/drive/13Af7Jw5x8udkUlnMwQWrNeFfKjQXuiw_?usp=sharing

---

## Ethical Considerations

- The dataset is **publicly available** from UCI and contains **no personally identifiable information (PII)**
- All client identifiers are anonymised numeric codes
- The dataset was used **exclusively for academic research**
- No data was shared with third parties
- Demographic features (SEX, EDUCATION, MARRIAGE) are included as per the original benchmark — any real-world deployment would require bias auditing and fairness analysis

---

## Future Work

- Apply **SMOTE** or **ADASYN** oversampling to better address class imbalance and improve recall
- Incorporate richer features: customer income, employment status, credit bureau data
- Explore **deep learning** approaches (fully connected networks, attention-based models)
- Apply **SHAP** or **LIME** for model explainability and feature attribution
- Extend to **real-time prediction** pipelines for production banking environments
- Validate across multiple institutions and regions for generalisation

---

## References

- Yeh, I.-C., & Lien, C.-H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473–2480.
- Lessmann, S., et al. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring. *European Journal of Operational Research*, 247(1), 124–136.
- Brown, I., & Mues, C. (2012). An experimental comparison of classification algorithms for imbalanced credit scoring data sets. *Expert Systems with Applications*, 39(3), 3446–3453.
- Xia, Y., et al. (2018). A boosted decision tree approach using Bayesian hyper-parameter optimization for credit scoring. *Expert Systems with Applications*, 78, 225–241.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD '16*, 785–794.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- UCI Machine Learning Repository. (2016). Default of Credit Card Clients Dataset. https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263–1284.

---

*M.Sc. Data Science | University of Hertfordshire | 7PAM2002*
