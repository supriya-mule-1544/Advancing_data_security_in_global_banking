# Credit Card Default Prediction Using Machine Learning

## MSc Data Science Project

### Project Overview

This project investigates credit card default prediction using machine learning techniques on the UCI Credit Card Default Dataset. The objective is to identify customers who are likely to default on their credit card payments, enabling financial institutions to improve risk management and lending decisions.

The project follows a complete machine learning pipeline including:

* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Feature Engineering
* Class Imbalance Handling using SMOTE
* Machine Learning Model Development
* Hyperparameter Tuning
* Model Evaluation and Comparison

---

## Business Problem

Credit card defaults represent a significant financial risk for banks and financial institutions. Early identification of high-risk customers can help reduce losses and support more informed credit decisions.

The dataset exhibits class imbalance, with approximately:

* 78% Non-Default Customers
* 22% Default Customers

Therefore, traditional accuracy metrics alone are insufficient for model evaluation.

---

## Dataset

**Source:** UCI Machine Learning Repository

**Dataset:** Default of Credit Card Clients Dataset

### Dataset Characteristics

* 30,000 customer records
* 24 original attributes
* Demographic information
* Credit limit information
* Repayment status history
* Monthly bill amounts
* Monthly payment amounts

### Target Variable

`default_payment_next_month`

* 0 = No Default
* 1 = Default

---

## Methodology

### 1. Exploratory Data Analysis

EDA was conducted to:

* Understand feature distributions
* Detect class imbalance
* Examine repayment behaviour
* Analyse feature correlations
* Identify important predictive variables

Key findings:

* Strong class imbalance
* PAY_0 showed the strongest correlation with default
* Bill amount variables exhibited high multicollinearity
* Financial variables were highly right-skewed

---

### 2. Data Preprocessing

The following preprocessing steps were performed:

* Removal of ID column
* Correction of categorical values
* Train-test split using stratification
* Feature scaling for Logistic Regression
* Data quality validation

---

### 3. Feature Engineering

To improve predictive performance, several behavioural features were created:

| Feature           | Description              |
| ----------------- | ------------------------ |
| avg_pay_delay     | Average repayment delay  |
| max_pay_delay     | Maximum repayment delay  |
| avg_bill_amt      | Average bill amount      |
| max_bill_amt      | Maximum bill amount      |
| total_bill_amt    | Total bill amount        |
| total_pay_amt     | Total payment amount     |
| utilization_ratio | Credit utilization ratio |
| pay_to_bill_ratio | Payment-to-bill ratio    |

These engineered features capture customer repayment behaviour more effectively than individual monthly variables.

---

### 4. Handling Class Imbalance

SMOTE (Synthetic Minority Oversampling Technique) was applied to the training dataset to address class imbalance.

Benefits:

* Improved minority-class learning
* Increased Recall
* Improved F1 Score
* Reduced bias toward majority class predictions

---

## Models Implemented

### Logistic Regression

Used as a baseline linear classification model.

### Random Forest

Ensemble learning algorithm based on bagging and decision trees.

### XGBoost

Gradient boosting algorithm selected for its strong performance on structured tabular datasets.

---

## Hyperparameter Tuning

GridSearchCV with Stratified 3-Fold Cross Validation was used to optimise model parameters.

Primary optimisation metric:

**F1 Score**

---

## Evaluation Metrics

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

F1 Score was selected as the primary evaluation metric because of the imbalanced nature of the dataset.

---

## Results

| Model               | F1 Score | ROC-AUC |
| ------------------- | -------- | ------- |
| Logistic Regression | ~0.43    | ~0.70   |
| Random Forest       | ~0.50    | ~0.75   |
| XGBoost             | ~0.51    | ~0.77   |

### Best Performing Model

**XGBoost**

Reasons:

* Highest F1 Score
* Highest ROC-AUC
* Strongest discrimination between default and non-default customers

---

## Feature Importance Insights

Random Forest feature importance analysis revealed that repayment behaviour variables are the strongest predictors of default.

Top features included:

* avg_pay_delay
* max_pay_delay
* PAY_0
* utilization_ratio
* pay_to_bill_ratio

This validates the effectiveness of the feature engineering process.

---

## Key Findings

* Repayment behaviour is the strongest predictor of default.
* Class imbalance significantly affects model performance.
* SMOTE improves detection of default customers.
* Ensemble methods outperform Logistic Regression.
* XGBoost achieved the best overall performance.

---

## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* XGBoost

---

## Project Structure

```text
├── Credit_Default_Prediction.ipynb
├── Project_Report.pdf
├── README.md
├── figures/
│   ├── eda_plots
│   ├── correlation_heatmap
│   ├── confusion_matrices
│   ├── roc_curves
│   └── feature_importance
└── data/
    └── credit_card_default.csv
```

## Future Improvements

* SHAP Explainability Analysis
* Cost-Sensitive Learning
* Threshold Optimisation
* Deep Learning Approaches
* External Dataset Validation
* Real-Time Credit Risk Scoring

---

## Academic Context

This project was completed as part of an MSc Data Science programme and demonstrates the application of machine learning techniques to real-world financial risk prediction problems.

---

## Author

Supriya

MSc Data Science

Machine Learning  | Financial Risk Modelling

