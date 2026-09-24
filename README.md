# Delinquency Risk Modeling — Datarisk

Probabilistic model designed to estimate, **for each invoice**, the risk of payment being delayed by **5 days or more**.  
The project covers the complete Data Science workflow of a technical case: data validation, EDA, feature engineering, *data leakage* prevention, temporal validation, model comparison, and final probability generation.

**Stack:** Python 3.12.5 · pandas · scikit-learn · XGBoost · Matplotlib · Seaborn · Jupyter

---

## Overview

The objective is not to classify an invoice as simply “good” or “bad” based on an arbitrary threshold, but rather to estimate:

\[
P(\text{delay} \geq 5\text{ days} \mid \text{information available at prediction time})
\]

The prediction unit is **one invoice**, and the output is a continuous probability between 0 and 1.

| Metric | Result |
|---|---:|
| Development invoices | **77,414** |
| Development customers | **1,248** |
| Test invoices | **12,275** |
| Event rate in development data | **7.02%** |
| Temporal holdout | **Apr/2021 → Jun/2021** |
| Selected model | **XGBoost** |
| Holdout ROC-AUC | **0.9291** |
| Holdout Log Loss | **0.1333** |
| Holdout Average Precision | **0.5808** |

> **Data:** the original datasets and the case statement/data dictionary are not redistributed in this repository. The notebook documents the expected structure and the complete modeling pipeline.

---

## The Problem

The target variable was constructed directly from payment and due dates:

```text
DAYS_LATE = PAYMENT_DATE - DUE_DATE

TARGET = 1  if DAYS_LATE >= 5
TARGET = 0  otherwise
```

Explicit boundary tests were included: 4 days → 0; 5 and 6 days → 1.

The development data covers **August 2018 through June 2021**. The test set corresponds to the following months, from **July through November 2021**. This temporal structure was central to the validation strategy.

---

## Pipeline

```mermaid
flowchart LR
    A[4 original datasets] --> B[Validation and joins]
    B --> C[Target construction]
    C --> D[EDA]
    D --> E[Feature engineering]
    E --> F[Temporal split]
    F --> G[Pipeline preprocessing]
    G --> H[LogReg / Random Forest / XGBoost]
    H --> I[Holdout selection]
    I --> J[Fit on full development data]
    J --> K[Test probabilities]
```

Several safeguards are intentional: `many-to-one` joins are validated, row counts are preserved, the original test order is protected through `_ROW_ID`, and imputation/encoding are fitted **inside the pipelines**, after the temporal split.

---

## Exploratory Data Analysis

### Target Imbalance

The positive class appears in **5,436 out of 77,414 invoices (7.02%)**.

<p align="center">
  <img src="assets/target_distribution.png" alt="Target distribution" width="620">
</p>

This imbalance is one of the reasons accuracy is not used as the main evaluation metric. Since the final output is a **probability**, the evaluation prioritizes probabilistic quality and discrimination.

### Behavior Over Time

The delinquency rate is not constant across cohorts: during the analyzed period, it ranged from approximately **4.14%** to **16.03%**.

<p align="center">
  <img src="assets/default_rate_over_time.png" alt="Invoice volume and delinquency rate by cohort" width="900">
</p>

This temporal variation reinforces the use of a future holdout instead of a random split that would mix past and future observations.

Other relevant EDA findings:

- the median is **28 invoices per customer**, with a long tail of recurring customers;
- **90.98%** of test customers also appear in the development data;
- `VALOR_A_PAGAR` and income are strongly right-skewed;
- income and number of employees contain missing values;
- the test set shows higher central values for some variables, although no formal drift test was performed;
- one `DDD` category appears only in the test set.

---

## Feature Engineering

The feature set was designed to remain interpretable and to use only information available at prediction time.

### Invoice and Time Information

- cohort year and month;
- invoice amount;
- rate;
- time between issue date and due date;
- time since customer registration.

### Financial and Customer Information

- previous-month income;
- number of employees;
- invoice amount / income;
- invoice amount / number of employees;
- segment, company size, DDD, ZIP code, email domain, and customer type.

### Customer History

Four historical features were created:

- `HIST_QTD_COBRANCAS`;
- `HIST_QTD_INADIMPLENCIAS`;
- `HIST_TAXA_INADIMPLENCIA`;
- `FLAG_SEM_HISTORICO`.

The most important aspect is **temporal consistency**: invoices are first aggregated by customer and cohort, and cumulative values are then shifted. Therefore, the features of a given cohort use only information from **previous cohorts**.

```text
customer + current cohort
        │
        ├── previous invoices
        ├── previous delinquencies
        └── previous historical delinquency rate

current-cohort target ──X──> current-cohort features
```

For the test set, customer history is frozen using only payments observed in the development period. Future predictions are never reused as if they were observed outcomes.

---

## Data Leakage Prevention

Separating information available at prediction time from future information is one of the central aspects of the project.

The following variables were excluded from the predictors:

```text
ID_CLIENTE
DATA_PAGAMENTO
DIAS_ATRASO
TARGET
_ROW_ID
raw dates used to construct derived features
```

The notebook also explicitly tests that:

- a customer's first month has no previous history;
- the second month uses only information from the first;
- the current cohort does not enter its own cumulative statistics;
- historical features are constant within each customer–cohort pair;
- the test set does not contain payment information or target values during feature construction;
- previously unseen customers are identified separately.

---

## Temporal Validation

Because the test set occurs chronologically after the development data, the evaluation strategy mirrors this scenario.

| Dataset | Period | Rows | Customers | Event Rate |
|---|---|---:|---:|---:|
| Train | Aug/2018 → Mar/2021 | 70,012 | 1,194 | 7.10% |
| Validation | Apr/2021 → Jun/2021 | 7,402 | 868 | 6.24% |
| Test | Jul/2021 → Nov/2021 | 12,275 | 976 | — |

For validation, customer history is also **frozen at the end of the training period**. Therefore, outcomes from April, May, or June 2021 do not update features for other observations inside the holdout itself.

---

## Models and Metrics

The following models were evaluated:

- Logistic Regression;
- Random Forest;
- Random Forest with `class_weight="balanced"`;
- XGBoost.

Logistic Regression receives standardized numerical features. Tree-based models use the same imputation strategy but without `StandardScaler`. Categorical variables are imputed and encoded with `OneHotEncoder(handle_unknown="ignore")`.

### Why Log Loss?

The case requires **probabilities**, not only class labels. For that reason, the primary metric is **Log Loss**, which penalizes overly confident incorrect predictions. ROC-AUC and Average Precision complement the evaluation by measuring discrimination.

### Temporal Holdout Results

| Model | ROC-AUC ↑ | Log Loss ↓ | Average Precision ↑ |
|---|---:|---:|---:|
| **XGBoost** | **0.9291** | **0.1333** | **0.5808** |
| Random Forest | 0.9205 | 0.1442 | 0.5719 |
| Logistic Regression | 0.8564 | 0.1701 | 0.4329 |
| Balanced Random Forest | 0.9283 | 0.3396 | 0.5740 |

XGBoost achieved the strongest overall performance on the temporal holdout.

One particularly useful result came from the balanced Random Forest: class weights slightly improved ROC-AUC and Average Precision compared with the unweighted Random Forest, but shifted the average predicted probability to **28.16%**, far above the **6.24%** event rate observed in validation. Log Loss deteriorated from **0.1442 to 0.3396**.

For this reason, class weighting was not adopted simply because the target was imbalanced.

<p align="center">
  <img src="assets/roc_curve.png" alt="ROC curves on the temporal holdout" width="47%">
  <img src="assets/precision_recall_curve.png" alt="Precision-Recall curves on the temporal holdout" width="47%">
</p>

---

## Final Model

The selected candidate was:

```python
XGBClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=0,
)
```

After the initial comparison, a small and deliberately limited refinement step was performed.

A configuration with 400 trees and `learning_rate=0.03` reduced Log Loss from **0.133252 to 0.133116**, an improvement of only **0.000136**. Since the gain was far below the minimum improvement defined beforehand and was accompanied by a slight reduction in ROC-AUC, the simpler baseline configuration was retained.

The goal was not to extract marginal improvements from the same holdout, but to avoid selecting a more complex configuration based on a practically negligible difference.

---

## What the Model Is Using

In the selected XGBoost model, the customer's **historical delinquency rate** appears as the most important feature, followed by invoice amount and the historical number of delinquent invoices.

<p align="center">
  <img src="assets/feature_importance.png" alt="Top XGBoost feature importances" width="850">
</p>

These importances are **predictive**, not causal. Correlated features may share importance, and one-hot encoded categories appear separately.

---

## Probability Checks

On the holdout:

- observed event rate: **6.24%**;
- average predicted probability: **5.22%**;
- difference: **−1.02 percentage points**;
- minimum predicted probability: **0.0017**;
- maximum predicted probability: **0.9493**.

<p align="center">
  <img src="assets/predicted_probability_distribution.png" alt="Distribution of predicted probabilities on validation data" width="780">
</p>

The difference in means suggests **global underestimation** during the validation period. This comparison is only a *sanity check* and does not replace a formal calibration analysis across risk buckets.

---

## Final Training and Output

After model selection, the chosen pipeline is refitted on all **77,414 development invoices** and applied to the **12,275 test invoices**.

The final output contains exactly:

```text
ID_CLIENTE
SAFRA_REF
PROBABILIDADE_INADIMPLENCIA
```

The generation step includes checks for:

- number of rows;
- column names and order;
- missing probabilities;
- probabilities outside `[0, 1]`;
- preservation of the original test order;
- accidental CSV index creation.

Because the test set does not contain the target, no performance metric is reported for it.

---

## Most Important Modeling Decisions

1. **Temporal validation instead of random splitting**  
   The test set occurs chronologically after the development period, so validation follows the same structure.

2. **Historical features built only from past information**  
   The `shift` is applied after aggregation by customer and cohort, preventing the current month's outcome from contaminating its own features.

3. **Probability quality over threshold-based metrics**  
   Log Loss is prioritized because the required output is a probability.

4. **Class imbalance does not automatically imply class weighting**  
   The balanced configuration slightly improved discrimination but severely distorted the probability scale.

5. **Conservative model refinement**  
   A Log Loss improvement of only 0.000136 was not considered sufficient to justify replacing the baseline with a more complex configuration.

---

## Repository Structure

```text
case-datarisk/
├── assets/
│   ├── default_rate_over_time.png
│   ├── feature_importance.png
│   ├── precision_recall_curve.png
│   ├── predicted_probability_distribution.png
│   ├── roc_curve.png
│   └── target_distribution.png
├── case_datarisk.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

`data/` and `submissao_case.csv` are local artifacts and do not need to be versioned.

---

## How to Run

### 1. Create the environment

```bash
python -m venv .venv
```

Activate the environment and install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare the data

The notebook expects a `data/` directory at the project root:

```text
data/
├── base_cadastral.csv
├── base_info.csv
├── base_pagamentos_desenvolvimento.csv
└── base_pagamentos_teste.csv
```

The files are read using `;` as the separator.

### 3. Run the notebook

Open:

```text
case_datarisk.ipynb
```

and run all cells in order from a restarted kernel.

The notebook performs data preparation, EDA, feature engineering, temporal validation, model training, and final output generation.

---

## Limitations

This project was developed as an analytical case and **should not be interpreted as a production-ready system**.

Main limitations:

- results are based on a single temporal holdout;
- no evaluation is available for periods after the provided test window;
- the average predicted probability underestimates the observed holdout event rate by approximately 1.02 percentage points;
- some inconsistent dates were preserved because the data dictionary did not define a correction rule;
- descriptive differences exist between development and test data, but no formal drift test was performed;
- XGBoost feature importances do not imply causal relationships;
- no operational decision threshold was defined because this would require business costs and objectives.

---

## Next Steps

To evolve this solution toward a more production-oriented setup, I would prioritize:

- **backtesting / walk-forward validation** across multiple temporal windows;
- formal **calibration** analysis and, if necessary, Platt scaling or isotonic regression;
- **feature drift** and prediction drift monitoring;
- threshold definition based on **business costs**, rather than an arbitrary value such as 0.5;
- observation-level explainability and feature stability analysis across future periods.

---

## Reproducibility

- `random_state=0` is used in stochastic components;
- dependencies are pinned in `requirements.txt`;
- imputation, scaling, and one-hot encoding are fitted inside `Pipeline`;
- the test set is not used for training, model selection, or evaluation;
- the final submission file is recreated directly from the four original datasets.

---

### Technologies

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `Jupyter`
