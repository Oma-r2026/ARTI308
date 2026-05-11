# Decision Trees & Random Forest – Loan Repayment Prediction

Predicting whether a borrower will fully repay their loan using LendingClub data from 2007–2010. The target variable is `not.fully.paid` (1 = defaulted, 0 = paid back). Dataset is pre-cleaned with no NA values.

## Features

13 input features including FICO score, interest rate, debt-to-income ratio, loan purpose, credit policy status, and delinquency history.
The `purpose` column is categorical and gets one-hot encoded before training.

## What's in here

**EDA**
- Overlapping FICO histograms split by `credit.policy` and then by `not.fully.paid` — shows how score distributions differ between groups
- Countplot of loan purpose broken down by repayment status
- Jointplot of FICO vs. interest rate
- lmplot of the same, separated by both `credit.policy` and `not.fully.paid` columns

**Data Prep**
One-hot encoding on `purpose` via `pd.get_dummies` with `drop_first=True`, then a standard 70/30 train-test split.

**Models**

| Model | Notes |
|---|---|
| Decision Tree | Default settings, no depth limit |
| Random Forest | 600 estimators |

Both evaluated with a confusion matrix and classification report.

**Key observation:** The Random Forest has a noticeably low recall for class 1 (defaulted loans) — it's heavily biased toward predicting class 0 because the dataset is imbalanced. The Decision Tree, despite being simpler, actually catches more defaults. Worth keeping in mind if the goal is to flag risky borrowers.

## Dependencies

```
pandas, numpy, matplotlib, seaborn, scikit-learn
```
