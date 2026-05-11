# ARTI308
This repository holds my work as a student for the Machine Learning course
# SVM Assignment – Iris Classification

Classifying iris flower species using a Support Vector Machine. The dataset has 150 samples across 3 species (Setosa, Versicolor, Virginica), each described by 4 measurements: sepal length, sepal width, petal length, and petal width.

## What's in here

**EDA**
Pairplot across all features colored by species — Setosa is clearly the most separable. Also a KDE plot of sepal length vs. width for Setosa specifically.

**Model**
A basic SVC with default settings, trained on a 70/30 train-test split. Evaluated with a confusion matrix and classification report.

**Hyperparameter Tuning**
GridSearchCV over `C` values `[0.1, 1]` and `gamma` values `[0.1, 0.01]` with an RBF kernel. The tuned model is re-evaluated to compare against the default.

## Dependencies

```
pandas, numpy, matplotlib, seaborn, scikit-learn
```
