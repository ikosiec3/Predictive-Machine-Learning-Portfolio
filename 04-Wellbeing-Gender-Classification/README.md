# Predicting Gender Based on Wellbeing and Lifestyle Habits

**Data Science and Business Analytics – Machine Learning Final Project**
**Authors:** Katarzyna Pełka, Izabela Kosiec

## Project Overview
The goal of this project is to apply and compare multiple machine learning algorithms for a classification task. We predict gender based on wellbeing, daily habits, and lifestyle data.

## Dataset
**Source:** Kaggle (Wellbeing and Lifestyle Dataset)
The dataset contains 15,972 observations and 24 variables describing health indicators, social behavior, stress levels, and work–life balance.
**Target variable:** `gender` (Female / Male)

## Key Preprocessing Steps
* Removal of an invalid categorical value in `DAILY_STRESS`.
* One-hot encoding of categorical variables.
* Train-test split with stratification.
* Feature scaling for models sensitive to scale.
* Correlation-based feature selection to reduce overfitting.
* Explicit handling of class imbalance (class weighting for Logistic Regression, `scale_pos_weight` for XGBoost).

## Models Used
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. XGBoost Classifier

## Evaluation & Results
* **Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC
* **Final model:** Class-weighted XGBoost with reduced feature set
* **Results:** Best F1-score ≈ 0.54

The moderate performance is expected due to strong overlap between classes. The results confirm that lifestyle and wellbeing variables contain only weak discriminatory signal for gender prediction, making this a challenging and realistic classification problem.

## Reproducibility
All experiments use a fixed random seed. Full preprocessing, modeling, and evaluation pipelines are included and can be fully reproduced by running the provided Jupyter Notebook. This project highlights the importance of addressing class imbalance and ethical awareness when predicting sensitive demographic attributes from behavioral data.