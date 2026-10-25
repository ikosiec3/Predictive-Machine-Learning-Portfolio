Machine Learning 2 – Final Project
Data Science and Business Analytics (2nd year)

Authors:
Katarzyna Pełka (432453)
Izabela Kosiec (441755)


PROJECT OVERVIEW

This project was prepared as a final assignment for the Machine Learning 2 course.
The goal of the project is to apply and compare multiple machine learning algorithms
for both regression and classification tasks using real-world datasets.

The project consists of two independent parts:
1) Regression task – predicting Airbnb prices in New York City
2) Classification task – predicting gender based on wellbeing and lifestyle data

For each task, several machine learning models were trained, tuned, evaluated,
and compared in a consistent and reproducible manner.


FILES INCLUDED

- Regression_project_final.ipynb
  Jupyter Notebook containing full regression analysis, code, and results

- Classification_proj_final.ipynb
  Jupyter Notebook containing full classification analysis, code, and results

- Regression_project_final.pdf
  PDF export with regression code output and results

- Classification_proj_final.pdf
  PDF export with classification code output and results

- Presentation.pdf
  PDF presentation summarizing both projects

- README.txt
  This file


REGRESSION PROJECT - Predicting Airbnb Nightly Prices in New York City

Dataset:
new_york_listings_2024.csv (Kaggle)

The dataset contains 20,758 observations and 22 variables describing Airbnb listings,
including location, property characteristics, host activity, availability, and user feedback.

Target variable:
- price (nightly price in USD)

Key preprocessing steps:
- Log-transformation of the target variable to handle heavy right-skewness
- Removal of the top 0.5% most expensive listings to improve model stability
- Conversion of categorical variables using one-hot encoding
- Simplification of high-cardinality variables (e.g. license, neighbourhood)
- Treatment of studios as 0 bedrooms
- Conversion of baths to numeric values with median imputation
- Capping extreme values for selected numeric variables
- Removal of identifier-like and non-informative columns

Models used:
1) Baseline model (mean predictor)
2) Random Forest Regressor (bagging with OOB validation)
3) Gradient Boosting Regressor
4) XGBoost Regressor
5) Ensemble model (Random Forest + XGBoost)

Evaluation metrics:
- RMSE
- MAE
- R²

Final model:
XGBoost Regressor with early stopping

Best test performance:
- RMSE ≈ 90.48 USD
- R² ≈ 0.555

Feature importance analysis confirmed that price is mainly driven by:
- Room type
- Property size (bedrooms, bathrooms, beds)
- Location (neighbourhood group)
- License status


CLASSIFICATION PROJECT - Predicting Gender Based on Wellbeing and Lifestyle Habits

Dataset:
Wellbeing and Lifestyle Dataset (Kaggle)

The dataset contains 15,972 observations and 24 variables describing daily habits,
health indicators, social behavior, stress levels, and work–life balance.

Target variable:
- gender (Female / Male)

Key preprocessing steps:
- Removal of an invalid categorical value in DAILY_STRESS
- One-hot encoding of categorical variables
- Train-test split with stratification
- Feature scaling for models sensitive to scale
- Correlation-based feature selection to reduce overfitting
- Explicit handling of class imbalance

Models used:
1) Logistic Regression
2) Random Forest Classifier
3) Gradient Boosting Classifier
4) XGBoost Classifier

Handling class imbalance:
- Class weighting for Logistic Regression
- scale_pos_weight for XGBoost

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Final model:
Class-weighted XGBoost with reduced feature set

Results:
- Best F1-score ≈ 0.54
- Moderate performance expected due to strong overlap between classes

The results confirm that lifestyle and wellbeing variables contain only weak
discriminatory signal for gender prediction, making this a challenging and realistic
classification problem.


REPRODUCIBILITY
- All experiments use a fixed random seed
- Full preprocessing, modeling, and evaluation pipelines are included
- Results can be fully reproduced by running the provided notebooks
- Code is structured in a clear and modular way


SUMMARY
This project demonstrates the application of modern machine learning methods to
both regression and classification problems using real-world data. It highlights the
importance of preprocessing, appropriate evaluation metrics, regularization, and
ethical awareness when working with complex and noisy datasets.

