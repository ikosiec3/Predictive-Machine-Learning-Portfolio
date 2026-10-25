# Predictive Machine Learning Portfolio

This repository contains four end-to-end machine learning projects, demonstrating a comprehensive approach to data preprocessing, exploratory data analysis (EDA), model selection, tuning, and evaluation.

##  Projects Overview

### [1. Apartment Price Regression](./01-Apartment-Price-Regression)
* **Goal:** Predict real estate prices based on property features, location, and local infrastructure.
* **Key Techniques:** Custom missing data imputation, log-transformations of right-skewed variables, handling multicollinearity.
* **Final Model:** XGBoost Regressor (Test RMSE: 95,074 PLN).

### [2. Insurance Claim Classification](./02-Insurance-Claim-Classification)
* **Goal:** Predict travel insurance claim approvals using customer demographics, trip details, and product characteristics.
* **Key Techniques:** Feature extraction, grouping rare categories, addressing severe class imbalance using SMOTETomek.
* **Final Model:** Support Vector Machine (SVM) (Balanced Accuracy: 0.78).

### [3. Airbnb Nightly Price Regression](./03-Airbnb-Price-Regression)
* **Goal:** Predict nightly prices of Airbnb listings in New York City.
* **Key Techniques:** Outlier removal (capping at 99.5th percentile), spatial data visualization, ensemble modeling comparison.
* **Final Model:** XGBoost Regressor with early stopping (Test RMSE: 90.48 USD).

### [4. Wellbeing & Gender Classification](./04-Wellbeing-Gender-Classification)
* **Goal:** Predict gender based on wellbeing, daily habits, and lifestyle survey data.
* **Key Techniques:** Correlation-based feature selection, handling highly overlapping distributions, class-weighting.
* **Final Model:** Class-weighted XGBoost Classifier (F1-score: 0.54).