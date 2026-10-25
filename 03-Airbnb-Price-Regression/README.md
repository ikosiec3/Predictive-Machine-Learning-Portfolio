# Airbnb Nightly Prices Prediction in NYC

**Data Science and Business Analytics – Machine Learning Final Project**
**Authors:** Katarzyna Pełka, Izabela Kosiec

## Project Overview
The goal of this project is to apply and compare multiple machine learning algorithms for a regression task using a real-world dataset. We predict nightly prices of Airbnb listings in New York City based on property characteristics, location, and host activity.

## Dataset
**Source:** Kaggle (new_york_listings_2024.csv)
The dataset contains 20,758 observations and 22 variables.
**Target variable:** `price` (nightly price in USD)

## Key Preprocessing Steps
* Log-transformation of the target variable to handle heavy right-skewness.
* Removal of the top 0.5% most expensive listings to improve model stability.
* Conversion of categorical variables using one-hot encoding.
* Simplification of high-cardinality variables (e.g. license, neighbourhood).
* Treatment of studios as 0 bedrooms and conversion of baths to numeric values with median imputation.
* Capping extreme values for selected numeric variables and removal of non-informative columns.

## Models Used
1. Baseline model (mean predictor)
2. Random Forest Regressor (bagging with OOB validation)
3. Gradient Boosting Regressor
4. XGBoost Regressor
5. Ensemble model (Random Forest + XGBoost)

## Evaluation & Results
* **Metrics:** RMSE, MAE, R²
* **Final model:** XGBoost Regressor with early stopping
* **Best test performance:** RMSE ≈ 90.48 USD, R² ≈ 0.555

Feature importance analysis confirmed that price is mainly driven by room type, property size (bedrooms, bathrooms, beds), location (neighbourhood group), and license status.

## Reproducibility
All experiments use a fixed random seed. Full preprocessing, modeling, and evaluation pipelines are included and can be fully reproduced by running the provided Jupyter Notebook. The project demonstrates the application of modern ML methods, highlighting the importance of preprocessing, appropriate evaluation metrics, and regularization.