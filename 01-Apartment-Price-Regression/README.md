# Apartment Price Prediction

**Data Science and Business Analytics – Machine Learning Final Project**
**Authors:** Katarzyna Pełka, Izabela Kosiec

## Project Overview
The goal of this project is to apply machine learning algorithms to a regression task, predicting apartment prices based on property features, location, and infrastructure.

## Dataset
**Source:** `appartments_train.csv` and `appartments_test.csv`.
**Target variable:** `price_z` (log-transformed to `log_price_z` for modeling).

## Key Preprocessing Steps
- Imputed missing `year_built` values with the median and created a `year_built_missing` categorical indicator.
- Filled missing distances (e.g., `dist_sch`, `dist_post`) with median values.
- Derived missing `floor_no` using half of `floor_max` or set to 5 if a lift is present, otherwise using the mode.
- Inferred missing `floor_max` based on `floor_no`, and `has_lift` based on `floor_max > 4`.
- Imputed `infrastructure_quality` using the mean quality for the specific `year_built`.
- Extracted temporal features (`src_year`, `src_month_num`, `src_quarter_str`) from the `src_month` variable.
- Applied log-transformations to right-skewed features (e.g., `dim_m2`, distances) and the target variable.
- Dropped highly correlated features (`green_space_ratio`, `infrastructure_quality`) to prevent multicollinearity.
- Built a preprocessing pipeline using `SimpleImputer` and `StandardScaler` for numerical features, and `OneHotEncoder` for categorical variables.

## Models Used
1. Ridge
2. Lasso
3. Elastic Net
4. SVR
5. Linear Regression
6. Random Forest Regressor
7. XGBoost Regressor

## Evaluation & Results
- **Hyperparameter Tuning:** Conducted using `RandomizedSearchCV` (for XGB, SVR, Elastic Net) and `GridSearchCV` (for others) with 5-fold cross-validation on a 10,000-observation sample.
- **Final model:** XGBoost Regressor.
- **Best Parameters:** `n_estimators=600`, `learning_rate=0.02`, `max_depth=8`, `subsample=0.7`, `colsample_bytree=0.8`.
- **Results:** The final model achieved a training RMSE of 81,921 PLN and a validation RMSE of 95,074 PLN, showing good generalization with limited overfitting (unlike Random Forest, which severely overfit on the training set).
- **Residual Analysis:** The model predicts housing prices with high accuracy, though precision decreases slightly for very expensive properties due to fewer observations and stronger nonlinear effects in that range.

## Reproducibility
All models and preprocessing steps use a fixed random seed (42) to ensure consistent results. The code is structured in a clear pipeline, allowing predictions on the test set to be transformed back from the logarithmic scale and seamlessly saved to a CSV file for final output.