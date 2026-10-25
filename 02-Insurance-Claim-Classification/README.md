# Insurance Claim Classification

**Data Science and Business Analytics – Machine Learning Final Project**
**Authors:** Katarzyna Pełka, Izabela Kosiec

## Project Overview
The goal of this project is to apply machine learning algorithms to a classification task, predicting travel insurance claim approvals based on customer demographics, trip details, and product characteristics.

## Dataset
**Source:** `insurance_train.csv` and `insurance_test.csv`.
**Target variable:** `claim_status` (0: Not Approved, 1: Approved).

## Key Preprocessing Steps
- Created a categorical indicator `reward_zero` to flag observations where the reward value is 0.
- Removed erroneous observations, such as records with a `customer_score` of 1.0.
- Imputed `person_age` anomalies (e.g., age 100) with the median age.
- Capped extreme `trip_length` values (> 4000) using the median from a specific location and product subset ('singapore' and 'TripGuard TicketCover').
- Grouped rare `location` categories (appearing in < 0.5% of observations) into an 'Other' category.
- Grouped `product_id` into broader product families (e.g., TripGuard, SecurePlan) to reduce cardinality and dropped the original column.
- Dropped near-zero variance features (e.g., `channel`) and features with very low correlation to the target (`support_interactions`, `person_age`).
- Applied one-hot encoding to the remaining categorical features (`person_gender`, `entity_type`, `agent_id`, `entity_a`, `location`, `prod_id_grouped`).

## Models Used
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. XGBoost Classifier

## Evaluation & Results
- **Class Imbalance Handling:** Firmly tested multiple resampling techniques on Logistic Regression to gauge performance, including Random Over/Under Sampler, SMOTE, ADASYN, Tomek Links, and SMOTETomek.
- **Hyperparameter Tuning:** Conducted using 5-fold cross-validation across comprehensive parameter grids for all models.
- **Final model:** SVM.
- **Best Parameters:** `C=1`, `gamma=0.01`, paired with `SMOTETomek` for resampling.
- **Results:** While cross-validation initially favored an SVM with Random Under Sampler, validation set performance was optimized by the `SMOTETomek` approach, yielding a final Balanced Accuracy of 0.7858.

## Reproducibility
The data splitting utilizes a fixed random seed (42) to ensure consistent stratifications. The full pipeline involves exporting the final trained model as `estimator.pkl` via `joblib`, allowing seamless generation of predictions for the test set, which are saved to a dedicated CSV file (`insurance_test_with_predictions.csv`).