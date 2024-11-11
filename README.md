# Predicting Hospitalization Risk in Dialysis Patients

This project provides a comparative analysis of four machine learning models to predict hospitalization risk among dialysis patients. The models include **Baseline Logistic Regression**, **Random Forest**, **Logistic Regression with Feature Engineering**, and **Logistic Regression with Interaction Terms**. The dataset includes clinical data such as age, duration on dialysis, diabetes, albumin levels, and hospitalization status.


## Dataset

The dataset `dialysis_data.csv` contains clinical information for each patient, including:
- `age`: Age of the patient
- `duration_on_dialysis`: Duration of time the patient has been on dialysis
- `diabetes`: Presence of diabetes (binary variable)
- `albumin_level`: Serum albumin level
- `hospitalization_status`: Target variable (1 if the patient was hospitalized, 0 otherwise)

## Requirements

To install necessary libraries, run:
```bash
pip install -r requirements.txt

##  Data Preprocessing
The script data_preprocessing.py performs data loading, missing value handling, feature engineering, and data scaling. It generates training and testing datasets which are saved as CSV files for consistency across models.

python data_preprocessing.py

##  Baseline Logistic Regression
The baseline_logistic_regression.py file trains and evaluates a baseline logistic regression model. It calculates AUC, Brier Score, sensitivity, specificity, and generates both ROC and Calibration curves.

python baseline_logistic_regression.py

## Random Forest Model
The script random_forest_model.py builds and tunes a Random Forest model using GridSearchCV. It evaluates performance metrics (AUC, Brier Score, etc.) and plots ROC and feature importance charts.

python random_forest_model.py
