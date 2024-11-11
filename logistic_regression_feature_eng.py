# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve, accuracy_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import PolynomialFeatures

# Load preprocessed data
X_train = pd.read_csv("X_train_scaled.csv")
X_test = pd.read_csv("X_test_scaled.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()  # Flatten array for compatibility
y_test = pd.read_csv("y_test.csv").values.ravel()

# Feature Engineering: Add polynomial features and interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'solver': ['liblinear', 'saga'],  # Solvers that support L1 and L2 regularization
    'penalty': ['l1', 'l2']  # Types of regularization
}

# Hyperparameter tuning with GridSearchCV
grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train_poly, y_train)
best_params = grid_search.best_params_
print("Best parameters found for Logistic Regression with Feature Engineering:", best_params)

# Train Logistic Regression model with best parameters
log_reg_feature_eng = LogisticRegression(**best_params, random_state=42)
log_reg_feature_eng.fit(X_train_poly, y_train)

# Model predictions and probabilities
y_pred_feature_eng = log_reg_feature_eng.predict(X_test_poly)
y_pred_proba_feature_eng = log_reg_feature_eng.predict_proba(X_test_poly)[:, 1]

# Performance metrics
feature_eng_auc = roc_auc_score(y_test, y_pred_proba_feature_eng)
feature_eng_brier = brier_score_loss(y_test, y_pred_proba_feature_eng)
accuracy = accuracy_score(y_test, y_pred_feature_eng)
precision = precision_score(y_test, y_pred_feature_eng)
recall = recall_score(y_test, y_pred_feature_eng)

# Confusion Matrix for sensitivity and specificity calculation
conf_matrix = confusion_matrix(y_test, y_pred_feature_eng)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Print performance metrics
print("Logistic Regression with Feature Engineering Performance Metrics:")
print(f"AUC: {feature_eng_auc:.2f}")
print(f"Brier Score: {feature_eng_brier:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_feature_eng)
plt.figure()
plt.plot(fpr, tpr, color='purple', label=f"Logistic Regression with Feature Engineering (AUC = {feature_eng_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression with Feature Engineering')
plt.legend()
plt.savefig('logistic_regression_feature_eng_roc.png')
plt.show()

# Plot Calibration Curve
plt.figure()
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba_feature_eng, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Logistic Regression with Feature Engineering')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Perfect Calibration")
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Probability')
plt.title('Calibration Curve for Logistic Regression with Feature Engineering')
plt.legend()
plt.savefig('logistic_regression_feature_eng_calibration.png')
plt.show()

# Save model metrics to a file for future reference
metrics = {
    "AUC": feature_eng_auc,
    "Brier Score": feature_eng_brier,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall (Sensitivity)": recall,
    "Sensitivity": sensitivity,
    "Specificity": specificity
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("logistic_regression_feature_eng_metrics.csv", index=False)

print("Logistic Regression with Feature Engineering analysis complete. Metrics, ROC curve, and calibration curve saved.")
