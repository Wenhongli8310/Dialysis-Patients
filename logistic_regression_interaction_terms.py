# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve, accuracy_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

# Load preprocessed data
X_train = pd.read_csv("X_train_scaled.csv")
X_test = pd.read_csv("X_test_scaled.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()  # Flatten array for compatibility
y_test = pd.read_csv("y_test.csv").values.ravel()

# Adding Interaction Terms
# PolynomialFeatures(degree=2, interaction_only=True) creates only interaction terms without higher-degree terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_interactions = poly.fit_transform(X_train)
X_test_interactions = poly.transform(X_test)
interaction_feature_names = poly.get_feature_names_out(X_train.columns)

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
grid_search.fit(X_train_interactions, y_train)
best_params = grid_search.best_params_
print("Best parameters found for Logistic Regression with Interaction Terms:", best_params)

# Train Logistic Regression model with best parameters
log_reg_interaction = LogisticRegression(**best_params, random_state=42)
log_reg_interaction.fit(X_train_interactions, y_train)

# Model predictions and probabilities
y_pred_interaction = log_reg_interaction.predict(X_test_interactions)
y_pred_proba_interaction = log_reg_interaction.predict_proba(X_test_interactions)[:, 1]

# Performance metrics
interaction_auc = roc_auc_score(y_test, y_pred_proba_interaction)
interaction_brier = brier_score_loss(y_test, y_pred_proba_interaction)
accuracy = accuracy_score(y_test, y_pred_interaction)
precision = precision_score(y_test, y_pred_interaction)
recall = recall_score(y_test, y_pred_interaction)

# Confusion Matrix for sensitivity and specificity calculation
conf_matrix = confusion_matrix(y_test, y_pred_interaction)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Print performance metrics
print("Logistic Regression with Interaction Terms Performance Metrics:")
print(f"AUC: {interaction_auc:.2f}")
print(f"Brier Score: {interaction_brier:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_interaction)
plt.figure()
plt.plot(fpr, tpr, color='orange', label=f"Logistic Regression with Interaction Terms (AUC = {interaction_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression with Interaction Terms')
plt.legend()
plt.savefig('logistic_regression_interaction_terms_roc.png')
plt.show()

# Plot Calibration Curve
plt.figure()
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba_interaction, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Logistic Regression with Interaction Terms')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Perfect Calibration")
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Probability')
plt.title('Calibration Curve for Logistic Regression with Interaction Terms')
plt.legend()
plt.savefig('logistic_regression_interaction_terms_calibration.png')
plt.show()

# Save model metrics to a file for future reference
metrics = {
    "AUC": interaction_auc,
    "Brier Score": interaction_brier,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall (Sensitivity)": recall,
    "Sensitivity": sensitivity,
    "Specificity": specificity
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("logistic_regression_interaction_terms_metrics.csv", index=False)

print("Logistic Regression with Interaction Terms analysis complete. Metrics, ROC curve, and calibration curve saved.")
