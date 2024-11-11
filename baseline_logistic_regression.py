# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve, auc, 
    accuracy_score, precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

# Load preprocessed data
X_train = pd.read_csv("X_train_scaled.csv")
X_test = pd.read_csv("X_test_scaled.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()  # Flatten array for compatibility
y_test = pd.read_csv("y_test.csv").values.ravel()

# Train baseline logistic regression model
baseline_log_reg = LogisticRegression()
baseline_log_reg.fit(X_train, y_train)

# Model predictions and probabilities
y_pred_baseline = baseline_log_reg.predict(X_test)
y_pred_proba_baseline = baseline_log_reg.predict_proba(X_test)[:, 1]

# Performance metrics
baseline_auc = roc_auc_score(y_test, y_pred_proba_baseline)
baseline_brier = brier_score_loss(y_test, y_pred_proba_baseline)
accuracy = accuracy_score(y_test, y_pred_baseline)
precision = precision_score(y_test, y_pred_baseline)
recall = recall_score(y_test, y_pred_baseline)

# Confusion Matrix for sensitivity and specificity calculation
conf_matrix = confusion_matrix(y_test, y_pred_baseline)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Print performance metrics
print("Baseline Logistic Regression Performance Metrics:")
print(f"AUC: {baseline_auc:.2f}")
print(f"Brier Score: {baseline_brier:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_baseline)
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f"Baseline Logistic Regression (AUC = {baseline_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Baseline Logistic Regression')
plt.legend()
plt.savefig('baseline_logistic_regression_roc.png')
plt.show()

# Plot Calibration Curve
plt.figure()
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba_baseline, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Baseline Logistic Regression')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Perfect Calibration")
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Probability')
plt.title('Calibration Curve for Baseline Logistic Regression')
plt.legend()
plt.savefig('baseline_logistic_regression_calibration.png')
plt.show()

# Save model metrics to a file for future reference
metrics = {
    "AUC": baseline_auc,
    "Brier Score": baseline_brier,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall (Sensitivity)": recall,
    "Sensitivity": sensitivity,
    "Specificity": specificity
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("baseline_logistic_regression_metrics.csv", index=False)

print("Baseline Logistic Regression analysis complete. Metrics and plots saved.")
