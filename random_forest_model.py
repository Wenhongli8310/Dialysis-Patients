# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, roc_curve, accuracy_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np

# Load preprocessed data
X_train = pd.read_csv("X_train_scaled.csv")
X_test = pd.read_csv("X_test_scaled.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()  # Flatten array for compatibility
y_test = pd.read_csv("y_test.csv").values.ravel()

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best parameters found for Random Forest:", best_params)

# Train Random Forest model with best parameters
rf_model = RandomForestClassifier(**best_params, random_state=42)
rf_model.fit(X_train, y_train)

# Model predictions and probabilities
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Performance metrics
rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
rf_brier = brier_score_loss(y_test, y_pred_proba_rf)
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)

# Confusion Matrix for sensitivity and specificity calculation
conf_matrix = confusion_matrix(y_test, y_pred_rf)
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Print performance metrics
print("Random Forest Performance Metrics:")
print(f"AUC: {rf_auc:.2f}")
print(f"Brier Score: {rf_brier:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_rf)
plt.figure()
plt.plot(fpr, tpr, color='green', label=f"Random Forest (AUC = {rf_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest')
plt.legend()
plt.savefig('random_forest_roc.png')
plt.show()

# Feature Importance Plot
feature_importances = rf_model.feature_importances_
feature_names = X_train.columns
indices = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), feature_importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Feature Importance in Random Forest Model')
plt.gca().invert_yaxis()  # Reverse order for better readability
plt.savefig('random_forest_feature_importance.png')
plt.show()

# Save model metrics to a file for future reference
metrics = {
    "AUC": rf_auc,
    "Brier Score": rf_brier,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall (Sensitivity)": recall,
    "Sensitivity": sensitivity,
    "Specificity": specificity
}
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv("random_forest_metrics.csv", index=False)

print("Random Forest analysis complete. Metrics, ROC curve, and feature importance plot saved.")
