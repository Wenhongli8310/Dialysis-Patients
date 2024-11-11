# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load dataset from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def handle_missing_values(data):
    """
    Handle missing values by filling them with the median of each column.
    Args:
        data (pd.DataFrame): Input data with possible missing values.
    Returns:
        pd.DataFrame: Data with missing values handled.
    """
    return data.fillna(data.median())

def create_features(data):
    """
    Create additional features including non-linear transformations and interaction terms.
    Args:
        data (pd.DataFrame): Original data.
    Returns:
        pd.DataFrame: Data with additional features.
    """
    data['age_squared'] = data['age'] ** 2
    data['log_dialysis_duration'] = np.log(data['duration_on_dialysis'] + 1)  # Avoid log(0)
    data['age_dialysis_interaction'] = data['age'] * data['duration_on_dialysis']
    return data

def split_data(data, target_column='hospitalization_status', test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    Args:
        data (pd.DataFrame): Input data.
        target_column (str): Column name of the target variable.
        test_size (float): Proportion of test set.
        random_state (int): Random seed for reproducibility.
    Returns:
        tuple: X_train, X_test, y_train, y_test datasets.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):
    """
    Standardize the feature data.
    Args:
        X_train (pd.DataFrame): Training data.
        X_test (pd.DataFrame): Testing data.
    Returns:
        tuple: Scaled X_train and X_test data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Load and preprocess data
file_path = 'dialysis_data.csv'  # Adjust the path as necessary
data = load_data(file_path)
data = handle_missing_values(data)
data = create_features(data)
X_train, X_test, y_train, y_test = split_data(data)
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# Saving processed data for model training
pd.DataFrame(X_train_scaled).to_csv('X_train_scaled.csv', index=False)
pd.DataFrame(X_test_scaled).to_csv('X_test_scaled.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Data preprocessing complete. Preprocessed data saved for model training.")
