import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

import imlightgbm as imlgb

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters for standard LightGBM model
params_standard = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    "early_stopping_rounds": 10,
}

# Train standard LightGBM model
lgb_standard = lgb.train(
    params_standard, train_data, num_boost_round=100, valid_sets=[test_data]
)

# Parameters for Imbalanced LightGBM model
params_imbalanced = {
    "objective": "binary_focal",  # binary_weighted
    "gamma": 2.0,  # alpha with binary_weighted
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    "early_stopping_rounds": 10,
}

# Train imbalanced LightGBM model
imlgb_focal = imlgb.train(
    params_imbalanced, train_data, num_boost_round=100, valid_sets=[test_data]
)

# Predict using standard LightGBM model
y_pred_standard = lgb_standard.predict(X_test)
y_pred_standard_binary = (y_pred_standard > 0.5).astype(int)

# Predict using Imbalanced LightGBM model
y_pred_focal = imlgb_focal.predict(X_test)
y_pred_focal_binary = (y_pred_focal > 0.5).astype(int)

# Evaluate models
accuracy_standard = accuracy_score(y_test, y_pred_standard_binary)
logloss_standard = log_loss(y_test, y_pred_standard)
rocauc_standard = roc_auc_score(y_test, y_pred_standard)

accuracy_focal = accuracy_score(y_test, y_pred_focal_binary)
logloss_focal = log_loss(y_test, y_pred_focal)
rocauc_focal = roc_auc_score(y_test, y_pred_focal)
print(
    f"Standard LightGBM - Accuracy: {accuracy_standard:.4f}, Log Loss: {logloss_standard:.4f}, rocauc: {rocauc_standard:.4f}"
)
print(
    f"LightGBM with Focal Loss - Accuracy: {accuracy_focal:.4f}, Log Loss: {logloss_focal:.4f}, rocauc: {rocauc_focal:.4f}"
)
