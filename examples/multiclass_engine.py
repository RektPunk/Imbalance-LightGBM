import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import imlightgbm as imlgb

# Generate dataset
X, y = make_classification(
    n_samples=5000,
    n_features=10,
    n_classes=3,
    n_informative=5,
    weights=[0.05, 0.15, 0.8],
    flip_y=0,
    random_state=42,
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Parameters for standard LightGBM model
params = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
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
    params, train_data, num_boost_round=100, valid_sets=[test_data]
)

# Predict using standard LightGBM model
y_pred_standard = lgb_standard.predict(X_test)
y_pred_standard_label = np.argmax(y_pred_standard, axis=1)

# Parameters for Imbalanced LightGBM model
params = {
    "objective": "multiclass_focal",  # multiclass_weighted
    "num_class": 3,
    "gamma": 2.0,  # alpha with binary_weighted
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42,
    "early_stopping_rounds": 10,
}

# Train Imbalanced LightGBM model
imlgb_focal = imlgb.train(
    params, train_data, num_boost_round=100, valid_sets=[test_data]
)

# Predict using Imbalanced LightGBM model
y_pred_focal = imlgb_focal.predict(X_test)
y_pred_focal_label = np.argmax(y_pred_focal, axis=1)


# Evaluate models
print("\nClassification Report for Standard:")
print(classification_report(y_test, y_pred_standard_label))
print("\nClassification Report for Imbalanced:")
print(classification_report(y_test, y_pred_focal_label))
