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

# Initialize the ImbalancedLGBMClassifier using binary focal loss
clf = imlgb.ImbalancedLGBMClassifier(
    objective="multiclass_focal",  # multiclass_weighted
    gamma=2.0,  # alpha with multiclass_weighted
    num_class=3,
    learning_rate=0.05,
    num_leaves=31,
)

# Train the classifier on the training data
clf.fit(X=X_train, y=y_train)

# Make predictions on the test data
y_pred_focal = clf.predict(X_test)


# Evaluate the model performance using accuracy, log loss, and ROC AUC
# Evaluate models
print("\nClassification Report:")
print(classification_report(y_test, y_pred_focal))
