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

# Initialize the ImbalancedLGBMClassifier using binary focal loss
clf = imlgb.ImbalancedLGBMClassifier(
    objective="binary_focal",  # binary_weighted
    gamma=2.0,  # alpha with binary_weighted
    learning_rate=0.05,
    num_leaves=31,
)

# Train the classifier on the training data
clf.fit(X=X_train, y=y_train)

# Make predictions on the test data
y_pred_focal = clf.predict(X_test)

# Convert probabilities into binary predictions (0 or 1) based on a threshold of 0.5
y_pred_focal_binary = (y_pred_focal > 0.5).astype(int)

# Evaluate the model performance using accuracy, log loss, and ROC AUC
accuracy_focal = accuracy_score(y_test, y_pred_focal_binary)
logloss_focal = log_loss(y_test, y_pred_focal)
rocauc_focal = roc_auc_score(y_test, y_pred_focal)

# Print the evaluation results
print(
    f"LightGBM with Focal Loss - Accuracy: {accuracy_focal:.4f}, Log Loss: {logloss_focal:.4f}, rocauc: {rocauc_focal:.4f}"
)
