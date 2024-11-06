# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample data: [hours studied, result (1=pass, 0=fail)]
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Predictions:", y_pred)
print("Probabilities:", y_pred_proba)
print("Accuracy:", accuracy)
