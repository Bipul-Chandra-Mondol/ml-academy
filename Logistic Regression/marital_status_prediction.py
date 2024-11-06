import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv("logistic_regression_data.csv")

# Add a target column with binary values (e.g., purchase made or not made)
df['Target'] = df['Marital_status'].apply(lambda x: 1 if x == 'Married' else 0)

# Convert categorical data to numerical format
df['Marital_status'] = df['Marital_status'].map({'Single': 0, 'Married': 1})

# Define features (X) and target (y)
X = df[['Age', 'Marital_status']]
y = df['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Predictions:", y_pred)
print("Probabilities:", y_pred_proba)
print("Accuracy:", accuracy)
