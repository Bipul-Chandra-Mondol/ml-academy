import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset (assuming it's a CSV file)
df = pd.read_csv('employe.csv')

# Prepare the data
X = df['Experience (Years)'].values.reshape(-1, 1)  # Independent variable
y = df['Salary'].values  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model performance
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')
print(f'Coefficient (slope): {model.coef_[0]:.2f}')
print(f'Intercept: {model.intercept_:.2f}')

# Create visualization
plt.figure(figsize=(10, 6))

# Plot the training data points
plt.scatter(X, y, color='blue', alpha=0.5, label='Actual Data')

# Plot the regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='red', label='Regression Line')

# Customize the plot
plt.xlabel('Experience (Years)')
plt.ylabel('Salary')
plt.title('Salary vs Experience Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)

# Add regression equation and R² to the plot
equation = f'y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}'
plt.text(0.05, 0.95, f'Regression Equation: {equation}\nR² = {r2:.2f}', 
         transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Function to predict salary for new experience values
def predict_salary(experience_years):
    return model.predict([[experience_years]])[0]

# Example prediction
example_experience = 5
predicted_salary = predict_salary(example_experience)
print(f'\nPredicted salary for {example_experience} years of experience: {predicted_salary:.2f}')