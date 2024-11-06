import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv('employe.csv')

# 1. Data Analysis and Visualization
plt.figure(figsize=(15, 5))

# Plot 1: Distribution of Salary
plt.subplot(131)
sns.histplot(df['Salary'], bins=30)
plt.title('Salary Distribution')
plt.xlabel('Salary')

# Plot 2: Distribution of Experience
plt.subplot(132)
sns.histplot(df['Experience (Years)'], bins=30)
plt.title('Experience Distribution')
plt.xlabel('Experience (Years)')

# Plot 3: Original Scatter Plot
plt.subplot(133)
sns.scatterplot(data=df, x='Experience(Years)', y='Salary')
plt.title('Salary vs Experience (Original)')

plt.tight_layout()
plt.show()

# 2. Handle Outliers using IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove outliers from both Salary and Experience
df_cleaned = df.copy()
df_cleaned = remove_outliers(df_cleaned, 'Salary')
df_cleaned = remove_outliers(df_cleaned, 'Experience(Years)')

# 3. Prepare the data
X = df_cleaned['Experience(Years)'].values.reshape(-1, 1)
y = df_cleaned['Salary'].values

# 4. Scale the features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 5. Split the scaled data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 6. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Make predictions
y_pred_scaled = model.predict(X_test)

# 8. Transform predictions back to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# 9. Calculate metrics
mse = mean_squared_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

print("After preprocessing:")
print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {np.sqrt(mse):.2f}')
print(f'R² Score: {r2:.2f}')

# 10. Visualization of improved model
plt.figure(figsize=(12, 6))

# Plot original data points
plt.scatter(X, df_cleaned['Salary'], color='blue', alpha=0.5, label='Cleaned Data')

# Plot regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_line_scaled = scaler_X.transform(X_line)
y_line_scaled = model.predict(X_line_scaled)
y_line = scaler_y.inverse_transform(y_line_scaled.reshape(-1, 1))

plt.plot(X_line, y_line, color='red', label='Regression Line')

plt.xlabel('Experience (Years)')
plt.ylabel('Salary')
plt.title('Salary vs Experience (Improved Model)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add regression equation and R² to the plot
equation = f'R² = {r2:.2f}'
plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# 11. Function to predict salary for new experience values
def predict_salary(experience_years):
    experience_scaled = scaler_X.transform([[experience_years]])
    salary_scaled = model.predict(experience_scaled)
    return scaler_y.inverse_transform(salary_scaled.reshape(-1, 1))[0][0]

# Example predictions
print("\nExample predictions:")
for exp in [2, 5, 10]:
    predicted_salary = predict_salary(exp)
    print(f'Predicted salary for {exp} years of experience: ${predicted_salary:,.2f}')

# 12. Print summary statistics
print("\nData Summary:")
print(f"Number of original records: {len(df)}")
print(f"Number of records after cleaning: {len(df_cleaned)}")
print(f"Percentage of data retained: {(len(df_cleaned)/len(df))*100:.1f}%")