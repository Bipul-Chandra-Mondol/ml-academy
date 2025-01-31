import pandas as pd
data =pd.read_csv('employee_dataset.csv')

# print(type(data))

# print(data.head())

# print(data.tail())

# print(data.shape)

# print(data.info())

# print(data.isnull().sum())

# 7. Find Percentage of Missing Value for All Attributes

# missing_percentage = ( data.isnull().sum() / len(data))*100
# print(missing_percentage)

# 7. Find Duplicate Data
# duplicate_rows = data.duplicated()
# print(duplicate_rows.sum())
# print(data[duplicate_rows])

import numpy as np

# Identify outliers using IQR
def find_outliers_iqr(data):
    outliers = {}
    for column in data.select_dtypes(include=np.number).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[column] = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

print("\nOutliers in the dataset:")
outliers = find_outliers_iqr(data)
for column, outlier_data in outliers.items():
    print(f"Outliers in {column}:")
    print(outlier_data)

# Drop rows with missing values
data_cleaned = data.dropna()

# Or drop columns with missing values
# data_cleaned = data.dropna(axis=1)

# Fill missing values with mean for numerical columns
data_filled = data.fillna(data.mean())

# Fill missing values with the mode for categorical columns
data_filled = data.fillna(data.mode().iloc[0])


# Drop duplicate rows
data_no_duplicates = data.drop_duplicates()


def remove_outliers_iqr(data):
    for column in data.select_dtypes(include=np.number).columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data

data_no_outliers = remove_outliers_iqr(data)
