import pandas
import numpy as np
from sklearn.preprocessing import MinMaxScaler,Normalizer
data = pandas.read_csv('employe.csv')
# print(data.head())

# max_vals = np.max(np.abs(data))

numerical_columns = ['Salary','Experience (Years)']

data_scaled = data.copy()
# max_vals = np.max(np.abs(data[numerical_columns]))
# print(max_vals)
# data_scaled = (data[numerical_columns] - max_vals)/max_vals
# print(data_scaled)
# min_vals = np.min(np.abs(data[numerical_columns]))

# min max scalling
'''scaler = MinMaxScaler()
data_scaled[numerical_columns] = scaler.fit_transform(data[numerical_columns])
print(data_scaled.head()) '''


normalizer = Normalizer()
data_scaled[numerical_columns] = normalizer.fit_transform(data[numerical_columns])
print(data_scaled[numerical_columns])
