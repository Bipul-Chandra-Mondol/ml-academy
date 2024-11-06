import pandas

data = pandas.read_csv('employe.csv')

# Using pandas get_dummies for one-hot encoding
data_one_hot = pandas.get_dummies(data, columns=['Position'])

print(data_one_hot.head())
