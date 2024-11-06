import pandas
from sklearn.preprocessing import LabelEncoder

data = pandas.read_csv('employe.csv')
# Create a label encoder
label_encoder = LabelEncoder()

# Apply label encoding to a column (e.g., 'Position')
data['Position_encoded'] = label_encoder.fit_transform(data['Position'])

print(data.head())
