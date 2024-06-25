import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading data...")
# Load the data
file_path = 'data/Sales History-revised-20240516.xlsx'
data = pd.read_excel(file_path, skiprows=5)

print("Renaming columns...")
# Rename the columns
data.columns = ['Index', 'Year', 'Residential_GWh', 'Residential_No', 'Farm_GWh', 'Farm_No', 
                'Commercial_GWh', 'Commercial_No', 'Industrial_GWh', 'Industrial_No', 
                'Total_GWh', 'Total_No']
data = data.drop(columns=['Index'])
data = data.drop([0, 1, 2])
data.fillna(0, inplace=True)

print("Cleaning 'Year' column...")
# Remove any rows where 'Year' is not numeric
data = data[pd.to_numeric(data['Year'], errors='coerce').notnull()]

print("Converting data types...")
# Convert data types
data = data.astype({
    'Year': 'int',
    'Residential_GWh': 'float',
    'Residential_No': 'int',
    'Farm_GWh': 'float',
    'Farm_No': 'int',
    'Commercial_GWh': 'float',
    'Commercial_No': 'int',
    'Industrial_GWh': 'float',
    'Industrial_No': 'int',
    'Total_GWh': 'float',
    'Total_No': 'int'
})

print("Splitting data into training and testing sets...")
# Split the data
X = data.drop(columns=['Total_GWh', 'Total_No'])
y = data['Total_GWh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Saving the split data...")
# Save the split data
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("Data preparation complete.")