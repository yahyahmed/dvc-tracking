import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()

# Create a Pandas dataframe for the features
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Create a Pandas dataframe for the target variable
y = pd.DataFrame(data=iris.target, columns=['target'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the preprocessed data to dataframes
X_train_processed = pd.DataFrame(X_train_scaled, columns=iris.feature_names)
X_test_processed = pd.DataFrame(X_test_scaled, columns=iris.feature_names)

# Concatenate the processed data with the target variable
train_data = pd.concat([X_train_processed, y_train], axis=1)
test_data = pd.concat([X_test_processed, y_test], axis=1)

# Save the preprocessed data to CSV files
train_data.to_csv('data_processed_train.csv', index=False)
test_data.to_csv('data_processed_test.csv', index=False)
