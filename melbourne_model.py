from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
melbourne_file_path = Path(__file__).parent / "melb_data.csv"

# Read the file into a variable melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

# Print summary statistics
print("\nMelbourne Data Summary:")
print(melbourne_data.describe())

# Print all columns
print("\nColumns in Melbourne Data:")
print(melbourne_data.columns)       # Prediction target: "Price"

# The Melbourne data has some missing values - drop these rows from data
melbourne_data = melbourne_data.dropna(axis=0)

# Review summary statistics again
print("\nNew Melbourne Data Summary (Incomplete Data Removed):")
print(melbourne_data.describe())

# Selecting the prediction target
y = melbourne_data.Price

# Choosing features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 
                      'Longtitude']
X = melbourne_data[melbourne_features]

# Review the data corresponding to features
print("\nMelbourne Feature Data Summary:")
print(X.describe())
print("\nMelbourne Feature Data Head:")
print(X.head())

# Define model. Specify a random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit the model
melbourne_model.fit(X, y)

# Make predictions
print("\nMaking predictions for the following 5 houses:")
print(X.head())
print("\nThe predictions are")
print(melbourne_model.predict(X.head()))

# Compare to actual values of home in data head
print("\nThe actual values are:")
print(melbourne_data['Price'].head().to_list())