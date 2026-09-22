from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = Path(__file__).parent / "iowa_train.csv"

# Read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Print summary statistics
print("\nIowa Data Summary:")
print(home_data.describe())

# Print the list of columns headers to find the name of the prediction target
print("\nColumns in Iowa Data:")
print(home_data.columns)

# Select the prediction target
y = home_data.SalePrice

# Create the list of features
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 
                 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Review statistics from X and print the top few lines
print("\nIowa Feature Data Summary:")
print(X.describe())
print("\nIowa Feature Data Head:")
print(X.head())

# Specify the model
# can set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X, y)

# Make predictions
predictions = iowa_model.predict(X)

print("\nMaking predictions for the following 5 houses in Iowa:")
print(X.head())
print("\nThe predicted values are:")
print(iowa_model.predict(X.head()))

# Compare to actual values of home in data head
print("\nThe actual values are:")
print(home_data['SalePrice'].head().to_list())