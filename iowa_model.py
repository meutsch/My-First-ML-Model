import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# This function is used to test how the number of leaf nodes 
# affects the mean absolute error of a model.

# The best max_leaf_nodes for this model and data is 100.

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """
    Builds a decision tree regressor with the given values and 
    calculates its mean absolute error.

    Parameters:
    max_leaf_nodes (int): The maximum number of leaves in the model
    train_X: The training data X-values (independent variable)
    val_X: The validation data X-values
    train_y: The training data y-values (dependent variable)
    val_y: The validation data y-values

    Returns:
    mae: The mean absolute error for the produced model.
    """
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# Path of the file to read
iowa_file_path = Path(__file__).parent / "iowa_train.csv"

# Read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# # Print summary statistics
# print("\nIowa Data Summary:")
# print(home_data.describe())

# # Print the list of columns headers to find the name of the prediction target
# print("\nColumns in Iowa Data:")
# print(home_data.columns)

# Select the prediction target
y = home_data.SalePrice

# Create the list of features
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 
                 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# Build forest model
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
iowa_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, iowa_preds))

"""
My MAE: 21857.15912981083
Kaggle MAE: 21857.15912981083
"""

# # Review statistics from X and print the top few lines
# print("\nIowa Feature Data Summary:")
# print(X.describe())
# print("\nIowa Feature Data Head:")
# print(X.head())

# # Build multiple models using get_mae() and compare them
# candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]   # best = 100
# for max_leaf_nodes in candidate_max_leaf_nodes:
#     my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
#     print("Max leaf nodes:  %d   \t\t Mean Absolute Error:  %d" 
#           %(max_leaf_nodes, my_mae))

# # Specify the models
# # can set a numeric value for random_state when specifying the model
# old_iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
# iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

# # Fit the models
# old_iowa_model.fit(X, y)
# iowa_model.fit(train_X, train_y)

# # Make predictions
# old_predictions = old_iowa_model.predict(X)
# val_predictions = iowa_model.predict(val_X)

# print("\nMaking predictions for the following 5 houses in Iowa:")
# print(X.head())
# print("\nFirst in-sample predictions (in $):")
# print(old_iowa_model.predict(X.head()))
# print("\nNew predictions after splitting data (in $):")
# print(iowa_model.predict(X.head()))

# # Compare to actual values of home in data head
# print("\nActual home values (in $):")
# print(home_data['SalePrice'].head().to_list())

# # Calculate and display Mean Absolute Error
# old_mae = mean_absolute_error(y, old_predictions)
# val_mae = mean_absolute_error(val_y, val_predictions)
# print("\nThe mean absolute error of the 1st model is: $", old_mae)
# print("\nThe mean absolute error of the 2nd model is: $", val_mae)