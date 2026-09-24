import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Models can suffer from either:
# Overfitting: capturing spurious patterns that won't recur in 
# the future, leading to less accurate predictions, or
# Underfitting: failing to capture relevant patterns, again
# leading to less accurate predictions.
#
# This function is used to test how the number of leaf nodes 
# affects the mean absolute error of a model.
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
melbourne_file_path = Path(__file__).parent / "melb_data.csv"

# Read the file into a variable melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

# # Print summary statistics
# print("\nMelbourne Data Summary:")
# print(melbourne_data.describe())

# # Print all columns
# print("\nColumns in Melbourne Data:")
# print(melbourne_data.columns)       # Prediction target: "Price"

# The Melbourne data has some missing values - drop these rows from data
filtered_melbourne_data = melbourne_data.dropna(axis=0)

# # Review summary statistics again
# print("\nNew Melbourne Data Summary (Incomplete Data Removed):")
# print(filtered_melbourne_data.describe())

# Selecting the prediction target
y = filtered_melbourne_data.Price

# Choosing features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 
                      'Longtitude']
X = filtered_melbourne_data[melbourne_features]

# Split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Build forest model
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

""" ERROR:
My MAE: 207226.35743972045
Kaggle MAE: 191669.7536453626
"""

# # Review the data corresponding to features
# print("\nMelbourne Feature Data Summary:")
# print(X.describe())
# print("\nMelbourne Feature Data Head:")
# print(X.head())

# # Build multiple models using get_mae() and compare them
# max_leaves = [5, 50, 500, 5000]
# for max_leaf_nodes in max_leaves:
#     my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
#     print("Max leaf nodes:  %d   \t\t Mean Absolute Error:  %d" 
#           %(max_leaf_nodes, my_mae))

# # Define the models
# old_melbourne_model = DecisionTreeRegressor(random_state=1)
# melbourne_model = DecisionTreeRegressor(random_state = 1)

# # Fit the models
# old_melbourne_model.fit(X, y)
# melbourne_model.fit(train_X, train_y)

# # get predicted prices on validation data
# old_predictions = old_melbourne_model.predict(X)
# val_predictions = melbourne_model.predict(val_X)

# # Make predictions
# print("\nMaking predictions for the following 5 houses in Melbourne:")
# print(X.head())
# print("\nFirst in-sample predictions (in $):")
# print(old_melbourne_model.predict(X.head()))
# print("\nNew predictions after splitting data (in $):")
# print(melbourne_model.predict(X.head()))

# # Compare to actual values of home in data head
# print("\nActual home values (in $):")
# print(filtered_melbourne_data['Price'].head().to_list())

# # Calculate and display Mean Absolute Error
# old_mae = mean_absolute_error(y, old_predictions)
# val_mae = mean_absolute_error(val_y, val_predictions)
# print("\nThe mean absolute error of the 1st model is: $", old_mae)
# print("\nThe mean absolute error of the 2nd model is: $", val_mae)

# ''' ERROR: 1st Model MAE
# My MAE: 1115.7467183128902
# Kaggle MAE: 434.71594577146544 '''

# ''' ERROR - 2nd Model MAE
# My MAE: 275312.2375726275
# Kaggle MAE: 265806.91478373145 '''