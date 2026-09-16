from pathlib import Path
import pandas as pd

""" Iowa Dataset """

# # Path of the file to read
# iowa_file_path = Path(__file__).parent / "iowa_train.csv"

# # Read the file into a variable home_data
# home_data = pd.read_csv(iowa_file_path)

# # Print summary statistics in next line
# print("Iowa Data Summary:")
# print(home_data.describe())

""" Melbourne Dataset """

# Path of the file to read
melbourne_file_path = Path(__file__).parent / "melb_data.csv"

# Read the file into a variable melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

# Print columns
print(melbourne_data.columns)
