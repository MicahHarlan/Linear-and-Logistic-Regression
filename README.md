# Linear-and-Logistic-Regression


This code contains data processing and analysis operations on two datasets: fighters.csv and tsunami.csv.

# Libraries used

pandas: for data manipulation and analysis
matplotlib: for data visualization
scikit-learn: for machine learning operations such as data preprocessing, model selection, and evaluation.
Functions

to_string(s): a helper function that takes a string input s and returns only the first word of that string.
Data Processing

Boxing Dataset (fighters.csv)
read_csv: reads the fighters.csv file and stores it in a pandas dataframe named data.
split: splits the height and reach columns by whitespace, and retains only the first value.
dataframe construction: constructs a new dataframe ans containing the following columns: Name, Wins, Losses, Draws, Ko_rate, Age, Height, and Country.
to_numeric: converts the age and height columns to numeric types.
groupby: groups the Country column by count and identifies the top 20 countries.
fix_countries: fixes the names of some countries to match the format used in the tsunami.csv dataset.
MinMaxScaler: normalizes the Wins and Losses columns using MinMaxScaler.
Tsunami Dataset (tsunami.csv)
read_csv: reads the tsunami.csv file and stores it in a pandas dataframe named data2.
dataframe construction: constructs a new dataframe disaster containing the following columns: Year, Validity, Latitude, Longitude, Deaths, Max Height, Houses Destroyed, Country, and Location.
to_numeric: converts the Latitude, Longitude, Total Deaths, and Total Houses Destroyed columns to numeric types.
Machine Learning Operations

train_test_split: splits the boxing dataframe into training and testing sets.
LinearRegression: fits a linear regression model to the training data and predicts the target variable for the testing data.
SimpleImputer: replaces missing values in the disaster dataframe using mean imputation.
OrdinalEncoder: encodes categorical features as ordinal numbers.
KBinsDiscretizer: discretizes continuous features into bins.
Binarizer: binarizes data (sets values above a threshold to 1 and below to 0).
LabelEncoder: encodes target variable as integers.
LogisticRegression: fits a logistic regression model to the training data and predicts the target variable for the testing data.
confusion_matrix: evaluates the performance of the logistic regression model using a confusion matrix.
