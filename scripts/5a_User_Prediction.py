''' title: "Analyze the pre-processed data"
    author: "Raza Mehar | Najam Mehdi | Pujan Thapa"
    date: "2023-12-03"
    description: Prediction based on user inputs.'''

# Import relevant packages and modules
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the datasets in dataframes
stores_df = pd.read_csv("3_stores_encoded.csv")

X = stores_df.drop("potential", axis = 1)
y = stores_df["potential"]

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size = 0.2, random_state = 42)

#random_forest = RandomForestRegressor(n_estimators = 100, max_depth = None, random_state = 42, criterion = "squared_error")
#random_forest.fit(X_train, y_train)

#Load the saved model
loaded_model = joblib.load("fater_model.joblib")

# Predict the potential on dummy data
path = 'dummy.csv'
dummy_data = pd.read_csv(path)
dummy_pred = loaded_model.predict(dummy_data)
print(dummy_pred)

