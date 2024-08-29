''' title: "Analyze the pre-processed data"
    author: "Raza Mehar | Najam Mehdi | Pujan Thapa"
    date: "2023-12-03"
    description: Model Selection.'''

# Import relevant packages and modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

# Load the datasets in dataframes
stores_df = pd.read_csv("3_stores_encoded.csv")

X = stores_df.drop("potential", axis = 1)
y = stores_df["potential"]

dt = DecisionTreeRegressor(max_depth = None, random_state = 42)
fr = RandomForestRegressor(n_estimators = 100, max_depth = None, random_state = 42)
gb = GradientBoostingRegressor(n_estimators = 100, max_depth = 5, learning_rate = 0.1, random_state = 42)
ab = AdaBoostRegressor(n_estimators = 100, learning_rate = 0.01, random_state = 42)

ensemble_regressors = [dt, fr, gb, ab]

'''A good standard value for k in k-fold cross-validation is 10, as empirical evidence shows. Experiments by Ron Kohavi on various real-world dataset
suggest that 10-fold cross-validation offers the best tradeoff between bias and variance'''
kf = KFold(n_splits = 10, shuffle = True, random_state = 42)

for regressor in ensemble_regressors:
    score = cross_val_score(estimator = regressor, X = X, y = y, cv = kf, scoring = "neg_mean_squared_error")
    rmse_score = (-score) ** 0.5
    mean_rmse = np.mean(rmse_score)
    std_rmse = np.std(rmse_score)

    print(f"Root Mean Squared Error of {regressor} is {rmse_score}")
    print(f"Mean RMSE: {mean_rmse}")
    print(f"Standard Deviation of RMSE: {std_rmse}")
    print("\n")
