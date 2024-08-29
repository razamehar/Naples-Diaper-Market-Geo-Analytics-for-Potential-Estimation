''' title: "Analyze the pre-processed data"
    author: "Raza Mehar | Najam Mehdi | Pujan Thapa"
    date: "2023-12-03"
    description: Model Selection & Hyperparameter Tuning.'''

# Import relevant packages and modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

# Load the datasets in dataframes
stores_df = pd.read_csv("3_stores_encoded.csv")

X = stores_df.drop("potential", axis = 1)
y = stores_df["potential"]

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size = 0.2, random_state = 42)

holdout_set = pd.concat([X_holdout, y_holdout], axis = 1)
holdout_set.to_csv("5_holdout_set.csv", index = False)

print("Initiating")

# Define hyperparameter grids for each model
param_grid_dt = {
    "max_depth": [None, 5],
    "random_state": [42]
}

param_grid_rf = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5],
    "random_state": [42]
}

param_grid_gb = {
    "n_estimators": [50, 100],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5],
    "random_state": [42]
}

param_grid_ab = {
    "n_estimators": [50, 100],
    "learning_rate": [0.01, 0.1],
    "random_state": [42]
}

# Create a dictionary with model names as keys and hyperparameter grids as values
models_params = {
    "DecisionTree:": (DecisionTreeRegressor(), param_grid_dt),
    "RandomForest": (RandomForestRegressor(), param_grid_rf),
    "GradientBoosting": (GradientBoostingRegressor(), param_grid_gb),
    "AdaBoost": (AdaBoostRegressor(), param_grid_ab)
}

'''A good standard value for k in k-fold cross-validation is 10, as empirical evidence shows. Experiments by Ron Kohavi on various real-world dataset
suggest that 10-fold cross-validation offers the best tradeoff between bias and variance'''
kf = KFold(n_splits = 10, shuffle = True, random_state = 42)

for model_name, (model, param_grid) in models_params.items():
    grid_search = GridSearchCV(model, param_grid = param_grid, cv = kf, scoring = "neg_mean_squared_error")
    grid_search.fit(X_train, y_train)
    
    # Print the best hyperparameters
    print(f"Best Hyperparameters for {model_name}:", grid_search.best_params_)