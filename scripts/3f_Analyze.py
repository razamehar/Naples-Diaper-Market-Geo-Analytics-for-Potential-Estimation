''' title: "Analyze the pre-processed data"
    author: "Raza Mehar | Najam Mehdi | Pujan Thapa"
    date: "2023-12-03"
    description: Feature Importance Analysis. Saving the model for future use.'''

# Import relevant packages and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load the datasets in dataframes
stores_df = pd.read_csv("3_stores_encoded.csv")

X = stores_df.drop("potential", axis = 1)
y = stores_df["potential"]

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size = 0.2, random_state = 42)

random_forest = RandomForestRegressor(n_estimators = 100, max_depth = None, random_state = 42, criterion = "squared_error")
fater_model = random_forest.fit(X_train, y_train)

y_train_pred = random_forest.predict(X_train)
y_holdout_pred = random_forest.predict(X_holdout)

# Computing MSE to measure the average of the squared differences between predicted and actual values. It heavily penalizes large erros due to squating opertion. This makes is sensitive to ourliers.
mse_train = mean_squared_error(y_train, y_train_pred)
mse_holdout = mean_squared_error(y_holdout, y_holdout_pred)

# Computing MSE to measure the average of the absolute differences between predicted and actual values. It heavily penalizes large erros due to squating opertion. This makes is sensitive to ourliers.
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_holdout = mean_absolute_error(y_holdout, y_holdout_pred)

# Computing R2 to measure how well the model explains the variance in the target variable (goodness of fit)
r2_train = r2_score(y_train, y_train_pred)
r2_holdout = r2_score(y_holdout, y_holdout_pred)

print(f"MSE train: {mse_train:.10f}, holdout {mse_holdout:.10f}")
print(f"MAE train: {mae_train:.10f}, holdout {mae_holdout:.10f}")
print(f"r2 train: {r2_train:.10f}, holdout {r2_holdout:.10f}")

n, k = stores_df.shape
residuals = (((y_holdout - y_holdout_pred) ** 2).sum()) / (n - k - 1)
standard_error = residuals ** 0.5
print(f"Standard Error of Predicted Values: {standard_error}")

feature_importances = random_forest.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance Analysis')
plt.show()

joblib.dump(fater_model, "fater_model.joblib")
print("\nTrained model has been saved as fater_model.joblib for future use.")