''' title: "Analyze the pre-processed data"
    author: "Raza Mehar | Najam Mehdi | Pujan Thapa"
    date: "2023-12-03"
    description: Decision Tree Interpretability.'''

# Import relevant packages and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, export_text, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz

# Load the datasets in dataframes
stores_df = pd.read_csv("3_stores_encoded.csv")

X = stores_df.drop("potential", axis = 1)
y = stores_df["potential"]

X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = DecisionTreeRegressor(max_depth = None, random_state = 42)
model.fit(X, y)

tree_rules = export_text(model, feature_names = X.columns.tolist())
print("Decision Tree Rules:\n", tree_rules)

# Visualize the decision tree graphically (requires Graphviz)
dot_data = export_graphviz(model, out_file = None, 
                           feature_names = X.columns.tolist(),  
                           filled = True, rounded = True,  
                           special_characters = True)  

graph = graphviz.Source(dot_data)
graph.render("decision_tree")
graph.view("decision_tree")