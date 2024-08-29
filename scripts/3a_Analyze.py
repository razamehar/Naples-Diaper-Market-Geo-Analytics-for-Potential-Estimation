''' title: "Analyze the pre-processed data"
    author: "Raza Mehar | Najam Mehdi | Pujan Thapa"
    date: "2023-12-03"
    description: Encoding. Correlation analysis.'''

# Import relevant packages and modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import kruskal, mannwhitneyu
import matplotlib.colors as mcolors

# Load the datasets in dataframes
stores_df = pd.read_csv("2_stores_preprocessed.csv")
potential_column = stores_df["potential"]

# Explore the initial rows, data types and dimensions of the dataset
print("Displaying the initial rows, data types and dimensions of the dataset.\n")
print(stores_df.head(), stores_df.info(), stores_df.shape)
print(stores_df.dtypes)

# Encode the relvant categorical features using One-Hot Encoding
columns_to_encode = ["store_type", "daytype", "time_slot", "datatype"]
encoded_df = pd.get_dummies(stores_df[columns_to_encode], columns = columns_to_encode)
encoded_df = pd.concat([potential_column, encoded_df], axis = 1)

# Explore the initial rows, data types and dimensions of the encoded datasets
print("\nDisplaying the initial rows, data types and dimensions of the encoded dataset.\n")
print(encoded_df.head(), encoded_df.info(), encoded_df.shape)
print(encoded_df.dtypes)

all_num_cols = ["store_size", "Parking", "population_m", "population_f", "population_age_00_04_yr", "population_age_05_14_yr", 
        "population_age_15_34_yr", "population_age_35_44_yr", "population_age_45_54_yr", "population_age_55_64_yr", "population_age_65_up_yr",
         "annual_average", "potential"]

# Perform descriptive statistics
print(stores_df[all_num_cols].describe())

cmap = mcolors.LinearSegmentedColormap.from_list('custom_blue', ['#FFFFFF', '#84ACC8'])

# Display heatmap of the linear correlation between numerical predictors and the response variable potential
corr_matrix = round(stores_df[all_num_cols].corr(), 2)
sns.set_theme(style = "ticks")
sns.heatmap(corr_matrix, annot = True, cmap = cmap, linewidths = 0.5)
plt.show()

# Drop population related features to reduce multi-collinearity keeping population_m and population_p only
stores_df.drop(["population", "population_age_00_04_yr", "population_age_05_14_yr", 
        "population_age_15_34_yr", "population_age_35_44_yr", "population_age_45_54_yr", "population_age_55_64_yr", "population_age_65_up_yr"], axis = 1, inplace = True)

# Test to check if the observed differences in potential values between store types and parking are statistically significant.
statistic, p_value = kruskal(stores_df["potential"][stores_df["store_type"] == "IPR"],
                        stores_df["potential"][stores_df["store_type"] == "SUP"],
                        stores_df["potential"][stores_df["store_type"] == "LIS"],
                        stores_df["potential"][stores_df["store_type"] == "SSD"],
                        stores_df["potential"][stores_df["store_type"] == "DIS"])

print(f"\np-value for Kruskal-Wallis test: {p_value:.3f}")

statistic, p_value = mannwhitneyu(stores_df["potential"][stores_df["Parking"] == 1],
                        stores_df["potential"][stores_df["Parking"] == 0])

print(f"\np_value for Mann-Whitney U test: {p_value:.3f}")

# Combine numerical features with the encoded categorical features.
stores_df.drop(["store_ID", "store_type", "Comune", "microcode", "daytype", "time_slot", "datatype", "Point", "potential"], axis = 1, inplace = True)
stores_df = pd.concat([stores_df, encoded_df], axis=1)           
print(stores_df.info())

# Create a new csv file called 3_stores_encoded.csv for further processing and analysis.
stores_df.to_csv("3_stores_encoded.csv", index = False)
print("Dataframe has been saved in a new csv file called: 3_stores_encoded.csv.\n")


