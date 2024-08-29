''' title: "Pre-process the merged dataset"
    author: "Raza Mehar | Najam Mehdi | Pujan Thapa"
    date: "2023-12-03"
    description: Handling missing values. Univariate and Bivariate analyses'''

# Import relevant packages and modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the datasets in dataframes
stores_df = pd.read_csv("1_stores_with_microcodes.csv")

# Explore the initial rows, data types and dimensions of the dataset
print("Displaying the initial rows, data types and dimensions of the dataset.\n")
print(stores_df.head(), stores_df.info(), stores_df.shape)
print(stores_df.dtypes)

# Check for duplicate values
print("Number of duplicae rows:", stores_df.duplicated().sum())

# Check for missing values
missing_values = stores_df.isnull()
missing_counts = missing_values.sum()
print("Displaying the features with respective counts of missing values.\n", missing_counts)

'''The missing_count indicates that values are missing in several columns. Separate lists of categorical and numerical column names
are being created. Numerical category columns will be used in histograms and Q-Q plots, The normality assessment will determine 
whether to use the mean or median when imputing missing values.'''

# Create list for categorical and numerical column names with missing values
categorical_cols = ["microcode", 
                    "daytype", 
                    "time_slot", 
                    "datatype"]
num_cols = ["population", 
        "population_m", 
        "population_f", 
        "population_age_00_04_yr", 
        "population_age_05_14_yr", 
        "population_age_15_34_yr", 
        "population_age_35_44_yr", 
        "population_age_45_54_yr", 
        "population_age_55_64_yr",
        "population_age_65_up_yr",
        "annual_average"]

# Plot histograms and qqplots of numerical features to check for normality
print("Generating the Histograms.\n")
fig, axes = plt.subplots(ncols = 4, nrows= 3, figsize = (10, 10))
sns.set_theme(style="ticks")

for i, ax in enumerate(axes.flat):
    if i < len(num_cols):
        sns.histplot(data=stores_df, x=num_cols[i], kde=True, ax=ax, edgecolor = ".3")
        
        ax.set_title(f'{num_cols[i]}')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels("")
        ax.set_yticklabels("")
    else:
        # If there are fewer columns than subplots, remove the empty subplots
        fig.delaxes(ax) 

plt.show()

print("Generating the QQPlots.\n")
fig, axes = plt.subplots(ncols = 4, nrows= 3, figsize = (10, 10))
sns.set_theme(style="ticks")

for i, ax in enumerate(axes.flat):
    if i < len(num_cols):
        sm.qqplot(data = stores_df[num_cols[i]], ax = ax, line = "45")
        ax.set_title(f'{num_cols[i]}')
        ax.set_xticklabels("")
        ax.set_yticklabels("")
        ax.tick_params(axis="x", which="both", bottom=False, top=False)
        ax.set_xlabel("")
        ax.set_ylabel("") 
    else:
        # If there are fewer columns than subplots, remove the empty subplots
        fig.delaxes(ax) 

plt.show()

'''Observing the histograms and Q-Q plots, we have established that the features do not follow a normal distribution. Therefore, 
imputing missing values will be performed using the median.'''

# Impute missing values in the categorical featutes using mode

## Converting relevant features to object type before imputing
for col in ["microcode", "daytype", "time_slot"]:
    stores_df[col] = stores_df[col].astype("object")

for col in categorical_cols:
    stores_df[col].fillna(stores_df[col].mode().iloc[0], inplace=True)

print("Displaying the values after imputation:")
print(stores_df[categorical_cols].isnull().sum())

# Impute missing values in the numerical features using median
for col in num_cols:
    stores_df[col].fillna(stores_df[col].median(), inplace=True)

print("Displaying the values after imputation:")
print(stores_df[num_cols].isnull().sum())

# Plot bar charts for all the categorical features for univariate analysis
print("Generating the Bar Charts.\n")
all_cat_cols = ["store_type", "Parking", "daytype", "time_slot", "datatype"]
label_mapping = {
    0: ["Libero Servizio", "Supermarket", "Discount Store", "Hypermarket", "Drug Store"],
    1: ["No", "Yes"],
    2: ["Weekday", "Weekend"],
    3: ["7am-10am", "10am-13pm", "13pm-14pm", "14pm-17pm", "17pm-20pm"],
    4: ["Under 18", "18-30 yr", "31-40", "41-50", "51-60", "Over 60", "Males", "Females"]
}

fig, axes = plt.subplots(ncols = 3, nrows= 2, figsize = (10, 10))
sns.set_theme(style="ticks")
for i,ax in enumerate(axes.flat):
    if i < len(all_cat_cols):
        sns.countplot(x = all_cat_cols[i], data=stores_df , ax = ax, edgecolor = "#446CAD", color = "#84ACC8")
        plt.xlabel(all_cat_cols[i])
        plt.ylabel("")
        if i in label_mapping:
            ax.set_xticklabels(label_mapping[i], rotation = 45)
    else:
        # If there are fewer columns than subplots, remove the empty subplots
        fig.delaxes(ax) 
plt.tight_layout()
plt.show()

# Plot histograms for all the numerical features for univariate analysis
print("Generating the Histograms.\n")
all_num_cols = ["store_size", "population", "population_m", "population_f", "population_age_00_04_yr", "population_age_05_14_yr", 
        "population_age_15_34_yr", "population_age_35_44_yr", "population_age_45_54_yr", "population_age_55_64_yr", "population_age_65_up_yr", 
        "annual_average", "potential"]
stores_df.hist(column = all_num_cols)

plt.ylabel("")
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
plt.tight_layout()
plt.show()

# Plot scatter plots for all the numerical features for bivariate analysis
print("Generating the scatter plots.\n")
fig, axes = plt.subplots(ncols = 5, nrows= 3, figsize = (10, 10))
sns.set_theme(style="ticks")
for i,ax in enumerate(axes.flat):
    if i < len(all_num_cols):
        sns.scatterplot(data = stores_df, x = all_num_cols[i], y = "potential", ax = ax, edgecolor = "#446CAD", color = "#84ACC8")
    else:
        # If there are fewer columns than subplots, remove the empty subplots
        fig.delaxes(ax) 
plt.tight_layout()
plt.show()

# Plot box plots for all the categorical features for bivariate analysis
print("Generating the box plots.\n")
fig, axes = plt.subplots(ncols = 3, nrows= 2, figsize = (10, 10))
sns.set_theme(style="ticks")
for i, ax in enumerate(axes.flat):
    if i < len(all_cat_cols):
        sns.boxplot(data=stores_df, x=all_cat_cols[i], y="potential", ax=ax, color="#84ACC8")
        if i in label_mapping:
            ax.set_xticklabels(label_mapping[i], rotation=45)
    else:
        # If there are fewer columns than subplots, remove the empty subplots
        fig.delaxes(ax)
plt.tight_layout()
plt.show()

# Create a new csv file called 2_stores_preprocessed.csv for further processing and analysis.
stores_df.to_csv("2_stores_preprocessed.csv", index = False)
print("Dataframe has been saved in a new csv file called: 2_stores_preprocessed.csv.")