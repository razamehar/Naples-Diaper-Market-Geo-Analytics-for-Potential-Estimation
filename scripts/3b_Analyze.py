''' title: "Analyze the pre-processed data"
    author: "Raza Mehar | Najam Mehdi | Pujan Thapa"
    date: "2023-12-03"
    description: Contingency table analyses. Store potential analyses. Geo-Spatial analysis.'''

# Import relevant packages and modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.wkt import loads

# Load the datasets in dataframes
stores_df = pd.read_csv("2_stores_preprocessed.csv")

# Explore the initial rows, data types and dimensions of the dataset
print("Displaying the initial rows, data types and dimensions of the dataset.\n")
print(stores_df.head(), stores_df.info(), stores_df.shape)
print(stores_df.dtypes)

# Create contingency table for potential with respect to store type, store size and parking
pot_categories = ["Low (0.001 - 0.241)", "Medium (0.241 - 0.601)", "High (0.601 - 0.841)"]
size_categories = ["Small (100 - 1000)", "Medium (1000 - 5000)", "Large (5000 - 12701)"]

stores_df["potential_cat"] = pd.cut(stores_df["potential"], bins = [0.001, 0.241, 0.601, 0.841], labels = pot_categories, include_lowest = True)
stores_df["size_cat"] = pd.cut(stores_df["store_size"], bins = [100, 1000, 5000, 12701], labels = size_categories, include_lowest = True)

ct_store_type = pd.crosstab(stores_df["store_type"], stores_df["potential_cat"], margins = True, margins_name = "Total")
ct_size = pd.crosstab(stores_df["size_cat"], stores_df["potential_cat"], margins = True, margins_name = "Total")
ct_parking = pd.crosstab(stores_df["Parking"], stores_df["potential_cat"], margins = True, margins_name = "Total")

print("\nContingency Table of Potential and Store Type:", ct_store_type)
print("\nContingency Table of Potential and Store Size:", ct_size)
print("\nContingency Table of Potential and Parking", ct_parking)

# Perform geo-spatial analysis on the yearly average population gravation across stores
stores_df['geometry'] = stores_df['Point'].apply(lambda x: loads(x))
gdf = gpd.GeoDataFrame(stores_df, geometry='geometry')

store_avg = stores_df.groupby("store_ID")["annual_average"].mean()
gdf = gdf.merge(store_avg, left_on='store_ID', right_index=True, how='left', suffixes=('', '_avg'))

# Perform geo-spatial analysis on the potential across stores
store_pot = stores_df.groupby("store_ID")["potential"].mean()
gdf = gdf.merge(store_avg, left_on='store_ID', right_index=True, how='left', suffixes=('', '_pot'))

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
sns.set_theme(style="ticks")

gdf.plot(column="potential", cmap="Blues", linewidth=0.8, edgecolor="0.8", legend=True, legend_kwds={"shrink": 0.4}, ax=ax)
ax.set_title("Average potential across stores")
ax.set_xticklabels("")
ax.set_yticklabels("")
ax.tick_params(axis="x", which="both", bottom=False, top=False)
ax.tick_params(axis="y", which="both", left=False, right=False)

plt.show()

# Display top 10 stores with the most yearly average population movement
store_pot.sort_values(ascending = False, inplace = True)
print(store_pot.head(10))
print(store_pot.describe())

# Perform the analyses on store_type, day_type, time_slot, and demographics with respect to yearly average population gravitation
daytype_avg = stores_df.groupby("daytype")["annual_average"].mean()
time_slot_avg = stores_df.groupby("time_slot")["annual_average"].mean()
datatype_avg = stores_df.groupby("datatype")["annual_average"].mean()

fig, axes = plt.subplots(1, 3, figsize = (12, 10))
sns.set_theme(style = "ticks")

# Plot Daytype Analysis
sns.barplot(x=daytype_avg.index, y=daytype_avg.values, ax=axes[0], edgecolor = "#446CAD", color = "#84ACC8")
axes[0].set_xlabel('Daytype')
axes[0].set_title('Day Type Analysis')
axes[0].set_yticklabels("")
axes[0].tick_params(axis = "y", which = "both", left =False, right = False)
axes[0].set_xticklabels(["Weekday", "Weekend"])

# Plot Time Slot Analysis
sns.barplot(x=time_slot_avg.index, y=time_slot_avg.values, ax=axes[1], edgecolor = "#446CAD", color = "#84ACC8")
axes[1].set_xlabel('Time Slot')
axes[1].set_title('Temporal Analysis')
axes[1].set_yticklabels("")
axes[1].tick_params(axis = "y", which = "both", left =False, right = False)
axes[1].set_xticklabels(["07am - 10am", "10am - 13pm", "13pm - 14pm", "14pm - 17pm", "17pm - 20pm"])

# Plot Demographic Insights
sns.barplot(x=datatype_avg.index, y=datatype_avg.values, ax=axes[2], edgecolor = "#446CAD", color = "#84ACC8")
axes[2].set_xlabel('Age Group and Gender')
axes[2].set_title('Demographic Trend Analysis')
axes[2].set_yticklabels("")
axes[2].tick_params(axis = "y", which = "both", left =False, right = False)
axes[2].set_xticklabels(["Under 18", "18 - 30", "31 - 40", "41 - 50", "51 - 60", "Over 61", "Females", "Males"])

plt.tight_layout(pad = 2.0)
plt.show()

stores_df.drop(["potential_cat", "size_cat"], axis = 1, inplace = True)