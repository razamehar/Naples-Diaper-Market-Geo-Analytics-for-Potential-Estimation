''' title: "Analyze the pre-processed data"
    author: "Raza Mehar | Najam Mehdi | Pujan Thapa"
    date: "2023-12-03"
    description: Pareto analysis.'''

# Import relevant packages and modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Load the datasets in dataframes
stores_df = pd.read_csv("2_stores_preprocessed.csv")

store_avg = stores_df.groupby("store_ID")["potential"].mean().sort_values(ascending = False).reset_index()
store_avg["Cumulative"] = store_avg["potential"].cumsum()
value_sum = store_avg["Cumulative"].tail(1).values
store_avg["Cumulative %"] = (store_avg["Cumulative"] / value_sum) * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
sns.set_theme(style = "ticks")

# Pareto chart for store_type
sns.barplot(x=store_avg["store_ID"], y=store_avg["potential"],
            order=store_avg.sort_values('potential', ascending=False)["store_ID"],
            ax=axes[0], edgecolor="#446CAD", color="#84ACC8")

axes[0].set_xticklabels("")
axes[0].tick_params(axis="x", which="both", bottom=False, top=False)

axes1_id = axes[0].twinx()
axes1_id.plot(store_avg.index, store_avg["Cumulative %"], color="red")
axes1_id.yaxis.set_major_formatter(PercentFormatter())

axes[0].set_title('Pareto Chart for store_ID')

# Pareto chart for store_type
store_avg_type = stores_df.groupby("store_type")["potential"].mean().sort_values(ascending=False).reset_index()
store_avg_type["Cumulative"] = store_avg_type["potential"].cumsum()
value_sum_type = store_avg_type["Cumulative"].tail(1).values
store_avg_type["Cumulative %"] = (store_avg_type["Cumulative"] / value_sum_type) * 100

sns.barplot(x=store_avg_type["store_type"], y=store_avg_type["potential"],
            order=store_avg_type.sort_values('potential', ascending=False)["store_type"],
            ax=axes[1], edgecolor="#446CAD", color="#84ACC8")
axes[1].tick_params(axis="x", which="both", bottom=False, top=False)

axes1_type = axes[1].twinx()
axes1_type.plot(store_avg_type.index, store_avg_type["Cumulative %"], color="red")
axes1_type.yaxis.set_major_formatter(PercentFormatter())

axes[1].set_title('Pareto Chart for store_type')

plt.tight_layout(pad=2.0)
plt.show()

# Display 20% of unique stores that generate 80% of combined potential
top_20_percent_stores = store_avg[store_avg["Cumulative %"] <= 80]
print(top_20_percent_stores)

print(f"\n{top_20_percent_stores.shape[0]} out of {store_avg.shape[0]} stores generate 80% of the combined potential.")

# Create a new csv file called 4_top_20_percent_stores.csv for record purpose only.
top_20_percent_stores.to_csv("4_top_20_percent_stores.csv", index = False)
print("\nList of top 20 percent stores has been saved in a new csv file called: 4_top_20_percent_stores.csv.")