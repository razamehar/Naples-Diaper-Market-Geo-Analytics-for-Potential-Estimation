''' title: "Analyze the pre-processed data"
    author: "Raza Mehar | Najam Mehdi | Pujan Thapa"
    date: "2023-12-03"
    description: Forecasting Based on Child Births and Ageing Population'''

# Import relevant packages and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

df_child = pd.read_csv("ISTAT_Child_Births_Data_2024.csv")
df_population = pd.read_csv("ISTAT_Population_Data_2024.csv")

print(df_child.info())
print(df_child.shape)

print(df_population.info())
print(df_population.shape)

df_child = df_child[df_child["Province"] == "Napoli"]
df_population = df_population[df_population["Provincia"] == "Napoli"]

df_child_grouped = df_child.groupby("Year").agg({"Live births": "sum"}).reset_index()
df_population_grouped = df_population.groupby(["Year", "Age"]).agg({"Total": "sum"}).reset_index()

df_95plus = df_population_grouped[df_population_grouped["Age"] == "95+"]
df_90_94 = df_population_grouped[df_population_grouped["Age"] == "90-94"]
df_85_89 = df_population_grouped[df_population_grouped["Age"] == "85-89"]
df_80_84 = df_population_grouped[df_population_grouped["Age"] == "80-84"]

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 10), sharex = True)
sns.set_theme(style = "ticks")
sns.lineplot(data = df_child_grouped, x = "Year", y = "Live births", color = "#84ACC8", ax = ax1)
ax1.set_ylabel("Live Births")
ax1.set_xlabel("")
ax1.set_title("Live Births Over the Years")

sns.lineplot(data = df_95plus, x = "Year", y = "Total", ax = ax2, label = "95+", color = "black")
sns.lineplot(data = df_90_94, x = "Year", y = "Total", ax = ax2, label = "90-94", color = "brown")
sns.lineplot(data = df_85_89, x = "Year", y = "Total", ax = ax2, label = "85-89", color ="darkblue")
sns.lineplot(data = df_80_84, x = "Year", y = "Total", ax = ax2, label = "80-84", color = "#84ACC8")
ax2.set_ylabel("Total Population")
ax2.set_xlabel("")
ax2.set_title("Ageing Population Over the Years")

plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer = True))
plt.show()
