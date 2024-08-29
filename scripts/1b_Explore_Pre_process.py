''' title: "Explore and pre-process the datasets"
    author: "Raza Mehar | Najam Mehdi | Pujan Thapa"
    date: "2023-11-15"
    description: Spatial Exploration (Extra Viusualization).'''

# Import relevant packages and modules
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.wkt import loads
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Load the datasets in dataframes
stores = pd.read_csv("stores_NA.csv")
shapes = pd.read_csv("shapes_NA.csv")
demographics = pd.read_csv("socio_demo_NA.csv")
gravitation = pd.read_csv("gravitation_NA.csv")

# Rename the column names to make them standard across the datasets
gravitation = gravitation.rename(columns= {"fasciaoraria": "time_slot", "media_annuale": "annual_average"})
stores = stores.rename(columns= {'Cod3HD': "store_ID", 'Insegna': "store_name", 'TipologiaPdV': "store_type", 'MQVEND': "store_size", 
                                 'Indirizzo': "address", 'Provincia': "province", 'Potenziale': "potential"})

# Create Shapely Point objects for each store
stores['Point'] = stores.apply(lambda row: Point(row['Long'], row['Lat']), axis=1)

# Convert 'geometry' column to Shapely Polygon objects
shapes['geometry'] = shapes['geometry'].apply(lambda x: loads(x))

gdf_shapes = gpd.GeoDataFrame(shapes, geometry='geometry')
gdf_stores = gpd.GeoDataFrame(stores, geometry='Point')

unique_stores = gdf_stores['store_name'].unique()

cmap = plt.get_cmap('tab20', len(unique_stores))
color_dict = {store:cmap(i) for i, store in enumerate(unique_stores)}
gdf_stores['color'] = gdf_stores['store_name'].map(color_dict)

na_provinces = gdf_stores[gdf_stores['province'].isna()]
na_provinces.shape[0]

legend_elements = [mpatches.Patch(facecolor=cmap(i), edgecolor=cmap(i), label=store)
    for i, store in enumerate(unique_stores)]

fig, ax = plt.subplots(figsize = (12, 10))
na_provinces.plot(ax = ax, marker='o', markersize = 5, color=na_provinces['color'])
plt.xlabel("")
plt.ylabel("")
ax.set_xticklabels([])
ax.set_yticklabels([])
leg = ax.legend(handles = legend_elements, loc = 'center left', bbox_to_anchor = (1, 0.5), title = "Stores", prop = {'size': 8})
plt.show()