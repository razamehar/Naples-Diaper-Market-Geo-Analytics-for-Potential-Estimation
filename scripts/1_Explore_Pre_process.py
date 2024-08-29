''' title: "Explore and pre-process the datasets"
    author: "Raza Mehar | Najam Mehdi | Pujan Thapa"
    date: "2023-11-15"
    description: Understanding the data structure, dimensions, and data types. Identifying relationships between datasets. 
                 Renaming the columns to make them standardized. Plotting the stores inside polygons to identify micro-codes. 
                 Merging the datasets.'''

# Import relevant packages and modules
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely.wkt import loads

# Load the datasets in dataframes
stores = pd.read_csv("stores_NA.csv")
shapes = pd.read_csv("shapes_NA.csv")
demographics = pd.read_csv("socio_demo_NA.csv")
gravitation = pd.read_csv("gravitation_NA.csv")

# Explore the initial rows, data types and dimensions of the datasets
print("Displaying the initial rows, data types and dimensions of the datasets.\n")
print("Displaying information for stores data frame.\n")
print(stores.head(), stores.info(), stores.shape)
print("Displaying information for shapes data frame.\n")
print(shapes.head(), shapes.info(), shapes.shape)
print("Displaying information for demographics data frame.\n")
print(demographics.head(), demographics.info(), demographics.shape)
print("Displaying information for gravitation data frame.\n")
print(gravitation.head(), gravitation.info(), gravitation.shape)

# Rename the column names to make them standard across the datasets
gravitation = gravitation.rename(columns= {"fasciaoraria": "time_slot", "media_annuale": "annual_average"})
stores = stores.rename(columns= {'Cod3HD': "store_ID", 'Insegna': "store_name", 'TipologiaPdV': "store_type", 'MQVEND': "store_size", 
                                 'Indirizzo': "address", 'Provincia': "province", 'Potenziale': "potential"})

# Create Shapely Point objects for each store
stores['Point'] = stores.apply(lambda row: Point(row['Long'], row['Lat']), axis=1)

# Convert 'geometry' column to Shapely Polygon objects
shapes['geometry'] = shapes['geometry'].apply(lambda x: loads(x))

# Check if each store is inside any polygon and assign microcode
stores['InsidePolygon'] = stores.apply(lambda row: any(row['Point'].within(polygon) for polygon in shapes['geometry']), axis=1)
stores.loc[stores['InsidePolygon'], 'microcode'] = stores.loc[stores['InsidePolygon'], 'Point'].apply(
    lambda point: shapes.loc[shapes['geometry'].apply(lambda polygon: point.within(polygon)), 'microcode'].values[0]
    if any(point.within(polygon) for polygon in shapes['geometry']) else None
)
# Join all the demographics and gravitation dataframes on "microcode" with stores dataframe and drop irrelevant features
stores = pd.merge(stores, demographics, on="microcode", how = "left")
stores = pd.merge(stores, gravitation, on="microcode", how = "left")
#stores = stores.drop(stores.columns[0], axis = 1)
stores.drop(["InsidePolygon", "Lat", "Long", "province_x", "district", "store_name", "address", "region", "province_y", "Unnamed: 0"], inplace=True, axis=1)

# Drop the irrelevant columns such as serial number, store name, address, province, region
print("Displaying the first rows of the merged dataframe after dropping irrelevant features.\n")
print(stores.head())

# Create a new csv file called 1_stores_with_microcodes for further processing and analysis.
stores.to_csv("1_stores_with_microcodes.csv", index = False)
print("Dataframe has been saved in a new csv file called: 1_stores_with_microcodes.")
