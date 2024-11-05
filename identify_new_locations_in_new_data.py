import pandas as pd
from pathlib import Path

new_data_dir = Path("/home/arr65/data/nzgd/processed_data_copy/cpt/data")

old_data_dir = Path("/home/arr65/vs30_data_input_data/parquet/data")

old_df = pd.read_parquet(old_data_dir, columns=["latitude", "longitude"])
old_df = old_df.drop_duplicates(keep="first")

new_df = pd.read_parquet(new_data_dir, columns=["record_name", "latitude", "longitude"])
new_df = new_df.drop_duplicates(keep="first")

print()





#
# import geopandas as gpd
# from shapely.geometry import Point
#
# # Define the points
# point1 = Point(174.763336, -36.848461)  # Example coordinates (longitude, latitude)
# point2 = Point(174.764336, -36.848461)  # Another point close to point1
#
# # Create GeoDataFrames
# gdf1 = gpd.GeoDataFrame(geometry=[point1], crs="EPSG:4326")
# gdf2 = gpd.GeoDataFrame(geometry=[point2], crs="EPSG:4326")
#
# # Convert to a projected coordinate system for accurate distance calculations
# gdf1 = gdf1.to_crs(epsg=3857)
# gdf2 = gdf2.to_crs(epsg=3857)
#
# # Define the radius in meters
# radius = 100  # 100 meters
#
# # Create a buffer around point1
# buffer = gdf1.buffer(radius)
#
# # Check if point2 is within the buffer
# is_within = gdf2.within(buffer.iloc[0])
#
# print(is_within.iloc[0])  # True if point2 is within the radius of point1