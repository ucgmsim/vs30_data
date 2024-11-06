import pandas as pd
from pathlib import Path
from tqdm import tqdm
import run_calculations

estimated_vs30_dir = Path("/home/arr65/data/nzgd/resources/vs30_map")

new_data_dir = Path("/home/arr65/data/nzgd/processed_data_copy/cpt/data")

old_data_dir = Path("/home/arr65/vs30_data_input_data/parquet/data")

metadata_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/metadata")

### Setting the station_id to be the record_name for compatibility with functions that use record_name
estimated_vs30_ll_df = pd.read_csv(estimated_vs30_dir/"non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll",
                                sep=" ",
                                header=None,
                                names=["longitude",
                                       "latitude",
                                       "record_name"])
estimated_vs30_value_df = pd.read_csv(estimated_vs30_dir/"non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.vs30",
                                   sep=" ", header=None, names=["record_name", "vs30"])

old_df = pd.read_parquet(old_data_dir, columns=["record_name", "latitude", "longitude"])

old_df = old_df.drop_duplicates(keep="first")

new_df = pd.read_parquet(new_data_dir, columns=["record_name", "latitude", "longitude"])
new_df = new_df.drop_duplicates(keep="first")

# Merge new_df and old_df with indicator to identify rows only in new_df
merged_df = new_df.merge(old_df, on=["record_name"], how="left", indicator=True)

# Select rows that are only in new_df
new_df_not_in_old = merged_df[merged_df["_merge"] == "left_only"].drop(columns=["_merge"])
new_df_not_in_old = new_df_not_in_old.rename(columns={
    'latitude_x': 'latitude',
    'longitude_x': 'longitude'})
new_df_not_in_old.drop(columns=["latitude_y", "longitude_y"], inplace=True)

if (metadata_dir / "closest_vs30_grid_point.csv").exists():
    closest_vs30_grid_point = pd.read_csv(metadata_dir / "closest_vs30_grid_point.csv")
else:
    closest_vs30_grid_point = run_calculations.calc_all_closest_cpt_dist(lon_lat_to_consider_df=new_df_not_in_old,
                                                            all_lon_lat_df=estimated_vs30_ll_df,
                                                            n_procs=7)
    pass

vs30_predictions = []
for index, row in tqdm(closest_vs30_grid_point.iterrows(),total=len(closest_vs30_grid_point)):
    matched_row = estimated_vs30_value_df[estimated_vs30_value_df["record_name"] == row["closest_cpt_name"]]
    vs30_predictions.append(matched_row["vs30"].values[0])

## Add the vs30_predictions to the closest_vs30_grid_point DataFrame
closest_vs30_grid_point["predicted_vs30_m_per_sec"] = vs30_predictions

closest_vs30_grid_point = closest_vs30_grid_point.rename(columns={
    "distance_to_closest_cpt_km": "distance_to_closest_grid_point_km",
    "closest_cpt_name": "closest_grid_point_name",
    "lon": "record_lon",
    "lat": "record_lat",
    "closest_cpt_lon": "closest_grid_point_lon",
    "closest_cpt_lat": "closest_grid_point_lat",
    "cpt_name": "record_name"})

closest_df = run_calculations.calc_all_closest_cpt_dist(lon_lat_to_consider_df=new_df_not_in_old,
                                                        all_lon_lat_df=old_df, n_procs=7)

merged_df = closest_vs30_grid_point.merge(closest_df[["cpt_name","distance_to_closest_cpt_km"]], left_on='record_name', right_on='cpt_name')
merged_df.drop(columns=["cpt_name"], inplace=True)
merged_df = merged_df.rename(columns={"distance_to_closest_cpt_km": "distance_to_closest_old_cpt_km"})

merged_df.to_csv(metadata_dir / "old_vs30_prediction_at_new_cpt_locations.csv")

