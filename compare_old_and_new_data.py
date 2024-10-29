"""
The main script for calculating Vs30 values from CPT data.
"""

import glob
import time
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import copy

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scipy import interpolate
import scipy

import config as cfg
import filtering
import load_sql_db
import numpy as np
from tqdm import tqdm

@dataclass
class OrganizedWithDepthRange:
    """
    Keeps track of which DataFrames have the largest and smallest depth ranges.
    """

    largest_depth_range: pd.DataFrame
    shortest_depth_range: pd.DataFrame


def organise_with_depth_range(df1: pd.DataFrame, df2: pd.DataFrame) -> OrganizedWithDepthRange:
    """
    Selects the DataFrame with the largest depth range.

    This function compares the depth ranges of two DataFrames and returns the one with the larger range.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame containing a 'Depth' column.
    df2 : pd.DataFrame
        DataFrame containing a 'Depth' column.

    Returns
    -------
    OrganizedWithDepthRange
        An instance of OrganizedWithDepthRange to indicate which Dataframes have the largest and shortest
        depth ranges.
    """

    d1_range = df1["Depth"].max() - df1["Depth"].min()
    d2_range = df2["Depth"].max() - df2["Depth"].min()

    if d1_range > d2_range:
        return OrganizedWithDepthRange(largest_depth_range=df1, shortest_depth_range=df2)
    else:
        return OrganizedWithDepthRange(largest_depth_range=df2, shortest_depth_range=df1)

def get_interpolated_df(organised_dfs: OrganizedWithDepthRange) -> pd.DataFrame:

    """
    Interpolates the DataFrame with the largest depth range onto the DataFrame with the smallest depth range so that
    every point in the smallest depth range has a corresponding point in the largest depth range.

    Parameters
    ----------
    organised_dfs : OrganizedWithDepthRange
        An instance of OrganizedWithDepthRange containing the DataFrames.

    Returns
    -------
    pd.DataFrame: The DataFrame with the largest depth range with interpolated onto the Depth values of the DataFrame
    with the smallest depth range.
    """

    qc_interp = interpolate.interp1d(organised_dfs.largest_depth_range["Depth"], organised_dfs.largest_depth_range["qc"], kind="linear", bounds_error=False)
    fs_interp = interpolate.interp1d(organised_dfs.largest_depth_range["Depth"], organised_dfs.largest_depth_range["fs"], kind="linear", bounds_error=False)
    u_interp = interpolate.interp1d(organised_dfs.largest_depth_range["Depth"], organised_dfs.largest_depth_range["u"], kind="linear", bounds_error=False)

    interpolated_df = organised_dfs.shortest_depth_range.copy()

    interpolated_df.loc[:,"qc"] = qc_interp(interpolated_df["Depth"])
    interpolated_df.loc[:,"fs"] = fs_interp(interpolated_df["Depth"])
    interpolated_df.loc[:,"u"] = u_interp(interpolated_df["Depth"])

    return interpolated_df



start_time = time.time()

config = cfg.Config()

old_to_new_match_df = pd.read_csv("/home/arr65/data/nzgd/stats_plots/sung_id_with_nzgd_match.csv")
new_converted_data_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/data")


data_dir = config.get_value("input_data_dir")

output_dir = Path(config.get_value("output_dir"))
output_dir.mkdir(parents=True, exist_ok=True)



filtered_out_df = pd.DataFrame(columns=["cpt_name", "reason", "reason_description"])

if config.get_value("input_data_format") == "csv":
    data_files = glob.glob(f"{data_dir}/*.csv")
    # cpts = [CPT.from_file(str(cpt_ffp)) for cpt_ffp in data_files]


if config.get_value("input_data_format") == "sql":

    engine = create_engine(f"sqlite:///{data_dir}/nz_cpt.db")
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    cpt_locs = load_sql_db.cpt_locations(session)
    initial_num_cpts = len(cpt_locs)

    # cpt_locs = cpt_locs[0:100]

    inconsistent_cpts = []
    old_cpt_not_in_new = []

    print("loading CPTs")
    #cpt_locs = cpt_locs[4:6]
    for row_n, cpt_loc in tqdm(enumerate(cpt_locs),total=len(cpt_locs)):

        if row_n % 1000 == 0:  # print every 1000
            print(f"{row_n + 1}/{len(cpt_locs)}: {cpt_loc.name}")

        cpt_records = load_sql_db.get_cpt_data(session, cpt_loc.name, columnwise=False)

        old_df = pd.DataFrame(cpt_records, columns=["Depth", "qc", "fs", "u"])

        needed_idx = cpt_loc.name == old_to_new_match_df["closest_nzgd_cpt_id"]
        # if np.sum(needed_idx) > 1:
        #     print()
        #     raise ValueError("more than one column of matching names")

        parquet_to_load = new_converted_data_dir / f"{cpt_loc.name}.parquet"
        if not parquet_to_load.exists():
            old_cpt_not_in_new.append(cpt_loc.name)
            continue

        new_df_with_metadata = pd.read_parquet(parquet_to_load)
        new_df_with_metadata = new_df_with_metadata[new_df_with_metadata["multiple_measurements"] == 0]
        new_df = new_df_with_metadata.drop(columns=["multiple_measurements", "record_name", "latitude", "longitude"])
        #new_df.loc[:, "Depth"] += 1

        organised_with_depth_range = organise_with_depth_range(old_df, new_df)

        interpolated_df = get_interpolated_df(organised_with_depth_range)

        residual = interpolated_df - organised_with_depth_range.shortest_depth_range

        

        print()
        #residual = scipy.stats.sigmaclip(residual.to_numpy, low=5, high=5)




        fractional_residual = residual / organised_with_depth_range.shortest_depth_range

        maximum_allowed_residual = 1e-2
        num_mismatch_points = np.sum(np.abs(residual.values) > maximum_allowed_residual)





        fraction_mismatch_points = num_mismatch_points / residual.size

        ### 3 sigma should include 99.7% of the data
        if fraction_mismatch_points > 1-0.997:

            import matplotlib as mpl
            mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer

            plt.subplot(3, 3, 1)
            plt.plot(interpolated_df["Depth"], interpolated_df["qc"], label="interpolated with largest depth range")
            plt.plot(organised_with_depth_range.shortest_depth_range["Depth"],
                     organised_with_depth_range.shortest_depth_range["qc"], linestyle="--",
                     label="original with shortest depth range")

            plt.subplot(3, 3, 2)
            plt.plot(interpolated_df["Depth"], interpolated_df["fs"], label="interpolated with largest depth range")
            plt.plot(organised_with_depth_range.shortest_depth_range["Depth"],
                     organised_with_depth_range.shortest_depth_range["fs"], linestyle="--",
                     label="original with shortest depth range")

            plt.subplot(3, 3, 3)
            plt.plot(interpolated_df["Depth"], interpolated_df["u"], label="interpolated with largest depth range")
            plt.plot(organised_with_depth_range.shortest_depth_range["Depth"],
                     organised_with_depth_range.shortest_depth_range["u"], linestyle="--",
                     label="original with shortest depth range")

            ###################################################################

            plt.subplot(3, 3, 4)
            plt.plot(interpolated_df["Depth"], residual["qc"])

            plt.subplot(3, 3, 5)
            plt.plot(interpolated_df["Depth"], residual["fs"])

            plt.subplot(3, 3, 6)
            plt.plot(interpolated_df["Depth"], residual["u"])

            plt.subplot(3, 3, 7)
            plt.plot(interpolated_df["Depth"], fractional_residual["qc"])

            plt.subplot(3, 3, 8)
            plt.plot(interpolated_df["Depth"], fractional_residual["fs"])

            plt.subplot(3, 3, 9)
            plt.plot(interpolated_df["Depth"], fractional_residual["u"])

            plt.show()
            print()



        print()


        max_allowed_residual = 1e-3

        num_invalid_residual_points = np.sum(max_allowed_residual > np.abs(residual))












        # try:
        #
        #     if (np.min(new_df3["Depth"]) <= np.min(cpt_records[:,0])) & (np.max(new_df3["Depth"]) >= np.max(cpt_records[:,0])):
        #         interp_onto = cpt_records
        #         array_to_interpolate = new_df3.to_numpy()
        #
        #     if (np.min(cpt_records[:,0]) < np.min(new_df3["Depth"])) & (np.max(cpt_records[:,0]) > np.max(new_df3["Depth"])):
        #         interp_onto = new_df3.to_numpy()
        #         array_to_interpolate = cpt_records
        #
        #     else:
        #         #print(f"{cpt_loc.name} has different depth ranges")
        #         inconsistent_cpts.append(cpt_loc.name)
        #         continue
        #
        #     new_interp = interpolate_one_array_onto_other(array_to_interpolate, interp_onto)
        #
        #     diff = new_interp - cpt_records
        #
        #     diff = diff[np.isfinite(diff)]
        #
        #     if not np.all(np.isclose(diff,0)):
        #         #print(f"{cpt_loc.name} is inconsistent with old data")
        #         inconsistent_cpts.append(cpt_loc.name)
        #
        #     # plt.figure()
        #     # plt.plot(new_df["qc"], new_df["Depth"], label="new")
        #     # plt.plot(cpt_records[:,1]+10, cpt_records[:,0], linestyle="--", label="old")
        #     #
        #     # plt.gca().invert_yaxis()
        #     #
        #     # plt.subplot(1, 3, 2)
        #     # plt.plot(new_df["fs"], new_df["Depth"], label="new")
        #     # plt.plot(cpt_records[:,2], cpt_records[:,0], linestyle="--", label="old")
        #     #
        #     # plt.gca().invert_yaxis()
        #     #
        #     # plt.subplot(1, 3, 3)
        #     # plt.plot(new_df["u"], new_df["Depth"], label="new")
        #     # plt.plot(cpt_records[:,3], cpt_records[:,0], linestyle="--", label="old")
        #     #
        #     # plt.gca().invert_yaxis()
        #     #
        #     # plt.legend()
        #     # plt.show()
        #
        #
        #
        # except Exception as e:
        #     #print(f"{cpt_loc.name} is inconsistent with old data")
        #     inconsistent_cpts.append(cpt_loc.name)
        #     continue


    print(f"Number of inconsistent CPTs: {len(inconsistent_cpts)}")
    print()