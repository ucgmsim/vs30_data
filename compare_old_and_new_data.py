"""
The main script for calculating Vs30 values from CPT data.
"""

import glob
import time
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from scipy import interpolate

import config as cfg
import filtering
import load_sql_db
import numpy as np
# from vs_calc import (
#     CPT,
#     cpt_vs_correlations,
#     vs30_correlations,
# )

start_time = time.time()

config = cfg.Config()

old_to_new_match_df = pd.read_csv("/home/arr65/data/nzgd/stats_plots/sung_id_with_nzgd_match.csv")
new_converted_data_dir = Path("/home/arr65/data/nzgd/standard_format_batch60/cpt/data")


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

    cpts = []

    print("loading CPTs")
    for row_n, cpt_loc in enumerate(cpt_locs):

        if row_n % 1000 == 0:  # print every 1000
            print(f"{row_n + 1}/{len(cpt_locs)}: {cpt_loc.name}")

        cpt_records = load_sql_db.get_cpt_data(session, cpt_loc.name, columnwise=False)

        filtered_out_entry = filtering.identify_no_data_in_cpt(
            cpt_loc.name, cpt_records
        )

        if filtered_out_entry is not None:
            filtered_out_df = pd.concat(
                [filtered_out_df, filtered_out_entry], ignore_index=True
            )
            continue

        # cpts.append(
        #     CPT(
        #         cpt_loc.name,
        #         cpt_records[:, 0],
        #         cpt_records[:, 1],
        #         cpt_records[:, 2],
        #         cpt_records[:, 3],
        #         cpt_loc.nztm_x,
        #         cpt_loc.nztm_y,
        #     )
        # )


        needed_idx = cpt_loc.name == old_to_new_match_df["closest_nzgd_cpt_id"]
        if np.sum(needed_idx) > 1:
            raise ValueError("more than one column of matching names")

        print()

        new_df = pd.read_parquet(new_converted_data_dir / f"{cpt_loc.name}.parquet")
        new_df2 = new_df[new_df["multiple_measurements"] == 0]
        new_df3 = new_df2.drop(columns=["multiple_measurements", "record_name", "latitude", "longitude"])

        new_df3_include_idx = (new_df3["Depth"] >= np.min(cpt_records[:, 0])) & (new_df3["Depth"] <= np.max(cpt_records[:, 0]))

        new_df4 = new_df3[new_df3_include_idx]

        ## new_depth = interpolate.interp1d(, )

        #ratio = cpt_records/new_df3[new_df3_include_idx]

        plt.figure()
        plt.subplot(1,3,1)
        plt.plot(new_df["qc"], new_df["Depth"], label="new")
        plt.plot(cpt_records[:,1]+10, cpt_records[:,0], linestyle="--", label="old")

        plt.gca().invert_yaxis()

        plt.subplot(1, 3, 2)
        plt.plot(new_df["fs"], new_df["Depth"], label="new")
        plt.plot(cpt_records[:,2], cpt_records[:,0], linestyle="--", label="old")

        plt.gca().invert_yaxis()

        plt.subplot(1, 3, 3)
        plt.plot(new_df["u"], new_df["Depth"], label="new")
        plt.plot(cpt_records[:,3], cpt_records[:,0], linestyle="--", label="old")

        plt.gca().invert_yaxis()

        plt.legend()
        plt.show()

        print()












        print()
