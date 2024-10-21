"""
The main script for calculating Vs30 values from CPT data.
"""

import glob
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import config as cfg
import filtering
import load_sql_db
import run_calculations
from vs_calc import (
    CPT,
    cpt_vs_correlations,
    vs30_correlations,
)

start_time = time.time()

config = cfg.Config()

old_to_new_match_df = pd.read_csv("/home/arr65/data/nzgd/stats_plots/sung_id_with_nzgd_match.csv")

print()


data_dir = config.get_value("input_data_dir")

output_dir = Path(config.get_value("output_dir"))
output_dir.mkdir(parents=True, exist_ok=True)



filtered_out_df = pd.DataFrame(columns=["cpt_name", "reason", "reason_description"])

if config.get_value("input_data_format") == "csv":
    data_files = glob.glob(f"{data_dir}/*.csv")
    cpts = [CPT.from_file(str(cpt_ffp)) for cpt_ffp in data_files]


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

        cpts.append(
            CPT(
                cpt_loc.name,
                cpt_records[:, 0],
                cpt_records[:, 1],
                cpt_records[:, 2],
                cpt_records[:, 3],
                cpt_loc.nztm_x,
                cpt_loc.nztm_y,
            )
        )



        old_to_new_match_df["cpt_name"] = old_to_new_match_df["cpt_name"].str.strip()

        print()
