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
import run_calculations
import load_sql_db
from vs_calc import (
    CPT,
    VsProfile,
    cpt_vs_correlations,
    vs30_correlations,
)



start_time = time.time()

config = cfg.Config()

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

    # num_cpt_to_do = 5000
    # cpt_locs = cpt_locs[0:5000]

    cpts = []

    print("loading CPTs")
    for row_n, cpt_loc in enumerate(cpt_locs):

        if row_n % 1000 == 0:  # print every 1000
            print(f"{row_n + 1}/{len(cpt_locs)}: {cpt_loc.name}")

        cpt_records = load_sql_db.get_cpt_data(session, cpt_loc.name, columnwise=False)

        filtered_out_entry = filtering.identify_no_data_in_cpt(cpt_loc.name, cpt_records)

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


print(f"time taken for loading: {(time.time() - start_time)/60.0} minutes")

dup_loc_names = filtering.get_dup_loc_names(
    min_CPT_separation_dist_m=config.get_value("min_CPT_separation_dist_m"),
    cpts=cpts,
    n_procs=config.get_value("n_procs"),
    output_dir=output_dir,
)

filtered_out_df = filtering.identify_cpts_to_filter_out(
    cpts=cpts,
    filters=[
        filtering.identify_location_duplication,
        filtering.identify_duplicated_depth_values,
        filtering.identify_values_less_than_threshold,
        # filtering.identify_repeated_digits,
        filtering.identify_insufficient_depth,
        filtering.identify_insufficient_depth_span,
    ],
    filter_params=[
        (config.get_value("min_CPT_separation_dist_m"), dup_loc_names),
        config.get_value("max_num_same_depth_values"),
        config.get_value("min_allowed_data_value"),
        # config.get_value("max_num_allowed_repeated_digits"),
        config.get_value("min_allowed_max_depth_m"),
        config.get_value("min_allowed_depth_span_m"),
    ],
    filtered_out_df=filtered_out_df,
    n_procs=config.get_value("n_procs"),
)

cpts = filtering.apply_filters_to_cpts(
    cpts, list(filtered_out_df["cpt_name"]), n_procs=config.get_value("n_procs")
)

print(f"time taken for filtering: {(time.time() - start_time)/60.0} minutes")

vs_calc_start_time = time.time()
vs30_results_df = run_calculations.calculate_vs30_from_all_cpts(
    cpts=cpts,
    cpt_vs_correlations=list(cpt_vs_correlations.CPT_CORRELATIONS.keys()),
    vs30_correlations=list(vs30_correlations.VS30_CORRELATIONS.keys()),
    n_procs=config.get_value("n_procs"),
)

print(
    f"time taken for vs calculation: {(time.time() - vs_calc_start_time)/60.0} minutes"
)

# Write output files
filtered_out_df.to_csv(output_dir / "filtered_out_with_all_reasons.csv", index=False)
vs30_results_df.to_csv(output_dir / "vs30_results.csv", index=False)
filtered_out_df.drop_duplicates(subset="cpt_name", keep="first").to_csv(
    output_dir / "filtered_out_only_first_reason.csv", index=False
)
summary_df = pd.DataFrame({"num_remaining": num_cpts_remaining, "num_skipped": num_skipped_for_reason, "reason": reasons })


print(f"total taken: {(time.time() - start_time)/60.0} minutes")
