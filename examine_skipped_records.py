"""
This script is used to examine the skipped records in the Vs30 estimation workflow.
"""
import pandas as pd
from pathlib import Path
import config as cfg

import numpy as np
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker

import load_sql_db

def count_unique_cpt_names_by_reason(df, reason_value):
    # Filter the DataFrame based on the reason
    filtered_df = df[df['reason'] == reason_value]

    # Count the number of unique values in the 'cpt_names' column
    unique_cpt_names_count = filtered_df['cpt_name'].nunique()

    return unique_cpt_names_count


config = cfg.Config()
data_dir = config.get_value("input_data_dir")
engine = create_engine(f'sqlite:///{data_dir}/nz_cpt.db')
DBSession = sessionmaker(bind=engine)
session = DBSession()

cpt_locs = load_sql_db.cpt_locations(session)


import config as cfg
config = cfg.Config()
output_dir = Path(config.get_value("output_dir"))

skipped_records_df = pd.read_csv(output_dir / "skipped_records.csv")

# If a CPT would be skipped for several reasons, they are all listed as rows in skipped_records_df
# Therefore, we only keep the first occurance which corresponds to when the CPT would first be skipped
skipped_records_df_dropped_duplicates = skipped_records_df.drop_duplicates(subset='cpt_name', keep='first')

# each skipped record should now only have one row (showing where it would first be skipped)
assert skipped_records_df_dropped_duplicates['cpt_name'].nunique() == len(skipped_records_df_dropped_duplicates)

num_cpts = len(cpt_locs)

reasons = sorted(skipped_records_df_dropped_duplicates['reason'].unique())

num_skipped_for_reason = [count_unique_cpt_names_by_reason(skipped_records_df_dropped_duplicates, reason) for reason in reasons]

num_cpts_remaining = num_cpts - np.cumsum(np.array(num_skipped_for_reason))

# print("num_skipped, reason")
# for i in range(len(reasons)):
#
#     print(f"{num_skipped_for_reason[i]}, {reasons[i]}")



summary_df = pd.DataFrame({"num_remaining": num_cpts_remaining, "num_skipped": num_skipped_for_reason, "reason": reasons })


unique_cpt_names_count = skipped_records_df['cpt_name'].nunique()


num_no_data = count_unique_cpt_names_by_reason(skipped_records_df, "Type 01: No data in record")

print(num_no_data)

# sung_skipped_records = pd.read_csv("/home/arr65/Data/cpt/outdir/zmax_20m_vs_cap/skipped_cpts", sep=" ",
#                                    header=None,
#                                    skiprows=1,
#                                    skipfooter=1)

sung_skipped_records = pd.read_csv("cpt2vs30/skipped_cpts.csv", sep=",")

sung_reasons = sorted(sung_skipped_records["reason"].unique())

sung_num_skipped_for_reason = [count_unique_cpt_names_by_reason(sung_skipped_records, reason) for reason in sung_reasons]

sung_summary_df = pd.DataFrame({"num_remaining": num_cpts - np.cumsum(np.array(sung_num_skipped_for_reason)), "num_skipped": sung_num_skipped_for_reason, "reason": sung_reasons })

#
# num_sung_skipped_records = sung_skipped_records["cpt_name"].nunique()

merged_df = sung_skipped_records.merge(skipped_records_df, how="outer", on="cpt_name", suffixes=("_sung", "_ours"))

merged_df.to_csv(output_dir/"merged_skipped_records.csv")

print()