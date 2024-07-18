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
skipped_records_df['cpt_name'] = skipped_records_df['cpt_name'].str.strip()
skipped_records_df['reason'] = skipped_records_df['reason'].str.strip()

# If a CPT would be skipped for several reasons, they are all listed as rows in skipped_records_df
# Therefore, we only keep the first occurance which corresponds to when the CPT would first be skipped
skipped_records_df_dropped_duplicates = skipped_records_df.drop_duplicates(subset='cpt_name', keep='first')

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

# sung_skipped_records = pd.read_csv("/home/arr65/Data/cpt/outdir/zmax_20m_vs_cap/skipped_cpts", sep=" ",
#                                    header=None,
#                                    skiprows=1,
#                                    skipfooter=1)

#sung_skipped_records = pd.read_csv("cpt2vs30/skipped_cpts.csv", sep=",")
sung_skipped_records = pd.read_csv("/home/arr65/Data/cpt/outdir/zmax_20m_vs_cap/skipped_cpts", sep=",")

sung_skipped_records["reason"] = sung_skipped_records["reason"].str.strip()

sung_skipped_records['cpt_name'] = sung_skipped_records['cpt_name'].str.strip()

sung_reasons = sorted(sung_skipped_records["reason"].unique())

sung_num_skipped_for_reason = [count_unique_cpt_names_by_reason(sung_skipped_records, reason) for reason in sung_reasons]

sung_summary_df = pd.DataFrame({"num_remaining": num_cpts - np.cumsum(np.array(sung_num_skipped_for_reason)), "num_skipped": sung_num_skipped_for_reason, "reason": sung_reasons })

#
# num_sung_skipped_records = sung_skipped_records["cpt_name"].nunique()

merged_df = sung_skipped_records.merge(skipped_records_df_dropped_duplicates, how="outer", on="cpt_name", suffixes=("_sung", "_ours"))

reason_conversion_dict = {""}

for i in range(len(reasons)):
    print(f"Num skipped for reason {reasons[i]} = {num_skipped_for_reason[i]}")

merged_df["reasons_match"] = merged_df["reason_sung"] == merged_df["reason_ours"]
# for i in range(len(merged_df)):
#     sr = merged_df["reason_sung"].iloc[i]
#     sm = merged_df["reason_ours"].iloc[i]
#
#     print("sr", sr)
#     print("sm", sm)
#
#     print()
#
#     if (not isinstance(sr, str) & (not isinstance(sm, str))):
#         merged_df.iloc[i,4] = False
#         print()
#         continue
#
#     if sr in sm:
#         merged_df.iloc[i,4] = True
#     else:
#         merged_df.iloc[i,4].iloc[i] = False



merged_df.to_csv(output_dir/"merged_skipped_records.csv")

merged_df_mismatch = merged_df[~merged_df["reasons_match"]]
merged_df_mismatch.to_csv(output_dir/"merged_skipped_records_mismatch.csv")

print()