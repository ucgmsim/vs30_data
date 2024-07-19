"""
This script is used to examine the skipped records in the Vs30 estimation workflow.
"""

from pathlib import Path

import pandas as pd

import config as cfg

config = cfg.Config()

data_dir = config.get_value("input_data_dir")
output_dir = Path(config.get_value("output_dir"))

skipped_records_df = pd.read_csv(output_dir / "filtered_out_with_all_reasons.csv")
skipped_records_df["cpt_name"] = skipped_records_df["cpt_name"].str.strip()
skipped_records_df["reason"] = skipped_records_df["reason"].str.strip()

# If a CPT would be skipped for several reasons, they are all listed as rows in skipped_records_df
# Therefore, we only keep the first occurance which corresponds to when the CPT would first be skipped
skipped_records_df_dropped_duplicates = skipped_records_df.drop_duplicates(
    subset="cpt_name", keep="first"
)

sung_skipped_records = pd.read_csv(
    "/home/arr65/Data/cpt/outdir/zmax_20m_vs_cap/skipped_cpts", sep=","
)
sung_skipped_records["reason"] = sung_skipped_records["reason"].str.strip()
sung_skipped_records["cpt_name"] = sung_skipped_records["cpt_name"].str.strip()

# Merge the two skipped records dataframes and see if the reasons for skipping match
merged_df = sung_skipped_records.merge(
    skipped_records_df_dropped_duplicates,
    how="outer",
    on="cpt_name",
    suffixes=("_sung", "_andrew"),
)
merged_df["reasons_match"] = merged_df["reason_sung"] == merged_df["reason_ours"]

merged_df.to_csv(output_dir / "post_merged_skipped_records.csv")

merged_df_mismatch = merged_df[~merged_df["reasons_match"]]
merged_df_mismatch.to_csv(output_dir / "post_merged_skipped_records_mismatch.csv")




