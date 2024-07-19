import pandas as pd
from pathlib import Path

my = pd.read_csv(Path("/home/arr65/vs30_data_output") / "vs30_results.csv")

su = pd.read_csv(Path("/home/arr65/Data/cpt/outdir/zmax_20m_vs_cap") / "vs30_results.csv")

condition1 = my["cpt_correlation"] == "mcgann_2015"

condition2 = my["vs30_correlation"] == "boore_2011"
#condition2 = my["vs30_correlation"] == "boore_2004"

filtered_my = my[condition1 & condition2]

merged_df = su.merge(
    filtered_my,
    how="outer",
    on="cpt_name",
    suffixes=("_sung", "_andrew"),
)

merged_df_vs30s = merged_df[["vs30_sung", "vs30_andrew"]]

merged_df_vs30s["diff"] = merged_df_vs30s["vs30_sung"] - merged_df_vs30s["vs30_andrew"]

# print the median diff
print("median diff:")
print(merged_df_vs30s["diff"].median())

merged_df_vs30s.to_csv(Path("/home/arr65/vs30_data_output") / "vs30_diff_andrew_sung.csv")

## Correlations
#mcgann_2015

# boore_2004
# boore_2011
