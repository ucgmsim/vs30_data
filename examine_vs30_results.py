from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

output_dir = Path("/home/arr65/vs30_data_output")

df = pd.read_csv(output_dir / "vs30_results.csv")

## Group on cpt_name and calculate the mean, median, range, min, and max of the vs30 values
group_mean = df.groupby("cpt_name")["vs30"].mean()
group_median = df.groupby("cpt_name")["vs30"].median()

group_range = df.groupby("cpt_name")["vs30"].apply(lambda x: x.max() - x.min())

group_min = df.groupby("cpt_name")["vs30"].min()
group_max = df.groupby("cpt_name")["vs30"].max()

# plot a histogram of the range, excluding the extreme outliers
lower_bound, upper_bound = np.percentile(group_range, [0, 99])

# Filter the data to exclude outliers
filtered_data = group_range[(group_range >= lower_bound) & (group_range <= upper_bound)]

plt.hist(filtered_data, bins=100)
plt.xlabel("range in Vs30 estimates")
plt.ylabel("count")
plt.title(
    "Range of Vs30 estimates for each CPT\nexcluding the highest 1% of range values"
)
plt.savefig(output_dir / "Vs30_range_histogram_excluding_highest_1pc.png", dpi=600)

plt.close("all")
# plot a histrogram of the range, including all extreme outliers
plt.hist(group_range, bins=1000)
plt.xlabel("range in Vs30 estimates")
plt.ylabel("count")
plt.title(
    "Range of Vs30 estimates for each CPT\nincluding the highest 1% of range values"
)
plt.savefig(output_dir / "Vs30_range_histogram.png", dpi=600)
