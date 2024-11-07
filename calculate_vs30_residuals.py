import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np

metadata_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/metadata")

old_vs30_predictions_at_new_locations = pd.read_csv(metadata_dir / "old_vs30_prediction_at_new_cpt_locations.csv")

new_vs30_predictions_at_new_locations = pd.read_csv(metadata_dir / "parquet_vs30_results.csv")
new_vs30_predictions_at_new_locations = new_vs30_predictions_at_new_locations.dropna(subset=['vs30', 'vs30_sd'])

residuals = []
log_residuals = []
old_vs30s = []
record_lon = []
record_lat = []

for idx, row in tqdm(new_vs30_predictions_at_new_locations.iterrows(),total=len(new_vs30_predictions_at_new_locations)):
    old_prediction_row = old_vs30_predictions_at_new_locations[old_vs30_predictions_at_new_locations["record_name"] == row["cpt_name"]]
    if old_prediction_row.empty:
        continue
    else:
        old_vs30 = old_prediction_row["predicted_vs30_m_per_sec"].values[0]
        new_vs30 = row["vs30"]
        residual = new_vs30 - old_vs30
        log_residual = np.log(new_vs30) - np.log(old_vs30)

        residuals.append(residual)
        log_residuals.append(log_residual)
        old_vs30s.append(old_vs30)
        record_lon.append(old_prediction_row["record_lon"].values[0])
        record_lat.append(old_prediction_row["record_lat"].values[0])



new_vs30_predictions_at_new_locations["residual_new_minus_old"] = residuals
new_vs30_predictions_at_new_locations["log_residual_ln_new_minus_ln_old"] = log_residuals
new_vs30_predictions_at_new_locations["old_vs30_predictions"] = old_vs30s
new_vs30_predictions_at_new_locations["record_lon"] = record_lon
new_vs30_predictions_at_new_locations["record_lat"] = record_lat

new_vs30_predictions_at_new_locations.to_csv(metadata_dir / "new_vs30_resdiuals.csv", index=False)




