import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

import config as cfg

config = cfg.Config()

output_dir = Path(config.get_value("output_dir"))

summary_df = pd.read_csv(output_dir / "summary.csv")

