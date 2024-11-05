
import matplotlib.pyplot as plt

import pandas as pd
import scipy

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

import download_nzgd_data.validation.load_sql_db as load_sql_db



old_data_dir = Path("/home/arr65/vs30_data_input_data/sql")
parquet_output_dir = Path("/home/arr65/vs30_data_input_data/parquet")



engine = create_engine(f"sqlite:///{old_data_dir}/nz_cpt.db")
DBSession = sessionmaker(bind=engine)
session = DBSession()

cpt_locs = load_sql_db.cpt_locations(session)
initial_num_cpts = len(cpt_locs)

inconsistent_cpts = []
old_cpt_not_in_new = []

print("loading CPTs")

for row_n, cpt_loc in tqdm(enumerate(cpt_locs), total=len(cpt_locs)):

    cpt_records = load_sql_db.get_cpt_data(session, cpt_loc.name, columnwise=False)

    old_df = pd.DataFrame(cpt_records, columns=["Depth", "qc", "fs", "u"])

    old_df.attrs["nztm_x"] = cpt_loc.nztm_x
    old_df.attrs["nztm_y"] = cpt_loc.nztm_y




    print()

