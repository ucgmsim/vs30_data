
import matplotlib.pyplot as plt

import pandas as pd
import scipy
import numpy as np

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
from qcore import coordinates, geo

import download_nzgd_data.validation.load_sql_db as load_sql_db


old_data_dir = Path("/home/arr65/vs30_data_input_data/sql")
parquet_output_dir = Path("/home/arr65/vs30_data_input_data/parquet_v2/data")
parquet_conversion_meta_output_dir = parquet_output_dir.parent / "meta"
parquet_conversion_meta_output_dir.mkdir(parents=True, exist_ok=True)


parquet_output_dir.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{old_data_dir}/nz_cpt.db")
DBSession = sessionmaker(bind=engine)
session = DBSession()

cpt_locs = load_sql_db.cpt_locations(session)
initial_num_cpts = len(cpt_locs)

inconsistent_cpts = []
old_cpt_not_in_new = []

cpt_with_size_zero = []
cpt_ids_with_other_exceptions = []
other_exceptions = []

for row_n, cpt_loc in tqdm(enumerate(cpt_locs), total=len(cpt_locs)):
    
    try:

        cpt_records = load_sql_db.get_cpt_data(session, cpt_loc.name, columnwise=False)
    
        if cpt_records.size == 0:
            cpt_with_size_zero.append(cpt_loc.name)
            continue
    
        old_df = pd.DataFrame(cpt_records, columns=["Depth", "qc", "fs", "u"])
    
        old_df.attrs["nztm_x"] = cpt_loc.nztm_x
        old_df.attrs["nztm_y"] = cpt_loc.nztm_y
    
        latlon = coordinates.nztm_to_wgs_depth(np.array([cpt_loc.nztm_y, cpt_loc.nztm_x]))
    
        old_df.attrs["lat"] = latlon[0]
        old_df.attrs["lon"] = latlon[1]
    
        old_df.attrs["cpt_name"] = cpt_loc.name
    
        old_df.insert(0, "record_name", cpt_loc.name)
        old_df.insert(1, "latitude", latlon[0])
        old_df.insert(2, "longitude", latlon[1])
    
        old_df.to_parquet(parquet_output_dir / f"{cpt_loc.name}.parquet")
    
    except Exception as e:
        cpt_ids_with_other_exceptions.append(cpt_loc.name)
        other_exceptions.append(e)


cpt_with_size_zero = pd.DataFrame(cpt_with_size_zero, columns=["cpt_name"])
cpt_with_size_zero.to_csv(parquet_conversion_meta_output_dir / "cpt_with_size_zero.csv")

cpt_ids_with_other_exceptions = pd.DataFrame({"cpt_name": cpt_ids_with_other_exceptions, "exception": other_exceptions})
