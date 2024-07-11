"""
The main script to run the Vs30 estimation workflow
"""



from collections import Counter
import config as cfg
import glob
from pathlib import Path
import pandas as pd
import time
import functools, operator

from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker

import load_sql_db
import filtering
from vs_calc import CPT, VsProfile, calculate_weighted_vs30

start_time = time.time()

config = cfg.Config()

data_dir = config.get_value("input_data_dir")

output_dir = Path(config.get_value("output_dir"))
output_dir.mkdir(parents=True, exist_ok=True)

skipped_records_df = pd.DataFrame(columns=["cpt_name", "reason"])


if config.get_value("input_data_format") == "csv":
    data_files = glob.glob(f"{data_dir}/*.csv")
    cpts = [CPT.from_file(str(cpt_ffp)) for cpt_ffp in data_files]



if config.get_value("input_data_format") == "sql":

    engine = create_engine(f'sqlite:///{data_dir}/nz_cpt.db')
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    locs = load_sql_db.cpt_locations(session)

    # num_cpt_to_do = 2000
    # locs = locs[:num_cpt_to_do]

    cpts = []

    print('loading CPTs')
    for row_n, loc in enumerate(locs):
        cpt_name = loc.name

        #print('loading cpt data for:', cpt_name)

        cpt_records = load_sql_db.get_cpt_data(session, cpt_name, columnwise=False)

        skip, skipped_records_df = filtering.no_data_in_cpt(cpt_name, cpt_records, skipped_records_df)
        if not skip:
            cpts.append(CPT(cpt_name, cpt_records[:,0], cpt_records[:,1], cpt_records[:,2], cpt_records[:,3], loc.nztm_x, loc.nztm_y))


print(f"time taken for loading: {(time.time() - start_time)/60.0} minutes")



cpts, skipped_records_df = filtering.filter_cpts(cpts,
                                               data_quality_filters=[filtering.duplicated_depth_values,
                                                        filtering.values_less_than_threshold,
                                                        filtering.repeated_digits,
                                                        filtering.insufficient_depth,
                                                        filtering.insufficient_depth_span],
                                               data_quality_filter_params=[None, # placeholder as duplicated_depth_values takes no parameters
                                                              config.get_value("min_allowed_data_value"),
                                                              config.get_value("max_num_allowed_repeated_digits"),
                                                              config.get_value("min_allowed_max_depth_m"),
                                                              config.get_value("min_allowed_depth_span_m")],
                                            min_separation_dist_m=config.get_value("min_separation_dist_m"),
                                            output_dir=output_dir,
                                            skipped_records_df=skipped_records_df)

skipped_records_df.to_csv(output_dir / "skipped_records.csv", index=False)

print(f"total taken: {(time.time() - start_time)/60.0} minutes")

