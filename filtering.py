from collections import Counter
import numpy as np
import functools, operator
from pathlib import Path
import time

import pandas as pd
import yaml

from vs_calc import CPT, VsProfile, calculate_weighted_vs30

from sung import loc_filter


# def log_error(skipped_fp, cpt_name, error):
#     skipped_fp.write(f"{cpt_name} - {error}\n")


def count_digits(arr):
    stringified = str(arr).replace("0", "").replace(".", "")
    return Counter(stringified)

def skipped_record_entry(cpt_name, reason):
    return pd.DataFrame({"cpt_name": [cpt_name], "reason": [reason]})


def no_data_in_cpt(cpt_name: str, cpt_record: np.array, skipped_recs_df: pd.DataFrame):
    if cpt_record.size == 0:
        return True, pd.concat([skipped_recs_df,skipped_record_entry(cpt_name, "Type 01: No data in record")],ignore_index=True)

    return False, skipped_recs_df

def duplicated_depth_values(cpt: CPT, skipped_records_df) -> bool:

    u, c = np.unique(cpt.depth, return_counts=True)
    if np.any([c > 1]):
        return True, pd.concat([skipped_records_df, skipped_record_entry(cpt.name, "Type 03: Duplicate depth detected - invalid CPT")],
                               ignore_index=True)

    return False, skipped_records_df




def values_less_than_threshold(cpt: CPT, threshold: float = 0.0) -> bool:
    """Check for negative values beyond a tolerance level in the CPT data

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the data to check.
    tolerance : float, optional
        The tolerance level for negative values, by default 0.0

    Returns
    -------
    True if any negative values are found, False otherwise
    """
    # Check for invalid negative readings
    if any(cpt.Fs < threshold) or any(cpt.Qc < threshold) or any(cpt.u < threshold):

        return True, skipped_record_entry(cpt.name, f"Type 04 : data values less than {threshold}")

    return False, None

def repeated_digits(cpt: CPT, max_num_allowed_repeated_digits) -> bool:
    if any(value > max_num_allowed_repeated_digits for fs_value in cpt.Fs for value in count_digits(fs_value).values()):
        return True, skipped_record_entry(cpt.name, f"Type 05 : More than {max_num_allowed_repeated_digits} repeated digits")
    return False, None

def insufficient_depth(cpt: CPT, min_allowed_max_depth_m: float = 5.0) -> bool:

    if np.max(cpt.depth) < min_allowed_max_depth_m:
        #log_error(skipped_fp, cpt_name, f"Type 06 : depth<5: {max_depth}")
        return True, skipped_record_entry(cpt.name, f"Type 06 : Maximum depth less than {min_allowed_max_depth_m} m")
    return False, None

def insufficient_depth_span(cpt: CPT, min_allowed_depth_span_m: float = 5.0) -> bool:

    if np.max(cpt.depth) - np.min(cpt.depth) < min_allowed_depth_span_m:
        #log_error(skipped_fp, cpt_name, f"Type 07 : depth range <5: {z_span}")
        return True, skipped_record_entry(cpt.name, f"Type 07 : depth span is less than {min_allowed_depth_span_m} m")
    return False, None

def filter_one_cpt_on_data_quality(cpt: CPT, filters: list[callable], filter_params: list) -> tuple[bool, pd.DataFrame]:

    filter_out = False
    filter_reasons = []
    for idx, filter in enumerate(filters):
        skip, reason = filter(cpt, filter_params[idx])

        filter_out |= skip
        filter_reasons.append(reason)

    if filter_out:
        return filter_out, pd.concat(filter_reasons,ignore_index=True)

    return filter_out, None

def filter_cpts_on_data_quality(cpts: list[CPT], filters: list[callable], filter_params: list,skipped_records: pd.DataFrame):

    passing_cpts = []

    for cpt in cpts:

        filter_out, reason = filter_one_cpt_on_data_quality(cpt,
                                                       filters=filters,
                                                       filter_params=filter_params)

        skipped_records = pd.concat([skipped_records, reason])

        if not filter_out:
            passing_cpts.append(cpt)

    return passing_cpts, skipped_records


def filter_cpts_on_location_duplicates(cpts, min_separation_dist_m, output_dir, skipped_records_df):
    passing_cpts = []

    dup_locs = find_duplicated_locations(cpts, min_separation_dist_m, output_dir)

    dup_locs = functools.reduce(operator.iconcat, list(dup_locs.values()), [])

    for cpt in cpts:

        if cpt.name in dup_locs:
            skipped_records_df = pd.concat([skipped_records_df, skipped_record_entry(cpt.name, "Type 02: Duplicate location")])

        else:
            passing_cpts.append(cpt)

    return passing_cpts, skipped_records_df

def filter_cpts(cpts: list[CPT], data_quality_filters: list[callable], data_quality_filter_params: list,
                min_separation_dist_m: float, output_dir: Path, skipped_records_df: pd.DataFrame):

    print('filtering on data quality')
    t1 = time.time()
    cpts, skipped_records = filter_cpts_on_data_quality(cpts, data_quality_filters, data_quality_filter_params, skipped_records_df)
    print(f"Time taken for filtering on data quality: {(time.time() - t1)/60} minutes")
    
    t2 = time.time()
    print('filtering on location duplicates')
    cpts, skipped_records = filter_cpts_on_location_duplicates(cpts, min_separation_dist_m, output_dir, skipped_records_df)
    print(f"Time taken for filtering on location duplicates: {(time.time() - t2)/60} minutes")

    return cpts, skipped_records

def find_duplicated_locations(locs, min_separation_dist_m, out_dir):

    dup_locs_yaml_file = out_dir/"dup_locs.yaml"
    if dup_locs_yaml_file.exists():
        with open(out_dir/"dup_locs.yaml", 'r') as f:
            dup_locs_dict = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        dup_locs_dict=loc_filter.locs_multiple_records(locs, min_separation_dist_m, stdout=False)
        #let's save this in a yaml file for future use
        with open(out_dir/"dup_locs.yaml","w") as f:
            yaml.safe_dump(dup_locs_dict,f)

    return dup_locs_dict
