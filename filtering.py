from collections import Counter
import numpy as np
import functools, operator
from pathlib import Path
import time
from typing import Optional

import pandas as pd
import yaml

from vs_calc import CPT, VsProfile, calculate_weighted_vs30
import matplotlib.pyplot as plt

from cpt2vs30 import loc_filter

def count_digits(arr):
    stringified = str(arr).replace("0", "").replace(".", "")
    return Counter(stringified)

def skipped_record_entry(cpt_name : str, reason: str, reason_description : str) -> pd.DataFrame:
    """
    Create a skipped record entry.

    Parameters
    ----------
    cpt_name : str
        The name of a CPT record.
    reason : str
        The reason for skipping the record.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns "cpt_name", "reason", and "reason_description".
    """

    return pd.DataFrame({"cpt_name": [cpt_name], "reason": [reason], "reason_description": [reason_description]})


def no_data_in_cpt(cpt_name: str, cpt_record: np.array) -> Optional[pd.DataFrame]:
    """
    Check if there is data in the CPT record.

    Parameters
    ----------
    cpt_name : str
        The name of a CPT record.
    cpt_record : np.array
        An array containing the CPT data.

    Returns
    -------
    pd.DataFrame or None
        If there is no data, returns a DataFrame with columns "cpt_name", "reason", and "reason_description".
        If there is data, returns None.
    """

    if cpt_record.size == 0:
        return skipped_record_entry(cpt_name, "Type 01", "No data in record")
    return None

def duplicated_depth_values(cpt: CPT, max_num_same_depth_values: int) -> Optional[pd.DataFrame]:

    """
    Check for duplicated depth values in the CPT data.

    Parameters
    ----------
    cpt : CPT
        The CPT object containing the data.
    max_num_same_depth_values : int
        The maximum number of the same depth values allowed
        (generally should only be one value at each depth).

    Returns
    -------
    pd.DataFrame or None
        If there are more than `max_num_same_depth_values`, returns a DataFrame with columns
        "cpt_name", "reason", and "reason_description".
        If there are not more than `max_num_same_depth_values`, returns None.
    """

    u, c = np.unique(cpt.depth, return_counts=True)
    if np.any([c > max_num_same_depth_values]):
        return skipped_record_entry(cpt.name, "Type 03","Duplicate depth detected - invalid CPT")
    return None

def values_less_than_threshold(cpt: CPT, threshold) -> Optional[pd.DataFrame]:

    """
    Check for values less than `threshold` in the CPT data.

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the data to check.
    threshold : float
        The threshold value to check against.

    Returns
    -------
    pd.DataFrame or None
        If there are values less than `threshold`, returns a DataFrame with columns
        "cpt_name", "reason", and "reason_description".
        If there are no values less than `threshold`, returns None
    """
    if any(cpt.Fs < threshold) or any(cpt.Qc < threshold) or any(cpt.u < threshold):

        return skipped_record_entry(cpt.name, f"Type 04", f"Data values less than {threshold}")

    return None

# def repeated_digits(cpt: CPT, max_num_allowed_repeated_digits) -> Optional[pd.DataFrame]:
#
#     """
#     Check for values less than `threshold` in the CPT data.
#
#     Parameters
#     ----------
#     cpt : CPT
#         A CPT object containing the data to check.
#     threshold : float
#         The threshold value to check against.
#
#     Returns
#     -------
#     pd.DataFrame or None
#         If there are values less than `threshold`, returns a DataFrame with columns
#         "cpt_name", "reason", and "reason_description".
#         If there are no values less than `threshold`, returns None
#     """
#
#     if any(value > max_num_allowed_repeated_digits for fs_value in cpt.Fs for value in count_digits(fs_value).values()):
#         print()
#         return skipped_record_entry(cpt.name, f"Type 05",
# f"More than {max_num_allowed_repeated_digits} repeated digits indicating a possible instrument problem")
#     return None

def repeated_digits(cpt: CPT, max_num_allowed_repeated_digits) -> Optional[pd.DataFrame]:

    """
    Check for values less than `threshold` in the CPT data.

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the data to check.
    threshold : float
        The threshold value to check against.

    Returns
    -------
    pd.DataFrame or None
        If there are values less than `threshold`, returns a DataFrame with columns
        "cpt_name", "reason", and "reason_description".
        If there are no values less than `threshold`, returns None
    """

    if any(value > max_num_allowed_repeated_digits for fs_value in cpt.Fs for value in count_digits(fs_value).values()):
        print()
        return skipped_record_entry(cpt.name, f"Type 05",
f"More than {max_num_allowed_repeated_digits} repeated digits indicating a possible instrument problem")
    return None

def repeated_digits_Andrew(cpt: CPT, max_num_allowed_repeated_digits) -> Optional[pd.DataFrame]:

    """
    Check for values less than `threshold` in the CPT data.

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the data to check.
    threshold : float
        The threshold value to check against.

    Returns
    -------
    pd.DataFrame or None
        If there are values less than `threshold`, returns a DataFrame with columns
        "cpt_name", "reason", and "reason_description".
        If there are no values less than `threshold`, returns None
    """

    found_exceeding_digit = False  # Flag to indicate if the condition is met

    for index, fs_value in enumerate(cpt.Fs):  # Step 1
        digit_counts = count_digits(fs_value)  # Step 2
        for value in digit_counts.values():  # Step 3
            if value > max_num_allowed_repeated_digits:  # Step 4

                print()
                found_exceeding_digit = True  # Set the flag to True if condition is met
                break  # Exit the loop as we only need one instance to satisfy the condition
        if found_exceeding_digit:
            break  # Exit the outer loop as well since the condition is satisfied

    if found_exceeding_digit:
        print()
        return skipped_record_entry(cpt.name, f"Type 05",
f"More than {max_num_allowed_repeated_digits} repeated digits indicating a possible instrument problem")
    return None

def insufficient_depth(cpt: CPT, min_allowed_max_depth_m: float = 5.0) -> Optional[pd.DataFrame]:

    if np.max(cpt.depth) < min_allowed_max_depth_m:
        return skipped_record_entry(cpt.name, f"Type 06", f"Maximum depth less than {min_allowed_max_depth_m} m")

    return None

def insufficient_depth_span(cpt: CPT, min_allowed_depth_span_m) -> Optional[pd.DataFrame]:

    if (np.max(cpt.depth) - np.min(cpt.depth)) < min_allowed_depth_span_m:
        return skipped_record_entry(cpt.name, f"Type 07",f"depth span less than {min_allowed_depth_span_m} m")
    return None

def find_duplicated_locations(cpt_locs, min_CPT_separation_dist_m, dup_locs_output_dir=None):

    if dup_locs_output_dir is None:
        dup_locs_dict = loc_filter.locs_multiple_records(cpt_locs, min_CPT_separation_dist_m, stdout=False)
        return functools.reduce(operator.iconcat, list(dup_locs_dict.values()), [])

    dup_locs_yaml_file = dup_locs_output_dir/"dup_locs.yaml"
    if dup_locs_yaml_file.exists():
        with open(dup_locs_output_dir/"dup_locs.yaml", 'r') as f:
            dup_locs_dict = yaml.load(f, Loader=yaml.SafeLoader)
    else:
        dup_locs_dict=loc_filter.locs_multiple_records(cpt_locs, min_CPT_separation_dist_m, stdout=False)
        #let's save this in a yaml file for future use
        with open(dup_locs_output_dir/"dup_locs.yaml","w") as f:
            yaml.safe_dump(dup_locs_dict,f)

    dup_locs_dict = loc_filter.locs_multiple_records(cpt_locs, min_CPT_separation_dist_m, stdout=False)

    return functools.reduce(operator.iconcat, list(dup_locs_dict.values()), [])

def filter_cpts_on_location_duplicates(cpts, min_CPT_separation_dist_m, skipped_records_df, dup_locs_output_dir=None):

    dup_locs_cpt_names = find_duplicated_locations(cpts, min_CPT_separation_dist_m, dup_locs_output_dir)

    if len(dup_locs_cpt_names) == 0:
        return cpts, skipped_records_df
    
    preserved_cpts = []

    for cpt in cpts:

        if cpt.name in dup_locs_cpt_names:
            skipped_records_df = pd.concat([skipped_records_df, skipped_record_entry(cpt.name, f"Type 02", f"CPT within {min_CPT_separation_dist_m} m of another CPT")])
            continue

        preserved_cpts.append(cpt)

    return preserved_cpts, skipped_records_df

def filter_single_cpt_on_data_quality(cpt: CPT, data_quality_filters: list[callable], data_quality_filter_params: list) -> Optional[pd.DataFrame]:

    skipped_records = []
    for idx, filter in enumerate(data_quality_filters):
        skipped_records.append(filter(cpt, data_quality_filter_params[idx]))

    # if skipped_records.count(None) == len(skipped_records):
    #     return None

    if any(isinstance(item, pd.DataFrame) for item in skipped_records):
        return pd.concat(skipped_records, ignore_index=True)
    return None


def filter_cpts(cpts: list[CPT], min_CPT_separation_dist_m: float,
                data_quality_filters: list[callable], data_quality_filter_params: list,
                skipped_records_df: pd.DataFrame,
                dup_locs_output_dir: Path = None):

    t0 = time.time()

    cpts, skipped_records_df = filter_cpts_on_location_duplicates(cpts, min_CPT_separation_dist_m, skipped_records_df, dup_locs_output_dir)

    data_quality_preserved_cpts = []

    t1 = time.time()

    for cpt in cpts:

        skipped_record_entry = filter_single_cpt_on_data_quality(cpt, data_quality_filters, data_quality_filter_params)

        if skipped_record_entry is not None:
            skipped_records_df = pd.concat([skipped_records_df, skipped_record_entry])
            continue

        data_quality_preserved_cpts.append(cpt)

    print(f"Time taken for filtering on data quality: {(time.time() - t1)/60} minutes")
    return data_quality_preserved_cpts, skipped_records_df




