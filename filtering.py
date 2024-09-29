"""
This module contains functions to filter out unusable CPT data.
These functions are adapted from earlier work by Sung Bae in
the cpt2vs30 package.
"""

import functools
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import run_calculations
from vs_calc import CPT


def count_digits(arr):
    """
    Count the number of repeated digits in a data value.
    This function was developed by Sung Bae.

    Parameters
    ----------
    arr : a numerical data value

    Returns
    -------
    Counter
        A Counter object containing the number of repeated digits in the data value.

    """
    stringified = str(arr).replace("0", "").replace(".", "")
    return Counter(stringified)


def filtered_out_entry(
    cpt_name: str, reason: str, reason_description: str
) -> pd.DataFrame:
    """
    Create a skipped record entry.

    Parameters
    ----------
    cpt_name : str
        The name of a CPT record.
    reason : str
        The short form reason (such as a code) for skipping the record.
    reason_description : str
        A more detailed description of the reason for skipping the record.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns "cpt_name", "reason", and "reason_description".
    """

    return pd.DataFrame(
        {
            "cpt_name": [cpt_name],
            "reason": [reason],
            "reason_description": [reason_description],
        }
    )


def identify_no_data_in_cpt(
    cpt_name: str, cpt_record: np.array
) -> Optional[pd.DataFrame]:
    """
    Identify CPT records that contain no data.

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
        Otherwise, returns None.
    """

    if cpt_record.size == 0:
        return filtered_out_entry(cpt_name, "Type 01", "No data in record")
    return None


def identify_duplicated_depth_values(
    cpt: CPT, max_num_same_depth_values: int
) -> Optional[pd.DataFrame]:
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
        Otherwise, returns None.
    """

    u, c = np.unique(cpt.depth, return_counts=True)
    if np.any([c > max_num_same_depth_values]):
        return filtered_out_entry(
            cpt.name, "Type 03", "Duplicate depth detected - invalid CPT"
        )
    return None


def identify_values_less_than_threshold(cpt: CPT, threshold) -> Optional[pd.DataFrame]:
    """
    Identify if a CPT's data has values less than `threshold`.

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
        Otherwise, returns None
    """
    if any(cpt.Fs < threshold) or any(cpt.Qc < threshold) or any(cpt.u < threshold):

        return filtered_out_entry(
            cpt.name, f"Type 04", f"Data values less than {threshold}"
        )

    return None


def identify_repeated_digits(
    cpt: CPT, max_num_allowed_repeated_digits
) -> Optional[pd.DataFrame]:
    """
    Identify if any data values have more than `max_num_allowed_repeated_digits` repeated digits
    in case these data were affected by instrumental issues.

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the data.
    max_num_allowed_repeated_digits : int
        The maximum allowable value of repeated digits in a data value.

    Returns
    -------
    pd.DataFrame or None
        If any data values have more than `max_num_allowed_repeated_digits` repeated digits,
        returns a DataFrame with columns "cpt_name", "reason", and "reason_description".
        Otherwise, returns None.
    """

    if any(
        value > max_num_allowed_repeated_digits
        for fs_value in cpt.Fs
        for value in count_digits(fs_value).values()
    ):
        # print()
        return filtered_out_entry(
            cpt.name,
            f"Type 05",
            f"More than {max_num_allowed_repeated_digits} repeated digits indicating a possible instrument problem",
        )
    return None


def identify_insufficient_depth(
    cpt: CPT, min_allowed_max_depth_m: float = 5.0
) -> Optional[pd.DataFrame]:
    """
    Identify if the CPT has a maximum depth is less than `min_allowed_max_depth_m`.

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the data.
    min_allowed_max_depth_m : float
        The minimum allowable maximum depth.

    Returns
    -------
    pd.DataFrame or None
        If the maximum depth is less than `min_allowed_max_depth_m`, returns a DataFrame with columns
        "cpt_name", "reason", and "reason_description".
        Otherwise, returns None.
    """

    if np.max(cpt.depth) < min_allowed_max_depth_m:
        return filtered_out_entry(
            cpt.name, f"Type 06", f"Maximum depth less than {min_allowed_max_depth_m} m"
        )

    return None


def identify_insufficient_depth_span(
    cpt: CPT, min_allowed_depth_span_m
) -> Optional[pd.DataFrame]:
    """
    Identify if the CPT has a depth span less than `min_allowed_depth_span_m`.

    Parameters
    ----------
    cpt : CPT
       A CPT object containing the data.
    min_allowed_depth_span_m : float
        The minimum allowable depth span.

    Returns
    -------
    pd.DataFrame or None
        If the depth span is less than `min_allowed_depth_span_m, returns a DataFrame with columns
        "cpt_name", "reason", and "reason_description".
        Otherwise, returns None.

    """

    if (np.max(cpt.depth) - np.min(cpt.depth)) < min_allowed_depth_span_m:
        return filtered_out_entry(
            cpt.name, f"Type 07", f"depth span less than {min_allowed_depth_span_m} m"
        )
    return None


def identify_location_duplication(
    cpt: CPT, sep_dist_dup_name_tup: tuple[float, list[str]]
) -> Optional[pd.DataFrame]:
    """
    Identify if a CPT is within `min_CPT_separation_dist_m` of another CPT.

    Parameters
    ----------
    cpt: CPT
        A CPT object containing the data.
    sep_dist_dup_name_tup: tuple[float, list[str]
        A tuple containing two items.
            `min_CPT_separation_dist_m` is the first item
            `duplicate_location_cpt_names` is the second item

    Returns
    -------
    pd.DataFrame or None
        If the CPT is within `min_CPT_separation_dist_m` of another CPT, returns a DataFrame with columns
        "cpt_name", "reason", and "reason_description".
        Otherwise, returns None.
    """

    min_CPT_separation_dist_m, duplicate_location_cpt_names = sep_dist_dup_name_tup

    if cpt.name in duplicate_location_cpt_names:
        return filtered_out_entry(
            cpt.name,
            f"Type 02",
            f"CPT within {min_CPT_separation_dist_m} m of another CPT",
        )
    return None


def identify_why_cpt_filtered_out(
    cpt: CPT, filters: list[callable], filter_params: list
) -> Optional[pd.DataFrame]:
    """
    Filter a single cpt using its data quality. This function
    needs no information about other cpts.

    Parameters
    ----------
    cpts : list[CPT]
        A list of CPT objects.
    filters : list[callable]
        A list of functions that take a CPT object and return a DataFrame if the CPT should be skipped.
    filter_params : list
        A list of parameters to pass to the data quality filters.
    Returns
    -------
    pd.DataFrame or None
        If the CPT should be skipped, returns a DataFrame with columns "cpt_name", "reason", and "reason_description".
        Otherwise, returns None.
    """

    skipped_records = []
    for idx, filter in enumerate(filters):
        skipped_records.append(filter(cpt, filter_params[idx]))

    if any(isinstance(item, pd.DataFrame) for item in skipped_records):
        return pd.concat(skipped_records, ignore_index=True)
    return None


def identify_cpts_to_filter_out(
    cpts: list[CPT],
    filters: list[callable],
    filter_params: list,
    filtered_out_df: pd.DataFrame,
    n_procs: int = 1,
):
    """
    Filter a list of cpts.
    This function needs the locations of all cpts.

    Parameters
    ----------
    cpt : CPT
        A CPT objects containing data.
    min_CPT_separation_dist_m : float
        The minimum required distance between two CPTs to not be considered duplicates.
    filters : list[callable]
        A list of functions that take a CPT object and return a DataFrame if the CPT should be skipped.
    filter_params : list
        A list of parameters to pass to the data quality filters.
    filtered_out_df : pd.DataFrame
        A DataFrame with columns "cpt_name", "reason", and "reason_description".
    dup_locs_output_dir: Path, optional
        If a Path, then a DataFrame containing the distance to the closest neighbouring CPT will be saved
        in this directory
        If None, the DataFrame will not be saved
    Returns
    -------
    pd.DataFrame or None
        If the CPT should be skipped, returns a DataFrame with columns "cpt_name", "reason", and "reason_description".
        Otherwise, returns None.
    """

    with multiprocessing.Pool(processes=n_procs) as pool:

        skipped_record_entry_list = pool.map(
            functools.partial(
                identify_why_cpt_filtered_out,
                filters=filters,
                filter_params=filter_params,
            ),
            cpts,
        )

    filtered_out_df = pd.concat(
        [filtered_out_df, *skipped_record_entry_list], ignore_index=True
    )

    return filtered_out_df


def apply_filter_to_single_cpt(cpt: CPT, names_to_filter_out: list[str]):
    """
    Filters out a CPT if it was previously identified as needing to be filtered out.

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the data.
    names_to_filter_out : list[str]
        A list of CPT names to filter out.

    Returns
    -------
    CPT or None
       - None if the CPT is filtered out
       - CPT otherwise
    """

    if cpt.name not in names_to_filter_out:

        return cpt

    else:
        return None


def apply_filters_to_cpts(cpts, names_to_filter_out, n_procs: int = 1):
    """
    Filters out a CPT if it was previously identified as needing to be filtered out.

    Parameters
    ----------
    cpts : list[CPT]
        A list of CPT objects.
    names_to_filter_out : list[str]
        A list of CPT names to filter out.
    n_procs: int
        The number of processes to use for the calculation.

    Returns
    -------
    list[CPT]
        A list of CPT objects that were not filtered out.
    """

    with multiprocessing.Pool(processes=n_procs) as pool:

        remaining_cpts = pool.map(
            functools.partial(
                apply_filter_to_single_cpt, names_to_filter_out=names_to_filter_out
            ),
            cpts,
        )

    remaining_cpts = [cpt for cpt in remaining_cpts if cpt is not None]

    return remaining_cpts


def get_dup_loc_names(
    min_CPT_separation_dist_m: float,
    cpts: list[CPT],
    n_procs: int = 1,
    output_dir: Optional[Path] = None,
    load_from_previous: Optional[Path] = None,
) -> list[str]:
    """
    Get a list of CPTs with duplicate locations.

    Parameters
    ----------
    min_CPT_separation_dist_m : float
        The minimum required distance between two CPTs to not be considered duplicates.
    cpts : list[CPT]
        A list of CPT objects.
    n_procs : int
        The number of processes to use for the calculation.
    output_dir : Path, optional
        If a Path, then a DataFrame containing the distance to the closest neighbouring CPT will be saved
        in this directory.
        If None, the DataFrame will not be saved.
    load_from_previous : Path, optional
        If a Path, then the list of CPT names with duplicate locations will be loaded from this file.
        If None, the list will be calculated.

    Returns
    -------
    list[str]
        A list of CPT names with duplicate locations.
    """

    if load_from_previous:
        return np.loadtxt(load_from_previous)

    all_dist_to_closest_cpt_df = run_calculations.get_all_dist_to_closest_cpt(
        cpts, n_procs=n_procs, output_dir=output_dir
    )

    duplicate_cpt_df = all_dist_to_closest_cpt_df[
        all_dist_to_closest_cpt_df["distance_to_closest_cpt_km"]
        < (min_CPT_separation_dist_m / 1000.0)
    ]

    unique_names = np.unique(duplicate_cpt_df[["cpt_name", "closest_cpt_name"]].values)

    if output_dir is not None:
        duplicate_cpt_df.to_csv(
            output_dir / "closest_cpt_distance_duplicate_locs.csv", index=False
        )
        np.savetxt(output_dir / "duplicate_cpt_names.txt", unique_names, fmt="%s")

    return list(unique_names)


def summarize_filtered_out_df(
    filtered_out_df: pd.DataFrame, initial_num_cpts: int
) -> pd.DataFrame:
    """
    Summarize the filtered out DataFrame.

    Parameters
    ----------
    filtered_out_df : pd.DataFrame
        A DataFrame containing the filtered out data.
    initial_num_cpts : int
        The initial number of CPTs before filtering.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the summarized data.
    """

    counts = filtered_out_df["reason"].value_counts()

    reason_code_description_dict = {
        "Type 01": "No data in record",
        "Type 02": "CPT too close to another CPT",
        "Type 03": "Duplicate depth detected - invalid CPT",
        "Type 04": "Data values less than threshold",
        "Type 05": "Repeated digits indicating a possible instrument problem",
        "Type 06": "Insufficient maximum depth",
        "Type 07": "Insufficient depth span",
    }

    description_list = [
        reason_code_description_dict.get(reason, reason) for reason in counts.index
    ]

    summary_df = pd.DataFrame(
        {
            "num_remaining": initial_num_cpts - counts.values,
            "num_skipped": counts,
            "reason": counts.index,
            "reason_description": description_list,
        }
    ).reset_index(drop=True)

    return summary_df
