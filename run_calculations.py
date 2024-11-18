"""
Functions to run the calculations for Vs30 estimation.
The Vs30 calculations were implemented by Joel Ridden
in the vs_calc package.
"""

import functools
import multiprocessing
from pathlib import Path
from typing import Optional
import numpy.typing as npt

import numpy as np
import pandas as pd
from qcore import coordinates, geo

from vs_calc import CPT, VsProfile


def calculate_vs30_from_single_cpt(
    cpt: CPT, cpt_vs_correlations: list[str], vs30_correlations: list[str]
) -> pd.DataFrame:
    """
    Calculate Vs30 values for a single cpt using several correlations.

    Parameters
    ----------
    cpt : CPT
        The CPT object for which to calculate Vs30 values.
    cpt_vs_correlations : list[str]
        A list of Vs correlations to use for the calculations.
    vs30_correlations : list[str]
        A list of Vs30 correlations to use for the calculations.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
            - cpt_name: the name of the CPT
            - nztm_x: the NZTM x coordinate of the CPT
            - nztm_y: the NZTM y coordinate of the CPT
            - cpt_correlation: the Vs correlation used for the CPT
            - vs30_correlation: the Vs30 correlation used for the CPT
            - vs30: the Vs30 value
            - vs30_sd: the standard deviation of the Vs30 value
    """

    results_df_list = []

    for cpt_vs_correlation in cpt_vs_correlations:

        for vs30_correlation in vs30_correlations:

            cpt_vs_profile = VsProfile.from_cpt(cpt, cpt_vs_correlation)

            cpt_vs_profile.vs30_correlation = vs30_correlation

            results_df_list.append(
                pd.DataFrame(
                    {
                        "cpt_name": [cpt.name],
                        "nztm_x": [cpt.nztm_x],
                        "nztm_y": [cpt.nztm_y],
                        "cpt_correlation": [cpt_vs_correlation],
                        "vs30_correlation": [cpt_vs_profile.vs30_correlation],
                        "vs30": [cpt_vs_profile.vs30],
                        "vs30_sd": [cpt_vs_profile.vs30_sd],
                    }
                )
            )

    return pd.concat(results_df_list, ignore_index=True)


def calculate_vs30_from_all_cpts(
    cpts: list[CPT],
    cpt_vs_correlations: list[str],
    vs30_correlations: list[str],
    n_procs: int = 1,
) -> pd.DataFrame:
    """
    Calculates Vs30 values for a list of cpts using several correlations.

    Parameters
    ----------
    cpts : list[CPT]
        A list of CPT objects.
    cpt_vs_correlations : list[str]
        A list of Vs correlations to use for the calculations.
    vs30_correlations : list[str]
        A list of Vs30 correlations to use for the calculations.
    n_procs : int
        The number of processes to use for the calculation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
            - cpt_name: the name of the CPT
            - nztm_x: the NZTM x coordinate of the CPT
            - nztm_y: the NZTM y coordinate of the CPT
            - cpt_correlation: the Vs correlation used for the CPT
            - vs30_correlation: the Vs30 correlation used for the CPT
            - vs30: the Vs30 value
            - vs30_sd: the standard deviation of the Vs30 value
    """

    with multiprocessing.Pool(processes=n_procs) as pool:

        results_df_list = pool.map(
            functools.partial(
                calculate_vs30_from_single_cpt,
                cpt_vs_correlations=cpt_vs_correlations,
                vs30_correlations=vs30_correlations,
            ),
            cpts,
        )

    return pd.concat(results_df_list, ignore_index=True)



def apply_nztm_to_ll(cpt):
    """
    Convert a CPT object's NZTM coordinates to longitudes and latitudes.

    Parameters
    ----------
    cpt : CPT
        A CPT objects.
    Returns
    -------
    pd.DataFrame
           A DataFrame with the following columns:
            - cpt_name: the name of the CPT
            - lon: the longitude of the CPT
            - lat: the latitude of the CPT
    """

    ## nztm_x and nztm_y are in the opposite order to the order in which they are used in the function
    ## as the function uses a different definition of x and y
    latlon = coordinates.nztm_to_wgs_depth(np.array([cpt.nztm_y, cpt.nztm_x]))
    return pd.DataFrame(
        {"cpt_name": [cpt.name], "lon": [latlon[1]], "lat": [latlon[0]]}
    )


def calc_all_ll(cpts: list[CPT], n_procs: Optional[int] = 1) -> pd.DataFrame:
    """
    For all CPTs, convert their NZTM coordinates to longitudes and latitudes.

    Parameters
    ----------
    cpt : CPT
        A CPT objects.
    Returns
    -------
    pd.DataFrame
           A DataFrame with the following columns:
            - cpt_name: the name of the CPT
            - lon: the longitude of the CPT
            - lat: the latitude of the CPT
    """

    with multiprocessing.Pool(processes=n_procs) as pool:

        all_ll = pool.map(apply_nztm_to_ll, cpts)

    return pd.concat(all_ll, ignore_index=True)

