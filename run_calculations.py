"""
Functions to run the calculations for Vs30 estimation.
The Vs30 calculations were implemented by Joel Ridden
in the vs_calc package.
"""
import functools
import multiprocessing

import pandas as pd


from vs_calc import (
    CPT,
    VsProfile)

def calculate_vs30_from_single_cpt(cpt: CPT, cpt_vs_correlations, vs30_correlations):

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
    cpts, cpt_vs_correlations, vs30_correlations, n_procs=1
):

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

