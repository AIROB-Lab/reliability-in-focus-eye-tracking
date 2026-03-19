from typing import List

import pandas as pd


def get_eyetracking_validity_stats(eyetracking_w_phases: pd.DataFrame, granularity: List[str] = None) -> pd.DataFrame:
    """Get the percentage of valid data points for the specified granularity.

    Parameters
    ----------
    eyetracking_w_phases : pd.DataFrame
        eyetracking data with phases from the TrustDataset
    granularity : list, optional
        list of columns to group by, by default ["participant", "cell", "run"]

    Returns
    -------
    pd.DataFrame
        DataFrame with the percentage and counts of valid data points for the specified granularity
    """
    if granularity is None:
        granularity = ["participant", "cell", "run"]

    et = eyetracking_w_phases[eyetracking_w_phases["phase"] != "NonePhase"].copy()
    et.set_index(["participant", "cell", "run", "mov", "phase"], inplace=True)
    irrelevant_cols = et.columns
    et["is_valid"] = et.filter(like="gaze_direction_validity").all(axis=1)
    # only keep is_valid column
    et.drop(irrelevant_cols, inplace=True, axis=1)

    # drop all rows where cell, run, mov, phase is nan
    et = et.reset_index().dropna(how="any")

    # count the number of valid data points and the total number of data points for the specified granularity
    et["data_point_counter"] = 1
    counts = et.groupby(granularity, observed=True).sum(numeric_only=True)
    counts["percent_valid"] = counts["is_valid"] * 100 / counts["data_point_counter"]
    return counts
