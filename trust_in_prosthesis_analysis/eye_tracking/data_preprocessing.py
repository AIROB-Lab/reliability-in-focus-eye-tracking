import json
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from trust_in_prosthesis_analysis.eye_tracking.array_handling import (
    bool_array_to_start_end_timestamps_array,
    merge_intervals,
)
from trust_in_prosthesis_analysis.eye_tracking.io_utils import (
    read_roi_data,
    read_eye_tracking_data,
    read_study_events,
    get_phase_change_events,
    get_mov_change_events,
    get_runs,
    get_study_breaks,
)


def clean_roi_data(
    roi_data: pd.DataFrame, gaze_validity_cols: pd.DataFrame, study_events: pd.DataFrame
) -> pd.DataFrame:
    """
    Cleans the ROI data by setting all invalid gaze data to False, adding a is_valid column
    and cutting the data to start and end according to the study events dataframe.

    Parameters
    ----------
    roi_data : pandas.DataFrame
        ROI data as loaded from the `read_roi_data` function.
    gaze_validity_cols : pandas.DataFrame
        DataFrame containing the validity of the gaze data (gaze_direction_validity columns from the eye_tracking_data).

    Returns
    -------
    roi_data : pandas.DataFrame
        Cleaned ROI data.
    """
    roi_data = roi_data.copy()
    roi_data = _clean_roi_validity(roi_data, gaze_validity_cols)
    print(
        f"#1 number of samples: {roi_data.shape[0]};"
        f" percentage of valid: {roi_data.is_valid.sum() / roi_data.is_valid.shape[0] * 100:.2f}"
    )
    roi_data = _cut_df_to_start_end(roi_data, study_events)
    print(
        f"#2 number of samples after cutting start and end: {roi_data.shape[0]};"
        f"percentage of valid: {roi_data.is_valid.sum() / roi_data.is_valid.shape[0] * 100:.2f}"
    )
    roi_data = _cut_study_breaks(roi_data, study_events)
    print(
        f"#3 number of samples after cutting breaks: {roi_data.shape[0]}; "
        f"percentage of valid: {roi_data.is_valid.sum() / roi_data.is_valid.shape[0] * 100:.2f}"
    )
    return roi_data


def _clean_roi_validity(roi_data: pd.DataFrame, gaze_validity_cols: pd.DataFrame) -> pd.DataFrame:
    """Sets all invalid gaze data to False in the ROI data and adds an is_valid column."""

    # roi_data["is_valid"] = gaze_validity_cols.all(axis=1) -> will be NaN if the indices are not perfectly aligned
    combined_invalidity = ~(gaze_validity_cols.all(axis=1))  # True when invalid -> better for start end array
    start_end_invalidity = pd.DataFrame(
        bool_array_to_start_end_timestamps_array(combined_invalidity.values, combined_invalidity.index.values),
        columns=["start", "end"],
    )

    roi_data["is_valid"] = True
    for start, end in start_end_invalidity.itertuples(index=False):
        # This seems to cause off by one errors in both directions:
        # the sample before the invalid area is set to False and the sample after the valid area is set to False.
        # roi_data.loc[start:end, "is_valid"] = False

        # This seems to work better: (The time between samples is usually ca. 8ms, so here we use 5ms as a buffer)
        roi_data.loc[start + pd.Timedelta(5, "ms") : end - pd.Timedelta(5, "ms"), "is_valid"] = False

    # determine the columns with the fixations, that should be set to False
    # Default are the hand, object and targets_hit columns
    target_cols = roi_data.columns.values
    target_cols = target_cols[
        (target_cols != "time_stamp_s") & (target_cols != "is_valid") & (target_cols != "targets_hit")
    ]

    # set the target columns to False where the gaze data is invalid
    roi_data.loc[~roi_data["is_valid"], target_cols] = False

    return roi_data


def clean_eye_tracking_data(eye_tracking_data: pd.DataFrame, study_events: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the eye tracking data by setting all invalid pupil diameter data to NA
    and cutting the data to start and end according to the study events dataframe.

    Parameters
    ----------
    eye_tracking_data : pandas.DataFrame
        Eye tracking data as loaded from the `read_eye_tracking_data` function.
    study_events : pandas.DataFrame
        DataFrame containing the study events as loaded from the `read_study_events` function.

    Returns
    -------
    eye_tracking_data : pandas.DataFrame
        Cleaned eye tracking data.
    """
    eye_tracking_data = eye_tracking_data.copy()
    eye_tracking_data = _clean_pupil_diameter(eye_tracking_data)
    eye_tracking_data = _cut_df_to_start_end(eye_tracking_data, study_events)
    eye_tracking_data = _cut_study_breaks(eye_tracking_data, study_events) # ToDo: Check if this gets ever called
    return eye_tracking_data


def _clean_pupil_diameter(eye_tracking_data):
    """Sets all values of the pupil diameter to NA where marked as invalid and deletes the validity cols."""
    eye_tracking_data = eye_tracking_data.assign(
        pupilDiameterRight_mm=lambda df: df["pupilDiameterRight_mm"].where(df["right_pupil_diameter_validity"], pd.NA),
        pupilDiameterLeft_mm=lambda df: df["pupilDiameterLeft_mm"].where(df["left_pupil_diameter_validity"], pd.NA),
    )
    eye_tracking_data.drop(["right_pupil_diameter_validity", "left_pupil_diameter_validity"], axis=1, inplace=True)
    return eye_tracking_data


def _cut_df_to_start_end(df: pd.DataFrame, study_events: pd.DataFrame) -> pd.DataFrame:
    study_start, study_end = get_study_start_and_end(study_events)
    return df.between_time(study_start.time(), study_end.time())


def _cut_study_breaks(df: pd.DataFrame, study_events: pd.DataFrame) -> pd.DataFrame:
    """Removes the data during study breaks."""
    df = df.copy()
    breaks = get_study_breaks(study_events)
    for _, row in breaks.iterrows():
        indexer = df.index.indexer_between_time(row.start.time(), row.end.time())
        df = df.reset_index().drop(indexer).set_index("time_stamp", drop=True)
    return df


def get_fixations_from_roi_data(
    roi_data: pd.DataFrame,
    relevant_cols: Optional[List[str]] = None,
    fillable_gap_size_s: float = 0.1,
    min_fixation_duration_s: float = 0.12,
    min_duration_between_fixations_s: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """
    Extracts the fixation phases from the ROI data.


    According to Lavoie et al., 2018 (https://doi.org/10.1167/18.6.18):

    - Brief periods (< 100ms) of missing data in each ROI fixation are filled in (-> `fillable_gap_size_s`)
    - Then, any brief fixations (< 100ms) are removed to avoid erroneous detection of fixations (e.g. fly-throughs)
      (This function uses 120ms as a default, see description of `min_fixation_duration_s` attribute below.)


    Lavoie et al. also define the following, which is not quite implemented here (yet?):

    - A fixation is said to occur when the distance between gaze vector and ROI is sufficiently small
      (Probably this is done in a similar way by unity directly and thus we already have the ROI data.)
    - Additionally, the velocity from gaze vector to ROI should be below 0.5 m/s. (TODO: should we implement this?)

    According to Lavoie et al. 2024 (https://doi.org/10.1167/jov.24.2.9), the minimum time between fixations
    in order to be classified as "distinct" is 100ms. However, if all gaps < 100ms are already filled, this
    requirement is already automatically satisfied.
    This function nevertheless allows to set the final minimal gap duration
    with the `min_duration_between_fixations_s` parameter.

    Parameters
    ----------
    roi_data : pandas.DataFrame
        ROI data as loaded from the `read_roi_data` function.
    relevant_cols : list of str
        Columns to extract the fixations from, default are the "hand" and "object" column.
    fillable_gap_size_s : float
        Maximum gap size in seconds between fixations to merge them into one fixation.
    min_fixation_duration_s : float
        Minimum duration of a fixation in seconds. The default is 120 ms,
        because this is the minimum duration needed to allow for information processing
        (see Wilson et al., 2010: https://doi.org/10.1007/s00464-010-0986-1).
    min_duration_between_fixations_s: float
        Minimum duration between the fixations in seconds. This is evaluated AFTER fly-throughs are deleted
        according to the min_fixation_duration_s criterion.
        If fixations are less than `min_fixation_duration_s` apart, they will be merged.
        If min_duration_between_fixations is less than or equal to fillable_gap_size_s, this will have no effect.

    Returns
    -------
    fixations : dict
        Dictionary containing a pd.DataFrame for each relevant column with the start and end indices and timestamps
        of the fixations.
    """

    if relevant_cols is None:
        relevant_cols = ["hand", "object"]

    fixations_dict = {}
    for target in relevant_cols:
        fixations_dict[target] = _get_fixations_from_bool_array(
            roi_data[target].values,
            roi_data["time_stamp_s"].values,
            fillable_gap_size_s,
            min_fixation_duration_s,
            min_duration_between_fixations_s,
        )

    # fixations = pd.concat(fixations_dict, axis=1)
    return fixations_dict


def _get_fixations_from_bool_array(
    bool_array: np.ndarray,
    timestamps_array: np.array,
    fillable_gap_size_s: float = 0.1,
    min_fixation_duration_s: float = 0.12,
    min_duration_between_fixations_s: float = 0.1,
) -> pd.DataFrame:

    # convert the boolean array to start and end timestamps
    start_end_timestamps = bool_array_to_start_end_timestamps_array(bool_array, timestamps_array)

    # merge the fixation periods that are less than fillable_gap_size apart
    merged_timestamps = merge_intervals(start_end_timestamps, fillable_gap_size_s)

    # only keep the fixation periods that are longer than min_fixation_duration
    merged_timestamps = np.delete(
        merged_timestamps, np.where(np.diff(merged_timestamps) < min_fixation_duration_s)[0], axis=0
    )

    # merge the fixation periods that are less than min_duration_between_fixations_s apart
    # TODO: do we need this? If fillable_gap_size >= min_duration_between_fixations, the following will do nothing.
    merged_timestamps = merge_intervals(merged_timestamps, min_duration_between_fixations_s)

    # get the indices for the start and end timestamps from the merged timestamps
    index_timestamp_series = pd.Series(np.arange(timestamps_array.shape[0]), index=timestamps_array)

    # The following does not always work, because the timestamps are not always unique:
    # merged_idxs = index_timestamp_series.loc[merged_timestamps.flatten()].values.reshape(-1, 2)

    # maybe a FIX: take the first occurrence of each timestamp
    _, unique_indices = np.unique(timestamps_array, return_index=True)
    index_timestamp_series = index_timestamp_series.iloc[unique_indices]
    merged_idxs = index_timestamp_series.loc[merged_timestamps.flatten()].values.reshape(-1, 2)

    return pd.DataFrame(
        np.hstack([merged_idxs, merged_timestamps]), columns=["start_idx", "end_idx", "start_time_s", "end_time_s"]
    ).convert_dtypes()  # to make dtype of the idxs int instead of float


def get_study_start_and_end(study_events: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the start and end time of the study from the study events DataFrame.

    The start time is defined as the StudyEvent with value "Start" and
    the end time is defined as the last StudyEvent within a Release phase.

    Parameters
    ----------
    study_events : pandas.DataFrame
        DataFrame containing the study events.

    Returns
    -------
    start_time : pandas.Timestamp
        Start time of the study.
    end_time : pandas.Timestamp
        End time of the study.
    """
    study_start = study_events[(study_events["event"] == "StudyEvent") & (study_events["value"] == "Start")].iloc[0]
    study_start = study_start["time_stamp"]
    study_end = study_events[(study_events["event"] == "StudyEvent") & (study_events["phase"] == "Release")].iloc[-1]
    study_end = study_end["time_stamp"]
    return study_start, study_end


def integrate_event_data_into_df(df: pd.DataFrame, study_events: pd.DataFrame) -> pd.DataFrame:
    """Adds the participant, cell, run, mov and phase column to the ROI data.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe, e.g. as loaded from the `read_roi_data` function.
    study_events : pandas.DataFrame
        DataFrame containing the study events as loaded from the `read_study_events` function.

    Returns
    -------
    roi_data : pandas.DataFrame
        ROI data with the participant, cell, run, mov and phase information.
    """
    df = df.copy()

    categories = {
        "phase": get_phase_change_events(study_events), # gets phase change infos
        "mov": get_mov_change_events(study_events), # gets mov change infos
        "run": get_runs(study_events), # gets run change infos
    }

    df["cell"] = pd.NA
    # go through phases, movs and runs and add the corresponding information to the data
    # (between start and end of the categories)
    for col in categories.keys():
        df[col] = None
        for name, start, end, cell in categories[col][[col, "start", "end", "cell"]].to_numpy():
            indexer = df.index.to_series().between(start, end)
            df.loc[indexer, col] = name
            df.loc[indexer, "cell"] = cell

    # add the participant and cell information to the roi_data
    df["participant"] = categories["run"].loc[0, "participant"]

    # change the datatype of the info cols to category
    for col in ["participant", "cell", "run", "mov", "phase"]:
        df[col] = df[col].astype("category")
    return df


def get_fixations_per_phase(
    roi_w_phases: pd.DataFrame, phase_change_events: pd.DataFrame, relevant_cols: Optional[List[str]] = None
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Returns a dictionary with the fixations for each phase.

    Parameters
    ----------
    roi_w_phases : pandas.DataFrame
        ROI data with the participant, cell, run, mov and phase information as
        loaded from the `integrate_event_data_into_roi` function.
    phase_change_events : pandas.DataFrame
        DataFrame containing the phase change events as loaded from the `get_phase_change_events` function.
    relevant_cols : list of str or None
        Columns to extract the fixations from, default are the "hand" and "object" column.

    Returns
    -------
    phase_dict : dict
        Dictionary with the fixations for each phase.
        :param phase_change_events:
    """
    if relevant_cols is None:
        relevant_cols = ["hand", "object"]

    phase_dict = {}
    for (phase_name,), phase in roi_w_phases.groupby(["phase"], observed=False):
        fixs_dict = {key: [] for key in relevant_cols}

        # get the fixation start and end timestamps for each tiny section of the phase
        for _, row in phase_change_events[phase_change_events["phase"] == phase_name].dropna().iterrows():
            section = phase.iloc[phase.index.indexer_between_time(row.start.time(), row.end.time())]
            if len(section) == 0:
                continue

            participant_id, cell_id, run_id, mov = section[["participant", "cell", "run", "mov"]].iloc[0]
            fixs = get_fixations_from_roi_data(section, relevant_cols=relevant_cols)

            # append the fixations of each target to the respective list in the dict
            for key, data in fixs.items():
                data["participant"] = participant_id
                data["cell"] = cell_id
                data["run"] = run_id
                data["mov"] = mov
                fixs_dict[key].append(data)

        # concatenate the fixations of each target to one dataframe
        for key, data in fixs_dict.items():
            if len(data) == 0:
                continue
            df = pd.concat(data)
            df = df.drop(["start_idx", "end_idx"], axis=1).reset_index(drop=True)
            fixs_dict[key] = df

        phase_dict[phase_name] = fixs_dict
    return phase_dict


if __name__ == "__main__":
    with open("../../config.json") as c:
        conf = json.load(c)
    data_per_particpant_folder = Path(conf["data_per_participant_folder"])
    data_root = data_per_particpant_folder / "VP_001/20240515_104109_fabio"
    eye_tracking_file = list(data_root.glob("*Eye_Tracking.csv"))[0]
    roi_file = list(data_root.glob("*ROI.csv"))[0]
    events_file = list(data_root.glob("*StudyEvents.csv"))[0]

    roi_data = read_roi_data(roi_file)
    et_data = read_eye_tracking_data(eye_tracking_file)
    study_events = read_study_events(events_file)

    gaze_validity_cols = et_data.filter(like="gaze_direction_validity")

    # This is for testing to test if it also works with misaligned timestamps:
    gaze_validity_cols.index += pd.Timedelta(0.001, unit="s")

    cleaned_roi_data = clean_roi_data(roi_data, gaze_validity_cols, study_events)

    # fixations = get_fixations_from_roi_data(cleaned_roi_data)
    fixations = get_fixations_per_phase(
        integrate_event_data_into_df(cleaned_roi_data, study_events), get_phase_change_events(study_events)
    )
    print(fixations)
