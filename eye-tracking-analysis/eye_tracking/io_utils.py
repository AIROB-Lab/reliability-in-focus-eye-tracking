import sys
from pathlib import Path

import pandas as pd
import numpy as np
import json

from pandas._libs import OutOfBoundsDatetime
import warnings

TARGET_COLS = ["target11_R", "target11_L", "target2", "home"]


def read_eye_tracking_data(eye_tracking_path: Path) -> pd.DataFrame:
    """
    Reads the eye tracking data from the specified csv file.
    If start_index_at_zero_seconds is set, the start time will be subtracted from the whole time index so that it
    starts at zero.

    Parameters
    ----------
    eye_tracking_path : Path
        Path to the eye tracking data.

    Returns
    -------
    eye_tracking_data : pandas.DataFrame
        Eye tracking data.
    """
    eye_tracking_data = pd.read_csv(eye_tracking_path, index_col=False)
    eye_tracking_data.index = pd.to_datetime(eye_tracking_data["time_stamp_s"], unit="s")
    eye_tracking_data.index.names = ["time_stamp"]

    eye_tracking_data = add_validity_cols(eye_tracking_data)
    return eye_tracking_data


def read_roi_data(roi_path: Path, add_target_cols: bool = True) -> pd.DataFrame:
    """
    Reads the ROI data from the specified csv file.
    If start_index_at_zero_seconds is set, the start time will be subtracted from the whole time index so that it
    starts at zero.

    Parameters
    ----------
    roi_path : Path
        Path to the ROI data.
    add_target_cols : bool
        Whether to read the entries in the "targets_hit" column and transform it to individual columns.

    Returns
    -------
    roi_data : pandas.DataFrame
        ROI data.
    """
    roi_data = pd.read_csv(
        roi_path,
        header=0,
        names=["time_stamp_s", "hand", "object", "target_hit_properties", "targets_hit"],
        index_col=False,
        sep=",",
    )
    roi_data.set_index(pd.to_datetime(roi_data["time_stamp_s"], unit="s"), inplace=True)
    roi_data.index.names = ["time_stamp"]

    if add_target_cols:
        roi_data = add_target_cols_to_roi_data(roi_data)
    return roi_data


def read_main_camera(camera_path: Path) -> pd.DataFrame:
    camera_data = pd.read_csv(camera_path, header=0, index_col=False, sep=";")
    camera_data.set_index(pd.to_datetime(camera_data.time_stamp_s, unit="s"), inplace=True)
    camera_data.index.names = ["time_stamp"]
    return camera_data


def read_elbow_tracker(tracker_path: Path) -> pd.DataFrame:
    tracker_data = pd.read_csv(tracker_path, header=0, index_col=False, sep=";")
    tracker_data.set_index(pd.to_datetime(tracker_data.time_stamp_s, unit="s"), inplace=True)
    tracker_data.index.names = ["time_stamp"]
    return tracker_data

# def bad_line_handler(bad_line):
#         print("Skipping bad line:", bad_line)
#         return None
    
def read_emg_raw(emg_path: Path) -> pd.DataFrame:
    emg_data = pd.read_csv(emg_path, header=10, skiprows=10, on_bad_lines="skip", index_col=False, sep=",",names=["time_stamp", "emg0", "emg1","emg2","emg3","emg4","emg5","emg6","emg7","emg8","emg9","emg10","emg11","emg12","emg13","emg14","emg15"])
    emg_data.set_index(pd.to_datetime(emg_data.time_stamp, unit="s"), inplace=True)
    # emg_data.index += pd.Timedelta(hours=2)
    emg_data.index.names = ["time_stamp"]
    return emg_data


def add_validity_cols(eye_tracking: pd.DataFrame) -> pd.DataFrame:
    """Parses the validity bitmasks ("eyeDataValidataBitMask[Left,Right]") and adds them as individual
    boolean columns to the DataFrame."""
    eye_tracking = eye_tracking.copy()
    validity_dfs = []
    for side in ["left", "right"]:
        col_name = f"eyeDataValidataBitMask{side.capitalize()}"
        decoded_bitmask = pd.DataFrame(
            np.unpackbits(eye_tracking[col_name].to_numpy().astype(np.uint8)[:, None], axis=1)[:, 3:],
            columns=[  # see: SRanipal docs for namespace ViveSR.anipal.Eye SingleEyeDataValidity
                f"{side}_pupil_pos_in_sensor_area_validity",
                f"{side}_eye_openness_validity",
                f"{side}_pupil_diameter_validity",
                f"{side}_gaze_direction_validity",
                f"{side}_origin_validity",
            ],
            index=eye_tracking.index,
        ).astype(bool)
        validity_dfs.append(decoded_bitmask)
    return pd.concat([eye_tracking] + validity_dfs, axis=1)


def add_target_cols_to_roi_data(roi_data: pd.DataFrame) -> pd.DataFrame:
    for col in TARGET_COLS:
        roi_data[col] = roi_data["targets_hit"].str.contains(col)
    return roi_data


def read_study_events(study_events_path: Path) -> pd.DataFrame:
    """
    Reads the study events from the specified csv file.

    Parameters
    ----------
    study_events_path : Path
        Path to the study events data.

    Returns
    -------
    study_events : pandas.DataFrame
        Study events data.
    """
    study_events = pd.read_csv(study_events_path, index_col=False)
    # study_events = study_events.replace({np.nan: None})
    study_events["phase"] = study_events["phase"].replace({np.nan: "NonePhase"})
    study_events["mov"] = study_events["mov"].replace({np.nan: "NoneMov"})

    # make last study event before cell change NoneMov, NonePhase, event=StudyBreak
    study_events.loc[
        (study_events.cell.shift(-1).diff() != 0) & (study_events.event == "PhaseChange"), "event"
    ] = "StudyBreak"
    study_events.loc[study_events.event == "StudyBreak", "mov"] = "NoneMov"
    study_events.loc[study_events.event == "StudyBreak", "phase"] = "NonePhase"
    study_events.loc[study_events.event == "StudyBreak", "value"] = "Start"

    for col in ["participant", "cell", "cell_name", "run", "mov", "phase", "event"]:
        study_events[col] = study_events[col].astype("category")

    try:
        study_events["time_stamp"] = pd.to_datetime(study_events["time_stamp_s"], unit="s")
    except OutOfBoundsDatetime as e:
        print(study_events)
        print(study_events["time_stamp_s"].min(), study_events["time_stamp_s"].max())
        print(
            pd.to_datetime(study_events["time_stamp_s"].min(), unit="s"),
            pd.to_datetime(study_events["time_stamp_s"].max(), unit="s"),
        )
        print(
            "This probably means, that something in your study events file is broken "
            "and some timestamps are missing their decimal point.",
            file=sys.stderr,
        )
        raise e
    return study_events


def read_questionnaire_data(
    questionnaire_path: Path, return_raw: bool = False, return_cleaned: bool = False
) -> pd.DataFrame:
    """Reads the questionnaire data from the specified csv file.
    If `return_raw` is set to True, the raw data is returned without any processing.
    Otherwise the data is cleaned and transformed into a DataFrame with one row per participant and cell.

    Parameters
    ----------
    questionnaire_path : Path
        Path to the questionnaire data.
    return_raw : bool
        Whether to return the raw data without any processing.
    return_cleaned : bool
        Whether to return the cleaned data (invalid rows excluded) without any processing.

    Returns
    -------
    df_qs : pandas.DataFrame
        Questionnaire data.
    """
    qs = pd.read_csv(questionnaire_path, header=1, encoding="utf-16", sep=";")

    if return_raw:
        return qs

    excluded_rows = [*range(8)] + [10, 32]
    qs = qs.drop(excluded_rows)
    if return_cleaned:
        return qs

    all_cells = ["#1", "#2", "#3", "#4", "#5", "#6"]

    # rename columns
    def rename_column(col: str):
        idx = col.find("#")
        if idx != -1:
            col = col[: idx - 1] + col[idx + 2 :]
        return col

    # Create an empty list to store dataframes
    dfs = []

    for p in qs["Participant ID:: [01]"].unique():
        qs_p = qs[qs["Participant ID:: [01]"] == p]
        for number in all_cells:
            df_p_cell = pd.DataFrame()
            columns_with_number = [col for col in qs_p.columns if number in col]
            df_p_cell = qs_p[columns_with_number]
            df_p_cell.columns = [rename_column(item) for item in df_p_cell.columns]
            df_p_cell.cell = number[1:]
            df_p_cell.insert(loc=0, column="participant", value="VP_0" + p[2:])
            df_p_cell.insert(loc=1, column="cell_id", value=number[1:])
            dfs.append(df_p_cell)
    df_qs = pd.concat(dfs)
    df_qs.reset_index(drop=True, inplace=True)
    return df_qs


def read_vr_questionnaire_data(questionnaire_path: Path) -> pd.DataFrame:
    df = read_questionnaire_data(questionnaire_path, return_cleaned=True)
    df = df[
        [
            "Participant ID:: [01]",
            "VR Experience",
            "FMSS Pre: Rate from no discomfort at all to severe discomfort",
            "FMSS Post: Rate from no discomfort at all to severe discomfort",
            "Letzte Seite, die im Fragebogen bearbeitet wurde",
        ]
    ]
    df.columns = ["participant", "vr_experience", "fmss_pre", "fmss_post", "last_page"]
    if not (df["last_page"] == 8).all():
        warnings.warn("There may be invalid rows in the cleaned questionnaire data: Not all last pages are 8.")
    df["participant"] = df["participant"].str.replace("1_", "VP_0")
    df.reset_index(drop=True, inplace=True)
    return df


def calculate_q_scores(df_qs: pd.DataFrame) -> pd.DataFrame:
    """Calculates the q scores of VEQ p.1 and 2 and Jian et al Trust questionnaire

    Args:
        df_qs (pd.DataFrame): dataframe with questionnaire answers

    Returns:
        pd.DataFrame: dataframe with additional scores
    """
    # veq
    df_qs["veq_p1"] = df_qs.filter(like="Daniel Roth p.1").mean(axis=1)
    df_qs["veq_p2"] = df_qs.filter(like="Daniel Roth p.2").mean(axis=1)

    # jian
    q_jian = df_qs.filter(like="Jian et al 43").copy()
    q_jian.iloc[:, 0:5] = 8 - q_jian.iloc[:, 0:5]

    df_qs["q_trust"] = q_jian.mean(axis=1)

    return df_qs


def get_study_breaks(study_events: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the study breaks from the study events data.

    Parameters
    ----------
    study_events : pandas.DataFrame
        Study events data as loaded from the `read_study_events` function.

    Returns
    -------
    breaks : pandas.DataFrame
        DataFrame containing the start and end timestamps of the study breaks (and participant, cell, run, mov information).
    """
    breaks = study_events[(study_events["event"] == "StudyBreak") | (study_events["event"] == "MovChange")].copy()
    breaks["start"] = breaks["time_stamp"]
    breaks["end"] = breaks["time_stamp"].shift(periods=-1)
    breaks = breaks[breaks["event"] == "StudyBreak"]
    breaks.drop(["time_stamp_s", "event", "value", "time_stamp"], axis=1, inplace=True)
    return breaks.dropna()


def get_phase_change_events(study_events: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the phase change events from the study events data.

    Parameters
    ----------
    study_events : pandas.DataFrame
        Study events data as loaded from the `read_study_events` function.

    Returns
    -------
    phases : pandas.DataFrame
        DataFrame containing the start and end timestamps of the phases (and participant, cell, run, mov information).
    """
    phases = study_events[(study_events["event"] == "PhaseChange") | (study_events["event"] == "MovChange")].copy()

    # kick out duplicate runs (keeping the last) by setting the phase to NonePhase
    # Group by the relevant columns and identify the last occurrence
    phases["is_last"] = (
        phases.groupby(["participant", "cell", "run", "mov", "phase"], observed=True).cumcount(ascending=False) == 0
    )

    # ToDo: Check this again for a participant with double recordings
    # Apply the NoneMov where it's not the last occurrence, preserving 'MovChange' as is
    phases["mov"] = phases.apply(
        lambda row: row["mov"] if row["event"] == "MovChange" or row["is_last"] else "NoneMov", axis=1
    )
    # Apply the NonePhase where it's not the last occurrence, preserving 'NonePhase' and 'MovChange' as is
    phases["phase"] = phases.apply(
        lambda row: row["phase"]
        if row["phase"] == "NonePhase" or row["event"] == "MovChange" or row["is_last"]
        else "NonePhase",
        axis=1,
    )

    # Drop the helper column used to identify last occurrences
    phases.drop(columns=["is_last"], inplace=True)

    # Add start and end columns
    phases["start"] = phases["time_stamp"]
    phases["end"] = phases["time_stamp"].shift(periods=-1)

    # exclude MovChange events 
    # ToDo optional to make dataframe nicer as there is always a remnant MovChange NonePhase from double recordings

    phases.drop(["time_stamp_s", "event", "value", "time_stamp"], axis=1, inplace=True)
    return phases


def get_mov_change_events(study_events: pd.DataFrame, keep_is_last_col: bool = False) -> pd.DataFrame:
    """
    Extracts the move change events from the study events data. 
    Warning: !! Mov includes NonePhase (until press of the next start button)

    Parameters
    ----------
    study_events : pandas.DataFrame
        Study events data as loaded from the `read_study_events` function.

    Returns
    -------
    moves : pandas.DataFrame
        DataFrame containing the start and end timestamps of the phases (and participant, cell, run, mov information).
    """
    moves = study_events[(study_events["event"] == "MovChange") | (study_events["event"] == "StudyBreak") ].copy()
    cell_ends = study_events[((study_events["event"] == "StudyEvent") & (study_events["value"] == "Stop"))].copy()
    # kick out duplicate runs (keeping the last one) by setting the mov to NoneMov
    moves["is_last"] = (
        moves.groupby(["participant", "cell", "run", "mov", "phase"], observed=True).cumcount(ascending=False) == 0
    )
    if not keep_is_last_col:
        moves.loc[~moves["is_last"], "mov"] = "NoneMov"

    # add cell ends
    cell_ends["is_last"] = True
    # concat and sort again by timestamp
    moves = pd.concat((moves, cell_ends), axis=0).sort_values(by="time_stamp_s")


    moves["start"] = moves["time_stamp"]
    moves["end"] = moves["time_stamp"].shift(periods=-1)
    moves = moves[moves["event"] == "MovChange"]

    cols_to_drop = ["time_stamp_s", "event", "value", "time_stamp"]
    if not keep_is_last_col:
        cols_to_drop.append("is_last")
    moves.drop(cols_to_drop, axis=1, inplace=True)
    return moves


def get_runs(study_events: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the runs from the study events data.

    The start time of a run is defined as the start time of the first Mov0 event and the end time of a run is defined as
    the end time of the last Mov2 event.

    Warning: !! Mov includes NonePhase (until press of the next start button)

    Parameters
    ----------
    study_events : pandas.DataFrame
        Study events data as loaded from the `read_study_events` function.

    Returns
    -------
    runs : pandas.DataFrame
        DataFrame containing the start and end timestamps of the runs (and participant, cell information).
    """
    moves = get_mov_change_events(study_events)
    moves.set_index(["participant", "cell", "run"], inplace=True)
    run_starts = moves[moves["mov"] == "Mov0"]["start"]  # automatically ignores previous movements of duplicate runs
    run_ends = moves[moves["mov"] == "Mov2"]["end"]
    runs = pd.concat([run_starts, run_ends], axis=1)
    runs.columns = ["start", "end"]
    return runs.reset_index()


def get_first_file_with_pattern(pattern, base_path, participant_id, subfolder):
    folder = base_path / participant_id / subfolder
    file = list(folder.glob(pattern))[0]
    return file


def concat_data_from_subfolders(base_path, participant_id, participant_subfolders, pattern, load_func):
    files = []
    for subdir in participant_subfolders:
        df = load_func(get_first_file_with_pattern(pattern, base_path, participant_id, subdir))

        # Identify subsequent duplicates (keep only the first occurrence)
        duplicated_indices = df.index.duplicated(keep="first")

        # Filter out the subsequent duplicates
        df = df[~duplicated_indices]

        if len(df) == 0:
            warnings.warn(f"No data found in {subdir} for {participant_id}.")
            continue

        files.append(df)
    return pd.concat(files, axis=0)



def load_concatenated_eye_tracking_data(base_path, participant_id, participant_subfolders, pattern):
    return concat_data_from_subfolders(
        base_path, participant_id, participant_subfolders, pattern, read_eye_tracking_data
    ).sort_index()


def load_concatenated_roi_data(base_path, participant_id, participant_subfolders, pattern):
    return concat_data_from_subfolders(
        base_path, participant_id, participant_subfolders, pattern, read_roi_data
    ).sort_index()


def load_concatenated_camera_data(base_path, participant_id, participant_subfolders, pattern):
    return concat_data_from_subfolders(
        base_path, participant_id, participant_subfolders, pattern, read_main_camera
    ).sort_index()


def load_concatenated_tracker_data(base_path, participant_id, participant_subfolders, pattern):
    return concat_data_from_subfolders(
        base_path, participant_id, participant_subfolders, pattern, read_elbow_tracker
    ).sort_index()


def load_concatenated_study_events(base_path, participant_id, participant_subfolders, pattern):
    return (
        concat_data_from_subfolders(base_path, participant_id, participant_subfolders, pattern, read_study_events)
        .sort_values(by=["time_stamp_s"])
        .reset_index(drop=True)
    )
def load_emg_data(base_path, participant_id, participant_subfolders, pattern):
    return concat_data_from_subfolders(
        base_path, participant_id, [""], pattern, read_emg_raw
    ).sort_index()


if __name__ == "__main__":
    with open("../../config.json") as c:
        conf = json.load(c)
    data_per_particpant_folder = Path(conf["data_per_participant_folder"])
    data_root = data_per_particpant_folder / "VP_000/20240515_114607_liv_1cell"
    eye_tracking_file = list(data_root.glob("*Eye_Tracking.csv"))[0]
    roi_file = list(data_root.glob("*ROI.csv"))[0]

    roi_data = read_roi_data(roi_file)
    print(roi_data.head())

    et_data = read_eye_tracking_data(eye_tracking_file)
    print(et_data.head())

    study_events_file = list(data_root.glob("*StudyEvents.csv"))[0]
    study_events = read_study_events(study_events_file)
    print(study_events.head())

class ExclusionObj():
    """Class to organize filter tool based on timestamps
    """
    def __init__(self, df_filter:pd.DataFrame, col_start:str, col_end:str):
        self.df_filter = df_filter.copy()
        self.col_start = col_start
        self.col_end = col_end

    def filter_timestamps_by_start_and_end(self, df_raw:pd.DataFrame):
        """Filters rows of a given df depending on if the column timestamp is somewhere in the df filter between any start and end

        Args:
            df_raw (pd.DataFrame): df to be filtered
            col_timestamp (str): name of the column with the timestamps
        """

        # Filter dataset based on intervals
        filtered_subsets = []
        for _, row in self.df_filter.iterrows():
            filtered = df_raw[row['start']:row['end']]
            filtered_subsets.append(filtered)
        df_processed = pd.concat(filtered_subsets)

        return df_processed.copy()

    def filter_start_end_by_start_end(self, df_raw:pd.DataFrame):
        pass
