import json
import pickle
import warnings
from pathlib import Path
from typing import Optional, Sequence
from functools import cached_property, lru_cache
from unittest.mock import inplace

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tpcp import Dataset
from tpcp.caching import hybrid_cache

from trust_in_prosthesis_analysis.dataset.utils import get_subject_dirs, add_performance_exclusion
from trust_in_prosthesis_analysis.eye_tracking.data_preprocessing import (
    clean_roi_data,
    integrate_event_data_into_df,
    get_fixations_from_roi_data,
    clean_eye_tracking_data,
    get_fixations_per_phase,
)
from trust_in_prosthesis_analysis.eye_tracking.features import get_exl_times_per_participant
from trust_in_prosthesis_analysis.eye_tracking.io_utils import (
    ExclusionObj,
    get_phase_change_events,
    get_mov_change_events,
    load_concatenated_camera_data,
    load_concatenated_eye_tracking_data,
    load_concatenated_roi_data,
    load_concatenated_study_events,
    load_concatenated_tracker_data,
    load_emg_data,
    read_questionnaire_data,
    calculate_q_scores,
    read_vr_questionnaire_data,
)

cached_load_questionnaire_data = lru_cache(maxsize=1)(read_questionnaire_data)


@lru_cache
def load_config():
    project_path = Path(__file__).parent.parent.parent
    with open(project_path / "config.json") as f:
        conf = json.load(f)
    return conf

MEMORY = joblib.Memory(location=load_config()["cache_folder"])


class TrustDataset(Dataset):
    use_cache: bool
    memory: joblib.Memory
    _participant_subfolders: Optional[dict] = None

    # constants
    CELL_NAME_MALFUNCTION_DELAY_MAPPING = {
        "A": {"malfunction": "None", "delay": "None"},
        "B": {"malfunction": "medium", "delay": "None"},
        "C": {"malfunction": "high", "delay": "None"},
        "D": {"malfunction": "None", "delay": "300ms"},
        "E": {"malfunction": "medium", "delay": "300ms"},
        "F": {"malfunction": "high", "delay": "300ms"},
    }

    # # Load the exclusion object from the pickle file
    # with open('outputs/exclObj', 'rb') as file:
    #     EXCL_OBJ = pickle.load(file)

    def __init__(
        self,
        base_path: Optional[Path] = None,
        use_cache: bool = True,
        memory: joblib.Memory = MEMORY,
        *,
        groupby_cols: Optional[Sequence[str]] = None,
        subset_index: Optional[Sequence[str]] = None,
    ):
        """Dataset class for the trust in prosthesis data.
        It uses the data provided in the data_per_participant_folder which can be specified either via the base_path
        parameter or the config.json file (see README.md).

        The properties/methods that can be accessed per cell and will also only give you
        the data for the current cell are:
        - study_events
        - roi_data_w_phases
        - eye_tracking_w_phases
        - fixations_per_phase
        - phase_change_events
        - mov_change_events
        - get_exl_times
        - cell_id
        - cell_name

        The following properties will always give you the data for the whole participant:
        - raw_eye_tracking_data
        - eye_tracking_data
        - raw_roi_data
        - roi_data
        - fixations


        Example:
        --------
        >>> ds = TrustDataset()
        >>> ds
        TrustDataset [4 groups/rows]
        <BLANKLINE>
         participant  cell_id cell_name
        0      VP_000        3         F
        1      VP_000        4         C
        2      VP_000        6         D
        3      VP_001        2         B

        Create subsets:

        >>> ds.get_subset(participant=["VP_000"])
        TrustDataset [3 groups/rows]
        <BLANKLINE>
         participant  cell_id cell_name
        0      VP_000        3         F
        1      VP_000        4         C
        2      VP_000        6         D

        >>> ds.get_subset(participant=["VP_000"], cell_id=[3])
        TrustDataset [1 groups/rows]
        <BLANKLINE>
         participant  cell_id cell_name
        0      VP_000        3         F

        Loop over the dataset by participant:

        >>> for d in ds.groupby(["participant"]):
        ...     print(d.participant_id)
        VP_000
        VP_001

        Access the data per participant:

        >>> d = ds.get_subset(participant=["VP_000"])
        >>> d.study_events
        >>> d.roi_data
        >>> d.eye_tracking_data
        >>> d.fixations

        Parameters
        ----------
        base_path : Path, optional
            path to the data_per_participant_folder, if not given it will be loaded from the config.json file
        use_cache : bool, optional
            if True the data will be cached with a hybrid cache, if false, only some data will be lru cached,
            by default True
        memory : joblib.Memory, optional
            memory object for caching, by default joblib.Memory(".cache")
        groupby_cols : list of str, optional
            columns to groupby, by default None
        subset_index : list of str, optional
            columns to use as index for the subset, by default None

        Returns
        -------
        TrustDataset
            dataset object

        """
        self.base_path = base_path
        self.use_cache = use_cache
        self.memory = memory

        super().__init__(
            groupby_cols=groupby_cols,
            subset_index=subset_index,
        )

    def create_index(self):
        # subject_ids = [
        #     subject_dir.name for subject_dir in get_subject_dirs(self.base_path, "VP_*")
        # ]
        subject_cells = [
            (f"VP_{p['id']:03}", c["cell_id"], c["cell_name"])
            for p in self.study_json["participants"]
            for c in p["cells"]
        ]

        index = [tup for tup in subject_cells if tup[0] in self.all_participant_subfolders.keys()]

        index_cols = ["participant", "cell_id", "cell_name"]
        index = pd.DataFrame(index, columns=index_cols)

        return index

    def get_base_path(self) -> Path:
        """Path to the data_per_participant_folder."""
        if self.base_path:
            return self.base_path
        return Path(load_config()["data_per_participant_folder"])

    @property
    def all_participant_subfolders(self) -> dict:
        """Returns a dict with the participant ids as keys and the subfolders as values.
        Used to create the dataset index."""
        participant_subfolders = {}
        for subject_dir in get_subject_dirs(self.get_base_path(), "VP_*"):
            participant_subfolders[subject_dir.name] = [subdir.name for subdir in subject_dir.glob("2024*/")]
        return participant_subfolders

    @property
    def study_json(self) -> dict:
        """Returns the study.json file as a dict."""
        path = self.get_base_path() / "study.json"
        with open(path) as f:
            return json.load(f)

    @property
    def participant_cell_name_mapping(self) -> dict:
        """Returns a dict with the participant id ("VP_XXX") and cell id ("Y") as key and the cell name as value."""
        return {
            (f"VP_{p['id']:03}", str(c["cell_id"])): c["cell_name"]
            for p in self.study_json["participants"]
            for c in p["cells"]
        }

    def get_cell_name(self, participant_id, cell_id) -> Optional[str]:
        """Returns the cell name for a participant and cell_id."""
        return self.participant_cell_name_mapping.get((f"VP_{participant_id:03}", str(cell_id)), None)

    def add_cell_name_col(self, df) -> pd.DataFrame:
        """Adds the cell name to a DataFrame df with columns 'participant' and 'cell_id'."""
        if "participant" not in df.columns or "cell_id" not in df.columns:
            raise ValueError("DataFrame has to have columns 'participant' and 'cell_id'")
        df["cell_name"] = df.apply(
            lambda row: self.participant_cell_name_mapping.get((f"VP_{row['participant']:03}", str(row["cell_id"])), None),
            axis=1,
        )
        return df

    def add_malfunction_and_delay_cols(self, df) -> pd.DataFrame:
        """Adds the malfunction and delay columns to a DataFrame df with column 'cell_name'.
        If the column 'cell_name' is not present, it will be added from columns 'participant' and 'cell_id'."""
        if "cell_name" not in df.columns:
            df = self.add_cell_name_col(df)
        df["malfunction"] = df["cell_name"].map(lambda x: self.CELL_NAME_MALFUNCTION_DELAY_MAPPING[x]["malfunction"])
        df["delay"] = df["cell_name"].map(lambda x: self.CELL_NAME_MALFUNCTION_DELAY_MAPPING[x]["delay"])
        return df

    def get_pastabox_results(self) -> pd.DataFrame:
        """Returns an overview over the pastabox test results for the current participant.

        This overview includes the following columns:
        - participant
        - cell
        - run
        - mov
        - cell_name
        - start (= timestamp of the start of the movement)
        - end (= timestamp of the end of the movement)
        - excl_bc_repetition (= whether the movement was excluded because the run was repeated)
        - failure_started (= whether a planned failure was started)
        - semi_success (= whether the movement was a semi success)
        - reason_for_classification_fail (= reason for the classification as failure)
        - time_of_event (= timestamp of the failure or success)
        - planned_failure_perc (= the percentage of the spatial movement where the failure was planned)
        - failure_planned (= whether a failure was planned in this movement)

        Returns
        -------
        pd.DataFrame
            DataFrame with the columns specified above.
        """
        if not self.is_single("participant"):
            raise ValueError("Can only be accessed for single participant")
        fns = self.get_failures_and_successes(keep_is_last_col=True).copy()
        fns.set_index(["participant", "cell", "run", "mov", "is_last"], inplace=True)
        mc = self.get_raw_mov_change_events()
        mc.set_index(["participant", "cell", "run", "mov", "is_last"], inplace=True)
        df = pd.concat([mc, fns[["failure_started", "semi_success", "value", "time_stamp"]]], axis=1)

        # change is_last to exclude because of repetition (=opposite)
        df["excl_bc_repetition"] = ~df.index.get_level_values("is_last")
        df.rename(
            columns={
                "value": "reason_for_classification_fail",
                "time_stamp": "time_of_event",
            },
            inplace=True,
        )
        df.index = df.index.droplevel("is_last")

        # add info on planned failures
        planned_fails = self.get_planned_fails_from_study_json()
        df["planned_failure_perc"] = planned_fails.set_index(["participant", "cell", "run", "mov"])["perc"]
        df["failure_planned"] = df["planned_failure_perc"] > 0

        df.drop(["phase"], inplace=True, axis=1)

        # add performance exclusion column
        # This is done later in a Exclusion notebook => this way its more comprehensible for the paper writing
        # df = add_performance_exclusion(df, performance_thr=perf_thr)
        return df

    def get_planned_fails_from_study_json(self):
        if not self.is_single("participant"):
            raise ValueError("Can only be accessed for single participant")
        planned_fails = []
        for info_per_cell in self.study_json["participants"][self.participant_number - 1]["cells"]:
            list_pc = info_per_cell["failrun"]
            list_pc = [dict(fr, cell=info_per_cell["cell_id"], participant=self.participant_number) for fr in list_pc]
            planned_fails.extend(list_pc)
        df = pd.DataFrame(planned_fails)[["participant", "cell", "run", "mov", "perc"]]
        # change mov col from number (e.g. 1) to string (e.g. "Mov1")
        df["mov"] = "Mov" + df["mov"].astype(str)
        return df

    @property
    def all_questionnaire_data(self) -> pd.DataFrame:
        """Returns a DataFrame with the VEQ and Q scores (+ respective questions) per participant and cell."""
        path = list(self.get_base_path().glob("data_*.csv"))[0]
        if self.use_cache:
            qs = cached_load_questionnaire_data(path)
        else:
            qs = read_questionnaire_data(path)

        qs = calculate_q_scores(qs.copy())
        qs["cell_name"] = qs.apply(
            lambda x: self.participant_cell_name_mapping.get((x["participant"], x["cell_id"]), None), axis=1
        )
        return qs

    @property
    def all_vr_question_data(self) -> pd.DataFrame:
        """Returns a DataFrame with the VR experience question data and the FMSS Pre and Post question data
        per participant."""
        path = list(self.get_base_path().glob("data_*.csv"))[0]
        qs = read_vr_questionnaire_data(path)
        return qs

    @property
    def participant_id(self) -> str:
        """Returns the participant id, e.g. "VP_000"."""
        if not self.is_single("participant"):
            raise ValueError
        return self.index["participant"][0]

    @property
    def participant_number(self) -> int:
        """Returns the participant number, e.g. 0."""
        return int(self.participant_id.split("_")[1])

    @property
    def cell_name(self) -> str:
        """Returns the cell name, e.g. "F"."""
        if not self.is_single("cell_name"):
            raise ValueError
        return self.index["cell_name"][0]

    @property
    def cell_id(self) -> int:
        """Returns the cell id, e.g. 3."""
        if not self.is_single("cell_id"):
            raise ValueError
        return self.index["cell_id"][0]

    @property
    def raw_eye_tracking_data(self) -> pd.DataFrame:
        """Returns the raw eye tracking data FOR THE WHOLE PARTICIPANT as a DataFrame."""
        if not self.is_single(["participant"]):
            raise ValueError("Can only be accessed for single participant")
        pattern = "*Eye_Tracking.csv"
        if self.use_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = hybrid_cache(self.memory, 4)(load_concatenated_eye_tracking_data)(
                    self.get_base_path(),
                    self.participant_id,
                    self.all_participant_subfolders[self.participant_id],
                    pattern,
                )
        else:
            data = load_concatenated_eye_tracking_data(
                self.get_base_path(), self.participant_id, self.all_participant_subfolders[self.participant_id], pattern
            )
        return data

    @cached_property
    # @property
    def eye_tracking_data(self) -> pd.DataFrame:
        """Returns the cleaned eye tracking data FOR THE WHOLE PARTICIPANT.
        Cleaned means it is cut to start and end according to the study events
        and the invalid gaze data is set to False."""
        return clean_eye_tracking_data(self.raw_eye_tracking_data, self.study_events)

    @cached_property
    def eye_tracking_w_phases(self) -> pd.DataFrame:
        """Returns the cleaned eye tracking data with the participant number, cell_id, mov, and phases
        integrated into the DataFrame.
        If the dataset is grouped by cell_id, only the data for the current cell is returned.
        """
        data = integrate_event_data_into_df(self.eye_tracking_data, self.study_events)
        if self.is_single(["cell_id"]):
            data = data[data["cell"] == self.cell_id]
        return data

    @property
    def camera_data(self) -> pd.DataFrame:
        """Returns camera data"""
        if not self.is_single(["participant"]):
            raise ValueError("Can only be accessed for single participant")
        pattern = "*Main Camera.csv"
        if self.use_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = hybrid_cache(self.memory, 4)(load_concatenated_camera_data)(
                    self.get_base_path(),
                    self.participant_id,
                    self.all_participant_subfolders[self.participant_id],
                    pattern,
                )
        else:
            data = load_concatenated_camera_data(
                self.get_base_path(), self.participant_id, self.all_participant_subfolders[self.participant_id], pattern
            )
        return data

    @cached_property
    def tracker_data(self) -> pd.DataFrame:
        """Returns camera data"""
        if not self.is_single(["participant"]):
            raise ValueError("Can only be accessed for single participant")
        pattern = "*RightElbowTracker.csv"
        if self.use_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = hybrid_cache(self.memory, 4)(load_concatenated_tracker_data)(
                    self.get_base_path(),
                    self.participant_id,
                    self.all_participant_subfolders[self.participant_id],
                    pattern,
                )
        else:
            data = load_concatenated_tracker_data(
                self.get_base_path(), self.participant_id, self.all_participant_subfolders[self.participant_id], pattern
            )
        return data
    
    @cached_property
    def emg_data(self) -> pd.DataFrame:
        """Returns EMG Data for the WHOLE participant"""
        if not self.is_single(["participant"]):
            raise ValueError("Can only be accessed for single participant")
        pattern = "*joiner.txt"
        if self.use_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = hybrid_cache(self.memory, 4)(load_emg_data)(
                    self.get_base_path(),
                    self.participant_id,
                    self.all_participant_subfolders[self.participant_id],
                    pattern,
                )
        else:
            data = load_emg_data(
                self.get_base_path(), self.participant_id, self.all_participant_subfolders[self.participant_id], pattern
            )
        return data

    @hybrid_cache(MEMORY, lru_cache_maxsize=4)    
    def emg_data_w_phases(self) -> pd.DataFrame:
        """Returns the cleaned emg data with the participant number, cell_id, mov, and phases
        integrated into the DataFrame.
        If the dataset is grouped by cell_id, only the data for the current cell is returned. 
        Of course not 100% accurate taking into consideration the delay between emg and phase labelling, 
        but neglectable in the scope of interaction
        """
        data = integrate_event_data_into_df(self.emg_data, self.study_events)
        if self.is_single(["cell_id"]):
            data = data[data["cell"] == self.cell_id]
            
        # filter data to samples where there is a participant, cell, round, mov
        data = data.dropna(subset=["participant", "cell", "run", "mov", "phase"])
         
        return data
    
    @property
    def raw_roi_data(self) -> pd.DataFrame:
        """Returns the raw roi data FOR THE WHOLE PARTICIPANT as a DataFrame."""
        if not self.is_single(["participant"]):
            raise ValueError("Can only be accessed for single participant")
        pattern = "*ROI.csv"
        if self.use_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = hybrid_cache(self.memory, 4)(load_concatenated_roi_data)(
                    self.get_base_path(),
                    self.participant_id,
                    self.all_participant_subfolders[self.participant_id],
                    pattern,
                )
        else:
            data = load_concatenated_roi_data(
                self.get_base_path(), self.participant_id, self.all_participant_subfolders[self.participant_id], pattern
            )
        return data

    @cached_property
    def roi_data(self) -> pd.DataFrame:
        """Returns the cleaned roi data FOR THE WHOLE PARTICIPANT.

        The data is cut to start and end and the invalid gaze data is set to False

        """
        gaze_validity_cols = self.eye_tracking_data.filter(like="gaze_direction_validity")
        return clean_roi_data(self.raw_roi_data, gaze_validity_cols, self.study_events)

    @cached_property
    # @property
    def roi_w_phases(self) -> pd.DataFrame:
        """Returns the cleaned roi data with the participant number, cell_id, mov, and phases
        integrated into the DataFrame.
        If the dataset is grouped by cell_id, only the data for the current cell is returned.
        """
        data = integrate_event_data_into_df(self.roi_data, self.study_events)
        if self.is_single(["cell_id"]):
            data = data[data["cell"] == self.cell_id]
        return data

    @cached_property
    def fixations(self) -> dict:
        """Returns the fixations on "hand", "object", "target11_L", "target11_r" and "target2" as a dict."""
        return get_fixations_from_roi_data(
            self.roi_data, relevant_cols=["hand", "object", "target11_L", "target11_R", "target2"]
        )

    @cached_property
    def fixations_per_phase(self) -> dict:
        """Returns the fixations on "hand", "object", "target11_L", "target11_r" and "target2" per phase as a dict.
        If the dataset is grouped by cell_id, only the data for the current cell is returned.
        """
        fixs = get_fixations_per_phase(
            self.roi_w_phases,
            get_phase_change_events(self.study_events),
            ["hand", "object", "target11_L", "target11_R", "target2"],
        )
        return fixs

    @property
    def study_events(self) -> pd.DataFrame:
        if not self.is_single(["participant"]):
            raise ValueError("Can only be accessed for single participant")
        pattern = "*StudyEvents.csv"
        if self.use_cache:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = hybrid_cache(self.memory, 4)(load_concatenated_study_events)(
                    self.get_base_path(),
                    self.participant_id,
                    self.all_participant_subfolders[self.participant_id],
                    pattern,
                )
        else:
            data = load_concatenated_study_events(
                self.get_base_path(), self.participant_id, self.all_participant_subfolders[self.participant_id], pattern
            )
        # check if participant number is correct (only use data which is not -1 for check)
        data_cln = data.loc[data.participant != -1].reset_index()
        if (
            len(list(data_cln.groupby(["participant"], observed=True))) != 1
            or data_cln.loc[0, "participant"] != self.participant_number
        ):
            warnings.warn(
                "Participant number in study events does not match the participant number in the dataset."
                "Check that all files are in the correct folders!"
            )

        if self.is_single(["cell_id"]):
            data = data[data["cell"] == self.cell_id]
        
        #apply timebased exclusion directly to event data so that all data so it is applied to all data later on
        # if self.EXCL_OBJ != None:
        #     data = self.EXCL_OBJ.filter_timestamps_by_start_and_end(data.set_index("time_stamp"))

        return data

    def get_failures_and_successes(self, keep_is_last_col=False) -> pd.DataFrame:
        """Returns the Failure and Success events as a DataFrame with an extra column `failure_started` that indicates
        whether a intended failure was induced or not."""
        if not self.is_single(["participant"]):
            raise ValueError("Can only be accessed for single participant")

        data = self.study_events
        successes = data[data["value"].str.contains("SUCCESS", case=True, na=False)]
        fails = data[data["value"].str.contains("FAIL", case=False, na=False)]
        all_results = pd.concat((successes, fails))
        all_results.sort_values("time_stamp_s", inplace=True)

        # add failure start as a column:
        failure_combinations = all_results[all_results["value"] == "Failure Start"][
            ["participant", "cell", "run", "mov"]
        ].drop_duplicates()
        all_results["failure_started"] = (
            all_results[["participant", "cell", "run", "mov"]]
            .apply(tuple, axis=1)
            .isin(failure_combinations.apply(tuple, axis=1))
        )
        # drop failure start entries
        all_results = all_results.loc[(all_results.value != "Failure Start") & (all_results.value != "Failure Ended")]

        all_results["semi_success"] = all_results["value"].isin(
            [
                "SUCCESS;reason: NaN;ori:up",
                "SUCCESS;reason: NaN;ori:N/A",
                "SUCCESS;reason: NaN;ori:upsideDown",
                "FAIL;reason:Object-Boundary-Touch;ori:up",
                "FAIL;reason:Object-Boundary-Touch;ori:upsideDown",
                "FAIL;reason:Object-Boundary-Touch;ori:N/A ",
            ]
        )

        # kick out duplicate runs (keeping the last one) by setting the mov to NoneMov
        all_results["is_last"] = (
            all_results.groupby(["participant", "cell", "run", "mov"], observed=True).cumcount(ascending=False) == 0
        )
        if not keep_is_last_col:
            all_results = all_results[all_results["is_last"]]
            all_results.drop("is_last", axis=1, inplace=True)

        return all_results

    @cached_property
    def phase_change_events(self) -> pd.DataFrame:
        """Returns the start and end of the phase change events as a DataFrame.
        If the dataset is grouped by cell_id, only the data for the current cell is returned."""
        return get_phase_change_events(self.study_events)

    @cached_property
    def mov_change_events(self) -> pd.DataFrame:
        """Returns the start and end of the movement change events as a DataFrame.
        If the dataset is grouped by cell_id, only the data for the current cell is returned.
        """
        return get_mov_change_events(self.study_events)

    def get_raw_mov_change_events(self) -> pd.DataFrame:
        """Returns the start and end of all movement change events (including invalid ones) as a DataFrame.
        If the dataset is grouped by cell_id, only the data for the current cell is returned.
        """
        return get_mov_change_events(self.study_events, keep_is_last_col=True)

    def get_exl_times(self, include_extra_info=False) -> pd.DataFrame:
        """Returns the eye arrival and leaving times at pickup and dropoff.
        If the dataset is grouped by cell_id, only the data for the current cell is returned.

        Parameters
        ----------
        include_extra_info : bool, optional
            if False, only the eye arrival and leaving times at pickup and dropoff are returned,
            else also the transport start and end times and other information is included, by default False

        Returns
        -------
        pd.DataFrame
            DataFrame with the eye arrival and leaving times at pickup and dropoff
        """
        if not self.is_single(["participant"]):
            raise ValueError("Can only be accessed for single participant")
        features = pd.DataFrame()

        features = get_exl_times_per_participant(self.fixations_per_phase, self.phase_change_events)

        if not include_extra_info:
            features = features[["eal_pickup_s", "eal_dropoff_s", "ell_pickup_s", "ell_dropoff_s"]]
        return features

    def plot_phases_and_movs(self, alpha=0.1, axs=None, remove_ticks=True):
        """Plots the phases and movements as colored areas (into axs if given).

        Parameters
        ----------
        alpha : float, optional
            transparency of the colored areas, by default 0.1
        axs : plt.Axes, optional
            axes to plot into, by default a new plot is created
        remove_ticks : bool, optional
            if True, the y-axis ticks are removed, by default True
        """
        phases = self.phase_change_events
        moves = self.mov_change_events

        color_map = {"Grasp": "y", "Reach": "r", "Release": "g", "Transport": "b", "NonePhase": "w"}

        if axs is None:
            fig, axs = plt.subplots(nrows=1, figsize=(10, 5))

        for phase_name, phase in phases.groupby("phase", observed=False):
            i = 0  # just for the label, so only one appears in the legend
            c = color_map[phase_name]
            for start, end in phase[["start", "end"]].to_numpy():
                axs.axvline(start, c="k", alpha=alpha / 2)
                axs.axvline(end, c="k", alpha=alpha / 2)
                axs.axvspan(start, end, alpha=alpha, color=c, label="_" * i + phase_name)
                i += 1

        hatch_map = {"Mov0": "o", "Mov1": "//", "Mov2": "-", "NoneMov": "."}
        for move_name, move in moves.groupby("mov", observed=False):
            i = 0
            h = hatch_map[move_name]
            for start, end in move[["start", "end"]].to_numpy():
                axs.fill(
                    [start, end, end, start],
                    [-100, -100, 200, 200],
                    closed=True,
                    fill=False,
                    hatch=h,
                    label="_" * i + move_name,
                    alpha=alpha,
                )
                i += 1

        axs.set_ylim([-0.2, 1.2])

        if remove_ticks:
            axs.tick_params(
                axis="y",  # changes apply to the y-axis
                which="both",  # both major and minor ticks are affected
                left=False,  # ticks along the bottom edge are off
                right=False,  # ticks along the top edge are off
                labelleft=False,
            )  # labels along the bottom edge are off

        axs.legend()

    def plot_fixations(
        self,
        axs=None,
        palette=sns.color_palette(),
        plot_start_end_lines=False,
        show_not_fixated=False,
        include_hand=False,
    ):
        """Plots the fixations on the different objects and if wanted the hand.
        The fixations are plotted as dots with different colors for the different objects.

        Parameters
        ----------
        axs : plt.Axes, optional
            axes to plot into, by default a new plot is created
        palette : list of colors, optional
            colors to use for the different objects, by default sns.color_palette()
        plot_start_end_lines : bool, optional
            if True, vertical lines are plotted at the start and end of each fixation, by default False
        show_not_fixated : bool, optional
            if True, the y-axis is set to also include the not fixated state, by default False
        include_hand : bool, optional
            if True, the hand fixations are also plotted, by default False
        """
        cmap_hand_obj = {
            "target11_L": [palette[4], palette[6]],
            "target11_R": [palette[2], palette[-1]],
            "target2": [palette[5], palette[7]],
            "object": [palette[1], palette[3]],
        }
        if include_hand:
            cmap_hand_obj["hand"] = [palette[0], palette[8]]
        if axs is None:
            fig, axs = plt.subplots(nrows=1, figsize=(10, 5))

        axs.set_yticks([-1.5, 0.55], ["not fixated", "fixated"])

        if plot_start_end_lines:
            # plot fixs start and ends
            for item, fixs in self.fixations.items():
                if item not in cmap_hand_obj.keys():
                    continue
                for i, (start, end) in enumerate(fixs[["start_time_s", "end_time_s"]].to_numpy()):
                    axs.axvline(
                        pd.Timestamp(start, unit="s"), c=cmap_hand_obj[item][0], label=i * "_" + f"start {item}"
                    )
                    axs.axvline(pd.Timestamp(end, unit="s"), c=cmap_hand_obj[item][1], label=i * "_" + f"end {item}")

        # plot actual fixations
        for i, target in enumerate(cmap_hand_obj.keys()):
            (self.roi_w_phases[target].astype(int) * 2 + 0.05 * i - 1.5).plot(
                style="o", color=cmap_hand_obj[target][0], ax=axs
            )

        if show_not_fixated:
            axs.set_ylim([-1.7, 0.9])
        else:
            axs.set_ylim([0.3, 0.9])

        axs.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    

if __name__ == "__main__":
    ds = TrustDataset()
    print(ds)

    d = ds[0]
    print("Study Events DF:")
    print(d.study_events)

    fig, axs = plt.subplots(nrows=1, figsize=(10, 5))
    d.plot_phases_and_movs(axs=axs, remove_ticks=False)
    d.plot_fixations(axs=axs, plot_start_end_lines=False, show_not_fixated=False, include_hand=True)
    plt.tight_layout()
    plt.show()
