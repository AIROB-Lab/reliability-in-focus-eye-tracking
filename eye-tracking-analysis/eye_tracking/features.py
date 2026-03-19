from typing import List, Literal, Union
import numpy as np
import pandas as pd
from pandas._libs.missing import NAType

from trust_in_prosthesis_analysis.eye_tracking.io_utils import get_phase_change_events


def fixation_percent(fixations: pd.DataFrame, total_time_s: float) -> float:
    """
    Calculate the percentage of time spent in fixations.
    Returns zero if no fixations are present or the total time is zero.

    As described by Wilson et al. 2010 [1].

    [1] M. Wilson, J. McGrath, S. Vine, J. Brewer, D. Defriend, und R. Masters,
    „Psychomotor control in a virtual laparoscopic surgery training environment:
    gaze control parameters differentiate novices from experts“, Surg Endosc,
    Bd. 24, Nr. 10, S. 2458–2464, Okt. 2010, doi: 10.1007/s00464-010-0986-1.


    Parameters
    ----------
    fixations : pandas.DataFrame
        DataFrame containing the start and end indices and timestamps of the fixations.
    total_time_s : float
        Total time of the recording in seconds.

    Returns
    -------
    float
        Percentage of time spent in fixations.
    """
    if len(fixations) == 0:
        return 0.0
    if total_time_s == 0:
        return 0.0

        # Sort intervals by start time
    fixations_sorted = fixations.sort_values("start_time_s").reset_index(drop=True)
    starts = fixations_sorted["start_time_s"].to_numpy()
    ends = fixations_sorted["end_time_s"].to_numpy()

    # Quick check for any overlap: if any start < previous end
    if np.all(starts[1:] >= ends[:-1]):
        # No overlaps, simply sum durations
        total_fixation_time = (ends - starts).sum()
        return (total_fixation_time / total_time_s) * 100

    # Otherwise, merge overlapping intervals
    intervals = list(zip(starts, ends))
    merged = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:  # Overlapping
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))  # Add last interval

    total_fixation_time = sum(end - start for start, end in merged)
    return (total_fixation_time / total_time_s) * 100


def number_of_fixations(fixations: pd.DataFrame) -> int:
    """
    Calculate the number of fixations.

    Parameters
    ----------
    fixations : pandas.DataFrame
        DataFrame containing the start and end indices and timestamps of the fixations.

    Returns
    -------
    int
        Number of fixations.
    """
    if len(fixations) == 0:
        return 0

    return len(fixations.dropna(how="any"))


def target_locking_strategy(
    fixations_targets: pd.DataFrame, fixations_hands: pd.DataFrame, total_time_s: float
) -> Union[NAType, float]:
    """
    Calculate the target locking strategy as described by Parr et al. [1].

    A more positive score indicates more time spent looking at the target than the hand, a negative score indicates
    more time spent looking at the hand than the target and a score of zero indicates equal time spent looking at the
    target and the hand (switching strategy).

    Returns NA if no fixations on the target are present.

    [1] J. V. V. Parr, S. J. Vine, N. R. Harrison, und G. Wood,
    „Examining the Spatiotemporal Disruption to Gaze When Using a Myoelectric Prosthetic Hand“,
    Journal of Motor Behavior, Bd. 50, Nr. 4, S. 416–425, Sep. 2017, doi: 10.1080/00222895.2017.1363703.


    Parameters
    ----------
    fixations_target : pandas.DataFrame
        DataFrame containing the start and end indices and timestamps of the fixations on the target.
    fixations_hand : pandas.DataFrame
        DataFrame containing the start and end indices and timestamps of the fixations on the hand.
    total_time_s : float
        Total time of the recording in seconds.

    Returns
    -------
    float/NA
        Target locking strategy or NA if no fixations on the target are present.
    """
    if len(fixations_targets) == 0:
        return pd.NA
    tls = fixation_percent(fixations_targets, total_time_s) - fixation_percent(fixations_hands, total_time_s)
    return tls


def get_phase_durations(study_events, groupby: List[str] = None, excl_nonephase_and_doubles = False) -> pd.DataFrame:
    """Calculate the duration of each phase.

    Per default, the phases are separated by participant and cell.
    This can be changed by specifying the groupby parameter.

    Parameters
    ----------
    study_events : pandas.DataFrame
        DataFrame containing the study events as loaded from the `read_study_events` function.
    groupby : list of str, optional
        Columns to group the data by. Default is ["participant", "cell", "phase"].
        The last column should always be "phase".

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the duration of each phase grouped by the specified columns.
    """
    phases = get_phase_change_events(study_events)
    phases["duration"] = phases["end"] - phases["start"]

    if excl_nonephase_and_doubles: # bug: the MoveChange none phases of the exclusions are still in, therefore need to be excluded
        phases = phases[phases.phase != "NonePhase"]

    if not groupby:
        groupby = ["participant", "cell", "phase"]
    return phases.groupby(groupby, observed=True)["duration"].sum()


def get_cell_durations(phase_change_events) -> pd.Series:
    """Calculate the duration of each cell per participant. The duration is the sum of all phases except "NonePhase".

    Parameters
    ----------
    phase_change_events : pandas.DataFrame
        DataFrame containing the phase change events as loaded from the `get_phase_change_events` function.

    Returns
    -------
    pandas.Series
        Series containing the duration of each cell per participant.
    """

    phase_change_events["duration"] = phase_change_events["end"] - phase_change_events["start"]
    phase_change_events = phase_change_events[phase_change_events.phase != "NonePhase"]
    return phase_change_events.groupby(["participant", "cell"], observed=True)["duration"].sum()


MODE_MOV_TARGETS = {
    "pickup": {
        "Mov0": "target2",
        "Mov1": "target11_R",
        "Mov2": "target11_L",
    },
    "dropoff": {
        "Mov0": "target11_R",
        "Mov1": "target11_L",
        "Mov2": "target2",
    },
}
MODE_PHASES = {
    "pickup": ["Reach", "Grasp", "Transport"],
    "dropoff": ["Transport", "Release"],
}


def eye_arrival_latency(fixations_per_phase: dict, phase_change_events: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame with the eye arrival latency at pickup and dropoff
    for each participant, cell, run and movement.

    The calculations are done as described in Lavoie et al. [1].

    [1] E. B. Lavoie et al., “Using synchronized eye and motion tracking to determine
    high-precision eye-movement patterns during object-interaction tasks,”
    Journal of Vision, vol. 18, no. 6, p. 18, Jun. 2018, doi: 10.1167/18.6.18.

    Parameters
    ----------
    fixations_per_phase : dict
        Dictionary containing the fixations per phase as loaded from the `read_fixations_per_phase` function.
    phase_change_events : pandas.DataFrame
        DataFrame containing the phase change events as loaded from the `get_phase_change_events` function.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the eye arrival latency at pickup and dropoff for each participant, cell, run and movement.

    """
    # Transport start time - target (pick-up location) fixation start time
    eal_df = _eye_x_latency("eal", fixations_per_phase, phase_change_events)
    eal_df["eal_pickup_s"] = eal_df["transport_start_time_s"] - eal_df["pickup_fixation_start_time_s"]
    eal_df["eal_dropoff_s"] = eal_df["transport_end_time_s"] - eal_df["dropoff_fixation_start_time_s"]
    return eal_df


def eye_leaving_latency(fixations_per_phase: dict, phase_change_events: pd.DataFrame):
    """Returns a DataFrame with the eye leaving latency at pickup and dropoff.

    The calculations are done as described in Lavoie et al. [1].

    [1] E. B. Lavoie et al., “Using synchronized eye and motion tracking to determine
    high-precision eye-movement patterns during object-interaction tasks,”
    Journal of Vision, vol. 18, no. 6, p. 18, Jun. 2018, doi: 10.1167/18.6.18.

    Parameters
    ----------
    fixations_per_phase : dict
        Dictionary containing the fixations per phase as loaded from the `read_fixations_per_phase` function.
    phase_change_events : pandas.DataFrame
        DataFrame containing the phase change events as loaded from the `get_phase_change_events` function.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the eye leaving latency at pickup and dropoff for each participant, cell, run and movement.
    """
    ell_df = _eye_x_latency("ell", fixations_per_phase, phase_change_events)
    ell_df["ell_pickup_s"] = ell_df["transport_start_time_s"] - ell_df["pickup_fixation_end_time_s"]
    ell_df["ell_dropoff_s"] = ell_df["transport_end_time_s"] - ell_df["dropoff_fixation_end_time_s"]
    return ell_df


def _eye_x_latency(
    feature_name: Literal["eal", "ell"], fixations_per_phase: dict, phase_change_events: pd.DataFrame
) -> pd.DataFrame:
    transport_times = (
        phase_change_events[phase_change_events.phase == "Transport"]
        .copy()
        .set_index(["participant", "cell", "run", "mov"])
    )
    exl_list = []
    for mode in MODE_MOV_TARGETS.keys():
        exl_list.append(_exl_for_mode(feature_name, mode, fixations_per_phase))

    # build DataFrame from exl list and transport times
    exl_df = pd.concat([*exl_list, transport_times[["start", "end"]].map(pd.Timestamp.timestamp)], axis=1)
    exl_df = exl_df.rename(columns={"start": "transport_start_time_s", "end": "transport_end_time_s"})
    return exl_df


# Currently EAL is defined as the start of the very first fixation (and not the start of the last fixation before pickup/dropoff) 
# Currently EEL is defined as the end of the very last fixation (and not the end of the first fixation after pickup/dropoff) 
def _exl_for_mode(
    feature_name: Literal["eal", "ell"], mode: Literal["pickup", "dropoff"], fixations_per_phase: dict
) -> pd.DataFrame:
    """Returns a DataFrame with the eye arrival latency at pickup and dropoff"""
    if feature_name == "eal":
        col_to_drop = "end_time_s"
        col_to_rename = "start"
    elif feature_name == "ell":
        col_to_drop = "start_time_s"
        col_to_rename = "end"
    else:
        raise ValueError(f"Feature {feature_name} is not supported, choose one of 'eal' or 'ell' instead.")

    # build Dataframe from fixations dict to make it usable and only include the desired phases
    relevant_fixations = []
    for p in MODE_PHASES[mode]:
        relevant_fixations.append(pd.concat(fixations_per_phase[p]))
    relevant_fixations = pd.concat(relevant_fixations).sort_values("start_time_s")

    exl_for_mode = []
    for wanted_mov, targ in MODE_MOV_TARGETS[mode].items():
        # choose the fixations on the corresponding target for the mode and movement
        if ((feature_name == "eal" and mode == "pickup") or (feature_name == "ell" and mode == "dropoff")):
            targ = ["object", targ]
        elif (feature_name == "ell" and mode == "pickup"):
            targ = ["object"]
        relevant_target_fixations = relevant_fixations.loc[targ].sort_values("start_time_s")
        relevant_target_fixations = relevant_target_fixations[relevant_target_fixations["mov"] == wanted_mov]
        # print(f"[{mode}] relevant_target_fixations for mov {wanted_mov}:\n", relevant_target_fixations)

        if feature_name == "eal":
            # take the first found fixation on the given target
            first_fixations = relevant_target_fixations.groupby(["participant", "cell", "run", "mov"]).first()
        else:  
            # feature is ell
            first_fixations = relevant_target_fixations.groupby(["participant", "cell", "run", "mov"]).last()
        exl_for_mode.append(first_fixations.drop(col_to_drop, axis=1))

    exl_for_mode = pd.concat(exl_for_mode)
    exl_for_mode = exl_for_mode.rename(columns={f"{col_to_rename}_time_s": f"{mode}_fixation_{col_to_rename}_time_s"})
    return exl_for_mode


def get_exl_times_per_participant(fixs_per_phase, phase_change_events):
    eal_df = eye_arrival_latency(fixs_per_phase, phase_change_events)
    ell_df = eye_leaving_latency(fixs_per_phase, phase_change_events)
    ell_df.drop(["transport_start_time_s", "transport_end_time_s"], axis=1, inplace=True)
    return pd.concat([eal_df, ell_df], axis=1)


# ##### Further ideas/plans ##########################
# percent change in Pupil size (PCPS) needs a baseline pupil diameter
# --> hard to implement and needs to be corrected for brightness
# see Zhang et al., 2016: https://ieeexplore.ieee.org/abstract/document/7844587

# Number of pupil size increases per second (maybe not this one)
# probably should also be corrected for brightness
# see White et al., 2017: https://ieeexplore.ieee.org/abstract/document/8071003 :
# > The first step was to eliminate all data points in which the eye tracking system lost
# track of a participant's eyes due to body movements, head rotation, or blinking in order
# to identify the valid data points for analysis. The percentage of valid data points in
# comparison to the number of points for the entire trial was determined. If participant
# testing yielded less than 80% of valid data points across both eyes for more than five trials,
# then the participant's data were excluded from the analysis. This level of invalid data
# points did not allow for a statistical reliable assessment of participant cognitive workload
# for the experiment. In some cases, participants did not follow instructions to maintain
# a constant body position, which caused the eye tracking system to lose track of their eyes.
# Pilot testing and an initial analysis of experiment data for the first few participants revealed
# that there were more valid data points for the right eye than the left, which was likely due to
# the location of the cameras, as part of the apparatus setup, relative to the participant's eyes.
# As a result, the NPI response was determined for the right eye across all participants.
# Eye tracking data was assessed to determine the number of times the pupil diameter increased over
# the course of the trial. Instances in which a participant blinked also needed to be filtered from
# the data analysis along with the six points immediately before and after the blink in order to ensure
# identification of pupil size changes due to workload versus pupil adaption to lighting.
# The MATLAB code counted any instance in which the pupil increased in size throughout an entire trial.
# Based on the length of the remaining filtered data file (purged of invalid data points or instances
# of blinking), the NPI was determined. >

# From Wilson et al., 2010:  https://doi.org/10.1007/s00464-010-0986-1:
# Fixation rate: number of fixations per second (per trial)
