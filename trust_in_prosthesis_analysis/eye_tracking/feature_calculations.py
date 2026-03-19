import pickle
import sys
from collections import defaultdict
import joblib
import pandas as pd
import json
from tpcp.caching import hybrid_cache

from trust_in_prosthesis_analysis.dataset.trust_dataset import TrustDataset, load_config
from trust_in_prosthesis_analysis.eye_tracking import features
from trust_in_prosthesis_analysis.eye_tracking.data_preprocessing import (
    get_fixations_per_phase,
    get_fixations_from_roi_data,
)
from trust_in_prosthesis_analysis.eye_tracking.features import (
    fixation_percent,
    number_of_fixations,
    target_locking_strategy,
    get_phase_durations,
    get_cell_durations,
)
import os

# tbd
from trust_in_prosthesis_analysis.eye_tracking.io_utils import ExclusionObj

MEMORY = joblib.Memory(location=load_config()["cache_folder"])

class FeatureCalculation:
    memory: joblib.Memory
    
    def __init__(self, ds=None):
        if ds != None:
            self.ds = ds
        else:
            self.ds = TrustDataset()
    
    def _calculate_fixation_features_per_phase(self, fixations_in_phase, duration_of_phase_s, phase, mov):
        feature_dict = {}
        for target, fixs_per_target in fixations_in_phase.items():
            feature_dict[f"fixation_percent_{target}"] = fixation_percent(fixs_per_target, duration_of_phase_s)
            feature_dict[f"number_of_fixations_{target}"] = number_of_fixations(fixs_per_target)

        # tls has only one value for the whole phase, but depends on the current phase
        if(phase == "Reach" or phase == "Grasp"):
            target = features.MODE_MOV_TARGETS["pickup"][mov]
            feature_dict["target_locking_strategy"] = target_locking_strategy(fixations_targets=pd.concat([fixations_in_phase["object"], fixations_in_phase[target]], axis=0), 
                                                                              fixations_hands=fixations_in_phase["hand"],
                                                                              total_time_s=duration_of_phase_s)                                       
        elif(phase == "Transport" or phase == "Release"):
            target = features.MODE_MOV_TARGETS["dropoff"][mov]
            feature_dict["target_locking_strategy"] = target_locking_strategy(fixations_targets=fixations_in_phase[target], 
                                                                              fixations_hands=pd.concat([fixations_in_phase["hand"], fixations_in_phase["object"]], axis=0), 
                                                                              total_time_s=duration_of_phase_s)
            
        elif(phase == "NonePhase"):
            feature_dict["target_locking_strategy"] = None
        else:
            raise Exception("Phase not known")
        
        # feature_dict["target_locking_strategy"] = target_locking_strategy(
        #     fixations_in_phase["object"], fixations_in_phase["hand"], duration_of_phase_s
        # )
        return feature_dict

    def _fixations_feature_dict2df(self, feat_dict, has_run=False, has_mov=False, has_phase=True, has_cell_name=False):
        df = pd.DataFrame(feat_dict)
        if not has_phase:
            df = df.T
            df.index = df.index.rename(["participant", "cell"])
            return df

        if has_mov:
            if has_run:
                df = df.T.stack().apply(pd.Series)
                if has_cell_name:
                    df.index = df.index.rename(["participant", "cell", "run", "mov", "cell_name", "phase"])
                else:
                    df.index = df.index.rename(["participant", "cell", "run", "mov", "phase"])
            else:
                df = df.T.stack().apply(pd.Series).apply(pd.Series)
                df.index = df.index.rename(["participant", "cell", "mov", "phase"])
        else:
            df = df.T.stack().apply(pd.Series).stack().apply(pd.Series)
            df.index = df.index.rename(["participant", "cell", "phase"])
        return df

    @hybrid_cache(MEMORY, lru_cache_maxsize=4)    
    def all_feature_calculations_per_run_mov_and_phase(self) -> pd.DataFrame:
        """
        Calculate the following features per participant, cell, run, movement, and phase:
            - fixation_percent_{target}: percentage of time spent fixating on the target
            - number_of_fixations_{target}: number of fixations on the target
            - target_locking_strategy: target locking strategy
            - duration_s: duration of the phase in seconds

        Parameters
        ----------
        dataset : TrustDataset
            dataset containing the data per participant
        Returns
        -------
        pd.DataFrame
            DataFrame containing the calculated features
        """

        mov_feature_dict = {}
        for d in self.ds.groupby(["participant"]):
            # get phase durations, as this is per phase (and not summed or similar) we dont need to exclude NonePhase
            phase_durs = get_phase_durations(d.study_events, groupby=["participant", "cell", "run", "mov", "phase"], excl_nonephase_and_doubles=False)

            for (pid, cell_id, run, mov), roi_per_mov in d.roi_w_phases.groupby(
                ["participant", "cell", "run", "mov"], observed=True
            ):
                if mov == "NoneMov":
                    continue
                print("Calculating for ", d.participant_id, ", cell", cell_id, ", run", run, ", Mov", mov, "...")
                cell_name = self.ds.get_cell_name(pid, cell_id)
                fixs_p_phase = get_fixations_per_phase(
                    roi_per_mov,
                    relevant_cols=["hand", "object", "target11_L", "target11_R", "target2"],
                    phase_change_events=d.phase_change_events,
                )
                phase_mov_feature_dict = {}
                for phase_name, fixs in fixs_p_phase.items():
                    if all(len(f) == 0 for f in fixs.values()):
                        # no fixations in this phase => Skip
                        continue
                    duration = phase_durs[pid, cell_id, run, mov, phase_name].total_seconds()
                    phase_mov_feature_dict[phase_name] = self._calculate_fixation_features_per_phase(fixations_in_phase=fixs, duration_of_phase_s=duration, phase=phase_name, mov=mov)
                    phase_mov_feature_dict[phase_name]["duration_s"] = duration
                mov_feature_dict[(d.participant_number, cell_id, run, mov, cell_name)] = phase_mov_feature_dict
        
        res = self._fixations_feature_dict2df(mov_feature_dict, has_run=True, has_mov=True, has_cell_name=True)

        res = res.reset_index()
        res = self.ds.add_malfunction_and_delay_cols(res)


        return res

    @hybrid_cache(MEMORY, lru_cache_maxsize=4)    
    def exl_features_per_cell_run_mov(self) -> pd.DataFrame:
        """Calculate the following features per participant, cell, run, and movement:
            - eye arrival latency at pickup and dropoff
            - eye leaving latency at pickup and dropoff

        Parameters
        ----------
        dataset : TrustDataset
            dataset containing the data per participant

        Returns
        -------
        pd.DataFrame
            DataFrame containing the calculated features
        """
        exls = [] 
        for d in self.ds.groupby(["participant"]):
            try:
                print("Calculating EAL/ELL features for", d.participant_id, "...")
                # as durations are per mov here, we need to exclude NonePhases otherwise include time "until the button is pressed again"
                mov_durs = get_phase_durations(d.study_events, groupby=["participant", "cell", "run", "mov"], excl_nonephase_and_doubles=True)
                exl = d.get_exl_times()
                exl = pd.concat([exl, mov_durs], axis=1)
                exls.append(exl)
            except Exception as e:
                print(f"Error while calculating EAL/ELL features for participant {d.participant_id}: {e}", file=sys.stderr)

        exl_df = pd.concat(exls, axis=0)
        
        # reset index
        exl_df = exl_df.reset_index()
        
        # sort by correct hierachy
        exl_df = exl_df.sort_values(['participant', 'cell', 'run', 'mov'], ascending=True)
        exl_df = exl_df.reset_index(drop=True)

        # add malfunctions and delays cols
        exl_df["cell_id"] = exl_df["cell"]
        exl_df = self.ds.add_malfunction_and_delay_cols(exl_df)
        exl_df.drop(columns="cell_id")
        
        return exl_df

    @hybrid_cache(MEMORY, lru_cache_maxsize=4)    
    def all_pastaBox_results(self, excl_double = True, excl_ps = False) -> pd.DataFrame:
        
        # 0. combine pasta box results
        pbrs = []
        for d in self.ds.groupby("participant"):
            print(f"Calculating pastabox results for {d.participant_id}...")
            pbr = d.get_pastabox_results()
            pbrs.append(pbr)
        pbrs = pd.concat(pbrs)

        # 1. double exclusion of single runs
        if excl_double:
            pbrs.excl_bc_repetition.astype("bool")
            # Eliminate extra runs from p. 13, by setting them to "rep" manually => CHECK AGAIN in original documents
            if(13 in pbrs.index.get_level_values('participant').unique()):
                pbrs.loc[(13, 6, 6), "excl_bc_repetition"] = pd.Series([False, False, False, True, True]).values
                pbrs = pbrs[~pbrs.excl_bc_repetition]
        
        # 2. exclusion due to eye tracking or performance
        if excl_ps:
            config = load_config() # load json config
            pbrs = pbrs[pbrs.index.get_level_values('participant').isin(config["incl_ps"])]
            print(pbrs.reset_index().participant.unique())
        
        # reset index
        pbrs=pbrs.reset_index()
        # add malfunctions and delays cols
        pbrs = self.ds.add_malfunction_and_delay_cols(pbrs)
        
        return pbrs


if __name__ == "__main__":
    DEBUG = True
    
    fc = FeatureCalculation()

    os.makedirs(os.path.dirname("outputs/"), exist_ok=True)

    pbrs = fc.all_pastaBox_results(excl_double=True, excl_ps= False)
    # pr already has double exclusion. Duration per phase is correct
    pr = fc.all_feature_calculations_per_run_mov_and_phase()
    # exl_feat already includes double exclusion. Also duration is correct => check again
    exl_feat = fc.exl_features_per_cell_run_mov()
    print(pr)

    # Not in debug
    if DEBUG == False:    
        pbrs.to_pickle("outputs/pbrs.pkl")
        exl_feat.to_pickle("outputs/exl_features_per_cell_run_mov.pkl")
        pr.to_pickle("outputs/fixation_features_per_run_mov_and_phase.pkl")
    