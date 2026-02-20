from warnings import simplefilter
import logging
import numpy as np
from scipy import stats
import pandas as pd
import torch

from force_regression.data.loaders.force import get_repetitions_for_task
from force_regression.data.preprocessing.force import downsample_force
import force_regression.utils.functions as fn
from force_regression.config.dataconfig import DataConfig
from force_regression.config.nomenclature import *
import force_regression.plotting.mu_plot as mu_plot

from configs.constants import *
simplefilter(action="ignore", category=pd.errors.PerformanceWarning) # for pandas performance warning

def sort_by_spike_times(data:dict):
    """
    Sorts all data in dict by MU ids
    """
    if MU_TIMESTAMP not in data:
        return data

    # Extract the first spike time from each tensor and sort the indices based on these times
    first_spikes = [t[0] for t in data[MU_TIMESTAMP]]
    sorted_indices = sorted(range(len(first_spikes)), key=lambda i: first_spikes[i])
    unsorted_data_keys = ['emg_w', 'best_exp', 'num firings', 'RoA',
                          'truth index', 'source', 'best_exp_idx', 'exponents']
    sorted_data = {key: [data[key][i] for i in sorted_indices] for key in data if key not in unsorted_data_keys}
    return sorted_data


def convert_to_s(spike_times, f_samp):
    return [i / f_samp for i in spike_times if i is not None]


def sort_list(ts):
    return sorted(ts, key=lambda x: x[0] if len(x) > 0 else float('inf'))


def cuda_to_numpy(list_tensors):
    return [i.cpu().detach().numpy() for i in list_tensors if i is not None]

def add_first_spike_time_col(mu_df:pd.DataFrame):
    """
    Adds a column to the dataframe containing the time of the first spike for each MU
    """
    mu_df[FIRST_SP_TIME] = mu_df.apply(lambda row: np.sort(row[SP_TIME])[0] if len(row[SP_TIME]) > 0 else np.nan,
                                       axis=1)
    return mu_df

def sort_rows_by_first_spike_time_per_rep(mu_df:pd.DataFrame):
    """
    Sorts the dataframe rows by the time of the first spike for each MU per repetition
    """
    assert FIRST_SP_TIME in mu_df.columns, f"Column {FIRST_SP_TIME} not found in dataframe"
    mu_df_sorted = mu_df.sort_values(by=[REP_ID, FIRST_SP_TIME])
    return mu_df_sorted


def bin_sp_times_using_mudf(time_bins, mu_df_task, config):
    """
    mu_count: total number of MUs across all electrodes
    """
    n_bins = time_bins.shape[0] - 1  # there are n-1 bins(intervals) in an array of n
    n_rows = mu_df_task.shape[0]

    mu_spike_count = np.zeros((n_rows, n_bins))
    for row in range(n_rows):  # each row contains a MU from a certain electrode
        bin_count = 0  # initialize bin count
        sp_times = np.array(mu_df_task.iloc[row][SP_TIME])
        if len(sp_times) > 0:
            sp_times = np.array(mu_df_task.iloc[row][SP_TIME] / config.f_samp)  
            bin_count, _, _ = stats.binned_statistic(sp_times, values=sp_times, bins=time_bins,
                                                                     statistic='count')
        else:
            if config.verbose:
                logging.info("No spikes to bin!")
            else:
                continue
        mu_spike_count[row, :] = bin_count
    return mu_spike_count


def get_start_end_times_for_mu_activity(mu_df:pd.DataFrame, config:DataConfig):
    """
    Extracts the start and end time from the mu_df considering whether the mu_df
    has the hold only phase or the entire force profile
    """
    if config.task_type == 'Trap':
        if config.segment_hold:
            assert np.sum(np.diff(mu_df[START_HOLD].values)) == 0, "Start time for hold is not the same for all fingers"
            assert np.sum(np.diff(mu_df[END_HOLD].values)) == 0, "End time for hold is not the same for all fingers"
            start_time_in_samples = mu_df[START_HOLD].values[0] * config.f_samp
            end_time_in_samples = mu_df[END_HOLD].values[0] * config.f_samp
        else:
            assert np.sum(np.diff(mu_df[START_TIME].values)) == 0, "Start time is not the same for all fingers"
            assert np.sum(np.diff(mu_df[END_TIME].values)) == 0, "End time is not the same for all fingers"
            start_time_in_samples = mu_df[START_TIME].values[0] * config.f_samp
            end_time_in_samples = mu_df[END_TIME].values[0] * config.f_samp
        config.tot_dur = (end_time_in_samples - start_time_in_samples)/config.f_samp
    return int(start_time_in_samples), int(end_time_in_samples)

def get_start_end_time_of_decomposition(mu_df:pd.DataFrame, config:DataConfig):
    """
    Parses the start and end time for when the decomposition was applied"""
    decomp_start_time_in_samples = int(mu_df[START_TIME].values[0] * config.f_samp)
    return decomp_start_time_in_samples



def get_common_mu(mu_df_sorted_dir:pd.DataFrame, base_mu_col:str):
    """
    Get the units that are active on both repetitions. This is done by 
    checking that the MU has at least one spike in each repetition.
    This is computed by checking on the FIRST_SP_TIME column
    if the MU has a spike in both reps, then the counter_cons_id will be 2 (ie the unit is present in both repetitions)
    """
    counter_cons_id = mu_df_sorted_dir.groupby([base_mu_col])[FIRST_SP_TIME].count()
    # extract the mu_ids that are common to both reps
    common_mu_ids = counter_cons_id[counter_cons_id == 2].index.values
    mu_df_sorted_dir = mu_df_sorted_dir[mu_df_sorted_dir[base_mu_col].isin(common_mu_ids)]
    return mu_df_sorted_dir

def is_tensor(x):
    """
    Checks whether input is a torch.tensor
    """
    if isinstance(x, np.ndarray):
        return False
    elif isinstance(x, torch.Tensor):
        return True

def count_spikes_in_windows(config:DataConfig, duration:float,
                            wins_start:float, wins_end:float,
                            spikes:pd.DataFrame, spikes_glob_ids:list):
    """
    Count the number of spikes in each bin for each MU. The spikes are counted in the bins defined by bins_start and
    bins_end. The spikes are counted for each MU and the result is stored in a dataframe.
    """
    n_mu = spikes.shape[0]
    if duration > 0: #handle case where start_time 0 and end -1. This happens when there are no spikes as start_time is set to 0 and end_time to -1
        sp_count_df = pd.DataFrame(np.zeros((len(wins_start),n_mu)), columns=spikes_glob_ids)
    else:
        sp_count_df = pd.DataFrame()
    nbins = len(wins_start)
    for mu, g_id in zip(range(n_mu), spikes_glob_ids):
        spikes_to_np = np.array(spikes.iloc[mu]) / config.f_samp     # convert to seconds 
        if spikes_to_np.shape[0] == 0:
            if config.verbose:
                logging.info("MU %s has no spikes", mu)
        else:
            for s in spikes_to_np:
                for b in range(nbins):
                    if s >= wins_start[b] and s< wins_end[b]:
                        sp_count_df.loc[b, g_id]+=1

    return sp_count_df



def select_dir_and_remap_ids_to_cons(mu_df_sorted:pd.DataFrame, config:DataConfig, mask_dir=True):
    """
    Selects the direction defined in the configuration id mask_dir is True. This ensures that the remapping is done only
    for the selected direction.
    Returns a copy of the segmented mu_df with a remaped MU ids to have consecutive ids.
    """
    if mask_dir:
        mu_df_sorted_dir = mu_df_sorted[mu_df_sorted[FING_DIR] == config.direction].copy()
    else:
        mu_df_sorted_dir = mu_df_sorted.copy()
    # map unique ids in mu_df_sorted to consecutive numbers
    mu_df_sorted_dir[CONS_MU_ID] = mu_df_sorted_dir[MU_ID].map(
        {k: v for v, k in enumerate(mu_df_sorted_dir[MU_ID].unique())})
    # if config.verbose:
    logging.debug(mu_df_sorted_dir[CONS_MU_ID].unique())

    return mu_df_sorted_dir

def filter_mvc_direction_rep_finger_mask(df:pd.DataFrame, mvc:int, direction:str,
                                        rep_id:int, finger_name:str):
    """
    Filters the dataframe by the specified mvc, direction and zero-indexed rep_id
    """
    mask = (df[REP_ID] == rep_id) & (df[FING_DIR]==direction) & (df[MVC_LVL]==mvc) & (df[FING_NAME_COL] == finger_name)
    return df[mask]

def create_cst_df(mu_df_sorted, config):
    """
    Create a dataframe with the cumulative spike train for each finger and electrode. To prepare this dataframe,
    we loop over the electrodes and append all the MU spike trains for each finger. The shape of the dataframe is (
    n_reps, n_fingers*n_electrodes*2) which is (2,15) The shape of the dataframe is (n_reps,
    n_fingers*n_electrodes*2) which is (2,15)
    """

    sp_count_gpby = mu_df_sorted.groupby([REP_ID, FING_NAME_COL, ELEC_NAME, FING_DIR]).sum(SP_COUNT)  

    col_names = [f'{elec_name}_{fing_name}' for fing_name in list(
        config.finger_label_map.keys()) for elec_name in config.electrodes]
    csp_df = pd.DataFrame(
        columns=col_names + [REP_ID] + [FING_DIR] + [MU_RATE] + [SP_COUNT] + [START_TIME] + [END_TIME] + [
            START_HOLD] + [END_HOLD] + [FIRST_SP_TIME])

    mu_df_sorted[REP_ID] = mu_df_sorted[REP_ID].astype(int)

    for rep_id in range(1, len(get_repetitions_for_task(config.task_type)) + 1):
        for _, config.fing_id in config.finger_label_map.items():
            for config.elec_name in config.electrodes:
                mask = (mu_df_sorted[REP_ID] == rep_id - 1) & (mu_df_sorted[ELEC_NAME] == config.elec_name) & (
                        mu_df_sorted[FING_NAME_COL] == fn.reverse_remap(config.fing_id, config.get_fingers())) & (mu_df_sorted[FING_DIR] == config.direction)

                # flatten the array of tensors
                if len(mu_df_sorted[mask][SP_TIME].values) > 0:  # in some cases there are no spikes
                    tmp_csp = np.concatenate(mu_df_sorted[mask][SP_TIME].values).ravel()

                    # re-sort the array of times: as we add another MU spike train, the order of the spikes might change
                    tmp_csp_sorted = np.sort(tmp_csp)

                    # add to DataFrame
                    csp_df.loc[rep_id - 1, f'{config.elec_name}_{fn.reverse_remap(config.fing_id, config.get_fingers())}'] = tmp_csp_sorted

                    # assert that the spike count is correct
                    assert len(tmp_csp_sorted) == sp_count_gpby.loc[(rep_id - 1, fn.reverse_remap(config.fing_id, config.get_fingers()),
                                                                     config.elec_name, config.direction), SP_COUNT], \
                        f"spike count mismatch for {fn.reverse_remap(config.fing_id, config.get_fingers())} {config.elec_name} rep_id {rep_id}"

        csp_df.loc[rep_id - 1, REP_ID] = rep_id - 1
        csp_df.loc[rep_id - 1, FING_DIR] = config.direction
        csp_df.loc[rep_id - 1, MU_RATE] = mu_df_sorted[mu_df_sorted[REP_ID] == rep_id - 1][MU_RATE].mean()
        csp_df.loc[rep_id - 1, SP_COUNT] = mu_df_sorted[mu_df_sorted[REP_ID] == rep_id - 1][SP_COUNT].sum()
        csp_df.loc[rep_id - 1, START_TIME] = mu_df_sorted[mu_df_sorted[REP_ID] == rep_id - 1][START_TIME].values[0]
        csp_df.loc[rep_id - 1, END_TIME] = mu_df_sorted[mu_df_sorted[REP_ID] == rep_id - 1][END_TIME].values[0]
        csp_df.loc[rep_id - 1, START_HOLD] = mu_df_sorted[mu_df_sorted[REP_ID] == rep_id - 1][START_HOLD].values[0]
        csp_df.loc[rep_id - 1, END_HOLD] = mu_df_sorted[mu_df_sorted[REP_ID] == rep_id - 1][END_HOLD].values[0]
        csp_df.loc[rep_id - 1, FIRST_SP_TIME] = mu_df_sorted[mu_df_sorted[REP_ID] == rep_id - 1][FIRST_SP_TIME].values[
            0]
    return csp_df


def create_fr_df(csp_df, config):
    fr_df = csp_df.copy()
    fr_df = fr_df[fr_df[FING_DIR] == config.direction].drop([FING_DIR, MU_RATE], axis=1)
    for col in fr_df.columns:
        if col != REP_ID:
            fr_df[col] = fr_df[col].apply(lambda x: len(x) / config.tot_dur if isinstance(x, np.ndarray) else 0)

    return fr_df


def scale_cst_per_electrode(df:pd.DataFrame, config):
    """
    Perform min-max scaling on the spike times for each electrode.
    """
    for elec in range(len(config.electrodes)):
        col_ids = np.arange(elec,15,3)
        df.iloc[:,col_ids] = (df.iloc[:,col_ids] - df.iloc[:,col_ids].min().min()) / (df.iloc[:,col_ids].max().max() - df.iloc[:,col_ids].min().min())
    return df

def get_dur_per_rep(mu_df_sorted, config):
    """
    Gets the duration of hold segment and entire task per repetition per mvc level
    """
    dur_per_rep = {}
    for mvc in mu_df_sorted[MVC_LVL].unique():
        for rep_id in range(1, len(get_repetitions_for_task(config.task_type)) + 1):
            mask = (mu_df_sorted[REP_ID] == rep_id - 1) & (mu_df_sorted[MVC_LVL] == mvc)
            dur_entire = mu_df_sorted[mask][END_TIME]  - mu_df_sorted[mask][START_TIME]

            assert (np.diff(dur_entire.values) == 0).all(), f"Some fingers do not have the same duration for mvc {mvc}. Please check the data."

            dur_per_rep[f'{rep_id}_entire_{int(mvc)}'] = dur_entire.iloc[0] 
            dur_hold = mu_df_sorted[mask][END_HOLD] - mu_df_sorted[mask][START_HOLD]
            dur_per_rep[f'{rep_id}_hold_{int(mvc)}'] = dur_hold.iloc[0]
        assert dur_per_rep[f'1_entire_{int(mvc)}'] == dur_per_rep[f'1_entire_{int(mvc)}'], f"Repetitions have different overall durations for mvc {mvc}. Please check the data"
        assert np.round(dur_per_rep[f'1_hold_{int(mvc)}'],5) == np.round(dur_per_rep[f'2_hold_{int(mvc)}'],5), f"Repetitions have different HOLD durations for mvc {mvc}. Please check the data"

    return dur_per_rep

