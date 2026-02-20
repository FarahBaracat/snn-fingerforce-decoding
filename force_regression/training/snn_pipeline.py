from __future__ import annotations

import os
import logging
from itertools import groupby
from omegaconf import DictConfig, OmegaConf, SCMode

import pickle as pkl
from math import ceil
import torch
import dill
from torch.utils.data import Subset
import pandas as pd
import numpy as np
import wandb
import force_regression.data.handlers.mu_handler as mhl
from force_regression.models.linear_regression import ConventionalRegressionOnEmg
import force_regression.utils.functions as fn
from force_regression.utils.io import create_folder
from configs.constants import *
import random
from scipy import stats
from torch.utils.data import Dataset
from force_regression.config.dataconfig import DataConfig
from force_regression.config.snnconfig import SNNConfig
from force_regression.data.loaders.force import get_repetitions_for_task
from force_regression.data.preprocessing.emg import apply_butter_lowpass, get_emg_electrode_bounds
from force_regression.data.preprocessing.force import compute_mean_force_over_windows
import force_regression.data.preprocessing.spikes as spk
import force_regression.data.noise as noise

# Import dataset classes from the separate module
from force_regression.data.datasets import EMGDataset, SnnMusDatasetPreparation, SnnEMGDatasetPreparation


def generate_data_paths(data_config, mvcs_string:str):
    """
    Generate the paths for the mu_df, force_df and data_config files where preprocessed data is saved
    instead of loading the data and preprocessing from scratch.
    """
    mu_df_file = os.path.join(data_config.snn_temp_data_path,
                              f'mu_df_sorted_{mvcs_string}_{data_config.decomp_dir.split("/")[-1]}_hold_{data_config.segment_hold}.pkl')
    force_df_file = os.path.join(data_config.snn_temp_data_path,
                                 f'multioutput_force_df_{mvcs_string}_{data_config.decomp_dir.split("/")[-1]}_hold_{data_config.segment_hold}.pkl')
    data_config_file = os.path.join(
        data_config.snn_temp_data_path, f'data_config_{mvcs_string}_{data_config.decomp_dir.split("/")[-1]}_hold_{data_config.segment_hold}.pkl')

    return mu_df_file,force_df_file,data_config_file


def load_mu_data_for_snn(data_config, load_from_file, mvcs_string, snn_default_params):
    """
    Loads data required for SNN training. If load_from_file is True, then the data is loaded from temp files to speed up the process.
    Else, the data is created and prepared and then saved to temp files.
    """
    mu_df_file, force_df_file, data_config_file = generate_data_paths(data_config, mvcs_string)

    if load_from_file: # if True access temp files. This is used to speed up the process
        mu_df_sorted = pd.read_pickle(mu_df_file)
        with open(force_df_file, 'rb') as handle:
            force_df = pkl.load(handle)
        logging.info("Loaded a pre-prepared mu_df and force_df from file")
        with open(data_config_file, 'rb') as handle:
            data_config = dill.load(handle)
            if hasattr(snn_default_params, 'task') and 'exp_dir' in snn_default_params.task:
                data_config.direction = snn_default_params.task["exp_dir"]
            else:
                data_config.direction = snn_default_params.exp_dir

    else: # if False create, prepare df and save to temp file
        _, _, mu_df_sorted, force_df = mhl.load_mu_data_to_df(data_config)
        mu_df_sorted.to_pickle(mu_df_file)
        with open(force_df_file , 'wb') as handle:
            pkl.dump(force_df, handle, protocol=pkl.HIGHEST_PROTOCOL)
        # save the updated config file
        with open(data_config_file, 'wb') as handle:
            dill.dump(data_config, handle, protocol=pkl.HIGHEST_PROTOCOL)
        logging.info("Creating the dataframes and saving them to temp files")
    return mu_df_sorted, force_df, data_config


def load_emg_data_for_snn(data_config, load_from_file:bool):
    """
    Loads iEMG data to be encoded and trained with SNN.
    Loading filtered iEMG data relies on the backend of ConventionalRegressionOnEmg.
    """
    # some dummy parameters to create the emg_reg object. They are not used in the training
    model_type= 'linear'        # model type to be used for regression: 'linear' | 'lasso'
    win_size_in_sec = 0.08      # analysis window size in seconds
    overlap_in_perc = 50        # percentage overlap between windows
    normalization_type =  'minmax'
    feature_list = ['rms']      # features to be extracted from EMG on each analysis windoow
    emg_type = 'intra'
    clip_feature = True         # Clip the feature based on the neighboring channels of the same grid/electrode
    shuffle_fingers_seed = 10   # seed used to shuffle the order of fingers before fitting the model
    post_process = False

    emg_reg = ConventionalRegressionOnEmg(model_type,
                                      data_config,
                                      emg_type,
                                      overlap_in_perc=overlap_in_perc,
                                      window_size_in_sec=win_size_in_sec,
                                      regression_data_parent_dir=data_config.convreg_temp_data_path,
                                      normalization_type=normalization_type,
                                      feature_list=feature_list,
                                      load_regression_data_from_file=load_from_file,
                                      clip_feature=clip_feature,
                                      post_process=post_process,
                                      shuffle_fingers_seed=shuffle_fingers_seed
                                      )

    emg_reg.create_emg_regression_df()
    return emg_reg.reg_data_df

def segment_tensor_with_overlap(data_tensor: torch.tensor, total_duration: float,
                                dt: int, window_size_in_sec: float,
                                overlap_ratio: float, sample_order_before_seg: list[str],
                                window_size_in_sec_conv:float=0.08):
    """
    Splits a tensor wrt to time (2nd dimension) into overlappiing windows of length bin_width.
    Only keeps the windows with size bin_width.
    window_size_in_sec: the size of the window in seconds for the SNN input presentation
    window_size_in_sec_conv: the size of the window in seconds for the LR model. This is used to check the last window
    """
    stride = window_size_in_sec * (1-overlap_ratio)

    start_ar = fn.compute_windows_start_times(window_size_in_sec, overlap_ratio, total_duration)
    start_ar_conv = fn.compute_windows_start_times(window_size_in_sec_conv, overlap_ratio, total_duration)
    conv_last_win_end = start_ar_conv[-1] + window_size_in_sec_conv
    splits = [data_tensor[:, int(start/dt): int(start/dt) + int(window_size_in_sec/dt)] for start in start_ar]

    logging.debug("Current slice start:%s. Window ends %s. Adding one more stride %s, the window would end at %s",
                  start_ar[-1], start_ar[-1]+window_size_in_sec, start_ar[-1]+stride, start_ar[-1]+stride+window_size_in_sec)
    last_win_start = start_ar[-1]
    last_win_end = last_win_start + window_size_in_sec
    while last_win_end < conv_last_win_end:
        last_win_start = start_ar[-1]+ stride
        last_win_end = last_win_start + window_size_in_sec
        if last_win_end > float(data_tensor.size(1)*dt):
            duration_diff = last_win_end - float(data_tensor.size(1)*dt)
            samples_to_append = int(np.round(duration_diff/dt))
            data_tensor = torch.cat([data_tensor, torch.zeros(data_tensor.size(0),
                                                              samples_to_append,
                                                              data_tensor.size(2))],dim=1)
            splits.append(data_tensor[:, int(last_win_start/dt): int(last_win_start/dt) + int(window_size_in_sec/dt)])

    if sample_order_before_seg is not None:
        sample_order_after_seg = sample_order_before_seg * len(splits)
    else:
        sample_order_after_seg = sample_order_before_seg
    assert all([s.size(1) == int(window_size_in_sec/dt) for s in splits])
    assert start_ar[-1]+stride+window_size_in_sec > total_duration
    start_ar = np.append(start_ar, last_win_start)
    return splits, sample_order_after_seg, start_ar



def split_tensors_for_rep(inputs:torch.tensor, labels:torch.tensor, duration:float, dt:float,
                               split_type:str, window_size_in_sec:float,
                               overlap_ratio:float=None,
                            finger_sample_order_before_seg:list[str]=None):
    """
    Splits a single repetition into smaller time bins.
    The returned input and labels tensors are organized as follows:
        - Given that all samples of rep 1 are in the even indeces of the input tensor, after splitting:
            - Segments of rep 1 of thumb: ids 0, 10, 20,... num_segments
            - Segments of rep 1 of index: ids 2, 12, 22,... num_segments
            - Segments of rep 1 of middle: ids 4, 14, 24,... num_segments
            - Segments of rep 1 of ring: ids 6, 16, 26,... num_segments
            - Segments of rep 1 of little: ids 8, 18, 28,... num_segments
        - Given that all samples of rep 2 are in the odd indeces of the input tensor, after splitting:
            - Segments of rep 2 of thumb: ids 1, 11, 21,... num_segments
            - ...etc
    args:
        - input: tensor of shape (num_samples=10, num_steps, num_inputs). Tensor arranged as follows:
            - Row 1: rep1 of thumb
            - Row 2: rep2 of thumb
            - Row 3: rep1 of index
            - Row 4: rep2 of index
            ...

    """
    if split_type == 'without_overlap':
        raise NotImplementedError("This function is not implemented yet")

    if split_type == 'with_overlap':
        assert overlap_ratio is not None, "Please specify overlap_perc to segment the repetition."
        inputs, _ , inputs_wins_start = segment_tensor_with_overlap(inputs, duration, dt, window_size_in_sec, overlap_ratio,
                                                finger_sample_order_before_seg)

        labels,finger_sample_order_after_seg, labels_wins_start  = segment_tensor_with_overlap(labels, duration, dt,
                                                                            window_size_in_sec,
                                                                            overlap_ratio,
                                                                            finger_sample_order_before_seg,
                                                                    )
        assert (inputs_wins_start == labels_wins_start).all(), "The start times of the windows are not the same for input and labels"
        n_splits = len(inputs)
        # Count the number of blocks for each unique integer
        if finger_sample_order_after_seg is not None:
            block_counts = {}
            for key, _ in groupby(finger_sample_order_after_seg):
                if key in block_counts:
                    block_counts[key] += 1
                else:
                    block_counts[key] = 1
            assert all([v == n_splits for v in block_counts.values()]), "The number of splits is not the same for all fingers"
            assert block_counts[list(block_counts.keys())[0]] == n_splits, "The number of splits is not the same for all fingers"
        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)
        return inputs, labels, n_splits, finger_sample_order_after_seg, inputs_wins_start


def reconstruct_rep_from_bins(concatenated_bins:np.ndarray, num_segments:int,
                               bin_width:float, overlap_perc:float, dt:float,
                               recorded_rep_duration:float)->torch.Tensor:
    """
    Reconstructs the original repetition from the segmented bins considering the overlap percentage"""
    single_rep_dur = get_repetition_duration_from_segments(num_segments, bin_width, overlap_perc)
    bins_starts = fn.compute_windows_start_times(bin_width, overlap_perc, recorded_rep_duration)
    bin_steps = int(bin_width/dt)

    n_fingers = concatenated_bins.shape[-1]
    assert concatenated_bins.shape[0]==1, "The input array should have a batch size of 1"
    reconstructed_rep = torch.zeros((ceil(single_rep_dur/dt), n_fingers, n_fingers))
    arr_pointer=0
    for fing_i in range(n_fingers):  # iterate over the fingers because the bins are concatenated along the finger axis as well
        for start_time in bins_starts:
            single_bin = concatenated_bins[0, arr_pointer: arr_pointer + bin_steps, :] #,:]
            reconstructed_rep[int(start_time/dt): int(start_time/dt) + bin_steps, fing_i,:] = single_bin
            arr_pointer += bin_steps
    return reconstructed_rep


def conc_var_tracker(tracker, is_target_tracker=False):
    """
    Concatenates the predictions made over the time segments of the repetition and the target of each segment
    depedning on the value of is_target_tracker, the input tracker is either a rec_tracker or a target_tracker
    if is_target_tracker is False, this is a rec_tracker. Else, it is a target_tracker
        - rec_tracker: list of rec. rec is a dict with keys: 'mem_LAYER','sp_LAYER', 'cur_LAYER

        Each variable in the rec dict is a tensor of shape: (batch_size, num_timesteps , prev_layer_size)

    """
    if not is_target_tracker:

        try:
            batch_size = tracker[0][SPK_IN].size(0)
        except:
            batch_size = tracker[0][EMG_IN].size(0)
    else:
        batch_size = tracker[0].size(0)
    # Step 1: Concatenate the batches first to the time dimension
    if batch_size > 1:
        logging.info("batch_size greater than 1: %d", batch_size)
        # reshape the tensors in rec_tracker to (num_timesteps*batch_size,1,  prev_layer_size)
        if not is_target_tracker:
            for rec in tracker:
                for var in rec.keys():
                    # concatenate the batch dimesion with the time dimension
                    # The right way to concatenate here is to ensure batch is the first dimension
                    rec[var] = rec[var].reshape(-1, rec[var].size(-1))
                    logging.debug("after flatten %s  shape: %s", var, rec[var].size())

                    # reshape to (num_timesteps*batch_size, 1,  prev_layer_size)
                    rec[var] = torch.unsqueeze(rec[var], dim=0)
                    logging.debug("after squeeze %s shape: %s", var, rec[var].size())
        else:
            # reshape the tensors in target_tracker to (num_timesteps*batch_size,1,  prev_layer_size)
            for i, target in enumerate(tracker):
                tracker[i] = target.reshape(-1, target.size(-1))
                tracker[i] = torch.unsqueeze(tracker[i], dim=0)
    # Step 2: Concatenate the samples from all time segments
    # parse layer names from the keys of the first rec
    # mem_out, sp_out, cur_out,mem_hid,etc
    conc_dict = {}
    conc_target = None
    if not is_target_tracker:
        var_names = list(tracker[0].keys())
        layer_names = [var_name.split('_')[1] for var_name in var_names]
        layer_names = list(set(layer_names))
        # concatenate segments (samples in case of batch 1) or number of fingers (in case batch >1) into a single tensor along time axis
        for var in var_names:
            conc_dict[var] = torch.cat([rec[var] for rec in tracker], dim=1)
    else:
        # concatenate targets along the time axis for the different batches
        conc_target = torch.cat([target for target in tracker], dim=1)  # works for batch >1
    return conc_dict, conc_target


def target_tracker_to_np(y):
    _, y_tens = conc_var_tracker(tracker=y, is_target_tracker=True)
    y_np = y_tens.squeeze(0).cpu().detach().numpy()
    return y_np


def convert_predictions_to_np(train_predicted_values_cache, train_true_values_cache,
                               test_predicted_values_cache, test_true_values_cache):
    """Convert raw prediction/target tracker caches to numpy arrays."""
    y_pred_test = target_tracker_to_np(test_predicted_values_cache)
    y_true_test = target_tracker_to_np(test_true_values_cache)
    y_pred_train = target_tracker_to_np(train_predicted_values_cache)
    y_true_train = target_tracker_to_np(train_true_values_cache)
    print(f"y_pred_test shape: {y_pred_test.shape}, y_true_test shape: {y_true_test.shape}")
    return y_pred_test, y_true_test, y_pred_train, y_true_train


def prepare_emg_split(emg_force_df, data_config, snn_config, mode):
    """
    Instantiate SnnEMGDatasetPreparation for one split mode ('train' or 'inf')
    and return (prepared_snndata, dataset).
    """
    prepared = SnnEMGDatasetPreparation(
        emg_force_df.copy(),
        data_config,
        snn_config,
        snn_config.task["select_dir"],
        mode,
        snn_config.task["shuffle_fingers_seed"],
    )
    dataset = prepared.create_snn_emg_dataset()
    return prepared, dataset


def build_unsegmented_emg_dataset(prepared_snndata, snn_config):
    """
    Build an unsegmented EMGDataset from a prepared inference split.
    Used for accurate operation counting during profiling.
    """
    return EMGDataset(
        prepared_snndata.inputs_unsegmented,
        None,
        prepared_snndata.labels_unsegmented,
        None,
        prepared_snndata.num_inputs,
        prepared_snndata.num_outputs,
        prepared_snndata.num_samples,
        None,
        prepared_snndata.single_sample_duration,
        snn_config.training["dt"],
        prepared_snndata.active_fingers_order,
        None,
    )


def retrive_rep_samples(dataset:EMGDataset):
    """
    In the older implmentation:  even ids belong to rep1, odd ids belong to rep2
    In the new implmentation, I rely on a list of rep_ids to split the dataset
    """
    rep1_dataset_ids = np.where(np.array(dataset.rep_ids_list)==1)[0]
    rep2_dataset_ids = np.where(np.array(dataset.rep_ids_list)==2)[0]
    return rep1_dataset_ids, rep2_dataset_ids


def assign_datasets_for_train_test(dataset_rep_1:Subset, dataset_rep_2:Subset, rep_on_rep:str):
    """
    Assigns the correct dataset for training and testing based on the rep_on_rep string
    """
    testing_rep = int(rep_on_rep.split('on')[1])
    training_rep = int(rep_on_rep.split('on')[0])

    if testing_rep == 1:
        test_dataset = dataset_rep_1
    else:
        test_dataset = dataset_rep_2

    if training_rep ==1:
        train_dataset  = dataset_rep_1
    else:
        train_dataset = dataset_rep_2
    return test_dataset,train_dataset


def get_repetition_duration_from_segments(num_seg, seg_dur, overlap_perc):
    """
    Calculates the total duration of a repetition given the number of segments,
    the duration of each segment and the overlap percentage"""

    total_duration = (num_seg - 1) * seg_dur*(1- overlap_perc) + seg_dur
    return total_duration


def map_decay_to_tau(decay_rate:float, dt:float)->float:
    """
    Computes the time constant tau from the decay rate.

    Parameters:
    decay_rate (float): The decay rate of the neuron or synapse.
    dt (float): The discretization time step of the SNN.

    Returns:
    float: The time constant tau.
    """
    return -dt / np.log(decay_rate)


def map_tau_to_decay(tau:float, dt:float)->torch.Tensor:
    """
    Computes the decay rate from the time constant tau.

    Parameters:
    tau (float): The time constant of the neuron or synapse.
    dt (float): The discretization time step of the SNN.

    Returns:
    torch.Tensor: The decay rate.
    """
    return torch.exp(-torch.tensor(dt) / torch.tensor(tau))


def create_subsets(data_config, snn_config, dataset:EMGDataset):
    """
    Create two subsets of the dataset for the two repetitions that will be used in cross-validation
    """
    train_fing_id = snn_config.task["train_fing_id"]

    rep1_dataset_ids,rep2_dataset_ids = retrive_rep_samples(dataset)
    consecutive_finger_sample_ids = np.concatenate([np.arange(i, rep1_dataset_ids.shape[0],
                                           data_config.n_ind_fingers) for i in train_fing_id])
    dataset_rep_1 = Subset(dataset, np.array(rep1_dataset_ids)[consecutive_finger_sample_ids])
    dataset_rep_2 = Subset(dataset, np.array(rep2_dataset_ids)[consecutive_finger_sample_ids])
    logging.debug("Samples in rep1: %d\nrep1_dataset_ids=%s\nSamples in rep2: %d\nrep2_dataset_ids=%s",
                  len(dataset_rep_1), rep1_dataset_ids, len(dataset_rep_2), rep2_dataset_ids)
    return dataset_rep_1, dataset_rep_2


def get_kfold_reps(snn_config, train_with_cv):
    """
    Get the rep_on_rep values to use for cross-validation.
    In case it is a cross-validation, the rep_on_rep values are ['1on2', '2on1']
    otherwise it is the value set in the snn_config
    """
    if train_with_cv:
        kfolds_reps = ['1on2', '2on1']
    else:
        kfolds_reps = [snn_config.rep_on_rep]
    return kfolds_reps



def create_mvcs_string(data_config):
    """
    Get the MVCs string based on the load_multi attribute in data_config.
    """
    if len(data_config.load_multi) > 1:
        return ''.join([str(m) for m in sorted(data_config.load_multi)])
    else:
        return str(data_config.mvc)  # a single mvc either 5 or 15


def create_wandb_config(snn_default_params, entity):
    """
    Create and return the SNN configuration.
    """
    wandb.init(project="iemg_snn", entity=entity, config=snn_default_params)
    return wandb.config

def create_wandb_config_hydra(cfg: DictConfig):
    """
    Create and return the SNN configuration.
    """
    if cfg.logging.wandb.use_wandb:
        wandb.init(entity=cfg.logging.wandb.entity, project=cfg.logging.wandb.project,
                   config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
        return wandb.config
    return cfg


def save_dataset_to_file(dataset:EMGDataset, mode:str, data_config, percent_omission=None):
    """
    Generate the name of the dataset file to be saved.
    """
    filename =  f'emgdataset_{mode}.pkl' if percent_omission is None else f'emgdataset_{mode}_noise_{percent_omission}.pkl'
    pkl.dump(dataset, open(os.path.join(data_config.snn_temp_data_path, filename), 'wb'))
    print(f"Saved {mode} dataset in {os.path.join(data_config.snn_temp_data_path, filename)}")



def create_snn_results_filename_suffix(snn_config, data_config, mvcs, kfolds_reps,
                                       train_with_cv:bool,
                                       rep_on_rep:str):
    """
    Generate the suffix to be attached to the pkl results file from SNN training.

    Filename format uses abbreviations to keep filenames shorter:
        - ep: epochs/num_iter
        - s: seed
        - n: noise mode (o=omission, a=addition, m=misattribution)
        - p: percent noise
        - ns: noise seed
        - tf1/tf2: tau filter 1/2
        - ts/tm: tau syn/mem (or si/mi if learnable - syn/mem init)
        - et/eth: encoder tau/threshold
        - al: aleaky parameters
    """
    # Determine noise percentage based on mode
    if snn_config.task['noise_mode'] == 'addition':
        percent_noise = snn_config.task['percent_addition']
    elif snn_config.task['noise_mode'] == 'omission':
        percent_noise = snn_config.task['percent_omission']
    elif snn_config.task['noise_mode'] == 'misattribution':
        percent_noise = snn_config.task['percent_misattribution']
        snn_config.task['noise_mode'] = 'miss'
    else:
        percent_noise = 0

    # Build filename with original parameter names
    file_prefix = f'snn_ep_{snn_config.training["num_iter"]}_seed_{snn_config.training["torch_seed"]}_taufilt1_{snn_config.decoder_type["first_filter_tau"]}'
    if snn_config.task['noise_mode'] is not None and snn_config.task['noise_mode'] != 'None':
        file_prefix = f'snn_ep_{snn_config.training["num_iter"]}_seed_{snn_config.training["torch_seed"]}_noise_{snn_config.task["noise_mode"]}_percent_{percent_noise}_ns_{snn_config.training["noise_torch_seed"]}_taufilt1_{snn_config.decoder_type["first_filter_tau"]}'

    if snn_config.decoder_type["topology"] in [SHALLOW_SPIKING_DOUBLE_FILTER, ENC_SHALLOW_SPIKING]:
        file_prefix += f'_taufilt2_{snn_config.decoder_type["second_filter_tau"]}'
    if snn_config.decoder_type["topology"] in [ENC_SHALLOW_SPIKING, ENC_SHALLOW_LEAKY]:
        if  snn_config.decoder_type["set_electrode_specific_parameters"]:
            file_prefix += f'_enctau1_{snn_config.decoder_type["enc_tau_mem"][0]}_enctau2_{snn_config.decoder_type["enc_tau_mem"][1]}_enctau3_{snn_config.decoder_type["enc_tau_mem"][2]}'
            file_prefix += f'_encthresh1_{snn_config.decoder_type["enc_spk_threshold"][0]}_encthresh2_{snn_config.decoder_type["enc_spk_threshold"][1]}_encthresh3_{snn_config.decoder_type["enc_spk_threshold"][2]}'
        else:
            file_prefix += f'_enctau_{snn_config.decoder_type["enc_tau_mem"]}_encthresh_{snn_config.decoder_type["enc_spk_threshold"]}'
    file_suffix = f'_dt_{snn_config.training["dt"]}_sign_{data_config.sign_mvc}_mvc_{mvcs}_kfold_{len(kfolds_reps)}'
    if not snn_config.decoder_type["learn_tau_syn"]:
        file_prefix += (f'_tausyn_{snn_config.decoder_type["tau_syn"]}')
    else:
        file_prefix += (f'_sinit_{snn_config.decoder_type["tau_syn"]}')
    if not snn_config.decoder_type["learn_tau_mem"]:
        file_prefix += (f'_taumem_{snn_config.decoder_type["tau_mem"]}')
    else:
        file_prefix += (f'_minit_{snn_config.decoder_type["tau_mem"]}')
    if snn_config.decoder_type["use_aleaky"]:
        file_prefix += f'_aleaky_tauadapt_{snn_config.decoder_type["threshold_tau_adapt"]}_scaleadapt_{snn_config.decoder_type["threshold_scale_adapt"]}'
    file_prefix += f'{file_suffix}_hold_{data_config.segment_hold}'
    if train_with_cv:
        return 'cv_' + file_prefix + '.pkl'
    return file_prefix + f'_rep_{rep_on_rep}.pkl'

def modify_snn_temp_data_path(data_config):
    """
    Update snn_temp_data to add a subfolder in case of load_multi
    """
    if len(data_config.load_multi) > 1:
        data_config.snn_temp_data_path = os.path.join(data_config.snn_temp_data_path, 'multi_mvc')
        create_folder(data_config.snn_temp_data_path)

def create_snn_results_and_figs_dirs(data_config, network_topology, noise_mode=None):
    """
    Update the results path in case of load_multi by creating a subfolder"""
    data_config.figs_snn_dir = f'SNN/{network_topology}' # figs_snn_dir updates the full path automatically figs_snn_path
    # results dir is the parent folder, results_path is the full path adding results_dir + subject
    # here we have to create the folder as the results_path is not updated automatically
    data_config.results_path = os.path.join(data_config.results_path, f'SNN/{network_topology}')
    create_folder(data_config.results_path)

    if noise_mode is not None and noise_mode != "None":
        data_config.results_path = os.path.join(data_config.results_path, f'noise/{noise_mode}')
        create_folder(data_config.results_path)

    if len(data_config.load_multi) > 1:
        data_config.results_path = os.path.join(data_config.results_path, 'multi_mvc')
        create_folder(data_config.results_path)

def write_results_to_files(data_config, metrics_df_cv, y_df_cv, trained_params_df_cv, file_name):
    """
    Write the results of the SNN training to a pkl file.
    """
    pkl.dump(metrics_df_cv, open(os.path.join(
    data_config.results_path, f'metrics_df_{file_name}'), 'wb'))
    pkl.dump(y_df_cv, open(os.path.join(
        data_config.results_path, f'y_df_{file_name}'), 'wb'))
    # Commented out because the filename is too long:
    # FIXTHIS
    # pkl.dump(trained_params_df_cv, open(os.path.join(
    #     data_config.results_path, f'trparams_df_{file_name}'), 'wb'))


def save_recorded_vars_in_inference(data_config, rec_ts:dict, test_rep:int, file_name:str):
    """
    Save the recorded variables in inference to a pkl file.
    """
    pkl.dump(rec_ts, open(os.path.join(
    data_config.results_path, f'rec_rep_{test_rep}_{file_name}'), 'wb'))
