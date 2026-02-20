from __future__ import annotations

import logging
import random
from itertools import groupby
from math import ceil

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import Dataset

from configs.constants import (
    CONS_MU_ID, END_HOLD, END_TIME, FING_DIR, FING_NAME_COL, MU_ID, MVC_LVL,
    REP_ID, SP_TIME, START_HOLD, START_TIME, TIME
)
from force_regression.config.dataconfig import DataConfig
from force_regression.config.loaders.force import get_repetitions_for_task
from force_regression.preprocessing.emg import get_emg_electrode_bounds
from force_regression.preprocessing.force import compute_mean_force_over_windows
import force_regression.preprocessing.spikes as spk
import force_regression.utils.functions as fn
import force_regression.utils.noise as noise


class EMGDataset(Dataset):
    """
    Dataset class for EMG data to be used in PyTorch DataLoader.
    """
    def __init__(self, input_data, input_data_noisy, labels, rep_ids_list,
                 num_inputs, num_outputs, n_samples, num_segments, rep_dur, dt,
                 active_fingers_order:dict[int,list[int]],
                 segments_start:dict[int,list[int]]):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_segments = num_segments
        self.rep_ids_list = rep_ids_list
        self.num_steps = input_data.size(1)
        self.n_samples = n_samples  # total number of samples (ie number of fingers * n reps)
        self.dt = dt     # diff equation discretization time step
        self.input = input_data
        self.input_noisy = input_data_noisy #TODO: rename the variable to input_noisy
        self.labels = labels
        self.rep_dur = rep_dur  # single rep duration in seconds prior to any splitting
        self.active_fingers_order = active_fingers_order
        self.segments_start = segments_start

    def __getitem__(self, idx):
        label = self.labels[idx]
        input_data = self.input[idx]
        noisy_input_data = self.input_noisy[idx] if self.input_noisy is not None else input_data
        return input_data, label, noisy_input_data

    def __len__(self):
        return len(self.labels)


class SnnMusDatasetPreparation():
    """
    Class to prepare the torch dataset with the MUs for the SNN models.
    """

    def __init__(self,  mu_df_sorted: pd.DataFrame,  force_df: pd.DataFrame, data_config: DataConfig,
                 network_config, select_dir: bool,
                 dataset_type:str, shuffle_fingers_seed: int = 1):
        self.data_config = data_config
        self.network_config = network_config
        self.mu_df_sorted = mu_df_sorted
        self.force_df = force_df
        self.window_size_in_sec = network_config.train_rep_binwidth if dataset_type == 'train' else network_config.inf_rep_binwidth
        self.select_dir = select_dir
        self.dataset_type = dataset_type
        self.shuffle_fingers_seed = shuffle_fingers_seed
        self.fingers_list = list(data_config.finger_label_map.values())
        self.dtype = torch.float
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.force_aux_cols = [REP_ID, TIME, FING_NAME_COL,FING_DIR, MVC_LVL]
        self.entire_rep = 'whole_rep'
        self.hold_phase = 'hold_phase'
        # number of samples is the number of repetitions times the number of fingers
        self.n_reps = len(mu_df_sorted[REP_ID].unique())
        self.num_samples = self.n_reps * data_config.n_ind_fingers * mu_df_sorted[MVC_LVL].nunique()
        self.num_outputs = data_config.n_ind_fingers
        self.num_inputs = None
        self.single_sample_duration = None
        self.inputs_unsegmented = None
        self.labels_unsegmented = None
        self.inputs_segmented = None
        self.labels_segmented = None
        self.inputs_unsegmented_noisy = None
        self.inputs_segmented_noisy = None
        self.rep_ids_segmented = None
        self.finger_sample_order = None
        self.num_wins_per_mvc = None
        self.wins_start_per_mvc = None
        self.active_fingers_order = None
        self.mvc_list = None
        self.finger_sample_order = None
        self._set_random_shuffle_seed()

    def _set_active_fingers_order(self, active_fingers_order:dict):
        """
        Set the order of active finger. Since the order is shuffled across the 2 reps
        the order is saved for each rep.
        """
        self.active_fingers_order = active_fingers_order

    def _set_finger_sample_order(self, finger_sample_order:list[str]):
        """
        Set the order of the fingers for each sample. This is used to create the dataset.
        """
        self.finger_sample_order = finger_sample_order

    def _set_noisy_inputs(self, inputs_unsegmented_noisy:torch.tensor,
                            inputs_segmented_noisy:torch.tensor):
        """
        Sets the inputs tensors with omitted spikes
        """
        self.inputs_unsegmented_noisy = inputs_unsegmented_noisy
        self.inputs_segmented_noisy = inputs_segmented_noisy

    def _set_dataset_variables(self, inputs_unsegmentd:torch.tensor, labels_unsegmented:torch.tensor,
                           inputs_segmented:torch.tensor, labels_segmented:torch.tensor,
                           rep_ids_segmented:torch.tensor,
                           finger_sample_order:dict[int,list[str]],
                           num_wins_per_mvc:list[int],
                           wins_start_per_mvc:dict[int,list[int]]):
        """
        Sets the inputs and labels tensors
        """
        self.inputs_unsegmented = inputs_unsegmentd
        self.labels_unsegmented = labels_unsegmented
        self.inputs_segmented = inputs_segmented
        self.labels_segmented = labels_segmented
        self.finger_sample_order = finger_sample_order
        self.rep_ids_segmented = rep_ids_segmented

        if len(num_wins_per_mvc) > 0:
            # assert all num_wins_per_mvc are the same
            assert all(x == num_wins_per_mvc[0] for x in num_wins_per_mvc), "Number of windows per mvc are not the same"
            self.num_wins_per_mvc = num_wins_per_mvc[0]
        self.wins_start_per_mvc = wins_start_per_mvc

    def _set_single_sample_duration(self, single_sample_duration:float):
        """
        Sets the duration of a single sample or input presentation to the SNN
        """
        self.single_sample_duration = single_sample_duration

    def _set_num_inputs(self, num_inputs:int):
        """
        Sets the number of inputs for the SNN model
        """
        self.num_inputs = num_inputs

    def _set_mvc_list(self, mvc_list:list[int]):
        """
        Sets the list of MVC levels
        """
        self.mvc_list = mvc_list

    def _set_random_shuffle_seed(self):
        """
        Sets seed for random shuffle of the fingers
        """
        random.seed(self.shuffle_fingers_seed)

    def retrieve_repetition_start_end_times(self, mu_df_for_finger:pd.DataFrame):
        """
        Retrieve the start and end time for this rep and when the decompositon started
        """
        # default values, this is the case when there are no spikes
        start_time_in_samples, end_time_in_samples = 0, -1
        if not mu_df_for_finger.empty:
            start_time_in_samples, end_time_in_samples = spk.get_start_end_times_for_mu_activity(mu_df_for_finger,
                                                                                                self.data_config)
        return start_time_in_samples, end_time_in_samples


    def segment_spikes_between_start_and_end_times(self, spikes_for_finger:pd.DataFrame,
                                                   start_time_in_samples:int,
                                                   end_time_in_samples:int):
        """
        Segment the spikes between the start and end times and shift the time reference to
        the start of the decomposition.
        """
        if spikes_for_finger.apply(spk.is_tensor).any():
            spikes_for_finger[SP_TIME] = spikes_for_finger[SP_TIME].apply(lambda x: x[torch.logical_and(x >= start_time_in_samples, x <= end_time_in_samples)] if len(x) > 0 else torch.tensor([]))
        else:
            spikes_for_finger[SP_TIME] = spikes_for_finger[SP_TIME].apply(lambda x: x[np.logical_and(x >= start_time_in_samples, x <= end_time_in_samples)] if len(x) > 0 else np.array([]))

        #shift the time reference to the start time of the previously applied mask
        spikes_for_finger[SP_TIME] = spikes_for_finger[SP_TIME].apply(lambda x: x - start_time_in_samples)
        return spikes_for_finger

    def extract_phase_durations_for_rep(self, mu_df_for_dir:pd.DataFrame,
                                        rep_id_one_indexed:int):
        """
        Gets the duration of hold segment and entire task per repetition per mvc level
        """
        phase_duration = {}
        for mvc in mu_df_for_dir[MVC_LVL].unique():
            mask = (mu_df_for_dir[REP_ID] == rep_id_one_indexed - 1) & (mu_df_for_dir[MVC_LVL] == mvc)
            dur_entire = mu_df_for_dir[mask][END_TIME]  - mu_df_for_dir[mask][START_TIME]
            assert (np.diff(dur_entire.values) == 0).all(), f"Some fingers do not have the same duration for mvc {mvc}. Please check the data."
            phase_duration[f'{self.entire_rep}_mvc{int(mvc)}'] = dur_entire.iloc[0]
            dur_hold = mu_df_for_dir[mask][END_HOLD] - mu_df_for_dir[mask][START_HOLD]
            phase_duration[f'{self.hold_phase}_mvc{int(mvc)}'] = dur_hold.iloc[0]
        return phase_duration

    def get_single_sample_duration(self, rep_phase_durations:dict[str,int], mvc:int):
        """
        Get the duration of a single sample or input presentation to the SNN depending
        on the segment_hold flag.
        """
        if self.data_config.segment_hold:
            return rep_phase_durations[f'{self.hold_phase}_mvc{int(mvc)}']
        return rep_phase_durations[f'{self.entire_rep}_mvc{int(mvc)}']

    def get_number_of_simulation_steps(self, rep_phase_durations:dict[str,int], mvc:int):
        """
        Get the number of simulation steps for each sample  depending on the segment_hold flag.
        """
        single_sample_duration = self.get_single_sample_duration(rep_phase_durations, mvc)
        number_sim_steps = int(single_sample_duration / self.network_config.dt)
        return number_sim_steps

    def init_inputs_and_labels_tensors_for_rep(self, mvc:int, rep_phase_durations:dict[str,int], num_inputs:int):
        """
        Initialize the inputs and labels tensors for a single repetition.
        """
        num_sim_steps = self.get_number_of_simulation_steps(rep_phase_durations, mvc)
        rep_num_samples = self.num_samples // self.n_reps
        logging.debug("Rep num_steps: %d  num samples: %d num inputs: %d  num outputs: %d", num_sim_steps, rep_num_samples, num_inputs, self.num_outputs)
        inputs = torch.zeros((rep_num_samples, num_sim_steps, num_inputs), dtype=self.dtype, device=self.device)
        labels = torch.zeros((rep_num_samples,num_sim_steps,  self.num_outputs), dtype=self.dtype, device=self.device)
        return inputs, labels

    def update_active_mu_dict(self, active_mu_per_finger:dict[str,list[int]],
                              finger_name:str, mu_df_for_finger:pd.DataFrame)->list[int]:
        """
        Updaes the list of active mu per finger. A MU is considered active if it holds at least one spike.
        """
        if not mu_df_for_finger.empty:
            active_mu_list = mu_df_for_finger.apply(lambda x: x[CONS_MU_ID] if len(x[SP_TIME]) > 0 else np.NaN,
                                                                axis=1).unique()
        else:
            active_mu_list = np.array([])
        active_mu_per_finger[finger_name] = active_mu_list[~np.isnan(active_mu_list)]
        logging.debug("finger:%s  active mu list:\n%s",finger_name, active_mu_list)
        return active_mu_per_finger

    def convert_samples_to_time(self, samples:np.ndarray):
        """
        Converts the samples to time in seconds.
        """
        if len(samples)>0:
            return samples /self.data_config.f_samp
        return samples

    def dense_to_sparse(self, mu_spikes_in_sec:list[float], total_duration:float):
        """
        Converts a list of spike times of a single MU in seconds to a sparse representation.
        Returns a tensor of shape (1, duration/dt) of 1s at location of spike times.

        """
        # we add dt to the max duration to make sure that the last spike is included in the last bin and no spikes are missed
        bin_range = np.arange(0, total_duration + self.network_config.dt, self.network_config.dt)
        sparse_sp = torch.zeros((1, len(bin_range)))
        if len(mu_spikes_in_sec) == 0:
            return sparse_sp
        sparse_sp_np, _, _ = stats.binned_statistic(mu_spikes_in_sec, values=mu_spikes_in_sec, bins=bin_range, statistic='count')
        sparse_sp = torch.from_numpy(sparse_sp_np.reshape(1,-1))
        assert torch.sum(sparse_sp) == len(mu_spikes_in_sec), f"spike count mismatched: {torch.sum(sparse_sp)} vs {len(mu_spikes_in_sec)}"
        return sparse_sp[:,:-1]

    def convert_spike_times_to_tensor(self, spikes_for_finger:pd.DataFrame, num_sim_steps:int, active_mus:list[int],
                                      single_sample_duration:float, num_inputs:int):
        """
        Converts the spike times to a tensor with the shape (num_sim_steps, num_mus)
        """
        sample_input_tensor = torch.zeros((num_sim_steps,num_inputs ), dtype=self.dtype, device=self.device)
        for mu_id in active_mus:
            mu_spikes_in_samples = spikes_for_finger[spikes_for_finger[CONS_MU_ID]== mu_id][SP_TIME].iloc[0]
            mu_spikes_in_sec = self.convert_samples_to_time(mu_spikes_in_samples)
            mu_sparse_sp = self.dense_to_sparse(mu_spikes_in_sec, single_sample_duration)
            sample_input_tensor[:, mu_id] = mu_sparse_sp
        return sample_input_tensor

    def create_inputs_and_labels_tensors(self):
        """
        Creates the inputs and labels tensors for the entire dataset.
        """
        # Import here to avoid circular import
        from force_regression.utils.snn_prepare import split_tensors_for_rep

        combined_inputs = []
        combined_inputs_omitted = []
        combined_inputs_added = []
        combined_inputs_jittered = []
        combined_inputs_misattributed = []

        combined_labels = []
        combined_inputs_segmented = []
        combined_inputs_omitted_segmented = []
        combined_inputs_added_segmented = []
        combined_inputs_jittered_segmented = []
        combined_inputs_misattributed_segmented = []

        combined_labels_segmented = []
        rep_ids_segmented = []
        finger_sample_order = {}
        active_fingers_task_order = {}
        wins_start_per_rep = {}

        n_reps = len(get_repetitions_for_task(self.data_config.task_type))
        for rep_id_one_indexed in range(1, n_reps + 1):
            # inputs, labels, inputs_after_seg, labels_after_seg, num_wins_per_mvc, sample_order, fingers_task_order, wins_start_per_mvc = self.create_inputs_and_labels_tensors_for_rep(rep_id_one_indexed)
            inputs, labels, fingers_task_order = self.create_inputs_and_labels_tensors_for_rep(rep_id_one_indexed)
            inputs_after_seg, labels_after_seg, num_wins_per_mvc, sample_order, wins_start_per_mvc = self.create_windowed_inputs_and_labels_for_rep(inputs, labels)
            inputs = torch.cat(inputs, dim=0)
            labels = torch.cat(labels, dim=0)

            if self.network_config.noise_mode is not None:
                if self.network_config.noise_mode == 'omission':
                    inputs_with_omission, _ = self.omit_spikes_from_inputs(inputs, self.network_config.percent_omission, self.network_config.noise_torch_seed)
                    inputs_with_omission_segmented, _, _, _, _ = self.create_windowed_inputs_and_labels_for_rep([inputs_with_omission], [labels])
                    combined_inputs_omitted.append(inputs_with_omission)
                    combined_inputs_omitted_segmented.append(inputs_with_omission_segmented)

                if self.network_config.noise_mode == 'addition':
                    inputs_with_addition, _ = self.add_spikes_to_inputs(inputs, self.network_config.percent_addition, self.network_config.noise_torch_seed)
                    inputs_with_addition_segmented, _, _, _, _ = self.create_windowed_inputs_and_labels_for_rep([inputs_with_addition], [labels])
                    combined_inputs_added.append(inputs_with_addition)
                    combined_inputs_added_segmented.append(inputs_with_addition_segmented)

                if self.network_config.noise_mode == 'location':
                    inputs_jittered, _ = noise.add_jitter(inputs, self.network_config.jitter_std, self.network_config.dt,
                                                          self.network_config.rand_indices_ratio)
                    inputs_jittered_segmented, _, _, _, _ = self.create_windowed_inputs_and_labels_for_rep([inputs_jittered], [labels])
                    combined_inputs_jittered.append(inputs_jittered)
                    combined_inputs_jittered_segmented.append(inputs_jittered_segmented)

                if self.network_config.noise_mode == 'misattribution':
                    inputs_miss, _= noise.misattribute_spikes(inputs, self.network_config.percent_misattribution)
                    inputs_miss_segmented, _, _, _, _ = self.create_windowed_inputs_and_labels_for_rep([inputs_miss], [labels])
                    combined_inputs_misattributed.append(inputs_miss)
                    combined_inputs_misattributed_segmented.append(inputs_miss_segmented)
            active_fingers_task_order[rep_id_one_indexed] = fingers_task_order
            wins_start_per_rep[rep_id_one_indexed] = wins_start_per_mvc

            finger_sample_order[rep_id_one_indexed] = sample_order
            combined_inputs.append(inputs)
            combined_labels.append(labels)
            combined_inputs_segmented.append(inputs_after_seg)
            combined_labels_segmented.append(labels_after_seg)
            rep_ids_segmented.extend([rep_id_one_indexed] * inputs_after_seg.size(0))

        inputs_unsegmented = torch.cat(combined_inputs, dim=0)
        labels_unsegmented = torch.cat(combined_labels, dim=0)
        inputs_segmented = torch.cat(combined_inputs_segmented, dim=0)
        labels_segmented = torch.cat(combined_labels_segmented, dim=0)

        if self.network_config.noise_mode is not None:
            if self.network_config.noise_mode == 'omission':
                inputs_unsegmented_omitted = torch.cat(combined_inputs_omitted, dim=0)
                inputs_segmented_omitted = torch.cat(combined_inputs_omitted_segmented, dim=0)
                self._set_noisy_inputs(inputs_unsegmented_omitted, inputs_segmented_omitted)
            elif self.network_config.noise_mode == 'addition':
                inputs_unsegmented_added = torch.cat(combined_inputs_added, dim=0)
                inputs_segmented_added = torch.cat(combined_inputs_added_segmented, dim=0)
                self._set_noisy_inputs(inputs_unsegmented_added, inputs_segmented_added)
            elif self.network_config.noise_mode == 'location':
                inputs_unsegmented_jittered = torch.cat(combined_inputs_jittered, dim=0)
                inputs_segmented_jittered = torch.cat(combined_inputs_jittered_segmented, dim=0)
                self._set_noisy_inputs(inputs_unsegmented_jittered,inputs_segmented_jittered)
            elif self.network_config.noise_mode == 'misattribution':
                inputs_unsegmented_misattributed = torch.cat(combined_inputs_misattributed, dim=0)
                inputs_segmented_misattributed = torch.cat(combined_inputs_misattributed_segmented, dim=0)
                self._set_noisy_inputs(inputs_unsegmented_misattributed,inputs_segmented_misattributed)
            else:
                raise ValueError(f"Noise mode {self.network_config.noise_mode} not recognized or no noise level specified.")

        self._set_dataset_variables(inputs_unsegmented, labels_unsegmented, inputs_segmented, labels_segmented,
                                    rep_ids_segmented, finger_sample_order, num_wins_per_mvc, wins_start_per_mvc)
        self._set_active_fingers_order(active_fingers_task_order)

    def omit_spikes_from_inputs(self, inputs:torch.tensor, noise_level_in_percent:int,
                                noise_torch_seed:int)->tuple[torch.tensor, torch.tensor]:
        """
        Omit spikes from the inputs tensor based on the specified noise level.
        The noise level is specified as a percentage of the total number of spikes in each sample.
        Inputs are the unsegmented inputs tensor (i.e before windowing)
        """
        torch.manual_seed(noise_torch_seed)
        print(f"noise seed:{torch.random.initial_seed()}")
        inputs_omitted, omission_mask = noise.omit_spikes(inputs, noise_level_in_percent)
        return inputs_omitted, omission_mask

    def add_spikes_to_inputs(self, inputs:torch.tensor, noise_level_in_percent:int,
                             noise_torch_seed:int)->tuple[torch.tensor, torch.tensor]:
        """
        Add spikes to the inputs tensor based on the specified noise level.
        This mode simulates false positive spikes by adding spikes at random time steps where there were no spikes.
        The noise level is specified as a percentage of the total number of non-spike time steps in each sample.
        Inputs are the unsegmented inputs tensor (i.e before windowing)
        """
        torch.manual_seed(noise_torch_seed)
        print(f"noise seed:{torch.random.initial_seed()}")
        inputs_added, addition_mask = noise.add_spikes(inputs, noise_level_in_percent)
        return inputs_added, addition_mask

    def segment_samples_into_window(self, inputs:torch.tensor, labels:torch.tensor,
                                    single_sample_duration:float,finger_sample_order:list[int]):
        """
        Segments the inputs and labels into windows.
        """
        # Import here to avoid circular import
        from force_regression.utils.snn_prepare import split_tensors_for_rep

        if self.network_config.split_rep:
            inputs_windows, labels_windows, num_wins, finger_sample_order_after_seg, wins_start= split_tensors_for_rep(inputs,
                                                                                                            labels,
                                                                                                            single_sample_duration,
                                                                                                            self.network_config.dt,
                                                                                                            self.network_config.split_type,
                                                                                                            self.window_size_in_sec,
                                                                                                            self.network_config.overlap_perc,
                                                                                                            finger_sample_order)
            logging.info("After spliting inputs shape: %s , labels shape: %s, num windows: %d", inputs_windows.shape, labels_windows.shape, num_wins)
            return inputs_windows, labels_windows, num_wins, finger_sample_order_after_seg, wins_start


    def create_inputs_and_labels_tensors_for_rep(self,rep_id_one_indexed:int):
        """
        Creates the inputs and labels tensors for a single repetition.
        """
        if self.data_config.common_only:
            self.mu_df_sorted = spk.get_common_mu(self.mu_df_sorted, base_mu_col=MU_ID)
        mu_df_sorted_dir = spk.select_dir_and_remap_ids_to_cons(self.mu_df_sorted, self.data_config,
                                                                self.select_dir)
        n_dir = len(mu_df_sorted_dir[FING_DIR].unique())
        assert n_dir == 1 if self.select_dir else n_dir == 2
        self._set_mvc_list(mu_df_sorted_dir[MVC_LVL].unique())
        force_measurement_columns = self.force_df.columns[~self.force_df.columns.isin(self.force_aux_cols)]
        rep_phase_durations = self.extract_phase_durations_for_rep(mu_df_sorted_dir, rep_id_one_indexed)

        num_inputs = len(mu_df_sorted_dir[CONS_MU_ID].unique())
        self._set_num_inputs(num_inputs)
        inputs_per_mvc = []
        labels_per_mvc = []
        active_mu_per_finger = {}
        finger_sample_order = []
        shuffled_fingers_list = random.sample(self.fingers_list, len(self.fingers_list))
        sample_row_id = 0

        for mvc in self.mvc_list:
            num_sim_steps = self.get_number_of_simulation_steps(rep_phase_durations, mvc)
            single_sample_duration = self.get_single_sample_duration(rep_phase_durations, mvc)
            if self.single_sample_duration != single_sample_duration:
                self._set_single_sample_duration(single_sample_duration)

            inputs, labels = self.init_inputs_and_labels_tensors_for_rep(mvc, rep_phase_durations, num_inputs)
            for self.data_config.fing_id in shuffled_fingers_list:
                finger_name = fn.reverse_remap(self.data_config.fing_id, self.data_config.get_fingers())
                mu_df_for_finger = spk.filter_mvc_direction_rep_finger_mask(mu_df_sorted_dir,
                                                                            mvc,
                                                                            self.data_config.direction,
                                                                            rep_id_one_indexed - 1,
                                                                            finger_name)
                spikes_for_finger = mu_df_for_finger[[SP_TIME, CONS_MU_ID]]
                force_df_for_finger = spk.filter_mvc_direction_rep_finger_mask(self.force_df,
                                                                                mvc,
                                                                                self.data_config.direction,
                                                                                rep_id_one_indexed - 1,
                                                                                finger_name)
                measured_forces = force_df_for_finger.loc[:,force_measurement_columns].values

                start_time_in_samples, end_time_in_samples = self.retrieve_repetition_start_end_times(mu_df_for_finger)
                spikes_for_finger = self.segment_spikes_between_start_and_end_times(spikes_for_finger,
                                                                                    start_time_in_samples,
                                                                                    end_time_in_samples)
                active_mu_per_finger = self.update_active_mu_dict(active_mu_per_finger,
                                                                  finger_name, mu_df_for_finger)
                sample_input_tensor = self.convert_spike_times_to_tensor(spikes_for_finger, num_sim_steps,
                                                                         active_mu_per_finger[finger_name],
                                                                         single_sample_duration,
                                                                         num_inputs)
                inputs[sample_row_id] = sample_input_tensor
                mean_force_over_wins, wins_start_times = compute_mean_force_over_windows(pd.DataFrame(measured_forces),
                                                                                         self.data_config,
                                                                                         self.network_config.dt,
                                                                                         overlap_ratio=0)
                logging.debug("mean force over windows: %s \n %s",len(wins_start_times), wins_start_times)
                if mean_force_over_wins.shape[0] != num_sim_steps:
                    logging.warning("mean force over windows shape mismatch: %d vs %d", mean_force_over_wins.shape[0], num_sim_steps)
                    if mean_force_over_wins.shape[0] < num_sim_steps:
                        mean_force_over_wins = np.pad(mean_force_over_wins, ((0, num_sim_steps - mean_force_over_wins.shape[0]), (0,0)), mode='constant', constant_values=0)
                    else:
                        mean_force_over_wins = mean_force_over_wins[:num_sim_steps]
                assert mean_force_over_wins.shape[0] == num_sim_steps, f"mean force over windows shape mismatch: {mean_force_over_wins.shape[0]} vs {num_sim_steps}"
                assert not np.isnan(mean_force_over_wins).any(), "There are nan values in the force_avg. Please check the force signal"
                labels[sample_row_id, :,:] = torch.from_numpy(mean_force_over_wins).to(self.device)
                sample_row_id += 1
                finger_sample_order.extend([finger_name] * inputs.size(0))

            self._set_finger_sample_order(finger_sample_order)
            inputs_per_mvc.append(inputs)
            labels_per_mvc.append(labels)
        return inputs_per_mvc, labels_per_mvc, shuffled_fingers_list

    def create_windowed_inputs_and_labels_for_rep(self, inputs_per_mvc:list[torch.tensor], labels_per_mvc:list[torch.tensor]):
        """
        Segments the inputs and labels into windows with overlap
        """
        inputs_wins = []
        labels_wins = []
        num_wins_per_mvc = []
        wins_start_per_mvc = {}
        for mvc_i, mvc in enumerate(self.mvc_list):
            inputs = inputs_per_mvc[mvc_i]
            labels = labels_per_mvc[mvc_i]
            mvc_inputs_over_wins, mvc_labels_over_wins, mvc_num_wins, finger_sample_order_after_seg, wins_start = self.segment_samples_into_window(inputs, labels,
                                                                                                                                       self.single_sample_duration,
                                                                                                                                       self.finger_sample_order)
            inputs_wins.append(mvc_inputs_over_wins)
            labels_wins.append(mvc_labels_over_wins)
            num_wins_per_mvc.append(mvc_num_wins)
            wins_start_per_mvc[mvc] = wins_start
        inputs_after_segment = torch.cat(inputs_wins, dim=0)
        labels_after_segment = torch.cat(labels_wins, dim=0)
        return inputs_after_segment, labels_after_segment, num_wins_per_mvc, finger_sample_order_after_seg, wins_start_per_mvc

    def create_snn_dataset(self):
        """
        Creates the EMGDataset object for the SNN model.
        """
        self.create_inputs_and_labels_tensors() # also sets the segmented and unsegmented
        dataset = EMGDataset(self.inputs_segmented,
                             self.inputs_segmented_noisy,
                             self.labels_segmented,
                             self.rep_ids_segmented,
                             self.num_inputs,
                             self.num_outputs,
                             self.num_samples,
                             self.num_wins_per_mvc,
                             self.single_sample_duration,
                             self.network_config.dt,
                             self.active_fingers_order,
                             self.wins_start_per_mvc)
        return dataset

    def create_rnn_dataset(self):
        """
        Creates the EMGDataset object for the RNN model.
        """
        self.create_inputs_and_labels_tensors() # also sets the segmented and unsegmented
        dataset = EMGDataset(self.inputs_unsegmented,
                             self.inputs_unsegmented_noisy,
                             self.labels_unsegmented,
                             None,
                             self.num_inputs,
                             self.num_outputs,
                             self.num_samples,
                             None,
                             self.single_sample_duration,
                             self.network_config.dt,
                             self.active_fingers_order,
                             None)
        return dataset

    def create_unsegmented_snn_dataset(self):
        """
        Creates an EMGDataset using unsegmented inputs for profiling purposes.
        This allows accurate operation counting on the full repetition without
        overlap-induced redundancy from windowed segmentation.

        The unsegmented data has shape [n_samples, timesteps, n_inputs] where
        timesteps corresponds to the full repetition duration.
        """
        # Ensure tensors are created (in case called before create_snn_dataset)
        if self.inputs_unsegmented is None:
            self.create_inputs_and_labels_tensors()

        dataset = EMGDataset(self.inputs_unsegmented,
                             self.inputs_unsegmented_noisy,
                             self.labels_unsegmented,
                             None,  # rep_ids not needed for profiling
                             self.num_inputs,
                             self.num_outputs,
                             self.num_samples,
                             None,  # num_segments = None for unsegmented
                             self.single_sample_duration,
                             self.network_config.dt,
                             self.active_fingers_order,
                             None)  # segments_start not applicable
        return dataset


class SnnEMGDatasetPreparation():
    """
    Class to prepare the torch dataset with the EMG data for the SNN models.
    """

    def __init__(self, emg_force_df:pd.DataFrame, data_config: DataConfig, snn_config, select_dir: bool,
                 dataset_type:str, shuffle_fingers_seed: int = 1):
        self.data_config = data_config
        self.snn_config = snn_config
        self.emg_force_df = emg_force_df
        self.emg_cols = emg_force_df.columns[emg_force_df.columns.str.contains('emg_filt')]
        self.force_cols = [col for col in emg_force_df.iloc[0]['force_rep1'].columns if 'force' in col]
        self.window_size_in_sec = snn_config.training['train_rep_binwidth'] if dataset_type == 'train' else snn_config.training['inf_rep_binwidth']
        self.select_dir = select_dir
        self.dataset_type = dataset_type
        self.shuffle_fingers_seed = shuffle_fingers_seed
        self.fingers_list = list(data_config.finger_label_map.values())
        self.dtype = torch.float
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.force_aux_cols = [REP_ID, TIME, FING_NAME_COL,FING_DIR, MVC_LVL]
        self.entire_rep = 'whole_rep'
        self.hold_phase = 'hold_phase'
        self.rectification_type = snn_config.decoder_type['rect_type']

        # number of samples is the number of repetitions times the number of fingers
        self.n_reps = 2
        self.num_samples = self.n_reps * data_config.n_ind_fingers
        self.num_outputs = data_config.n_ind_fingers
        self.num_inputs = emg_force_df.iloc[0]['emg_filt_rep1'].shape[0]
        self.single_sample_duration = emg_force_df.iloc[0]['emg_filt_rep1'].shape[1] / data_config.f_samp
        self.num_sim_steps = int(self.single_sample_duration / self.snn_config.training["dt"])

        self.downsampling_factor = int(self.data_config.f_samp * self.snn_config.training["dt"])

        self.inputs_unsegmented = None
        self.labels_unsegmented = None
        self.inputs_segmented = None
        self.labels_segmented = None
        self.rep_ids_segmented = None
        self.finger_sample_order = None
        self.num_wins = None
        self.wins_start = None
        self.active_fingers_order = None
        self._set_random_shuffle_seed()

    def _set_active_fingers_order(self, active_fingers_order:dict):
        """
        Set the order of active finger. Since the order is shuffled across the 2 reps
        the order is saved for each rep.
        """
        self.active_fingers_order = active_fingers_order

    def _set_random_shuffle_seed(self):
        """
        Sets seed for random shuffle of the fingers
        """
        random.seed(self.shuffle_fingers_seed)


    def _set_dataset_variables(self, inputs_unsegmentd:torch.tensor, labels_unsegmented:torch.tensor,
                           inputs_segmented:torch.tensor, labels_segmented:torch.tensor,
                           rep_ids_segmented:torch.tensor,
                           finger_sample_order:dict[int,list[str]],
                           num_wins:list[int],
                           wins_start:dict[int,list[int]]):
        """
        Sets the inputs and labels tensors
        """
        self.inputs_unsegmented = inputs_unsegmentd
        self.labels_unsegmented = labels_unsegmented
        self.inputs_segmented = inputs_segmented
        self.labels_segmented = labels_segmented
        self.finger_sample_order = finger_sample_order
        self.rep_ids_segmented = rep_ids_segmented
        self.num_wins = num_wins
        self.wins_start = wins_start


    def rectify_emg(self, emg:np.ndarray):
        """
        Rectifies the EMG signal by taking the absolute value.
        """
        if self.rectification_type == 'full':
            return np.abs(emg)
        elif self.rectification_type == 'half':
            return emg.apply(lambda x: np.where(x>0, x,0)) #np.where(emg > 0, emg, 0)
        else:
            raise ValueError(f"Unknown rectification type: {self.rectification_type}. Please use 'full' or 'half'.")

    def clip_emg_channel_to_neigboring(self, emg:np.ndarray):
        """
        Clips the EMG signal to the mean value of the surrounding channles per electrode.
        The condition for clipping is that the channel value is larget than 3 times the mean value of the surrounding channels.
        """
        clipped_emg = emg.copy()
        channel_ranges_per_electrode = get_emg_electrode_bounds('intra')
        channel_neighbours = fn.create_ch_neighbors_dict('intra')

        for _, channel_range in enumerate(channel_ranges_per_electrode):
            # Get the mean value of the surrounding channels
            electrode_mean = np.mean(emg[channel_range[0]:channel_range[1],:], axis=0)
            electrode_std = np.std(emg[channel_range[0]:channel_range[1],:], axis=0)
            for rel_ch_id, channel in enumerate(range(channel_range[0], channel_range[1])):
                # Get surrounding channels
                surrounding_channels = [c for c in channel_neighbours[rel_ch_id+1] if c != -1]
                surrounding_channels_global_ids = np.array(surrounding_channels)-1 + channel_range[0]
                # Get the mean value of the surrounding channels
                surrounding_channels_mean = np.mean(emg[surrounding_channels_global_ids, :], axis=0)
                # Clip the EMG signal to the mean value of the surrounding channels
                outlier_time_points = np.where(emg[channel, :] > electrode_mean +  3 * electrode_std)[0]
                if len(outlier_time_points) > 0:
                    clipped_emg[channel, outlier_time_points] = surrounding_channels_mean[outlier_time_points].reshape(1,-1)
        return clipped_emg

    def apply_clipping_to_finger_tasks(self, emg_df:pd.DataFrame):
        """
        Applies the clipping to the EMG signal for all repetitions.
        """
        emg_df_clipped = emg_df.copy()
        for emg_col in self.emg_cols:
            emg_df_clipped[emg_col] = emg_df[emg_col].apply(self.clip_emg_channel_to_neigboring)
        return emg_df_clipped

    def clip_to_match_sim_steps(self, input_signal:np.ndarray):
        """
        Clips the EMG signal to match the number of simulation steps.
        Discrepencies arise when the number of samples is not divisible by the downsampling factor.
        """
        return input_signal[:self.num_sim_steps,:]

    def downsample_signal(self, input_signal:np.ndarray):
        """
        Downsamples the force/emg signal to match the number of simulation steps.
        Discrepencies arise when the number of samples is not divisible by the downsampling factor.
        """
        return input_signal[::self.downsampling_factor,: ]

    def create_unsegmented_inputs_and_labels_tensors(self):
        """
        Prepares the input and labels tensors for the snn model.
        Returns EMGDataset object.
        """
        dtype = torch.float
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.snn_config.decoder_type["clip_channels"]:
            emg_df_clipped = self.apply_clipping_to_finger_tasks(self.emg_force_df)
            emg_reg_df_rectified = emg_df_clipped.copy()
        else:
            emg_reg_df_rectified = self.emg_force_df.copy()
        emg_reg_df_rectified[self.emg_cols] = emg_reg_df_rectified[self.emg_cols].apply(self.rectify_emg)

        inputs = torch.zeros((self.num_samples, self.num_sim_steps, self.num_inputs), dtype=dtype, device=device)
        labels = torch.zeros((self.num_samples, self.num_sim_steps, self.num_outputs), dtype=dtype, device=device)

        finger_sample_order = []
        rep_ids = []
        sample_row_counter = 0
        finger_task_order_per_rep = {}
        for rep_id_one_indexed in range(1, self.n_reps + 1):
            shuffled_fingers_list = random.sample(self.fingers_list, len(self.fingers_list))
            finger_task_order_per_rep[rep_id_one_indexed] = shuffled_fingers_list
            for self.data_config.fing_id in shuffled_fingers_list:
                emg_reg_df_finger = emg_reg_df_rectified[emg_reg_df_rectified['finger_id'] == self.data_config.fing_id]
                emg_finger = emg_reg_df_finger[f'emg_filt_rep{rep_id_one_indexed}'].values[0].T
                emg_finger = self.downsample_signal(emg_finger)
                emg_finger = self.clip_to_match_sim_steps(emg_finger)
                inputs[sample_row_counter] = torch.from_numpy(emg_finger).to(device)

                # Get the force signal for this finger
                force_finger_df = emg_reg_df_finger[f'force_rep{rep_id_one_indexed}'].values[0]
                force_finger = force_finger_df[self.force_cols].to_numpy()
                force_finger = self.downsample_signal(force_finger)
                force_finger = self.clip_to_match_sim_steps(force_finger)
                labels[sample_row_counter] = torch.from_numpy(force_finger).to(device)

                # Update finger sample order for one repetion
                if rep_id_one_indexed ==1:
                    finger_sample_order.append(self.data_config.fing_id)
                rep_ids.append(rep_id_one_indexed)
                sample_row_counter += 1
        return inputs, labels, finger_sample_order, rep_ids, finger_task_order_per_rep

    def segment_samples_into_window(self, inputs:torch.tensor, labels:torch.tensor,
                                    single_sample_duration:float,finger_sample_order:list[int]):
        """
        Segments the inputs and labels into windows.
        """
        # Import here to avoid circular import
        from force_regression.utils.snn_prepare import split_tensors_for_rep

        if self.snn_config.task["split_rep"]:
            inputs_windows, labels_windows, num_wins, finger_sample_order_after_seg, wins_start= split_tensors_for_rep(inputs,
                                                                                                            labels,
                                                                                                            single_sample_duration,
                                                                                                            self.snn_config.training["dt"],
                                                                                                            self.snn_config.task["split_type"],
                                                                                                            self.window_size_in_sec,
                                                                                                            self.snn_config.task["overlap_perc"],
                                                                                                            finger_sample_order)
            logging.info(f"After spliting inputs shape: {inputs_windows.shape} , labels shape: {labels_windows.shape}, num windows: {num_wins}")
            return inputs_windows, labels_windows, num_wins, finger_sample_order_after_seg, wins_start


    def create_segmented_inputs_and_labels_tensors(self, inputs:torch.tensor, labels:torch.tensor,
                                                  finger_sample_order:list[int], rep_ids:list[int]):
        """
        Segments the inputs and labels into windows.
        """
        inputs_wins = []
        labels_wins = []
        rep_ids_segmented = []
        combined_finger_sample_order =[]
        for rep_id_one_indexed in range(1, self.n_reps + 1):
            # Filter the inputs and labels for this rep
            inputs_rep = inputs[np.array(rep_ids)==rep_id_one_indexed]
            labels_rep = labels[np.array(rep_ids)==rep_id_one_indexed]
            inputs_wins_per_rep, labels_wins_per_rep, num_wins, finger_sample_order_after_seg, wins_start = self.segment_samples_into_window(inputs_rep,
                                                                                                            labels_rep,
                                                                                                            self.single_sample_duration,
                                                                                                            finger_sample_order)
            inputs_wins.append(inputs_wins_per_rep)
            labels_wins.append(labels_wins_per_rep)
            rep_ids_segmented.extend([rep_id_one_indexed] * inputs_wins_per_rep.size(0))
            combined_finger_sample_order.extend(finger_sample_order_after_seg)

        inputs_segmented = torch.cat(inputs_wins, dim=0)
        labels_segmented = torch.cat(labels_wins, dim=0)
        return inputs_segmented, labels_segmented, num_wins, combined_finger_sample_order, wins_start, rep_ids_segmented

    def create_snn_emg_dataset(self):
        """
        Creates the EMGDataset object for the SNN model.
        """
        inputs, labels, finger_sample_order, rep_ids, finger_task_order = self.create_unsegmented_inputs_and_labels_tensors()
        inputs_segmented, labels_segmented, num_wins, finger_sample_order_after_seg, wins_start, rep_ids_segmented = self.create_segmented_inputs_and_labels_tensors(inputs,
                                                                                                            labels,
                                                                                                            finger_sample_order,
                                                                                                            rep_ids)

        wins_start_dict = {self.data_config.mvc: wins_start} # to maintain the same format as with the snn_mu_dataset
        self._set_dataset_variables(inputs, labels, inputs_segmented, labels_segmented,
                                    rep_ids_segmented, finger_sample_order_after_seg,
                                    num_wins, wins_start_dict)
        self._set_active_fingers_order(finger_task_order)
        dataset = EMGDataset(self.inputs_segmented,
                             None,
                             self.labels_segmented,
                             self.rep_ids_segmented,
                             self.num_inputs,
                             self.num_outputs,
                             self.num_samples,
                             self.num_wins,
                             self.single_sample_duration,
                             self.snn_config.training["dt"],
                             self.active_fingers_order,
                             self.wins_start)
        return dataset
