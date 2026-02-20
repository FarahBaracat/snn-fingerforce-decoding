import os
import re
import logging
import random
import pickle as pkl
import time
import tracemalloc
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import  Lasso, LinearRegression
import numpy as np
from force_regression.evaluation import metrics as m
from force_regression.data.loaders.force import get_repetitions_for_task
from force_regression.data.preprocessing.emg import get_emg_electrode_bounds, apply_butter_lowpass
import force_regression.data.preprocessing.spikes as spk
from force_regression.data.preprocessing.force import compute_mean_force_over_windows
from force_regression.utils.functions import load_from_pickle, save_to_pickle
from force_regression.data.handlers.force_handler import prepare_data_for_mvc_and_dir, load_data_and_segment_force
from force_regression.data.handlers.mu_handler import load_mu_data_to_df
from force_regression.config.dataconfig import DataConfig
import force_regression.utils.feature_extraction as fe
import force_regression.utils.functions as fn
from configs.constants import *


DTYPE_FP32 = np.float32
class ConventionalLinearRegression:
    """
    Class performing conventional least squareslinear regression on MU spike trains or EMG signals.
    """
    def __init__(self, linear_model:str, data_config:DataConfig, emg_type:str,
                 regression_data_parent_dir:str, overlap_in_perc:float,
                 window_size_in_sec:float,
                 feature_type:str,
                 select_dir:str=True,
                 normalization_type:str='minmax',
                 post_process:bool=False,
                 post_process_cutoff:float=2,
                 shuffle_fingers_seed:int=1,
                 noise_seed:int =100,
                 percent_noise:float=None,
                 noise_mode:str='omission'):
        
        self.model_type = linear_model
        self.feature_type = feature_type
        self.data_config = data_config
        self.emg_type = emg_type
        self.regression_data_parent_dir = regression_data_parent_dir

        self.noise_seed = noise_seed
        self.noise_mode = noise_mode
        self.percent_noise = percent_noise
        self.kcv = 2 # number of cross-validation folds: there are only 2 reps in total
        self.metrics_list= [RMSE, MAE, R2]
        self.random_split_seed = 1
        self.shuffle_fingers_seed = shuffle_fingers_seed
        self.overlap_in_perc= overlap_in_perc

        self.window_size_in_sec = window_size_in_sec
        self.window_size_in_samples = int(self.window_size_in_sec * data_config.f_samp)
        self.overlap_in_samples = fe.get_overlap_in_samp(self.window_size_in_samples,
                                                         self.overlap_in_perc)
        self.sampling_freq = int(1/(self.overlap_in_perc/100 * self.window_size_in_sec))

        self.select_dir = select_dir
        self.normalization_type = normalization_type
        self.post_process = post_process
        self.post_process_cutoff = post_process_cutoff # in Hz
        self.windows_bounds_for_finger = {}
        self.windows_count_for_finger = {}

        self._set_random_shuffle_seed()

    def _create_sub_directory(self, subdirectory:str):
        """
        Create a sub-directory for the regression results.
        """
        regression_data_path = os.path.join(self.regression_data_parent_dir, subdirectory)
        if not os.path.exists(regression_data_path):
            os.makedirs(regression_data_path)
            logging.info("Created directory for regression data: %s", regression_data_path)
        return regression_data_path
    
    def _set_random_shuffle_seed(self):
        """
        Sets seed for random shuffle of the fingers
        """
        random.seed(self.shuffle_fingers_seed)


    def initialize_regression_model(self):
        """
        Initialize the regression model.
        """
        if self.model_type == 'linear':
            return LinearRegression(fit_intercept=False)
        if self.model_type == 'lasso':
            return Lasso(fit_intercept=False, alpha=0.01)
        print(f"Model type {self.model_type} not recognized. Exiting.")

    
    def normalize_train_val_input(self,input_train, input_val):
        """
        Normalize the input data for training and validation.
        """
        scaler = initialize_scaler(self.normalization_type)
        input_train = scaler.fit_transform(input_train)
        input_val = scaler.transform(input_val)
        return input_train, input_val
      

    def prepare_convreg_predictions_df(self, y_pred_train, y_pred_test,
                                       target_train, target_test,
                                       one_indexed_test_rep, fold_index, input_dur,
                                       fing_list:list[str],
                                       y_pred_train_post:None, y_pred_test_post:None) -> pd.DataFrame:
        """
        Prepares a dataframe with the predictions for the conventional regression model.
        """
        df_columns = [Y_PRED_TRAIN, Y_TRUE_TRAIN, Y_PRED_TEST, Y_TRUE_TEST,
                     TEST_ON_REP, FOLD_COL, INPUT_DUR_COL, FING_ORDER_COL]
        if y_pred_train_post is not None:
            df_columns += [Y_PRED_TRAIN_POST, Y_PRED_TEST_POST]

        y_pred_df = pd.DataFrame(columns=df_columns)
        y_pred_df.loc[0, Y_PRED_TRAIN] = y_pred_train
        y_pred_df.loc[0, Y_TRUE_TRAIN] = target_train
        y_pred_df.loc[0, Y_PRED_TEST] = y_pred_test
        y_pred_df.loc[0, Y_TRUE_TEST] = target_test
        y_pred_df.loc[0, TEST_ON_REP] = one_indexed_test_rep
        y_pred_df.loc[0, FOLD_COL] = fold_index
        y_pred_df.loc[0, INPUT_DUR_COL] = input_dur
        y_pred_df.loc[0, FING_ORDER_COL] = fing_list   # tile with the finger name

        if y_pred_train_post is not None:
            y_pred_df.loc[0, Y_PRED_TRAIN_POST] = y_pred_train_post
            y_pred_df.loc[0, Y_PRED_TEST_POST] = y_pred_test_post

        return y_pred_df

    def post_process_predictions(self, y_pred:np.ndarray, cutoff:float,
                                 pred_sampling_freq:int)->np.ndarray:
        """
        Post-process the predictions by low-pass filtering to smoothen the predictions.
        """
        y_pred_post = np.zeros_like(y_pred)
        for output in range(y_pred.shape[1]):
            y_pred_post[:,output] = apply_butter_lowpass(y_pred[:, output],
                                                        cutoff=cutoff,
                                                        fs=pred_sampling_freq)
        return y_pred_post

    def build_force_per_finger(self, force: np.ndarray,
                               finger_order: list,
                               dataset_type: str,
                               fold_i: int) -> dict:
        """
        Stitch the force given the feature windows bounds.
        This function undoes the feature extraction and stitches the force back to create a single vector.
        """
        force_per_finger_df = pd.DataFrame()
        pointer = 0
        total_windows_count = 0
        for finger in finger_order:
            finger_windows = self.windows_count_for_finger[fold_i][finger][dataset_type]
            temp_df = pd.DataFrame(force[pointer: pointer + finger_windows, :])
            temp_df[FING_ID_COL] = finger
            temp_df[TIME] = np.arange(0,len(temp_df)/self.sampling_freq, 1/self.sampling_freq)[:finger_windows]
            force_per_finger_df = pd.concat([force_per_finger_df, temp_df], axis=0)
            pointer += finger_windows
            total_windows_count+= finger_windows
        # assert total_windows_count == force.shape[0], f"Number of windows of unfolded version [{total_windows_count}] doesn't match orignal force size [{force.shape[0]}]"

        return force_per_finger_df

    def unfold_force_df(self, y_df_cv:pd.DataFrame, force_column_name:str):
        """
        Post-process the force predictions to a dictionary per finger
        """
        unfolded_force_df = pd.DataFrame()
        dataset_type = force_column_name.split('_')[2]
        dataset_type = 'val' if dataset_type=='test' else dataset_type
        for fold_i in y_df_cv[FOLD_COL].unique():
            fold_df = y_df_cv[y_df_cv[FOLD_COL]==fold_i]
            finger_order = fold_df[FING_ORDER_COL].iloc[0]

            temp_df = pd.DataFrame()
            force = fold_df[force_column_name].iloc[0]
            force_per_finger_df = self.build_force_per_finger(force, finger_order, dataset_type, fold_i)
            temp_df = pd.concat([temp_df, pd.DataFrame(force_per_finger_df)], axis=1)
            temp_df[FOLD_COL] = fold_i
            temp_df[TEST_ON_REP] = fold_df[INPUT_DUR_COL].iloc[0]
            temp_df[INPUT_DUR_COL] = fold_df[TEST_ON_REP].iloc[0]
            unfolded_force_df = pd.concat([unfolded_force_df, temp_df], axis=0)
        return unfolded_force_df

    def post_process_y_cv_df(self, y_df_cv:pd.DataFrame):
        """
        Post-process the y_df_cv into a dict with keys are y_df columns and values
        are DataFrames with unfolded force in time.
        """
        force_aux_cols = [TEST_ON_REP, FOLD_COL, INPUT_DUR_COL, FING_ORDER_COL]
        force_cols = y_df_cv.drop(columns=force_aux_cols).columns
        post_processed_forces = {}
        for force_col in force_cols:
            post_processed_forces[force_col] = self.unfold_force_df(y_df_cv, force_col)
        return post_processed_forces
    
    def prepare_trained_models_df(self,trained_models, model_type:str,
                                test_rep:int, fold_idx:int)->pd.DataFrame:
        """
        Prepare a dataframe with the trained models.
        """
        trained_models_df = pd.DataFrame(columns=['trained_model'])
        trained_models_df.loc[0, 'trained_model'] = trained_models
        trained_models_df.loc[0, 'model_type'] = model_type
        trained_models_df.loc[0, FOLD_COL] = int(fold_idx)
        trained_models_df.loc[0, TEST_ON_REP] = int(test_rep)
        return trained_models_df

    def _generate_results_filenames(self):
        """
        Generate the metrics_df and y_df pickle file names.
        """
        metrics_df_filename = None
        y_df_filename = None
        y_dict_filename = None
        trained_models_filename = None
        if self.feature_type == INPUT_TYPE_EMG_GLOB:
            n_fing = "_".join(str(i) for i in list(self.data_config.finger_label_map.values()))
            sign_mvc = self.data_config.sign_mvc
            mvc = self.data_config.mvc
            basename = f"{self.emg_type}_ws{self.window_size_in_sec}_ol{self.overlap_in_perc}_{sign_mvc}_{mvc}_{RESULTS_FILE_ID_EMG_GLOB}_nfing_{n_fing}_shuffleseed_{self.shuffle_fingers_seed}_{self.model_type}_hold_{self.data_config.segment_hold}.pkl"
            metrics_df_filename = f"metrics_df_{basename}"
            y_df_filename = f"y_df_{basename}"
            y_dict_filename = f'y_dict_{basename}'
            trained_models_filename = f'trained_models_{basename}'

        if self.feature_type == INPUT_TYPE_SP_COUNT:
            n_fing = "_".join(str(i) for i in list(self.data_config.finger_label_map.values()))
            sign_mvc = self.data_config.sign_mvc
            mvc = self.data_config.mvc
            if self.percent_noise is not None:
                # unify the file naming convention between the baseline and the SNN for easier parsing
                noise_name = self.noise_mode if self.noise_mode !='misattribution' else 'miss'
                basename = f"{self.emg_type}_ws{self.window_size_in_sec}_ol{self.overlap_in_perc}_{sign_mvc}_{mvc}_{RESULTS_FILE_ID_SP_COUNT}_nfing_{n_fing}_shuffleseed_{self.shuffle_fingers_seed}_{self.model_type}_hold_{self.data_config.segment_hold}_noise_{noise_name}_percent_{self.percent_noise}_ns_{self.noise_seed}.pkl"
            else:
                basename = f"{self.emg_type}_ws{self.window_size_in_sec}_ol{self.overlap_in_perc}_{sign_mvc}_{mvc}_{RESULTS_FILE_ID_SP_COUNT}_nfing_{n_fing}_shuffleseed_{self.shuffle_fingers_seed}_{self.model_type}_hold_{self.data_config.segment_hold}.pkl"
            metrics_df_filename = f"metrics_df_{basename}"
            y_df_filename = f"y_df_{basename}"
            y_dict_filename = f'y_dict_{basename}'
            trained_models_filename = f'trained_models_{basename}'

        return metrics_df_filename, y_df_filename, y_dict_filename, trained_models_filename
    
    def prepare_input_data_for_profiling(self, reg_data_df:pd.DataFrame,
                                         non_mus_cols)->np.ndarray:
        "Prepare input dataframe for profiling"
            # force_a
            # non_mus_cols = self.data_config.force_cols_list + self.force_aux_cols

        profile_input = reg_data_df.drop(columns=non_mus_cols).to_numpy()
        return profile_input.astype(DTYPE_FP32)

    def profile_model(self, trained_models_df:pd.DataFrame,
                      reg_data_df:pd.DataFrame,
                        non_mus_cols:list,
                      verbose:bool=True) -> dict:
        """
        Profile the model for inference time, MACs, memory usage and model size on disk.

        Args:
            trained_models_df: DataFrame containing trained models
            reg_data_df: DataFrame containing regression data
            non_mus_cols: List of columns to exclude from input data
            verbose: If True, print profiling summary

        Returns:
            Dictionary with profiling results matching SNN format
        """
        # Get a single trained model from the dataframe
        trained_model = trained_models_df.iloc[0]['trained_model']

        # Prepare input data for profiling
        input_data = self.prepare_input_data_for_profiling(reg_data_df, non_mus_cols)

        # Static metric: Footprint - number of coefficients x size of each coefficient
        n_inputs = trained_model.coef_.shape[1]
        n_outputs = trained_model.coef_.shape[0]
        n_coefficients = n_inputs * n_outputs
        if trained_model.intercept_ is not None and trained_model.intercept_ > 0:
            n_coefficients += n_outputs  # Add bias terms
        coefficient_size_bytes = 4 if DTYPE_FP32 == np.float32 else 8  # 4 bytes for float32
        model_size_bytes = n_coefficients * coefficient_size_bytes

        # Computational costs: MACs per inference
        # For each output: n_inputs multiplications + (n_inputs - 1) additions = n_inputs MACs
        macs_per_window = n_outputs * n_inputs

        # Get number of windows per sample (one finger = one sample)
        n_windows_per_sample = 0
        if self.windows_count_for_finger:
            fold_0_windows = self.windows_count_for_finger.get(0, {})
            # Get windows for the first finger (all fingers have similar window counts)
            if fold_0_windows:
                first_finger_id = next(iter(fold_0_windows))
                finger_windows = fold_0_windows[first_finger_id]
                if isinstance(finger_windows, dict):
                    n_windows_per_sample = finger_windows.get('val', 0)
                else:
                    n_windows_per_sample = finger_windows

        total_macs_per_sample = macs_per_window * n_windows_per_sample

        # Spike counting ops
        # For each spike: check all windows (2 comparisons per window) + 1 addition when found
        # Total spikes can be computed from input_data (sum of all spike counts)
        total_spikes_per_sample = 0
        spike_count_comparisons_per_sample = 0
        spike_count_additions_per_sample = 0
        spike_count_ops_per_sample = 0

        if n_windows_per_sample > 0 and input_data.shape[0] >= n_windows_per_sample:
            # Calculate average spikes per sample from the input data
            # input_data contains spike counts per window per MU
            # Sum across all windows and MUs for one sample gives total spikes
            n_samples = input_data.shape[0] // n_windows_per_sample
            total_spikes_all_samples = input_data.sum()
            total_spikes_per_sample = int(total_spikes_all_samples / n_samples)

            # Spike counting operations per sample:
            # - Comparisons: each spike checked against all windows (2 comparisons per window)
            # - Additions: one per spike (incrementing the counter)
            spike_count_comparisons_per_sample = total_spikes_per_sample * n_windows_per_sample * 2
            spike_count_additions_per_sample = total_spikes_per_sample
            spike_count_ops_per_sample = spike_count_comparisons_per_sample + spike_count_additions_per_sample

        # Profile wall clock time and memory per sample (one sample = one finger = n_windows_per_sample windows)
        profiling_samples = []
        total_windows = input_data.shape[0]

        if n_windows_per_sample > 0:
            n_samples_to_profile = total_windows // n_windows_per_sample
        else:
            n_samples_to_profile = total_windows
            n_windows_per_sample = 1  # fallback: treat each window as a sample

        for i in range(n_samples_to_profile):
            # Get all windows for this finger/sample
            start_idx = i * n_windows_per_sample
            end_idx = start_idx + n_windows_per_sample
            sample_windows = input_data[start_idx:end_idx].astype(DTYPE_FP32)

            # Start memory tracking
            tracemalloc.start()

            # Measure wall-clock time for processing all windows in one sample (one finger)
            wall_start = time.perf_counter()
            _ = trained_model.predict(sample_windows)
            wall_end = time.perf_counter()

            # Get memory stats
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            profiling_samples.append({
                'wall_clock_time': (wall_end - wall_start) * 1e6,  # Convert to microseconds
                'cpu_memory': peak_mem  # Peak memory in bytes
            })

        # Aggregate results
        wall_clock_times = np.array([s['wall_clock_time'] for s in profiling_samples])
        cpu_memories = np.array([s['cpu_memory'] for s in profiling_samples])

        results = {
            # Per-sample arrays
            'wall_clock_time_per_sample': wall_clock_times.tolist(),
            'cpu_memory_per_sample': cpu_memories.tolist(),

            # Wall-clock time stats (in microseconds)
            'peak_wall_clock_time_us': float(np.max(wall_clock_times)),
            'avg_wall_clock_time_us': float(np.mean(wall_clock_times)),
            'std_wall_clock_time_us': float(np.std(wall_clock_times)),

            # CPU memory stats (in bytes)
            'peak_cpu_memory_bytes': int(np.max(cpu_memories)),
            'avg_cpu_memory_bytes': float(np.mean(cpu_memories)),
            'std_cpu_memory_bytes': float(np.std(cpu_memories)),

            # Model size and parameters
            'model_size_bytes': model_size_bytes,
            'n_coefficients': n_coefficients,
            'n_inputs': n_inputs,
            'n_outputs': n_outputs,

            # Computational cost - Linear regression inference
            'macs_per_window': macs_per_window,
            'n_windows_per_sample': n_windows_per_sample,
            'total_macs_per_sample': total_macs_per_sample,

            # Computational cost - Spike counting (preprocessing)
            'total_spikes_per_sample': total_spikes_per_sample,
            'spike_count_comparisons_per_sample': spike_count_comparisons_per_sample,
            'spike_count_additions_per_sample': spike_count_additions_per_sample,
            'spike_count_ops_per_sample': spike_count_ops_per_sample,
            'total_ops_per_sample': total_macs_per_sample + spike_count_ops_per_sample,

            # Metadata
            'num_samples_profiled': n_samples_to_profile,
            'model_type': self.model_type
        }

        if verbose:
            print("\n" + "=" * 70)
            print("Conventional Linear Regression Profiling Results")
            print("=" * 70)
            print(f"Samples profiled: {results['num_samples_profiled']}")
            print(f"\n[Model Parameters]")
            print(f"  Coefficients: {results['n_coefficients']:,}")
            print(f"  Model size:   {results['model_size_bytes']:,} bytes ({results['model_size_bytes'] / 1024:.2f} KiB)")
            print(f"\n[Computational Cost - Linear Regression]")
            print(f"  MACs per window:       {results['macs_per_window']:,}")
            print(f"  Windows per sample:    {results['n_windows_per_sample']}")
            print(f"  Total MACs per sample: {results['total_macs_per_sample']:,}")
            print(f"\n[Computational Cost - Spike Counting]")
            print(f"  Total spikes per sample:  {results['total_spikes_per_sample']:,}")
            print(f"  Comparisons per sample:   {results['spike_count_comparisons_per_sample']:,}")
            print(f"  Additions per sample:     {results['spike_count_additions_per_sample']:,}")
            print(f"  Spike count ops/sample:   {results['spike_count_ops_per_sample']:,}")
            print(f"\n[Total Computational Cost]")
            print(f"  Total ops per sample:     {results['total_ops_per_sample']:,}")
            print(f"\n[CPU Memory]")
            print(f"  Peak:    {results['peak_cpu_memory_bytes']:,} bytes ({results['peak_cpu_memory_bytes'] / 1024:.2f} KiB)")
            print(f"  Average: {results['avg_cpu_memory_bytes']:,.0f} bytes ({results['avg_cpu_memory_bytes'] / 1024:.2f} KiB)")
            print(f"  Std:     {results['std_cpu_memory_bytes']:,.0f} bytes ({results['std_cpu_memory_bytes'] / 1024:.2f} KiB)")
            print(f"\n[Wall-Clock Time (inference)]")
            print(f"  Peak:    {results['peak_wall_clock_time_us']:.3f} us")
            print(f"  Average: {results['avg_wall_clock_time_us']:.3f} us")
            print(f"  Std:     {results['std_wall_clock_time_us']:.3f} us")
            print("=" * 70 + "\n")

        return results

    def save_profiling_results(self, profiling_results: dict,
                                is_noise_experiment: bool = False,
                                noise_mode: str = 'omission'):
        """
        Save profiling results to pickle and log file.

        Args:
            profiling_results: Dictionary with profiling metrics from profile_model()
            is_noise_experiment: If True, save to noise subdirectory
            noise_mode: Type of noise experiment ('omission', 'misattribution', etc.)
        """
        # Generate filename based on experiment config
        sign_mvc = self.data_config.sign_mvc
        mvc = self.data_config.mvc
        subject = fn.remap_subject(self.data_config.subject, self.data_config.subj_map)

        basename = f"{subject}_sign_{sign_mvc}_mvc_{mvc}_feature_{self.feature_type}"

        # Determine save directory
        if DTYPE_FP32 == np.float32:
            subdir = os.path.join(self.data_config.results_path, 'baseline_fp32')
            if is_noise_experiment:
                subdir = os.path.join(subdir, 'noise', noise_mode)
            os.makedirs(subdir, exist_ok=True)
        else:
            subdir = self.data_config.results_path

        # Save as pickle
        profiling_pkl_filename = f"profiling_{basename}.pkl"
        pkl.dump(profiling_results, open(os.path.join(subdir, profiling_pkl_filename), 'wb'))

        # Save as human-readable log file
        profiling_log_filename = f"profiling_{basename}.log"
        log_path = os.path.join(subdir, profiling_log_filename)

        with open(log_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Conventional Linear Regression Profiling Results\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Model type: {profiling_results.get('model_type', 'N/A')}\n")
            f.write(f"Samples profiled: {profiling_results.get('num_samples_profiled', 0)}\n\n")

            f.write("[Model Parameters]\n")
            f.write(f"  Coefficients: {profiling_results.get('n_coefficients', 0):,}\n")
            model_size = profiling_results.get('model_size_bytes', 0)
            f.write(f"  Model size:   {model_size:,} bytes ({model_size / 1024:.2f} KiB)\n")
            f.write(f"  Inputs:       {profiling_results.get('n_inputs', 0)}\n")
            f.write(f"  Outputs:      {profiling_results.get('n_outputs', 0)}\n\n")

            f.write("[Computational Cost - Linear Regression]\n")
            f.write(f"  MACs per window:       {profiling_results.get('macs_per_window', 0):,}\n")
            f.write(f"  Windows per sample:    {profiling_results.get('n_windows_per_sample', 0)}\n")
            f.write(f"  Total MACs per sample: {profiling_results.get('total_macs_per_sample', 0):,}\n\n")

            f.write("[Computational Cost - Spike Counting]\n")
            f.write(f"  Total spikes per sample:  {profiling_results.get('total_spikes_per_sample', 0):,}\n")
            f.write(f"  Comparisons per sample:   {profiling_results.get('spike_count_comparisons_per_sample', 0):,}\n")
            f.write(f"  Additions per sample:     {profiling_results.get('spike_count_additions_per_sample', 0):,}\n")
            f.write(f"  Spike count ops/sample:   {profiling_results.get('spike_count_ops_per_sample', 0):,}\n\n")

            f.write("[Total Computational Cost]\n")
            f.write(f"  Total ops per sample:     {profiling_results.get('total_ops_per_sample', 0):,}\n\n")

            peak_mem = profiling_results.get('peak_cpu_memory_bytes', 0)
            avg_mem = profiling_results.get('avg_cpu_memory_bytes', 0)
            std_mem = profiling_results.get('std_cpu_memory_bytes', 0)
            f.write("[CPU Memory]\n")
            f.write(f"  Peak:    {peak_mem:,} bytes ({peak_mem / 1024:.2f} KiB)\n")
            f.write(f"  Average: {avg_mem:,.0f} bytes ({avg_mem / 1024:.2f} KiB)\n")
            f.write(f"  Std:     {std_mem:,.0f} bytes ({std_mem / 1024:.2f} KiB)\n\n")

            f.write("[Wall-Clock Time (inference)]\n")
            f.write(f"  Peak:    {profiling_results.get('peak_wall_clock_time_us', 0):.3f} us\n")
            f.write(f"  Average: {profiling_results.get('avg_wall_clock_time_us', 0):.3f} us\n")
            f.write(f"  Std:     {profiling_results.get('std_wall_clock_time_us', 0):.3f} us\n\n")

            f.write("=" * 70 + "\n")

        logging.info("Profiling results saved to %s:\n  %s\n  %s",
                     subdir, profiling_pkl_filename, profiling_log_filename)

    def save_results(self, metrics_cv_df:pd.DataFrame,
                     y_cv_df:pd.DataFrame, y_dict:dict,
                     trained_models_df:pd.DataFrame,
                     is_noise_experiment:bool=False,
                     noise_mode:str='omission',
                     is_sweep_experiment:bool=False):
        """
        Save the results of the regression.
        """
        metrics_df_filename, y_df_filename, y_dict_filename, trained_models_filename = self._generate_results_filenames()
        # create subfolder if it doesn't exist
        if DTYPE_FP32 == np.float32:
            subdir = os.path.join(self.data_config.results_path, 'baseline_fp32')
            if is_noise_experiment:
                subdir = os.path.join(subdir, 'noise', noise_mode)
            if is_sweep_experiment:
                subdir = os.path.join(subdir, 'sweep_wind')
            os.makedirs(subdir, exist_ok=True)
        else:
            subdir = self.data_config.results_path
        pkl.dump(metrics_cv_df,
                 open(os.path.join(subdir, metrics_df_filename), 'wb'))
        pkl.dump(y_cv_df,
                 open(os.path.join(subdir, y_df_filename), 'wb'))
        pkl.dump(trained_models_df, open(os.path.join(subdir, trained_models_filename), 'wb'))
        pkl.dump(y_dict, open(os.path.join(subdir, y_dict_filename), 'wb'))
        logging.info("Results saved to %s:\n %s \n %s \n %s \n %s",subdir, metrics_df_filename, y_df_filename, y_dict_filename, trained_models_filename)


class ConventionalRegressionOnMu(ConventionalLinearRegression):
    """
    Class performing conventional linear regression on features of the MUs decomposed
    from the EMG.
    """

    def __init__(self, linear_model: str, data_config: DataConfig, emg_type: str,
                 regression_data_parent_dir: str, overlap_in_perc: float,
                 post_process:bool,
                 window_size_in_sec: float,
                 select_dir: str = True,
                 load_regression_data_from_file:bool=False,
                 shuffle_fingers_seed:int=1,
                 ):
        ConventionalLinearRegression.__init__(self, linear_model, data_config,
                                              emg_type,
                                              regression_data_parent_dir, overlap_in_perc,
                                              window_size_in_sec,
                                              feature_type=INPUT_TYPE_SP_COUNT,
                                              select_dir=select_dir,
                                              post_process=post_process,
                                              shuffle_fingers_seed=shuffle_fingers_seed
                                              )
        self.regression_subdir = 'mu_regression'
        self.regression_data_path = self._create_sub_directory(self.regression_subdir)
        self.fingers_list = list(data_config.finger_label_map.values())
        self.feature_type = INPUT_TYPE_SP_COUNT
        self.force_aux_cols = [REP_ID, TIME, FING_NAME_COL,FING_DIR, MVC_LVL]
        self.mu_df_cols = [SP_TIME, MU_ID, MU_RATE, ELEC_NAME, FING_NAME_COL, SP_COUNT, START_TIME, END_TIME]
        self.mu_df = None
        self.mu_df_sorted = None
        self.mu_summary_df = None
        self.force_df = None
        self.reg_data_df = None
        self.active_mus_across_reps = None
        self.reg_active_fingers_order = None
        self.load_regression_data_from_file = load_regression_data_from_file

    def _set_mu_dfs(self, mu_df:pd.DataFrame, mu_df_sorted:pd.DataFrame,
                    mu_summary_df:pd.DataFrame):
        """
        Set the MU dataframes.
        """
        self.mu_df = mu_df
        self.mu_df_sorted = mu_df_sorted
        self.mu_summary_df = mu_summary_df

    def _set_force_df(self, force_df:pd.DataFrame):
        self.force_df = force_df

    def _set_feature_windows_for_finger(self,
                                        windows_se_for_finger:dict,
                                        windows_count_for_finger:dict):
        """
        Set the feature windows for each finger
        """
        self.windows_bounds_for_finger = windows_se_for_finger
        self.windows_count_for_finger = windows_count_for_finger

    def _set_reg_data_df(self, reg_data_df:pd.DataFrame):
        """
        Set the regression data dataframe.
        """
        self.reg_data_df = reg_data_df

    def _set_reg_active_fingers_order(self, reg_active_fingers_order:dict):
        """
        Set the order of active finger. Since the order is shuffled across the 2 reps
        the order is saved for each rep.
        """
        self.reg_active_fingers_order = reg_active_fingers_order

    def _set_active_mus_across_reps(self, active_mus_across_reps:dict):
        """
        Set the active MUs across repetitions
        """
        self.active_mus_across_reps = active_mus_across_reps

    def _add_aux_cols_to_force_df(self,force_df:pd.DataFrame,
                                  direction:str, rep_id:int, mvc_lvl:int,
                                  finger_name:str)->pd.DataFrame:
        """
        Add the auxiliary columns to the force dataframe.
        """
        force_df[REP_ID] = rep_id
        force_df[FING_DIR] = direction
        force_df[MVC_LVL] = mvc_lvl
        force_df[FING_NAME_COL] = finger_name
        force_df[TIME] = np.arange(0, force_df.shape[0] / self.data_config.f_samp, 1/self.data_config.f_samp)
        return force_df
    

    def create_mu_dfs_file_names(self):
        """
        Create file names for the MU dataframes.
        The filenames do not contain a sign next to the MVC as the dataframes contain the data
        from both flexion and extension directions
        """
        task_name = re.sub(r'(_[1-5]_)', '_', self.data_config.task_name().replace('-', ''),
                           count=1)
        file_suffix = f"{task_name}_{self.emg_type}_hold_{self.data_config.segment_hold}.pkl"
        mu_df_filename = f"mu_df_{file_suffix}"
        mu_df_sorted_filename = f"mu_df_sorted_{file_suffix}"
        mu_summary_df_filename = f"mu_summary_df_{file_suffix}"
        return mu_df_filename, mu_df_sorted_filename, mu_summary_df_filename
    
    def create_force_df_file_name(self):
        """
        Create file names for the force data. Similarly to the mu_dfs the force_df
        contains the data for both directions (i.e. flexion and extension)
        """
        task_name = re.sub(r'(_[1-5]_)', '_', self.data_config.task_name().replace('-', ''),
                           count=1)
        file_name = f"force_df_{task_name}_{self.emg_type}_hold_{self.data_config.segment_hold}.pkl"
        return file_name

    def create_regression_file_name(self):
        """
        Create file names for the regression data
        """
        file_name = f"muregdf_{self.data_config.task_name()}_{self.emg_type}_hold_{self.data_config.segment_hold}_comononly_{self.data_config.common_only}.pkl"
        return file_name

    def get_sampling_interval_from_time_column(self, input_df:pd.DataFrame)->float:
        """
        Get the input duration from the time column.
        """
        unique_rep_id = input_df[REP_ID].unique()
        rep_sampling_intervals = []
        for rep_id in unique_rep_id:
            rep_data = input_df[input_df[REP_ID]==rep_id]
            sampling_interval = np.diff(rep_data[TIME])[0]
            rep_sampling_intervals.append(np.round(sampling_interval, 3))
        assert len(set(rep_sampling_intervals))==1, f"Input duration mismatch: {rep_sampling_intervals}"
        return rep_sampling_intervals[0]
    
    def delete_all_zeros_columns(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Delete columns with all zeros.
        """
        zero_cols = df.columns[(df[df.columns]==0).all()]
        df = df.drop(columns=zero_cols, inplace=True)
    
    def load_and_save_regression_data_to_dfs(self):
        """
        Load the MU data and save it to dataframes.
        """
        mu_df, mu_summary_df, mu_df_sorted, force_df  = load_mu_data_to_df(self.data_config)
        self._set_mu_dfs(mu_df, mu_df_sorted, mu_summary_df)
        mu_df_filename, mu_df_sorted_filename, mu_summary_df_filename = self.create_mu_dfs_file_names()
        save_to_pickle(mu_df, os.path.join(self.regression_data_path, mu_df_filename), True)
        save_to_pickle(mu_df_sorted, os.path.join(self.regression_data_path, mu_df_sorted_filename), True)
        save_to_pickle(mu_summary_df, os.path.join(self.regression_data_path, mu_summary_df_filename), True)
        logging.info("MU data saved to:\n %s \n %s \n %s", mu_df_filename, mu_df_sorted_filename, mu_summary_df_filename)
        force_df_filename = self.create_force_df_file_name()
        self._set_force_df(force_df)
        save_to_pickle(force_df, os.path.join(self.regression_data_path, force_df_filename), True)
        logging.info("Force data saved to:\n %s", force_df_filename)

        del force_df

    def load_saved_regression_data_to_dfs(self):
        """
        Load the saved regression data to dataframes.
        """
        mu_df_filename, mu_df_sorted_filename, mu_summary_df_filename = self.create_mu_dfs_file_names()
        mu_df = load_from_pickle(os.path.join(self.regression_data_path, mu_df_filename), True)
        mu_df_sorted = load_from_pickle(os.path.join(self.regression_data_path, mu_df_sorted_filename), True)
        mu_summary_df = load_from_pickle(os.path.join(self.regression_data_path, mu_summary_df_filename), True)
        self._set_mu_dfs(mu_df, mu_df_sorted, mu_summary_df)

        force_df_filename = self.create_force_df_file_name()
        force_df = load_from_pickle(os.path.join(self.regression_data_path, force_df_filename), True)
        self._set_force_df(force_df)
    
    def create_mu_regression_df(self)->pd.DataFrame:
        """
        Apply preprocessing to the MU spike train data for all repetitions and 
        concatenate to create a single dataframe for regression.
        """
        if not self.load_regression_data_from_file:
            self.load_and_save_regression_data_to_dfs()
        else:
            self.load_saved_regression_data_to_dfs()
        
        reg_data_df = pd.DataFrame()
        n_reps = len(get_repetitions_for_task(self.data_config.task_type))
        active_mus_across_reps = {}
        reg_active_fingers_task_order = {}
        wins_se_across_reps = {}
        wins_count_across_reps = {}
        for rep_id_one_indexed in range(1, n_reps + 1):
            rep_df, active_mus_per_finger, fingers_task_order, wins_se, wins_count = self.create_regression_df_for_rep(rep_id_one_indexed)
            active_mus_across_reps[rep_id_one_indexed] = active_mus_per_finger
            reg_active_fingers_task_order[rep_id_one_indexed] = fingers_task_order
            reg_data_df = pd.concat([reg_data_df, rep_df], axis=0)
            logging.info("Rep %d %s all repetitions: %s", rep_id_one_indexed, rep_df.shape, reg_data_df.shape)

            wins_se_across_reps[rep_id_one_indexed-1] = wins_se
            wins_count_across_reps[rep_id_one_indexed-1] = wins_count
        self.delete_all_zeros_columns(reg_data_df)
        self._set_reg_data_df(reg_data_df)
        self._set_active_mus_across_reps(active_mus_across_reps)
        self._set_reg_active_fingers_order(reg_active_fingers_task_order)
        self._set_feature_windows_for_finger(wins_se_across_reps, wins_count_across_reps)

    def retrieve_rep_and_decomp_start_end_times(self, mu_df_for_finger:pd.DataFrame):
        """
        Retrieve the start and end time for this rep and when the decompositon started
        """
        # default values, this is the case when there are no spikes
        start_time_in_samples, end_time_in_samples = 0, -1
        decomp_start_time_in_samples = 0
        if not mu_df_for_finger.empty:
            start_time_in_samples, end_time_in_samples = spk.get_start_end_times_for_mu_activity(mu_df_for_finger,
                                                                                                    self.data_config)
            decomp_start_time_in_samples = spk.get_start_end_time_of_decomposition(mu_df_for_finger, self.data_config)
        return start_time_in_samples, end_time_in_samples, decomp_start_time_in_samples

    def create_spike_count_over_windows_df(self, spikes_for_finger:pd.DataFrame,
                                           finger_name:str,
                                           spikes_glob_ids:list,
                                           start_time_in_samples:int,
                                           end_time_in_samples:int,
                                           decomp_start_time_in_samples: int):
        """
        # Get the bins over which the spikes will be counted. The bins need to be shifted to account
        # for the start time of the decomposition and the start time of the repetition
        # """
        finger_rep_duration = (end_time_in_samples - start_time_in_samples)/self.data_config.f_samp
        logging.info("%s duration: %s", finger_name, finger_rep_duration)
        wins_start_times = np.array([])
        wins_end_times = np.array([])
        if finger_rep_duration > 0:
            wins_start_times = fn.compute_windows_start_times(self.window_size_in_sec,
                                                            self.overlap_in_perc/100,
                                                            finger_rep_duration)
            # shift the windows to account for the start time of the decomposition and the start time of the repetition
            wins_start_times = wins_start_times + start_time_in_samples / self.data_config.f_samp - decomp_start_time_in_samples/self.data_config.f_samp
            wins_end_times = wins_start_times + self.window_size_in_sec

        spike_counts_over_windows = spk.count_spikes_in_windows(self.data_config, finger_rep_duration,
                                                                wins_start_times, wins_end_times,
                                                                spikes_for_finger, spikes_glob_ids)
        
        return spike_counts_over_windows, wins_start_times, wins_end_times

    def create_regression_df_for_rep(self, rep_id_one_indexed:int):
        """
        Prepare the dataset for the regression model
        Returns a single dataframe containing all the convolved MU spike
        trains from all MUs identified from the fingers and their corresponding forces. The force is downsampled to match
        the length of the convolved spike trains.

        The way the dataframe is structured is the following:
            - Each integer column is the time series (convolved spike
        train) of a single MU. The number of columns is equal to the total number of MUs (across all fingers) - There are
        5 columns for the force: one for each finger 
            - There is an additional time column. Time is discretized with a
        time step of dt sec (step in the discretization of the spike times)
            - There is an additional rep_id column. This
        is used to identify the repetition of the task. Rep id can be either 1 or 2.
        """
        active_mu_per_finger = {}
        mu_df_sorted_dir = spk.select_dir_and_remap_ids_to_cons(self.mu_df_sorted, self.data_config, self.select_dir)
        unique_cons_ids = mu_df_sorted_dir[CONS_MU_ID].unique()

        n_dir = len(mu_df_sorted_dir[FING_DIR].unique())
        assert n_dir == 1 if self.select_dir else n_dir == 2

        reg_df = pd.DataFrame()
        if self.data_config.common_only:
            mu_df_sorted_dir = spk.get_common_mu(mu_df_sorted_dir, CONS_MU_ID)
        force_measurement_columns = self.force_df.columns[~self.force_df.columns.isin(self.force_aux_cols)]
        
        shuffled_fingers_list = random.sample(self.fingers_list, len(self.fingers_list))
        windows_se_for_finger = {}
        windows_count_for_finger = {}
        for self.data_config.fing_id in shuffled_fingers_list:
            finger_name = fn.reverse_remap(self.data_config.fing_id, self.data_config.get_fingers())
            for mvc in mu_df_sorted_dir[MVC_LVL].unique():
                mu_df_for_finger = spk.filter_mvc_direction_rep_finger_mask(mu_df_sorted_dir,
                                                                            mvc,
                                                                            self.data_config.direction,
                                                                            rep_id_one_indexed - 1,
                                                                            finger_name)
                spikes_for_finger = mu_df_for_finger[SP_TIME]
                spikes_glob_ids = list(mu_df_for_finger[CONS_MU_ID])
                logging.debug("spikes: %s  spikes_glob_ids: %d", spikes_for_finger.shape, len(spikes_glob_ids))

                force_df_for_finger = spk.filter_mvc_direction_rep_finger_mask(self.force_df,
                                                                               mvc,
                                                                               self.data_config.direction,
                                                                               rep_id_one_indexed - 1,
                                                                               finger_name)
                measured_forces = force_df_for_finger.loc[:,force_measurement_columns].values
                start_time_in_samples, end_time_in_samples, decomp_start_time_in_samples = self.retrieve_rep_and_decomp_start_end_times(mu_df_for_finger)

                spikes_for_finger = self.segment_spikes_between_start_and_end_times(spikes_for_finger,
                                                                                    start_time_in_samples,
                                                                                    end_time_in_samples,
                                                                                    decomp_start_time_in_samples)

                spike_count_df, wins_start_times, _ = self.create_spike_count_over_windows_df(spikes_for_finger,
                                                                                              finger_name,
                                                                                              spikes_glob_ids,
                                                                                              start_time_in_samples,
                                                                                              end_time_in_samples,
                                                                                              decomp_start_time_in_samples)
                logging.debug('%s: First 4 windows start times: %s', finger_name, wins_start_times[:5])
                if self.data_config.verbose:
                    logging.debug("%s  finger_df:%d reg_df:%d", finger_name, len(spike_count_df), len(reg_df))

                force_over_wins_df, wins_start_times_force = self.create_force_over_windows_df(force_measurement_columns,
                                                                                               finger_name,
                                                                                               mvc,
                                                                                               measured_forces)
                windows_count_for_finger[self.data_config.fing_id] = len(wins_start_times_force)

                if len(wins_start_times)>0:
                    if len(wins_start_times_force) != len(wins_start_times):
                        min_wins = min(len(wins_start_times), len(wins_start_times_force))
                        wins_start_times = wins_start_times[:min_wins]
                        wins_start_times_force = wins_start_times_force[:min_wins]
                        logging.warning("Force and spike windows start times mismatch. Truncated to min wins:%d", min_wins)
                        force_over_wins_df = force_over_wins_df.iloc[:min_wins]
                        spike_count_df = spike_count_df.iloc[:min_wins]
                    assert np.allclose(wins_start_times-wins_start_times[0], wins_start_times_force-wins_start_times_force[0]), "Force and spike windows start times mismatch"
                    windows_se_for_finger[self.data_config.fing_id] = wins_start_times
                    # windows_count_for_finger[self.data_config.fing_id] = len(wins_start_times)
                else:
                    windows_se_for_finger[self.data_config.fing_id] = np.array([])
                    # windows_count_for_finger[self.data_config.fing_id] = 0
                    logging.warning("No spikes for finger %s", finger_name)

                reg_df_for_finger_rep = self.combine_input_output_df(unique_cons_ids, spike_count_df, force_over_wins_df)
                reg_df_for_finger_rep[TIME] = wins_start_times_force - wins_start_times_force[0]  # relying on the force since it is always non-empty
                reg_df = pd.concat([reg_df, reg_df_for_finger_rep],axis=0).reset_index(drop=True)
                active_mu_per_finger = self.update_active_mu_dict(active_mu_per_finger,
                                                            finger_name, mu_df_for_finger)

        reg_df = reg_df.fillna(0)
        reg_df[REP_ID] = rep_id_one_indexed - 1
        return reg_df, active_mu_per_finger, shuffled_fingers_list, windows_se_for_finger, windows_count_for_finger

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

    def combine_input_output_df(self, unique_cons_ids:list[int],
                                spike_count_df:pd.DataFrame,
                                force_over_wins_df:pd.DataFrame)->pd.DataFrame:
        """
        Concatenates the input and output features (spike_count df and force_df) for the given finger into
        a single df.
        """
        combined_spikes_force_df = pd.concat([spike_count_df, force_over_wins_df], axis=1).reset_index(drop=True)
        inactive_mus_for_finger = list(set(unique_cons_ids) - set(list(combined_spikes_force_df.columns)))
        combined_spikes_force_df[inactive_mus_for_finger] = 0
    
        return combined_spikes_force_df

    def create_force_over_windows_df(self, force_measurement_columns:list[str],
                                     finger_name:str, mvc:int, measured_forces:np.ndarray):
        """
        Create a dataframe with the average forces over windows to match the spike count dataframe.
        """
        force_df = pd.DataFrame(measured_forces)
        mean_force_over_wins, wins_start_times = compute_mean_force_over_windows(force_df, self.data_config,
                                                               self.window_size_in_sec,
                                                               self.overlap_in_perc/100)

        mean_force_over_wins_df = pd.DataFrame(mean_force_over_wins, columns=force_measurement_columns)
        mean_force_over_wins_df[FING_NAME_COL] = finger_name
        mean_force_over_wins_df[FING_DIR] = self.data_config.direction
        mean_force_over_wins_df[MVC_LVL] = mvc

        if self.data_config.verbose:
            print(f"{finger_name}  force_avg:{mean_force_over_wins.shape} ")
        return mean_force_over_wins_df, wins_start_times

    def segment_spikes_between_start_and_end_times(self, spikes_for_finger:pd.DataFrame,
                                                   start_time_in_samples:int,
                                                   end_time_in_samples:int,
                                                   decomp_start_time_in_samples:int):
        """
        Segment the spikes between the start and end times and shift the time reference to
        the start of the decomposition.
        """
        if spikes_for_finger.apply(spk.is_tensor).any():
            spikes_for_finger = spikes_for_finger.apply(lambda x: x[torch.logical_and(x >= start_time_in_samples, x <= end_time_in_samples)] - decomp_start_time_in_samples if len(x) > 0 else torch.tensor([]))
        else:
            spikes_for_finger = spikes_for_finger.apply(lambda x: x[np.logical_and(x >= start_time_in_samples, x <= end_time_in_samples)] - decomp_start_time_in_samples if len(x) > 0 else np.array([]))
        return spikes_for_finger

    def restructure_wins_related_dicts(self, fold_i:int,
                                    train_rep_id:int,
                                    val_rep_id:int):
        """
        Restructure the windows_bounds_per_finger and windows_count_per_finger dicts
        to follow the same format in the EMG regression.
        """
        restructured_wins_se_dict = self._restructure_dict(self.windows_bounds_for_finger,
                                                           train_rep_id,
                                                           val_rep_id,
                                                           fold_i)
        restructured_wins_count_dict = self._restructure_dict(self.windows_count_for_finger,
                                                              train_rep_id,
                                                              val_rep_id,
                                                              fold_i)
        self._set_feature_windows_for_finger(restructured_wins_se_dict,
                                             restructured_wins_count_dict)


    def _restructure_dict(self, original_dict, train_rep_id, val_rep_id, fold_i):
        """
        Helper function to restructure the dictionary from dict[rep_id]=list[int]
        to dict[fold_i][finger_id] = {'train': list[int], 'val': list[int]}
        """
        restructured_dict = {0:{}, 1:{}}
        train_data = original_dict[train_rep_id]
        val_data = original_dict[val_rep_id]
        for finger in train_data.keys():
            restructured_dict[fold_i][finger] = {'train': train_data[finger], 'val': val_data[finger]}
            restructured_dict[fold_i + 1][finger] = {'val': train_data[finger], 'train': val_data[finger]}
        return restructured_dict
    
    def cross_validate_model(self):
        """
        Perform cross-validation on the MU spike train data.
        """
        sampling_interval = self.get_sampling_interval_from_time_column(self.reg_data_df)
        assert sampling_interval== np.round(self.window_size_in_sec*self.overlap_in_perc/100, 3), f"Input sampling interval mismatch: {sampling_interval} vs {1/self.sampling_freq}"

        if self.reg_data_df.empty:
            print("No data to perform regression on.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        kf = KFold(n_splits=self.kcv, random_state=self.random_split_seed, shuffle=True)
        kfold_split = kf.split(self.reg_data_df[REP_ID].unique())
        non_mus_cols = self.data_config.force_cols_list + self.force_aux_cols

        trained_models = []
        trained_models_df = pd.DataFrame()
        metrics_df_cv = pd.DataFrame()
        y_df_cv = pd.DataFrame()
        for fold_i, (train_rep, val_rep) in enumerate(kfold_split):
            val_rep = val_rep[0]
            train_rep = train_rep[0]
            input_train, target_train, input_val, target_val = self.prepare_training_validation_data(self.reg_data_df,
                                                                                                     non_mus_cols,
                                                                                                     train_rep,
                                                                                                     val_rep)
           
            if self.normalization_type is not None:
                input_train, input_val = self.normalize_train_val_input(input_train,input_val)
            print(f"\nFitting on input_train rep {train_rep+1}: {input_train.shape}  y_train:{target_train.shape}")

            reg_model = self.initialize_regression_model()

            # set datatype to float32
            input_train = input_train.astype(DTYPE_FP32)
            input_val = input_val.astype(DTYPE_FP32)
            target_train = target_train.astype(DTYPE_FP32)
            target_val = target_val.astype(DTYPE_FP32)

            reg_model.fit(input_train, target_train)
            trained_models.append(reg_model)

            y_pred_val = reg_model.predict(input_val)
            y_pred_train = reg_model.predict(input_train)

            y_pred_val_post = None
            y_pred_train_post = None
            if self.post_process:
                y_pred_val_post = self.post_process_predictions(y_pred_val, self.post_process_cutoff,
                                                                self.sampling_freq)
                y_pred_train_post = self.post_process_predictions(y_pred_train, self.post_process_cutoff,
                                                                  self.sampling_freq)

            fold_metrics_df = m.prepare_convreg_metrics_df(y_pred_train,target_train,
                                                           y_pred_val, target_val,
                                                           val_rep+1,fold_i,self.window_size_in_sec,
                                                           self.reg_active_fingers_order[val_rep+1],
                                                           self.metrics_list,
                                                           y_pred_train_post, y_pred_val_post)
            fold_y_df = self.prepare_convreg_predictions_df(y_pred_train, y_pred_val,
                                                            target_train.to_numpy(),
                                                            target_val.to_numpy(),
                                                            val_rep+1, fold_i,
                                                            self.window_size_in_sec,
                                                            self.reg_active_fingers_order[val_rep+1],
                                                            y_pred_train_post, y_pred_val_post)
            
            fold_trained_model_df = self.prepare_trained_models_df(reg_model, self.model_type,
                                                                   val_rep, fold_i)
            metrics_df_cv = pd.concat([metrics_df_cv, fold_metrics_df], axis=0)
            y_df_cv = pd.concat([y_df_cv, fold_y_df], axis=0)
            trained_models_df = pd.concat([trained_models_df, fold_trained_model_df], axis=0)

            if fold_i == 0:
                self.restructure_wins_related_dicts(fold_i, train_rep, val_rep)
        return metrics_df_cv, y_df_cv, trained_models_df

    def prepare_training_validation_data(self, reg_data_df:pd.DataFrame,
                                         aux_col:list[str],
                                         train_rep:int, val_rep:int):
        """
        Prepare the training and validation data for the regression."""
        fold_data_train = reg_data_df[reg_data_df[REP_ID]== train_rep]
        fold_data_test = reg_data_df[reg_data_df[REP_ID]==val_rep]

        input_train = fold_data_train.drop(columns=aux_col)
        target_train = fold_data_train[self.data_config.force_cols_list]
        input_val = fold_data_test.drop(columns=aux_col)
        target_val = fold_data_test[self.data_config.force_cols_list]
        return input_train,target_train,input_val,target_val
    

  
    
class ConventionalRegressionOnEmg(ConventionalLinearRegression):
    """
    Class performing conventional linear regression on features of the EMG signals
    """
    def __init__(self, linear_model:str, data_config:DataConfig, emg_type:str,
                 regression_data_parent_dir:str, overlap_in_perc:float,
                 window_size_in_sec:float,
                 feature_list:list,
                 normalization_type:str,
                 post_process:bool,
                 multioutput=True,
                 load_regression_data_from_file:bool=False,
                 select_dir:str=True,
                 clip_feature:bool=True,
                 shuffle_fingers_seed:int=1,
                ):
        
        ConventionalLinearRegression.__init__(self, linear_model, data_config,
                                              emg_type,
                                              regression_data_parent_dir, overlap_in_perc,
                                              window_size_in_sec,
                                              feature_type=INPUT_TYPE_EMG_GLOB,
                                              select_dir=select_dir,
                                              normalization_type=normalization_type,
                                              post_process=post_process,
                                              shuffle_fingers_seed=shuffle_fingers_seed
                                              )
        self.regression_subdir = 'emg_regression'
        self.regression_data_path = self._create_sub_directory(self.regression_subdir)
        self.feature_type = INPUT_TYPE_EMG_GLOB
        self.feature_list = feature_list
        self.multioutput = multioutput
        self.fingers_list = list(data_config.finger_label_map.values())
        self.emg_rep1_col = 'emg_filt_rep1'
        self.emg_rep2_col  = 'emg_filt_rep2'
        self.force_rep1_col = 'force_rep1'
        self.force_rep2_col = 'force_rep2'
        self.finger_id_col = 'finger_id'
        self.nreps = 2
        self.clip_feature = clip_feature #clip value to average of neighboring channels
        self.df_columns = [self.emg_rep1_col, self.emg_rep2_col,
                            self.force_rep1_col, self.force_rep2_col,
                            self.finger_id_col]
        self.load_regression_data_from_file = load_regression_data_from_file
        self.reg_data_df = None

    def _set_reg_data_df(self, reg_data_df:pd.DataFrame):
        """
        Set the regression data dataframe.
        """
        self.reg_data_df = reg_data_df

    def _set_feature_windows_for_finger(self,
                                        windows_se:dict[int,tuple[np.ndarray, np.ndarray]],
                                        windows_count:int,
                                        fold_i:int):
        """
        Set the feature windows for the finger.
        """
        self.windows_bounds_for_finger[fold_i] = windows_se
        self.windows_count_for_finger[fold_i] = windows_count

    def create_regression_file_name(self):
        """
        Create the file name that contains the regression DataFrame.
        """
        file_name = f'finger_{self.data_config.fing_id}_regfiltdict_{self.data_config.task_name()}_{self.emg_type}_hold_{self.data_config.segment_hold}.pkl'
        return file_name

    def create_emg_regression_df(self):
        """
        Loads data for training linear model.
        Either loads filtered and prepared emg and force data or reload from original file

        data_path: os.path.join(notebook_dir, emg_temp_data_path)
        """
        if not self.load_regression_data_from_file:
            self.load_and_save_regression_data_for_finger_to_dict()
        reg_data_df = self.organize_regression_data_into_df()
        self._set_reg_data_df(reg_data_df)

    def load_and_save_regression_data_for_finger_to_dict(self):
        """
        Loads emg and force data and saves it to a dictionary.
        """
        for self.data_config.fing_id in list(self.data_config.finger_label_map.values()):
            _, forces_in_perc_mvc, emg_data_dict = load_data_and_segment_force(config=self.data_config)
            emg_filt_rep1, emg_filt_rep2, force_df_rep1, force_df_rep2 = prepare_data_for_mvc_and_dir(
                    self.data_config, emg_data_dict, forces_in_perc_mvc,self.emg_type)
            regression_dict = {self.df_columns[0]: emg_filt_rep1,
                               self.df_columns[1]: emg_filt_rep2,
                               self.df_columns[2]: force_df_rep1,
                               self.df_columns[3]: force_df_rep2}

            file_name = self.create_regression_file_name()
            save_to_pickle(regression_dict,
                               file_path=os.path.join(self.regression_data_path, file_name))
            del regression_dict

    def organize_regression_data_into_df(self)->pd.DataFrame:
        """
        Organize the regression data into a dataframe.
        """
        reg_data_df = pd.DataFrame(columns=self.df_columns)
        for row, self.data_config.fing_id in enumerate(list(self.data_config.finger_label_map.values())):
            file_name = self.create_regression_file_name()
            reg_dict = load_from_pickle(os.path.join(self.regression_data_path, file_name))
            logging.info("Loaded regression data for finger %s", self.data_config.fing_id)
            reg_data_df.loc[row, :] = [reg_dict[k] for k in reg_dict.keys()] + [self.data_config.fing_id]
        return reg_data_df
 
    def cross_validate_model(self):
        """
        Perform cross-validation on the EMG data.
        """
        reg_data_df = self.reg_data_df
        logging.info("Input features freq: %s Hz", self.sampling_freq)

        kf = KFold(n_splits=self.kcv, random_state=self.random_split_seed, shuffle=True)
        kfold_split = kf.split(np.arange(self.nreps))

        trained_models = []
        trained_models_df = pd.DataFrame()
        metrics_df_cv = pd.DataFrame()
        y_df_cv = pd.DataFrame()
        for fold_i, (train_rep, val_rep) in enumerate(kfold_split):
            # there are only 2 reps in total so train_rep and test_rep are 1-element arrays
            train_rep = train_rep[0] + 1
            val_rep = val_rep[0] + 1
            input_train, target_train, input_val, target_val, fingers_order, wins_se, wins_count = self.prepare_training_validation_data(reg_data_df,
                                                                                                                    train_rep,
                                                                                                                    val_rep,
                                                                                                                    )
            self._set_feature_windows_for_finger(wins_se, wins_count, fold_i)
            if self.normalization_type is not None:
                input_train, input_val = self.normalize_train_val_input(input_train,input_val)

            print(f'x train min: {input_train.min()}, x train max: {input_train.max()}')
            print(f'x test min: {input_val.min()}, x test max: {input_val.max()}')
            print(f'Window size: {self.window_size_in_samples/self.data_config.f_samp*1000} ms. Overlap ({self.overlap_in_perc} %): {self.overlap_in_samples/self.data_config.f_samp*1000} ms')
            print(f"X_train: {input_train.shape}  X_test: {input_val.shape}    y_train:{target_train.shape}   y_test:{target_val.shape}")

            reg_model = self.initialize_regression_model()

            # set data type
            input_train = input_train.astype(DTYPE_FP32)
            input_val = input_val.astype(DTYPE_FP32)
            target_train = target_train.astype(DTYPE_FP32)
            target_val = target_val.astype(DTYPE_FP32)

            reg_model.fit(input_train, target_train)
            trained_models.append(reg_model)

            y_pred_val = reg_model.predict(input_val)
            y_pred_train = reg_model.predict(input_train)
            
            y_pred_val_post = None
            y_pred_train_post = None
            if self.post_process:
                y_pred_val_post = self.post_process_predictions(y_pred_val, self.post_process_cutoff,
                                                                self.sampling_freq)
                y_pred_train_post = self.post_process_predictions(y_pred_train, self.post_process_cutoff,
                                                                  self.sampling_freq)

            fold_metrics_df = m.prepare_convreg_metrics_df(y_pred_train, target_train,
                                                           y_pred_val, target_val,
                                                           val_rep, fold_i, self.window_size_in_sec,
                                                           fingers_order,
                                                           self.metrics_list,
                                                           y_pred_train_post, y_pred_val_post)
            fold_y_df = self.prepare_convreg_predictions_df(y_pred_train, y_pred_val,
                                                            target_train, target_val,
                                                            val_rep, fold_i, self.window_size_in_sec,
                                                            fingers_order,
                                                            y_pred_train_post,y_pred_val_post)
            
            fold_trained_model_df = self.prepare_trained_models_df(reg_model, self.model_type,
                                                                   val_rep, fold_i)
            metrics_df_cv = pd.concat([metrics_df_cv, fold_metrics_df], axis=0)
            y_df_cv = pd.concat([y_df_cv, fold_y_df], axis=0)
            trained_models_df = pd.concat([trained_models_df, fold_trained_model_df], axis=0)
            trained_models.append(reg_model)
        return metrics_df_cv, y_df_cv, trained_models_df
    
   
    def prepare_training_validation_data(self, reg_data_df: pd.DataFrame, train_rep: int, val_rep: int):
        """
        Prepare the training and validation data for the regression.
        The function extracts the features from the EMG data and the force data.
        """
        electrode_bounds = get_emg_electrode_bounds(self.emg_type)
        ch_neigh_dict = fn.create_ch_neighbors_dict(self.emg_type)

        all_input_train_features, all_output_train_force = [], []
        all_input_val_features, all_output_val_force = [], []
        shuffled_fingers_list = random.sample(self.fingers_list, len(self.fingers_list))
        logging.info("Train rep: %s, val rep: %s |Shuffled fingers list: %s", train_rep, val_rep, shuffled_fingers_list)
        windows_se_for_finger = {}
        windows_count_for_finger = {}
        for self.data_config.fing_id in shuffled_fingers_list:
            reg_finger_df = reg_data_df[reg_data_df[self.finger_id_col] == self.data_config.fing_id]
            emg_train_filt = reg_finger_df[f'emg_filt_rep{train_rep}'].iloc[0]
            emg_test_filt = reg_finger_df[f'emg_filt_rep{val_rep}'].iloc[0]
            force_train = reg_finger_df[f'force_rep{train_rep}'].iloc[0]
            force_test = reg_finger_df[f'force_rep{val_rep}'].iloc[0]

            if self.clip_feature:
                emg_train_filt = fe.clip_feature_to_mean(emg_train_filt, electrode_bounds,
                                                         ch_neigh_dict)
                emg_test_filt = fe.clip_feature_to_mean(emg_test_filt, electrode_bounds,
                                                        ch_neigh_dict)

            if self.multioutput:  # in case of multioutput, we use all the fingers
                force_train_ft = force_train[self.data_config.force_cols_list]
                force_test_ft = force_test[self.data_config.force_cols_list]
            else: # in case of single output, we use only the finger of interest
                force_train_ft = force_train[self.data_config.force_cols_list[self.data_config.fing_id-1]]
                force_test_ft = force_test[self.data_config.force_cols_list[self.data_config.fing_id-1]]
            input_train_features, output_train_force, wins_se_tr, wins_count_tr = fe.get_feature_over_windows_for_input_output(emg_train_filt, force_train_ft,
                                                                              self.window_size_in_samples,
                                                                              overlap_in_samples=self.overlap_in_samples,
                                                                              features=self.feature_list)
            input_val_features, output_val_force, wins_se_val, wins_count_val = fe.get_feature_over_windows_for_input_output(emg_test_filt, force_test_ft,
                                                                          self.window_size_in_samples,
                                                                          overlap_in_samples=self.overlap_in_samples,
                                                                          features=self.feature_list)
            logging.debug("X_train: %s  X_test: %s    y_train: %s   y_test: %s", input_train_features.shape, input_val_features.shape, output_train_force.shape, output_val_force.shape)
                
            if not self.multioutput:
                output_train_force = np.ravel(output_train_force)
                output_val_force = np.ravel(output_val_force)

            all_input_train_features.append(input_train_features)
            all_output_train_force.append(output_train_force)
            all_input_val_features.append(input_val_features)
            all_output_val_force.append(output_val_force)
            windows_se_for_finger[self.data_config.fing_id] = {'train': wins_se_tr, 'val': wins_se_val}
            windows_count_for_finger[self.data_config.fing_id] = {'train': wins_count_tr, 'val': wins_count_val}

        input_train = np.vstack(all_input_train_features)
        target_train = np.concatenate(all_output_train_force)
        input_val = np.vstack(all_input_val_features)
        target_val = np.concatenate(all_output_val_force)
        return input_train, target_train, input_val, target_val, shuffled_fingers_list, windows_se_for_finger, windows_count_for_finger


class ConventionalRegressionOnMUwithNoise(ConventionalLinearRegression):
    """
    This class implements a conventional regression model on muscle units (MUs) with noise on input during inference.
    """
    def __init__(self, linear_model: str, data_config: DataConfig, emg_type: str,
                 regression_data_parent_dir: str, overlap_in_perc: float,
                 post_process:bool,
                 snndata,
                 snn_dt:float,
                 window_size_in_sec: float,
                 select_dir: str = True,
                 load_regression_data_from_file:bool=False,
                 shuffle_fingers_seed:int=1,
                 percent_noise:float = 0,
                 noise_mode:str='omission'
                 ):
        ConventionalLinearRegression.__init__(self, linear_model, data_config,
                                              emg_type,
                                              regression_data_parent_dir, overlap_in_perc,
                                              window_size_in_sec,
                                              feature_type=INPUT_TYPE_SP_COUNT,
                                              select_dir=select_dir,
                                              post_process=post_process,
                                              shuffle_fingers_seed=shuffle_fingers_seed,
                                              percent_noise=percent_noise,
                                              noise_mode=noise_mode
                                              )
        self.regression_subdir = 'mu_regression_with_noise'
        self.regression_data_path = self._create_sub_directory(self.regression_subdir)
        self.fingers_list = list(data_config.finger_label_map.values())
        self.feature_type = INPUT_TYPE_SP_COUNT
        self.force_aux_cols = [REP_ID, TIME, FING_NAME_COL,FING_DIR, MVC_LVL]
        self.force_df = None
        self.sp_count_df = None
        self.noisy_sp_count_df = None
        self.load_regression_data_from_file = load_regression_data_from_file

        self.snndata = snndata
        self.snn_dt = snn_dt
        self.reg_active_fingers_order = self.snndata.active_fingers_order
        self.inputs = self.snndata.inputs_unsegmented
        self.labels = self.snndata.labels_unsegmented
        self.noisy_inputs = self.snndata.inputs_unsegmented_noisy if percent_noise > 0 else self.snndata.inputs_unsegmented
        self.n_mus = self.inputs.size(-1)
        self.rep_duration_in_samples = self.inputs.size(1)

        self.windows_start, self.windows_end = self.compute_windows_start_end_samples()
        
    def _set_reg_df(self, sp_count_df:pd.DataFrame, noisy_sp_count_df:pd.DataFrame, force_df:pd.DataFrame, ):
        self.force_df = force_df
        self.sp_count_df = sp_count_df
        self.noisy_sp_count_df = noisy_sp_count_df

    def _set_feature_windows(self, windows_se:tuple[np.ndarray, np.ndarray], windows_count:int):
        """
        Set the feature windows for the finger following the same format as in the EMG and MU class
        """
        for fold_i in range(2):
            self.windows_bounds_for_finger[fold_i] = {}
            self.windows_count_for_finger[fold_i] = {}
            for finger_id in self.reg_active_fingers_order[fold_i+1]:
                self.windows_bounds_for_finger[fold_i][finger_id] ={'train': windows_se, 'val':windows_se}
                self.windows_count_for_finger[fold_i][finger_id] = {'train': windows_count, 'val': windows_count}


    def compute_windows_start_end_samples(self):
        """
        Compute the start and end samples for sliding windows.
        """
        single_window_nsteps = int(self.window_size_in_sec / self.snn_dt)
        window_stride_nsteps = int((self.window_size_in_sec * (100 - self.overlap_in_perc) / 100) / self.snn_dt)
        windows_start = []
        windows_end = []
        win_start = 0
        while win_start + window_stride_nsteps <= self.rep_duration_in_samples:
            win_end = win_start + single_window_nsteps
            if win_end > self.rep_duration_in_samples:
                break
            windows_start.append(win_start)
            windows_end.append(win_end)
            win_start += window_stride_nsteps

        self._set_feature_windows(windows_se=(np.array(windows_start), np.array(windows_end)),
                                             windows_count=len(windows_start),
                                             )
        return windows_start, windows_end
    
    def prepare_regression_dataframes(self):
        """Count spikes in specified windows of the input tensor and average the force over same windows. 
        Return a DataFrame with organized entries per finger,
        repetition"""
        n_mus = self.inputs.size(-1)
        n_target_fingers = self.labels.size(-1)
        sp_count_df = pd.DataFrame()
        noisy_sp_count_df = pd.DataFrame()

        force_df = pd.DataFrame()
        for rep_id in [1,2]:
            shuffled_fingers_list = self.reg_active_fingers_order[rep_id]
            inputs_for_rep = self.inputs[(rep_id-1) * self.data_config.n_ind_fingers:rep_id * self.data_config.n_ind_fingers, :, :]
            noisy_inputs_for_rep = self.noisy_inputs[(rep_id-1) * self.data_config.n_ind_fingers:rep_id * self.data_config.n_ind_fingers, :, :]
            labels_for_rep = self.labels[(rep_id-1) * self.data_config.n_ind_fingers:rep_id * self.data_config.n_ind_fingers, :, :]

            print(f"Rep {rep_id}, retrieve inputs index: {(rep_id-1) * self.data_config.n_ind_fingers} - {rep_id * self.data_config.n_ind_fingers}")
            print(f"Inputs for rep shape: {inputs_for_rep.shape}, fingers: {shuffled_fingers_list}")
            for finger_task in range(inputs_for_rep.size(0)):
                sp_count_df_for_finger = pd.DataFrame(np.zeros((len(self.windows_start), n_mus)))
                noisy_sp_count_df_for_finger = pd.DataFrame(np.zeros((len(self.windows_start), n_mus)))

                force_df_for_finger = pd.DataFrame(np.zeros((len(self.windows_start), n_target_fingers)))

                inputs_for_finger = inputs_for_rep[finger_task]
                noisy_inputs_for_finger = noisy_inputs_for_rep[finger_task]
                print(f"Finger task {finger_task}, input shape: {inputs_for_finger.shape}")
                for win_i, (win_start, win_end) in enumerate(zip(self.windows_start, self.windows_end)):
                    sp_count_df_for_finger.iloc[win_i] = inputs_for_finger[win_start:win_end].sum(dim=0).numpy()
                    noisy_sp_count_df_for_finger.iloc[win_i] = noisy_inputs_for_finger[win_start:win_end].sum(dim=0).numpy()
                    force_df_for_finger.iloc[win_i] = labels_for_rep[finger_task, win_start:win_end].mean(dim=0).numpy()
                
                # append auxillary columns
                sp_count_df_for_finger[REP_ID] = rep_id
                sp_count_df_for_finger[FING_ID_COL] = shuffled_fingers_list[finger_task]
                sp_count_df = pd.concat([sp_count_df, sp_count_df_for_finger], axis=0)

                noisy_sp_count_df_for_finger[REP_ID] = rep_id
                noisy_sp_count_df_for_finger[FING_ID_COL] = shuffled_fingers_list[finger_task]
                noisy_sp_count_df = pd.concat([noisy_sp_count_df, noisy_sp_count_df_for_finger], axis=0)

                force_df_for_finger[REP_ID] = rep_id
                force_df_for_finger[FING_ID_COL] = shuffled_fingers_list[finger_task]
                force_df = pd.concat([force_df, force_df_for_finger], axis=0)

        self._set_reg_df(sp_count_df, noisy_sp_count_df,force_df)

    def cross_validate_model(self):
        """
        Perform cross-validation on the MU spike train data.
        """

        if self.sp_count_df.empty:
            print("No data to perform regression on.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        kf = KFold(n_splits=self.kcv, random_state=self.random_split_seed, shuffle=True)
        kfold_split = kf.split(self.sp_count_df[REP_ID].unique())
        aux_cols = [REP_ID, FING_ID_COL, FING_NAME_COL, TIME, MVC_LVL, FING_DIR]

        trained_models = []
        trained_models_df = pd.DataFrame()
        metrics_df_cv = pd.DataFrame()
        y_df_cv = pd.DataFrame()
        for fold_i, (train_rep, val_rep) in enumerate(kfold_split):
            val_rep = val_rep[0] + 1
            train_rep = train_rep[0] + 1
            input_train, target_train, input_val, target_val = self.prepare_training_validation_data(aux_cols,
                                                                                                     train_rep,
                                                                                                     val_rep)

            if self.normalization_type is not None:
                input_train, input_val = self.normalize_train_val_input(input_train,input_val)
            print(f"\nFitting on input_train rep {train_rep+1}: {input_train.shape}  y_train:{target_train.shape}")
            print(f"Evaluating on input_val rep {val_rep+1}: {input_val.shape}  y_val:{target_val.shape}")
            reg_model = self.initialize_regression_model()
            if input_train.shape[0] != target_train.shape[0]:
                min_samples = np.min([input_train.shape[0], target_train.shape[0]])
                input_train = input_train[:min_samples,:]
                target_train = target_train[:min_samples,:]
                print(f"Fitting model with {min_samples} samples due to mismatch in training input and target shapes.")
            if input_val.shape[0] != target_val.shape[0]:
                min_samples = np.min([input_val.shape[0], target_val.shape[0]])
                input_val = input_val[:min_samples,:]
                target_val = target_val[:min_samples,:]
                for finger in self.windows_count_for_finger[0].keys():
                    self.windows_count_for_finger[fold_i][finger]['val'] = min_samples
                print(f"Fitting model with {min_samples} samples due to mismatch in validation input and target shapes.")
            reg_model.fit(input_train, target_train)
            trained_models.append(reg_model)

            y_pred_val = reg_model.predict(input_val)
            y_pred_train = reg_model.predict(input_train)

            y_pred_val_post = None
            y_pred_train_post = None
            if self.post_process:
                y_pred_val_post = self.post_process_predictions(y_pred_val, self.post_process_cutoff,
                                                                self.sampling_freq)
                y_pred_train_post = self.post_process_predictions(y_pred_train, self.post_process_cutoff,
                                                                  self.sampling_freq)

            fold_metrics_df = m.prepare_convreg_metrics_df(y_pred_train,target_train,
                                                           y_pred_val, target_val,
                                                           val_rep,fold_i,self.window_size_in_sec,
                                                           self.reg_active_fingers_order[val_rep],
                                                           self.metrics_list,
                                                           y_pred_train_post, y_pred_val_post)
            fold_y_df = self.prepare_convreg_predictions_df(y_pred_train, y_pred_val,
                                                            target_train,
                                                            target_val,
                                                            val_rep, fold_i,
                                                            self.window_size_in_sec,
                                                            self.reg_active_fingers_order[val_rep],
                                                            y_pred_train_post, y_pred_val_post)
            
            fold_trained_model_df = self.prepare_trained_models_df(reg_model, self.model_type,
                                                                   val_rep, fold_i)
            metrics_df_cv = pd.concat([metrics_df_cv, fold_metrics_df], axis=0)
            y_df_cv = pd.concat([y_df_cv, fold_y_df], axis=0)
            trained_models_df = pd.concat([trained_models_df, fold_trained_model_df], axis=0)
        return metrics_df_cv, y_df_cv, trained_models_df

    def prepare_training_validation_data(self, aux_cols:list[str], train_rep:int, val_rep:int):
        """
        Prepare the training and validation data for the regression."""
        cols_to_drop = [c for c in aux_cols if c in self.sp_count_df.columns]

        input_train = self.sp_count_df[self.sp_count_df[REP_ID]== train_rep].drop(columns=cols_to_drop)
        input_val = self.noisy_sp_count_df[self.noisy_sp_count_df[REP_ID]==val_rep].drop(columns=cols_to_drop)

        cols_to_drop = [c for c in aux_cols if c in self.force_df.columns]
        target_val = self.force_df[self.force_df[REP_ID]==val_rep].drop(columns=cols_to_drop).to_numpy()
        target_train = self.force_df[self.force_df[REP_ID]==train_rep].drop(columns=cols_to_drop).to_numpy()

        return input_train, target_train, input_val, target_val


def initialize_scaler(normalization_type:str):
    """
    Initialize the scaler based on the normalization type."""
    if normalization_type=='zscore':
        return StandardScaler()
    if normalization_type == 'minmax':
        return MinMaxScaler()