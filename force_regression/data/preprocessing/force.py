import logging
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
import force_regression.utils.functions as fn
from force_regression.config.dataconfig import DataConfig
from configs.constants import *


def downsample_force(force_mvc:pd.DataFrame,
                     dt:float,
                     config:DataConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Downsample the force to match the resolution of simulation dt
    """
    current_force = force_mvc.dropna()  # dropping nan values in case this repetition is slightly shorter than the other
    tot_dur = len(current_force) / config.f_samp
    logging.debug("tot_dur: %s", tot_dur)

    # create a time vector for the force over which the binning will be performed
    force_time_ax = np.arange(0, current_force.shape[0] / config.f_samp, 1 / config.f_samp)
    # Average the force to match len of X.
    bin_range = np.arange(0, tot_dur + dt, dt) # number of "ticks"

    if current_force.shape[1] > 1:
        force_avg_df = pd.DataFrame()
       # bin each output unit independently
        for output in range(current_force.shape[1]):
            force_vect = current_force.iloc[:, output]
            ouput_force_avg, _, _ = binned_statistic(force_time_ax, values=force_vect, bins=bin_range, statistic='mean')
            force_avg_df[f'force_{output}'] = ouput_force_avg
        force_avg = force_avg_df.values
    else:
        force_vect = np.ravel(current_force)
        force_avg, _, _ = binned_statistic(force_time_ax, values=force_vect, bins=bin_range, statistic='mean')
    return force_avg[:-1], bin_range[:-1]


def compute_mean_force_over_windows(force_df:pd.DataFrame, config:DataConfig,
                                    window_size_in_sec:float,
                                    overlap_ratio:float):
    """
    force_df: time x 5 columns (one for each loadcell)
    """
    duration = len(force_df) / config.f_samp
    wins_start_times = fn.compute_windows_start_times(window_size_in_sec, overlap_ratio, duration)
    force_df_copy = force_df.copy()
    force_df_copy[TIME] = np.arange(0, duration, 1/config.f_samp)
    binned_force_ar = np.empty((len(wins_start_times), len(force_df.columns)))
    for fing_force in force_df.columns:
        for i, win_st in enumerate(wins_start_times):
            win_et = win_st + window_size_in_sec
            force_bin = force_df_copy[(force_df_copy[TIME] >= win_st) & (force_df_copy[TIME] < win_et)][fing_force]
            binned_force_ar[i, fing_force] = force_bin.mean()
    return binned_force_ar, wins_start_times


def downsample_signal(signal, window_size):
    num_windows = signal.shape[1] // window_size
    downsampled = np.array([signal[:, i*window_size:(i+1)*window_size].mean(axis=1) for i in range(num_windows)]).T
    x_axis_downsampled = np.arange(window_size / 2, signal.shape[1], window_size)
    return downsampled, x_axis_downsampled

