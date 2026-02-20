import numpy as np
import pandas as pd


def calculate_ssc(signal):
    # Slope sign changes
    return (np.diff(np.sign(np.diff(signal))) != 0).sum()


def calculate_wamp(signal, threshold=0.01):
    # Willison Amplitude
    return (np.abs(np.diff(signal)) > threshold).sum()


def calculate_wl(signal):  
    # Waveform length
    return np.sum(np.abs(np.diff(signal)))

def calculate_zc(signal):
    # Zero crossing
    return ((signal[:-1] * signal[1:]) < 0).sum()

def calculate_rms(signal):
    # Root mean square
    return np.sqrt(np.mean(signal ** 2))

def calculate_mav(signal):
    # Mean absolute value
    return np.mean(np.abs(signal))

def calculate_var(signal):
    # Variance
    return np.var(signal)

def compute_feature_over_windows(input_data, windows_start:list[int],
                                 windows_end:list[int],
                                 is_force:bool=False,
                                 features: list = None):
    """
    Compute features over windows for the input data.
    """
    if features is None:
        features = ['rms', 'mav', 'var', 'zc', 'wamp', 'wl', 'ssc']

    feature_functions = {
        'rms': calculate_rms,
        'mav': calculate_mav,
        'var': calculate_var,
        'zc': calculate_zc,
        'wamp': calculate_wamp,
        'wl': calculate_wl,
        'ssc': calculate_ssc
    }

    feature_over_windows = []
    data_columns = np.arange(input_data.shape[0]) if not is_force else input_data.columns
    for channel in data_columns:
        window_features = []
        for win_start, win_end in zip(windows_start, windows_end):
            window = input_data[channel][win_start : win_end]
            feature_vector = [feature_functions[feat](window) for feat in features if feat in feature_functions]
            window_features.append(feature_vector)
        feature_over_windows.append(window_features)

    if len(features) == 1:
        feature_over_windows = np.array(feature_over_windows).squeeze(-1)
    else:
        feature_over_windows = np.array(feature_over_windows).reshape(len(data_columns), -1)
    return feature_over_windows.T


def get_feature_over_windows_for_input_output(emg:np.ndarray, force_mvc:pd.DataFrame,
                                              window_size_in_samples:int,
                                              overlap_in_samples:int,
                                              features:list = None):
    """
    Compute features over windows for input and output data.
    emg is of shape (n_channels, n_samples), force_mvc is a dataframe with columns as force sensors
    """
    assert len(force_mvc) == len(emg[0]), "Force and EMG should have the same length"
    step_size_in_samples = window_size_in_samples - overlap_in_samples
    windows_start = np.arange(0, len(force_mvc) - window_size_in_samples + 1, step_size_in_samples)
    windows_end = windows_start + window_size_in_samples

    emg_features = compute_feature_over_windows(emg, windows_start,
                                                windows_end,
                                                is_force=False,
                                                features=features)
    force_features = compute_feature_over_windows(force_mvc, windows_start,
                                                    windows_end,
                                                    is_force=True,
                                                    features=features)
    wins_se = (windows_start, windows_end)
    wins_count = len(windows_start)
    return emg_features, force_features, wins_se, wins_count
    # for _, emg_channel in enumerate(emg):
    #     channel_features = []
    #     for win_start in windows_start:
    #         win_end = win_start + window_size
    #         window = emg_channel[win_start : win_end]
    #         feature_vector = [feature_functions[feat](window) for feat in features if feat in feature_functions]
    #         channel_features.append(feature_vector)

    #     input_feature.append(channel_features)

    # for force_sensor in force_mvc.columns:
    #     force_features = []
    #     for win_start in windows_start: #range(0, len(force_mvc) - window_size + 1, step_size):
    #         win_end = win_start + window_size
    #         window_y_train = force_mvc[force_sensor][win_start : win_end]#[win_start:win_start + window_size]
    #         feature_vector_y = [feature_functions[feat](window_y_train) for feat in features if feat in feature_functions]
    #         force_features.append(feature_vector_y)
    #     output_feature.append(force_features)

    # if len(features) == 1:
    #     input_feature = np.array(input_feature).squeeze(-1)
    #     output_feature = np.array(output_feature).squeeze(-1)
    # else:
    #     input_feature = np.array(input_feature).reshape(len(emg), -1)
    #     output_feature = np.array(output_feature).reshape(force_mvc.shape[1], -1)

    # return input_feature.T, output_feature.T


def get_mean_sd_per_grid(emg_filt, segments):
    mean_emg_per_grid, sd_emg_per_grid = {}, {}
    for grid in range(len(segments)):
        mean_emg_per_grid[grid] =  np.mean(emg_filt[segments[grid][0]:segments[grid][1],:], axis=0)
        sd_emg_per_grid[grid] =    np.std(emg_filt[segments[grid][0]:segments[grid][1],:], axis=0)
    return mean_emg_per_grid, sd_emg_per_grid


def get_overlap_in_samp(win_size_in_samp:int, overlap_percentage: int):
    """
    Returns number of samples that are overlapped between two consecutive windows.
    """
    return int(overlap_percentage * win_size_in_samp/100)


def clip_feature_to_mean(emg_filt, segments, ch_neigh_dict):
    """
    Clips feature value that are "larger than 3x sd of the mean of all 64" electrodes in 
    a grid per time window.
    """
    # get each grid mean feat
    mean_emg_per_grid, sd_emg_per_grid = get_mean_sd_per_grid(emg_filt, segments)
    clipped_ch = False
    # check on each channel per grid if it exceeds the mean + 3*sd
    # CHECKTHIS: do we need to rectify? would this change the mean of the grid
    for grid in range(len(segments)):
        for ch in range(segments[grid][0],segments[grid][1],1):
            mask = emg_filt[ch,:] > (mean_emg_per_grid[grid] + 3 * sd_emg_per_grid[grid])
            if np.any(mask):
                clipped_ch = True
                # print(f'Clipping channel {ch} in grid {grid}')
                # get the relative channel id
                rel_ch_id = ch - segments[grid][0] + 1
                assert rel_ch_id in list(ch_neigh_dict.keys()), "channel not in dict"
                # get the neighbors of this channel and convert to absolute channel id
                neighbors = np.array(ch_neigh_dict[rel_ch_id]) + segments[grid][0] - 1
                print(f"Outlier channel {ch} in grid {grid} with neighbors {neighbors} {np.where(mask)[0].shape}")
                # replace the clipped channel with the mean of its neighbors
                mean_ch = emg_filt[neighbors,:].mean(axis=0)
                emg_filt[ch,np.where(mask)[0]] = mean_ch[np.where(mask)[0]]
    if clipped_ch:
        print(f'Clipped some channels')
    return emg_filt

