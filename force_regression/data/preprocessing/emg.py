import numpy as np
from scipy.signal import butter, filtfilt
import force_regression.utils.functions as fn

def create_butter_hp_coef(cutoff:float, fs:int, order=5):
    """
    Create the butterworth high-pass filter coefficients
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def apply_butter_highpass(data:np.ndarray, cutoff:float, fs:int, order=5):
    """
    Apply the butterworth high-pass filter to the data
    """
    b, a = create_butter_hp_coef(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def create_butter_lp_coef(cutoff:float, fs:int, order=5):
    """
    Create the butterworth low-pass filter coefficients
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_butter_lowpass(data:np.ndarray, cutoff:float, fs:int, order=5):
    """
    Apply the butterworth low-pass filter to the data
    """
    b, a = create_butter_lp_coef(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def convert_otb_to_thinfilm_mapping(emg_type:str):
    """
    Remaps the OTB electrode numbering to the ThinFilm electrode numbering.
    This utility function returns the correct ordereing of channels for each intra electrode/grid
    """
    if emg_type == 'intra':
        mapping = {
            1: 25, 2: 24, 3: 23, 4: 26, 5: 27, 6: 22, 7: 21, 8: 28, 9: 29, 10: 20,
            11: 19, 12: 30, 13: 31, 14: 18, 15: 17, 16: 32, 17: 33, 18: 16, 19: 15, 20: 34,
            21: 35, 22: 14, 23: 13, 24: 36, 25: 37, 26: 12, 27: 11, 28: 38, 29: 39, 30: 10,
            31: 9, 32: 40, 33: 5, 34: 8, 35: 7, 36: 6, 37: 3, 38: 2, 39: 1, 40: 4
        }
    elif emg_type == 'surf':
        mapping = {i: i for i in range(1, 65)}
    inv_mapping = {v: k-1 for k, v in mapping.items()}

    return inv_mapping


def filter_emg(emg:np.ndarray, emg_type:str,
               f_samp:int=10240, filter_order:int=5):
    """
    Filters the EMG signal using a high-pass and low-pass filter.
    """
    cutoff, cutoff_lp = get_emg_filters_cutoff_freq(emg_type)
    emg_filt = []
    for channel in emg:
        channel_filtered = apply_butter_highpass(channel, cutoff, f_samp, filter_order)
        channel_filtered = apply_butter_lowpass(channel_filtered, cutoff_lp, f_samp, filter_order)
        emg_filt.append(channel_filtered)
    return np.array(emg_filt)


def check_bad_channels(emg):
    """
    Check for bad channels and replace with zeros
    """

    # Calculate the global mean and standard deviation across all channels
    global_std = np.std(emg)
    std_threshold = global_std * 3

    for i, channel in enumerate(emg):
        channel_std = np.std(channel)
        if channel_std > std_threshold:
            emg[i] = np.zeros_like(channel)
    return emg

def get_emg_electrode_bounds(emg_type:str):
    """
    The channel bounds in each EMG electrode/grid.
    """
    intra_electrodes_bounds = [(0, 40), (40, 80), (80, 120)]
    surf_electrodes_bounds = [(0, 64), (64, 128), (128, 192)]
    if emg_type == 'intra':
        bounds = intra_electrodes_bounds
    elif emg_type == 'surf':
        bounds = surf_electrodes_bounds
    else:
        raise ValueError('emg_type must be either intra or surf')
    return bounds

def get_emg_filters_cutoff_freq(emg_type:str):
    """
    Returns the cutoff frequency for the high-pass and low-pass filters for the EMG signal.
    """
    if emg_type == 'intra':
        cutoff = 100
        cutoff_lp = 4400
    elif emg_type == 'surf':
        cutoff = 20
        cutoff_lp = 500
    else:
        raise ValueError('emg_type must be either intra or surf')
    return cutoff, cutoff_lp


def segment_emg_data(emg_type:str, emg_data_dict:dict[str,np.ndarray],
                     rep1_start:int, rep1_end:int,
                     rep2_start:int, rep2_end:int):
    """
    Segments the EMG data for the two repetitions
    """
    ch_map = convert_otb_to_thinfilm_mapping(emg_type)
    n_channels_per_electrode = fn.get_ch_count_for_emg(emg_type) # nchannles per electrode
    ordered_electrode_channels = [ch_map[i] for i in range(1, n_channels_per_electrode+1)]
    electrode_type_columns = [electrode for electrode in emg_data_dict.keys() if emg_type in electrode]
    n_electrodes = len(electrode_type_columns)

    data_for_emg_type = np.concatenate([emg_data_dict[electrode][ordered_electrode_channels] for electrode in electrode_type_columns], axis=0)
    emg_rep1 = data_for_emg_type[:, rep1_start: rep1_end]
    emg_rep2 = data_for_emg_type[:, rep2_start: rep2_end]
    assert emg_rep1.shape[0] == n_electrodes * n_channels_per_electrode, "EMG rep1 shape is not correct"
    assert emg_rep2.shape[0] == n_electrodes * n_channels_per_electrode, "EMG rep2 shape is not correct"
    return emg_rep1,emg_rep2