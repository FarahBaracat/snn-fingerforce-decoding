import pickle as pkl
import numpy as np
import pandas as pd

def remap_subject(subject, mapping):
    """ Remap a subject name to its anonymous ID. """
    return mapping.get(subject, "Unknown")


def reverse_remap(anonymous_id, mapping):
    """ Reverse remap an anonymous ID to its original subject name. """
    reverse_mapping = {v: k for k, v in mapping.items()}
    return reverse_mapping.get(anonymous_id, "Unknown")


def get_ch_count_for_emg(emg_type):
    """
    Returns the number of channels for the given EMG type.
    """
    if emg_type == 'intra':
        return 40
    elif emg_type == 'surf':
        return 64
    else:
        raise ValueError(f"Unknown EMG type: {emg_type}")

def grid_layout_func(emg_type):
    """
    Surface grids used are the 13 x 5 with 4 mm inter-electrode distance.
    The top left corner is missing one channel for a total of 64 channels.
    """
    if emg_type == 'surf':
        return 13, 5
    else:
        return 20, 2


def intra_grid_layout():
    """
    40 channels
    """
    n_cols = 2
    n_rows = 20
    return n_rows, n_cols


def recorded_mvc_per_sensor(config)->np.ndarray:
    """
    Lists of recorded maximal voluntary contractions (MVC) for each sensor.
    The first 5 sensors are extension and the last 5 are flexion.
    """
    if remap_subject(config.subject, config.subj_map) == 'S1':
        rec_mvcs = [0.081, 0.061, 0.049, 0.035, 0.049,
                0.208, 0.188, 0.165, 0.152, 0.179]
    elif remap_subject(config.subject, config.subj_map) == 'S2':
        rec_mvcs = [0.079, 0.048, 0.025, 0.023, 0.060,
                0.337, 0.200, 0.088, 0.058, 0.025]
    else:
        raise ValueError(f"Unknown subject: {config.subject}")
    return np.array(rec_mvcs).reshape(-1, 1)


def get_baselines_trap_for_pseudo(config):
    if remap_subject(config.subject, config.subj_map) == 'S1':
        baselines = [-2.791, -2.814, -2.701, -2.806, -0.704,
                     -2.797, -2.834, -0.655, -2.707, 0.465]
    elif remap_subject(config.subject, config.subj_map) == 'S2':
        baselines = [-2.786, -2.806, -2.692, -2.795, -0.689,
                     -2.803, -2.824, -0.636, -2.698, 0.456]
    else:
        raise ValueError(f"Unknown subject: {config.subject}")
    return baselines


#### surface grid layout 13 x 5
def find_key_from_ch(ch, ch_group_per_row):
    for key, value in ch_group_per_row.items():
        if ch in value:
            return key
    return None

def layout_channels(emg_type):

    if emg_type == 'surf':
        ch_group_per_row = {0: [-1, 25, 26, 51, 52],   # top to bottom
                            1: [1, 24, 27, 50, 53],
                            2: [2, 23, 28, 49, 54],
                            3: [3, 22, 29, 48, 55],
                            4: [4, 21, 30, 47, 56],
                            5: [5, 20, 31, 46, 57],
                            6: [6, 19, 32, 45, 58],
                            7: [7, 18, 33, 44, 59],
                            8: [8, 17, 34, 43, 60],
                            9:  [9, 16, 35, 42, 61],
                            10: [10, 15, 36, 41, 62],
                            11: [11, 14, 37, 40, 63],
                            12: [12, 13, 38, 39, 64]}
    else:
        ch_group_per_row = {0: [1, 2],
                            1: [3, 4],
                            2: [5, 6],
                            3: [7, 8],
                            4: [9, 10],
                            5: [11, 12],
                            6: [13, 14],
                            7: [15, 16],
                            8: [17, 18],
                            9: [19, 20],
                            10: [21, 22],
                            11: [23, 24],
                            12: [25, 26],
                            13: [27, 28],
                            14: [29, 30],
                            15: [31, 32],
                            16: [33, 34],
                            17: [35, 36],
                            18: [37, 38],
                            19: [39, 40]}
    return ch_group_per_row


def reshape_sta_grid(emg_type, sta,  window):
    if sta.shape[1] > sta.shape[0]:
        sta = sta.T

    channel_layout = layout_channels(emg_type)
    channel_order = [ch-1 for group in channel_layout.values() for ch in group]
    reshaped_array = sta[:, channel_order].T
    n_rows, n_cols = grid_layout_func(emg_type)
    sta = reshaped_array.reshape(n_rows, n_cols, window)

    return sta 

def get_grid_layout(emg_type) -> dict:
    """
    A layout of a single grid. Connector is facing down and is on the bottom.
    """
    n_rows, n_cols = grid_layout_func(emg_type)
    ch_group_per_row = layout_channels(emg_type)
    
    assert (list(ch_group_per_row.keys()) == np.arange(n_rows)).all()
    
    ch_group_per_col = {}
    for col in range(n_cols):
        ch_group_per_col[col] = [ch_group_per_row[row][col] for row in ch_group_per_row.keys()]


    grid_layout = {'ch_group_per_row':ch_group_per_row,
                   'ch_group_per_col': ch_group_per_col,
                  }
    return grid_layout


def create_ch_neighbors_dict(emg_type:str)->dict:
    """
    Gets the neighboring channels for each channel in the grid (in case of surface grid) or 
    electrode (in case of intra).
    """
    n_ch_per_grid =  get_ch_count_for_emg(emg_type)
    grid_layout = get_grid_layout(emg_type)
    neighboring_dict = {}
    n_rows, n_cols = grid_layout_func(emg_type)

    if emg_type == 'surf':
        edges_ch = [grid_layout['ch_group_per_row'][0],         # top
                    grid_layout['ch_group_per_row'][n_rows-1],  # bottom
                    grid_layout['ch_group_per_col'][0],         # left
                    grid_layout['ch_group_per_col'][n_cols-1]]  # right

        for ch in range(1, n_ch_per_grid+1, 1):
            # get the corresponding row and col to this ch
            row = find_key_from_ch(ch, grid_layout['ch_group_per_row'])
            col = np.where(
                np.array(grid_layout['ch_group_per_row'][row]) == ch)[0][0]
            # if ch in corner_ch:
            if ch == 52:
                neighboring_dict[ch] = [51, 53]
            elif ch == 12:
                neighboring_dict[ch] = [11, 13]
            elif ch == 64:
                neighboring_dict[ch] = [39, 63]

            elif ch in np.concatenate(edges_ch):
                if ch in edges_ch[0]:  # top edge
                    neighboring_dict[ch] = [
                        grid_layout['ch_group_per_row'][row][col-1],
                        grid_layout['ch_group_per_row'][row][col+1],
                        grid_layout['ch_group_per_row'][row+1][col]
                    ]

                elif ch in edges_ch[1]:  # bottom edge
                    neighboring_dict[ch] = [
                        grid_layout['ch_group_per_row'][row-1][col],
                        grid_layout['ch_group_per_row'][row-1][col+1],
                        grid_layout['ch_group_per_row'][row-1][col-1]]
                elif ch in edges_ch[2]:  # left edge
                    neighboring_dict[ch] = [grid_layout['ch_group_per_row'][row][col+1],
                                            grid_layout['ch_group_per_row'][row-1][col],
                                            grid_layout['ch_group_per_row'][row+1][col],

                                            ]
                elif ch in edges_ch[3]:  # right edge
                    neighboring_dict[ch] = [grid_layout['ch_group_per_row'][row][col-1],
                                            grid_layout['ch_group_per_row'][row-1][col],
                                            grid_layout['ch_group_per_row'][row+1][col],
                                            ]
            else:

                neighboring_dict[ch] = [grid_layout['ch_group_per_row'][row][col-1],
                                        grid_layout['ch_group_per_row'][row][col+1],
                                        grid_layout['ch_group_per_row'][row-1][col],
                                        grid_layout['ch_group_per_row'][row+1][col],
                                        ]
    elif emg_type == 'intra':
        edges_ch = [grid_layout['ch_group_per_row'][0],         # top
                    grid_layout['ch_group_per_row'][n_rows-1],  # bottom
                    grid_layout['ch_group_per_col'][0],         # left
                    grid_layout['ch_group_per_col'][n_cols-1]]  # right
        
        for ch in range(1, n_ch_per_grid+1, 1):
            # get the corresponding row and col to this ch
            row = find_key_from_ch(ch, grid_layout['ch_group_per_row'])
            col = np.where(np.array(grid_layout['ch_group_per_row'][row]) == ch)[0][0]

            # the neighbours in the intramuscular are only the up and down channels
            neighboring_dict[ch] = [grid_layout['ch_group_per_row'][row - 1][col] if row - 1 >= 0 else -1,
                grid_layout['ch_group_per_row'][row + 1][col] if row + 1 < len(grid_layout['ch_group_per_row']) else -1]

    else:
        raise ValueError(f"Unknown EMG type: {emg_type}")
    return neighboring_dict


def save_to_pickle(data, file_path:str, is_dataframe:str=False):
    """
    Save data to a pickle file.
    """
    if is_dataframe:
        data.to_pickle(file_path)
    else:
        with open(file_path,'wb') as h:
            pkl.dump(data, h, protocol = pkl.HIGHEST_PROTOCOL)

def load_from_pickle(file_path:str, is_dataframe:bool=False):
    """
    Load data from a pickle file.
    """
    if is_dataframe:
        return pd.read_pickle(file_path)
    
    try:
        with open(file_path, 'rb') as h:
            return pkl.load(h)
    except:  # due to pickle incompatibility with pandas version
        print(f"Probably this file contains a df. Please double check! In the meanwhile, I will load to df...")
        return pd.read_pickle(file_path)
    


def compute_windows_start_times(window_size_in_sec:float, overlap_ratio:float,
                                total_rep_duration:float):
    """
    Calculate the start times of the bins given the bin width, 
    overlap percentage and the total duration.
    """
    stride = window_size_in_sec * (1 - overlap_ratio)
    start_ar = np.arange(0, total_rep_duration - window_size_in_sec, stride)
    if start_ar[-1] + stride + window_size_in_sec <= total_rep_duration:
        start_ar = np.append(start_ar, start_ar[-1] + stride)
    return start_ar



def convert_to_mvc(data:np.ndarray)->np.ndarray:
    """
    Convert the data to %MVC.
    """
    return data * 100