import logging
import pandas as pd
import numpy as np
from force_regression.data.loaders.force import load_data_and_segment_force
from force_regression.config.dataconfig import DataConfig
import force_regression.utils.functions as fn
from force_regression.data.preprocessing.emg import filter_emg, check_bad_channels, segment_emg_data
from configs.constants import FING_DIR, REP_ID, FING_NAME_COL, MVC_LVL, TIME
from force_regression.config.nomenclature import *


def create_singleoutput_force_df(config:DataConfig):
    """
    Prepares the force data for each finger task considering only the target finger.
    Singleoutput refers to loading only the measured forces for the target finger.
    """

    force_per_finger = {}
    if config.segment_hold:
        rep1_start, rep1_end, rep2_start, rep2_end = define_bounds_for_hold_segment(config, is_force_cut=True)
    
    # for _, config.fing_id in config.finger_label_map.items():
    for _, config.fing_id in {'thumb':1}.items():
        force_df = pd.DataFrame()
        for config.mvc in config.load_multi:
            # for config.direction in [FLEX, EXT]:
            for config.direction in [EXT]:
                assert config.sign_mvc == -1 if config.direction ==  EXT else 1, "Sign of MVC is not correct"
                force_dict_per_rep,_,_ = load_data_and_segment_force(config=config)
                force_df_for_direction = pd.DataFrame()
                for rep_id, rep_data in force_dict_per_rep.items():
                    if rep_data is not None:
                        normalized_cut_force = rep_data[FORCE]
                        force_df_for_direction = pd.concat([force_df_for_direction,
                                                  pd.DataFrame({rep_id: normalized_cut_force})],
                                                  ignore_index=True, axis=1)

                        # force_df_for_direction = force_df_for_direction.loc[rep_start:rep_end]
                        if config.verbose:
                            print(
                                f"rep_id:{rep_id} {fn.reverse_remap(config.fing_id, config.get_fingers())} force shape:{force_df_for_direction.shape} dur:{normalized_cut_force.shape[0] / config.f_samp} ")
                force_df_for_direction[FING_DIR] = config.direction
                force_df_for_direction[MVC_LVL] = config.mvc
                force_df = pd.concat([force_df, force_df_for_direction], axis=0, ignore_index=True)
                if config.verbose:
                    logging.debug("force direction df: %s %s\n", force_df_for_direction.shape, list(force_df_for_direction.columns))
                    logging.debug("force df: %s %s\n", force_df.shape, force_df.columns)
            force_per_finger[fn.reverse_remap(config.fing_id, config.get_fingers())] = force_df
    return force_per_finger

#TODO: check if I can use the same function for both single and multioutput
def create_multioutput_force_df(config:DataConfig):
    """
    Prepares the force data considering all fingers, directions and mvc levels.
    Multioutput refers to loading all 5 fingers for each target task and not only the
    measured forces for the target finger.
    """
    force_df = pd.DataFrame()
    rep1_id_one_indexed, rep2_id_one_indexed = 1, 2
    for config.mvc in config.load_multi:
        for config.sign_mvc in [-1,1]:
        # for config.sign_mvc in [-1]:
            force_direction = FLEX if config.sign_mvc==1 else EXT
            for _, config.fing_id in config.finger_label_map.items():
            # for _, config.fing_id in {'thumb':1}.items():
                logging.debug("Loading finger %s", config.fing_id)
                _, forces_in_perc_mvc, emg_data_dict = load_data_and_segment_force(config=config)
                force_rep_id_1, force_rep_id_2 = prepare_data_for_mvc_and_dir(config, emg_data_dict,
                                                                              forces_in_perc_mvc,
                                                                              prepare_emg=False)
                force_rep_id_1[REP_ID] = rep1_id_one_indexed - 1
                force_rep_id_2[REP_ID] = rep2_id_one_indexed - 1

                force_rep_id_1[FING_DIR] = force_direction
                force_rep_id_2[FING_DIR] = force_direction

                force_rep_id_1[MVC_LVL] = config.mvc
                force_rep_id_2[MVC_LVL] = config.mvc
                
                print(fn.reverse_remap(config.fing_id, config.get_fingers()))
                force_rep_id_1[FING_NAME_COL] = fn.reverse_remap(config.fing_id, config.get_fingers())
                force_rep_id_2[FING_NAME_COL] = fn.reverse_remap(config.fing_id, config.get_fingers())
                
                force_df = pd.concat((force_df, force_rep_id_1, force_rep_id_2))
    return force_df


def add_aux_cols_to_force_df(data_config,
                              force_df: pd.DataFrame,
                              direction: str, rep_id: int, mvc_lvl: int,
                              finger_name: str) -> pd.DataFrame:
    """
    Add the auxiliary columns to the force dataframe.
    """
    force_df[REP_ID] = rep_id
    force_df[FING_DIR] = direction
    force_df[MVC_LVL] = mvc_lvl
    force_df[FING_NAME_COL] = finger_name
    force_df[TIME] = np.arange(0, force_df.shape[0] / data_config.f_samp, 1/data_config.f_samp)
    return force_df
    
def organize_forces_dict_into_df(data_config, forces_in_perc_mvc,
                                 fing_name, mvc_lvl, direction)->pd.DataFrame:
    """
    Organizes the forces dict of dicts into a DataFrame
    """
    force_df = pd.DataFrame()
    # for fing_name in forces_dict.keys():
        # for mvc_lvl in forces_dict[fing_name].keys():
            # for direction in forces_dict[fing_name][mvc_lvl].keys():
    force_df_rep1, force_df_rep2 = prepare_data_for_mvc_and_dir(data_config, None,
                                                                forces_in_perc_mvc,
                                                                prepare_emg=False)
    force_df_rep1 = add_aux_cols_to_force_df(data_config, force_df_rep1, direction, 0, mvc_lvl, fing_name)
    force_df_rep2 = add_aux_cols_to_force_df(data_config,force_df_rep2, direction, 1, mvc_lvl, fing_name)
    force_df = pd.concat((force_df, force_df_rep1, force_df_rep2))
    return force_df


def prepare_data_for_mvc_and_dir(config, emg_data_dict, forces_in_perc_mvc,
                                 emg_type:str='intra', order=5,
                                 prepare_emg:bool=True):
    """
    Loads and filters both repetitions. Segments the data.
    Returns emg and force data for rep1 and rep2.
    """
    rep1_start, rep1_end = config.rep1_s, config.rep1_e
    # FIXTHIS: rep2_s_cut is empty. This would be crucial when using the whole force profile
    # This is likely to rely on rep2_s and rep2_e
    rep2_start, rep2_end = config.rep2_s, config.rep2_e
    if config.segment_hold:
        rep1_start, rep1_end, rep2_start, rep2_end = define_bounds_for_hold_segment(config)
    logging.info("Segmenting data for rep1 from %s to %s and rep2 from %s to %s", rep1_start/config.f_samp, rep1_end/config.f_samp, rep2_start/config.f_samp, rep2_end/config.f_samp)
    force_df_rep1, force_df_rep2 = create_force_df(config, forces_in_perc_mvc,
                                                   rep1_start, rep1_end,
                                                   rep2_start, rep2_end)
    
    if prepare_emg:
        emg_rep1_filt, emg_rep2_filt = prepare_emg_repetitions(config, emg_type, order,
                                                               emg_data_dict,
                                                               rep1_start, rep1_end,
                                                               rep2_start, rep2_end)
        return emg_rep1_filt, emg_rep2_filt, force_df_rep1, force_df_rep2
    return force_df_rep1, force_df_rep2

def define_bounds_for_hold_segment(config:DataConfig, is_force_cut:bool=False):
    """
    Defines the start and end points for the hold segment for both repetitions.
    These are the points with which the force and EMG will be segmented.
    """
    # when segmenting emg we use the global time axis; i.e. on the uncut scale
    if is_force_cut:
        rep1_start = config.rep1_force_cut[LANDMARK].end_rise
        rep1_end =  config.rep1_force_cut[LANDMARK].start_fall
        rep2_start = config.rep2_force_cut[LANDMARK].end_rise
        rep2_end = config.rep2_force_cut[LANDMARK].start_fall
    else:
        rep1_start = config.rep1_force_uncut[LANDMARK].end_rise
        rep1_end =  config.rep1_force_uncut[LANDMARK].start_fall
        rep2_start = config.rep2_force_uncut[LANDMARK].end_rise
        rep2_end = config.rep2_force_uncut[LANDMARK].start_fall
    return rep1_start, rep1_end, rep2_start, rep2_end


def prepare_emg_repetitions(config:DataConfig, emg_type:str, order:int,
                            emg_data_dict:dict[str,np.ndarray],
                            rep1_start:int, rep1_end:int,
                            rep2_start:int, rep2_end:int):
    """
    Segments the EMG data for the two repetitions in the region of interest
    and filters the data.
    """
    emg_rep1, emg_rep2 = segment_emg_data(emg_type, emg_data_dict,
                                        rep1_start, rep1_end,
                                        rep2_start, rep2_end)
    emg_rep1_filt = filter_emg(emg_rep1, emg_type, f_samp=config.f_samp, filter_order=order)
    emg_rep2_filt = filter_emg(emg_rep2, emg_type, f_samp=config.f_samp, filter_order=order)
    emg_rep1_filt = check_bad_channels(emg_rep1_filt)
    emg_rep2_filt = check_bad_channels(emg_rep2_filt)
    return emg_rep1_filt,emg_rep2_filt



def create_force_df(config:DataConfig, forces_in_perc_mvc:np.ndarray,
                    rep1_start:int, rep1_end:int,
                    rep2_start:int, rep2_end:int):
    """
    Organizes the force data for both repetitions in DataFrames.
    Force df are organized in columns for each sensor and time in rows.
    Columns are the relevant sensors for the task direction and TIME. 
    For example, for flexion task, the columns are the flexion sensors.
    """
    # num_fingers = config.n_ind_fingers
    first_ext_sensor_id = config.first_extension_sensor_id
    first_flex_sensor_id = config.first_flexion_sensor_id
    n_ext_sensors = config.n_extension_sensors
    n_flex_sensors = config.n_flexion_sensors
    if config.direction == EXT:
        force_rep1 = forces_in_perc_mvc[first_ext_sensor_id:n_ext_sensors, rep1_start: rep1_end]
        force_rep2 = forces_in_perc_mvc[first_ext_sensor_id:n_ext_sensors, rep2_start: rep2_end]
    else:
        force_rep1 = forces_in_perc_mvc[first_flex_sensor_id: first_flex_sensor_id + n_flex_sensors, rep1_start: rep1_end]
        force_rep2 = forces_in_perc_mvc[first_flex_sensor_id: first_flex_sensor_id + n_flex_sensors, rep2_start: rep2_end]
    force_df_rep1, force_df_rep2 = pd.DataFrame(), pd.DataFrame()
    force_df_rep1[config.force_cols_list] = force_rep1.T[:,:config.n_ind_fingers]
    force_df_rep1[TIME] = np.arange(0, force_df_rep1.shape[0]/config.f_samp, 1/config.f_samp)
    
    force_df_rep2[config.force_cols_list] = force_rep2.T[:,:config.n_ind_fingers]
    force_df_rep2[TIME] = np.arange(0, force_df_rep2.shape[0]/config.f_samp, 1/config.f_samp)
    return force_df_rep1, force_df_rep2

