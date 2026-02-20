import os
import pandas as pd
import numpy as np
import torch
import re
from force_regression.config.dataconfig import DataConfig, default_electrodes
from force_regression.data.loaders.decomposition import load_pickle
from force_regression.data.loaders.force import load_data_and_segment_force
from force_regression.data.handlers.force_handler import organize_forces_dict_into_df
from force_regression.data.preprocessing.mu_quality import post_process_units, post_process_units_tracking
from force_regression.plotting.time_series_plot import plot_force_before_after_cut
from force_regression.config.nomenclature import *
import force_regression.data.preprocessing.spikes as spk
import force_regression.utils.functions as fn
from configs.constants import *


def init_empty_dfs(columns_list:list[str]):
    """
    Initializes empty dataframes for mu and mu summary data.
    """
    # columns = [SP_TIME, MU_ID, MU_RATE, ELEC_NAME, FING_NAME_COL, SP_COUNT, START_TIME, END_TIME]
    # if df_type =='mu_df':
    init_df = pd.DataFrame(columns=columns_list)
    # elif df_type =='mu_summary_df':
        # init_df = pd.DataFrame(columns=default_electrodes() + [FING_DIR])
    # else:
        # init_df = pd.DataFrame()
    return init_df

def compute_isi_for_all_mus(config:DataConfig,
                            sp_times_for_all_mus:list[torch.tensor])->list[torch.tensor]:
    """
    Computes the ISI for a list of lists. Each element in the first list is a list of spike times for a single MU.
    """
    isi_for_all_mus = [np.round(torch.cat((torch.tensor([0]), torch.diff(u))).detach().numpy() / config.f_samp, 2) * 1000
                       for u in sp_times_for_all_mus]

    return isi_for_all_mus


def create_mudf_row_for_electrode_finger(config: DataConfig,
                                        units: list[torch.tensor],
                                        contents: dict,
                                        fing_name: str,
                                        rep_id: int,
                                        n_mu: int,
                                        last_indexed_mu: int)->pd.DataFrame:
    """
    Creates a row in the dataframe for
    """

    isi_for_all_mus = compute_isi_for_all_mus(config, units)
    if config.task_type == TRAP:
        df_row = pd.DataFrame({
            SP_TIME: units,
            # TODO: remove this line after the data is redecomposed
            FILTERS: contents['filters'] if 'filters' in contents.keys() else None,
            MU_ID: np.arange(last_indexed_mu, last_indexed_mu + n_mu),
            MU_RATE: contents[FR_FIRST_REP] if rep_id == 0 else contents[FR_SECOND_REP],
            ISI: isi_for_all_mus,
            ELEC_NAME: config.electrode_name,
            FING_NAME_COL: np.repeat(fing_name, n_mu),
            REP_ID: rep_id,
            FING_DIR: config.direction,
            # CHECKTHIS: shouldnt START_TIME for rep 0 be 0 since the spikes are already shifted considering rep1_s
            START_TIME: [(getattr(config, f'rep{int(rep_id + 1)}_s') - config.time_removed) / config.f_samp if rep_id == 1 else getattr(config, f'rep{int(rep_id + 1)}_s') / config.f_samp][0],
            END_TIME: [(getattr(config, f'rep{int(rep_id + 1)}_e') - config.time_removed) / config.f_samp if rep_id == 1 else getattr(config, f'rep{int(rep_id + 1)}_e') / config.f_samp][0],
            START_RAMP: (config.points_cut[0 + rep_id*4]) / config.f_samp,
            START_HOLD: (config.points_cut[1 + rep_id*4]) / config.f_samp,
            END_HOLD:   (config.points_cut[2 + rep_id*4]) / config.f_samp,
            END_RAMP:   (config.points_cut[3 + rep_id*4]) / config.f_samp,
            MVC_LVL: config.mvc,

        })
    else:
        df_row = pd.DataFrame({
                    SP_TIME: units,
                    MU_ID: np.arange(last_indexed_mu, last_indexed_mu + n_mu),
                    MU_RATE: contents[FR_FIRST_REP] if rep_id == 0 else contents[FR_SECOND_REP],
                    ISI: isi_for_all_mus,
                    ELEC_NAME: config.electrode_name,
                    FING_NAME_COL: np.repeat(fn.reverse_remap(config.fing_id, config.get_fingers()), n_mu),
                    REP_ID: rep_id,
                    FING_DIR: config.direction,
                    START_TIME: getattr(config, f'rep{int(rep_id + 1)}_s') / config.f_samp,
                    END_TIME: getattr(config, f'rep{int(rep_id + 1)}_e')  / config.f_samp,
                    MVC_LVL: config.mvc,

                })
    return df_row
    
def count_active_units_in_rep(rep_units: list[torch.tensor])->int:
    """
    Counts the number of active units in a repetition
    """
    n_active_units = 0
    for i in rep_units:
        if len(i) > 0:
            n_active_units += 1
    return n_active_units

def load_and_postprocess_decomposed_mus(file_path:str, config:DataConfig, last_mu_id:int):
    """
    Loads the decomposed units from file and post-process
    """
    contents = load_pickle(file_path)
    contents = spk.sort_by_spike_times(contents)

    contents, units_rep1, units_rep2, units_for_reps = post_process_units(mu_dict=contents, config=config)
    n_active_mus_rep1 = count_active_units_in_rep(units_rep1)
    n_active_mus_rep2 = count_active_units_in_rep(units_rep2)
    assert len(units_rep1)==len(units_rep2), "Number of units in rep1 and rep2 should be the same"
    total_active_mus = len(units_rep1)
    mu_data = []
    fing_name = fn.reverse_remap(config.fing_id, config.get_fingers())
    for rep_id, units in enumerate(units_for_reps, start=0):
        if units:
            df_row = create_mudf_row_for_electrode_finger(config, units, contents,
                                                         fing_name, int(rep_id),
                                                         total_active_mus, last_mu_id)
            mu_data.append(df_row)
    del contents
    mu_df = pd.concat(mu_data, ignore_index=True) if mu_data else pd.DataFrame()
    return mu_df, n_active_mus_rep1, n_active_mus_rep2, total_active_mus

def get_electrode_filenames_for_task(config:DataConfig)->list[str]:
    """
    Gets all electrode files corresponding to the task. Each file from surf and intra MUs is decomposed
    separately.
    """
    electrodes_files = [f for f in os.listdir(config.decomp_dir) if config.task_name() in f]
    return electrodes_files



def load_mu_data_to_df(config: DataConfig)->pd.DataFrame:
    """
    Loads decomposed motor units to DataFrame.
    TODO: add the description of the mu_df columns (e.g GLOB_MU_ID vs MU_ID, SP_COUNT, etc)
        GLOB_MU_ID: is a global motor id untracked across reps.
    """
    mu_df_columns = [SP_TIME, MU_ID, MU_RATE, ELEC_NAME, FING_NAME_COL, SP_COUNT, START_TIME, END_TIME]
    mu_df_sum_columns = default_electrodes() + [FING_DIR]

    mu_df, mu_summary_df = init_empty_dfs(mu_df_columns), init_empty_dfs(mu_df_sum_columns)
    last_mu_id = 0
    counter_plot = 0
    initial_sign_mvc = config.sign_mvc
    initial_emg_type = config.emg_type

    # loaded_forces_dict = {}
    combined_forces_df = pd.DataFrame()
    for _, config.fing_id in config.finger_label_map.items():
        fing_name = fn.reverse_remap(config.fing_id, config.get_fingers())
        # loaded_forces_dict[fing_name] = {}
        for config.mvc in config.load_multi:
            # loaded_forces_dict[fing_name][config.mvc] = {}
            
            for config.direction in [FLEX, EXT]:
                finger_electrode_summary_df = init_empty_dfs(mu_df_sum_columns)
                assert config.sign_mvc == -1 if config.direction == EXT else 1, "Sign of MVC is not correct"
                electrodes_files = get_electrode_filenames_for_task(config)
                _,forces_in_perc_mvc,_ = load_data_and_segment_force(config=config)
                print(f"{fing_name} | {config.direction} | start rep1 {config.rep1_s/config.f_samp}")
                # loaded_forces_dict[fing_name][config.mvc][config.direction] = forces_in_perc_mvc
                
                # force_df = self.organize_forces_dict_into_df(forces_dict)
                force_df = organize_forces_dict_into_df(config, forces_in_perc_mvc,
                                                        fing_name, config.mvc, config.direction)
                combined_forces_df = pd.concat([combined_forces_df, force_df], ignore_index=True)

                if counter_plot == 0:
                    if config.task_type == TRAP:
                        plot_force_before_after_cut(config)
                for file in electrodes_files:
                    file_path = os.path.join(config.decomp_dir, file).replace("\\", "/")
                    config.electrode_name = next((e for e in config.electrodes if e in file), None)
                    if config.electrode_name:
                        electrode_mu_df, n_mu_rep1, n_mu_rep2, n_mu = load_and_postprocess_decomposed_mus(file_path,
                                                                                                          config,
                                                                                                          last_mu_id)
                        mu_df = pd.concat([mu_df, electrode_mu_df], ignore_index=True)
                        finger_electrode_summary_df.loc[fing_name, config.electrode_name] = [n_mu_rep1, n_mu_rep2]
                        last_mu_id += n_mu
                    else:
                        print(f"Electrode name not found in file: {file_path}")
                counter_plot += 1
                finger_electrode_summary_df[FING_DIR] = config.direction
                finger_electrode_summary_df[MVC_LVL] = config.mvc
                mu_summary_df = pd.concat([mu_summary_df, finger_electrode_summary_df], ignore_index=True)

    tot_1 = mu_summary_df.ext_digitorum.map(lambda x: x[0]) + mu_summary_df.ext_pollicis.map(lambda x: x[0]) + mu_summary_df.flx_digitorum.map(lambda x: x[0])
    tot_2 = mu_summary_df.ext_digitorum.map(lambda x: x[1]) + mu_summary_df.ext_pollicis.map(lambda x: x[1]) + mu_summary_df.flx_digitorum.map(lambda x: x[1])

    mu_summary_df[TOTAL_MUS_REP1] = tot_1
    mu_summary_df[TOTAL_MUS_REP2] = tot_2
    mu_df = spk.add_first_spike_time_col(mu_df)
    mu_df_sorted = spk.sort_rows_by_first_spike_time_per_rep(mu_df)

    mu_df_sorted[SP_COUNT] = mu_df_sorted[SP_TIME].apply(lambda x: len(x))
    mu_df_sorted[GLOB_MU_ID] = np.arange(mu_df_sorted.shape[0])
    config.reset_sign_emg_type(initial_sign_mvc, initial_emg_type)
    return mu_df, mu_summary_df, mu_df_sorted, combined_forces_df


def load_mu_data_to_df_tracking(config: DataConfig):
    mu_df, mu_summary_df = init_empty_dfs()
    last_mu_id = 0

    counter_plot = 0
    for _, config.fing_id in config.finger_label_map.items():
        for config.direction in ['flex', 'ext']:
            _, temp_df = init_empty_dfs()
            
            assert config.sign_mvc == -1 if config.direction == 'ext' else 1, "Sign of MVC is not correct"
            elec_files = [f for f in os.listdir(os.path.join(config.decomp_dir, "rep1")) if config.task_name() in f]

            _,_,_= load_data_and_segment_force(config=config)

            if counter_plot == 0:
                if config.task_type == 'Trap':
                    plot_force_before_after_cut(config)

            for file in elec_files:
                config.electrode_name = next((e for e in config.electrodes if e in file), None)
                rep_dir = os.path.join(config.decomp_dir, "rep1")
                file_path_rep1 = os.path.join(rep_dir, file).replace("\\", "/")
                path_rep2 = rep_dir.replace("rep1", "rep2")

                file_name_base = os.path.basename(file).split('_start')[0]  
                file_name_pattern = re.escape(file_name_base) + fr'_start\d+_end\d+_{config.emg_type}_{config.electrode_name}\.pkl'
                regex = re.compile(file_name_pattern)
                matching_file = [f for f in os.listdir(path_rep2) if regex.match(f)][0]

                file_path_rep2 = os.path.join(path_rep2, matching_file).replace("\\", "/")
                if config.electrode_name:
                    tmp_df, n_mu_rep1, n_mu_rep2, n_mu = process_file_tracking(file_path_rep1, file_path_rep2, config, last_mu_id)
                    mu_df = pd.concat([mu_df, tmp_df], ignore_index=True)
                    temp_df.loc[
                        fn.reverse_remap(config.fing_id, config.get_fingers()), config.electrode_name] = [n_mu_rep1, n_mu_rep2] 
                    
                    last_mu_id += n_mu
                else:
                    print(f"Electrode name not found")
            counter_plot += 1
            temp_df[FING_DIR] = config.direction
            mu_summary_df = pd.concat([mu_summary_df, temp_df], ignore_index=True)

    tot_1 = mu_summary_df.ext_digitorum.map(lambda x: x[0]) + mu_summary_df.ext_pollicis.map(lambda x: x[0]) + mu_summary_df.flx_digitorum.map(lambda x: x[0])
    tot_2 = mu_summary_df.ext_digitorum.map(lambda x: x[1]) + mu_summary_df.ext_pollicis.map(lambda x: x[1]) + mu_summary_df.flx_digitorum.map(lambda x: x[1])

    mu_summary_df[TOTAL_MUS_REP1] = tot_1
    mu_summary_df[TOTAL_MUS_REP2] = tot_2
    mu_df = spk.add_first_spike_time_col(mu_df)
    mu_df_sorted = spk.sort_rows_by_first_spike_time_per_rep(mu_df)

    mu_df_sorted[SP_COUNT] = mu_df_sorted[SP_TIME].apply(lambda x: len(x))
    mu_df_sorted[GLOB_MU_ID] = np.arange(mu_df_sorted.shape[0])

    return mu_df, mu_summary_df, mu_df_sorted



def process_file_tracking(file_path_rep1, file_path_rep2, config, last_mu_id):
    contents_rep1 = load_pickle(file_path_rep1)
    contents_rep1 = spk.sort_by_spike_times(contents_rep1)

    contents_rep2 = load_pickle(file_path_rep2)
    contents_rep2 = spk.sort_by_spike_times(contents_rep2)

    # Get cleaned, cut region of interest and glue the repetitions together. 
    # Shift spike times according to this new time axis where 0 corresponds to config.rep1_s
    contents_rep1, contents_rep2, units_rep1, units_rep2, reps = post_process_units_tracking(data_rep1=contents_rep1, data_rep2=contents_rep2, configuration=config)

    n_mu_rep1, n_mu_rep2 = 0, 0
    for i in units_rep1:
        if len(i) > 0:
            n_mu_rep1 += 1

    # if config.task_type == 'Trap':
    for i in units_rep2:
        if len(i) > 0:
            n_mu_rep2 += 1

    # n_mu = len(units_rep1)  # problem is this
    mu_data = []
    for rep_id in range(2):
        units = reps[rep_id]
        n_mu = len(units)  # TOCHECK
        contents = contents_rep1 if rep_id == 0 else contents_rep2
        if units:

            mu_data.append(pd.DataFrame({
                SP_TIME: units,
                FILTERS: contents['filters'] if 'filters' in contents.keys() else [],
                MU_ID: np.arange(last_mu_id, last_mu_id + n_mu),
                MU_RATE: contents['fr rep 1'] if rep_id == 0 else contents['fr rep 2'],
                ISI: [np.round(torch.cat((torch.tensor([0]), torch.diff(u))).detach().numpy() / config.f_samp, 2) * 1000
                    for u in units],
                ELEC_NAME: config.electrode_name,
                FING_NAME_COL: np.repeat(fn.reverse_remap(config.fing_id, config.get_fingers()), n_mu),
                REP_ID: int(rep_id),
                FING_DIR: config.direction,
                START_TIME: [(getattr(config, f'rep{int(rep_id + 1)}_s') - config.time_removed)  / config.f_samp if rep_id == 1 else getattr(config, f'rep{int(rep_id + 1)}_s')/ config.f_samp][0],  
                END_TIME: [(getattr(config, f'rep{int(rep_id + 1)}_e') - config.time_removed)  / config.f_samp if rep_id == 1 else getattr(config, f'rep{int(rep_id + 1)}_e')/ config.f_samp][0],
                START_RAMP: (config.points_cut[0 + rep_id*4] )/ config.f_samp,
                START_HOLD: (config.points_cut[1 + rep_id*4] )/ config.f_samp,
                END_HOLD:   (config.points_cut[2 + rep_id*4] )/ config.f_samp,
                END_RAMP:   (config.points_cut[3 + rep_id*4] )/ config.f_samp,
            }))


    del contents
    return pd.concat(mu_data, ignore_index=True) if mu_data else pd.DataFrame(), n_mu_rep1, n_mu_rep2, n_mu