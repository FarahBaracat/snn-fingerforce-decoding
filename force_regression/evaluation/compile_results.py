import re
import os
import logging
import itertools
from typing import List, Dict
from scipy import stats
from matplotlib import gridspec
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from force_regression.config.dataconfig import DataConfig
import force_regression.utils.functions as fn
from configs.constants import *


def parse_params_from_filename(filename:str)->dict:
    """
    Parses the parameters from the string filename.
    """
    params = filename.split('_')
    fing_matches = re.search(r'nfing_((?:\d+_)*\d+)', filename)
    pattern_hold = r'hold_(True|False)'
    match = re.search(pattern_hold, filename)
    hold_flag = match.group(1)
    hold_flag = hold_flag == 'True' if hold_flag is not None else None

    noise_seed_match = re.search(r'ns_(\d+)', filename)
    noise_mode_match = re.search(r'noise_(\w+)_percent', filename)
    noise_level_match = None
    if noise_seed_match is not None:
        noise_level_match = re.search(r'percent_(\d+)', filename)
    
    params_dict = {
        EMG_COL: params[2],
        WS_COL: float(params[3].split('ws')[-1]), #in seconds
        SIGN_MVC_COL: int(params[5]),
        OL_COL: int(params[4].split('ol')[-1]),  #in percentage
        MVC_COL: int(params[6]),
        REG_FING_COL: list(map(int,fing_matches.group(1).split('_'))) if fing_matches is not None else list(np.arange(1,6,1)),
        SEGMENT_HOLD_COL: hold_flag,
        NOISE_SEED_COL: int(noise_seed_match.group(1)) if noise_seed_match is not None else None,
        NOISE_LEVEL_COL: int(noise_level_match.group(1)) if noise_seed_match is not None else None,
        NOISE_MODE_COL: noise_mode_match.group(1) if noise_seed_match is not None else None,
    }

    return params_dict
    

def create_model_name_column(df:pd.DataFrame)->pd.DataFrame:
    """
    Creates a new column in the dataframe with the model name,
    parsed from the emg_type and input_type columns.
    """
    return df.apply(lambda x:f"{x[EMG_COL]} {x[INPUT_TYPE_COL]}", axis=1)


def generate_stat_annotation(p_value:float, test_type:str):
    """
    Generates the annotation text for the p-value.
    """
    if p_value < 0.05:
        if p_value < 1e-4:   # for p < 1e-05
            p_text = "****"
        
        elif p_value < 1e-3: # for p < 0.001
            p_text = "***"

        elif p_value < 1e-2:  # for p < 0.01
            p_text = "**"

        else:
            p_text = f"p={p_value}"
    else:
        p_text = f"ns (p={p_value:.3f})" if test_type == 't-test' else "std-ns"

    return p_text



def load_and_parse_df_data(subject_list:List[str], data_type:str,
                      config:dict,
                      results_dir:str,
                      is_multi=False, is_sweep=False,
                      is_noise_analysis=False,
                      noise_mode:str='omission'):
    """
    Loads and organizes the data needed for comparison between EMG(surf/intr) + feature(global/MUs)
    in a compiled_df. This data can either be the metrics or the y_df.
    """
    compiled_df = pd.DataFrame()

    for sub in subject_list:
        data_config = DataConfig(root_dir=config['root_dir'],
                                 root_results_dir=config['root_results_dir'],
                                 subject=fn.reverse_remap(
                                     sub, config['subject_mappings']),
                                 task_type="Trap", finger_type="Individual", day="Day 1",
                                 emg_type="surf", f_samp=10240, subj_map=config['subject_mappings'],
                                 segment_hold=True, verbose=False, common_only=True, images=False,
                                 time_to_cut=1, results_dir=results_dir)
        data_config.results_path = os.path.join(data_config.results_path, 'baseline_fp32')
        if is_multi:
            data_config.results_path = os.path.join(data_config.results_path, 'multi_mvc')
        if is_sweep:
            data_config.results_path = os.path.join(data_config.results_path, 'sweep_wind')
        if is_noise_analysis:
            data_config.results_path = os.path.join(data_config.results_path, f'noise/{noise_mode}')

        for filename in os.listdir(data_config.results_path):
            if data_type in filename and ('sweep' not in filename) and ('Pseudo' not in filename):
                df = pd.read_pickle(os.path.join(data_config.results_path, filename))
                ismu_results_file = bool(RESULTS_FILE_ID_SP_COUNT in filename)
                params = parse_params_from_filename(filename)
                df[SEGMENT_HOLD_COL] = params[SEGMENT_HOLD_COL]
                df[SUBJECT_COL] = sub
                df[EMG_COL] = params[EMG_COL]
                df[SIGN_MVC_COL] = params[SIGN_MVC_COL]
                df[MVC_COL] = params[MVC_COL]
                df[DIR_COL] = df[SIGN_MVC_COL].map({-1: 'Extension', 1: 'Flexion'})
                df[WS_COL] = params[WS_COL] if WS_COL in params else None
                df[OL_COL] = params[OL_COL] if OL_COL in params else None
                df[REG_FING_COL] =  [params[REG_FING_COL]] * df.shape[0]
                df[INPUT_TYPE_COL] = INPUT_TYPE_SP_COUNT if ismu_results_file else INPUT_TYPE_EMG_GLOB
                if ismu_results_file:
                    if params[MVC_COL]==15 and params[EMG_COL]=='intra':
                        print(filename)

                if is_noise_analysis:
                    df[NOISE_SEED_COL] = params[NOISE_SEED_COL]
                    df[NOISE_LEVEL_COL] = params[NOISE_LEVEL_COL]
                    df[NOISE_MODE_COL] = params[NOISE_MODE_COL]
                compiled_df = pd.concat([compiled_df, df], axis=0, ignore_index=True)
    logging.info("Loaded data for %s subjects: %s\nMVC:%s\nWindow size: %s\nOverlap: %s\nSign MVC: %s\n",
                 compiled_df[SUBJECT_COL].nunique(), compiled_df[SUBJECT_COL].unique(),
                 compiled_df[MVC_COL].unique(), compiled_df[WS_COL].unique(),
                 compiled_df[OL_COL].unique(), compiled_df[SIGN_MVC_COL].unique())

    return compiled_df

def parse_encoding_params(df_snn:pd.DataFrame, filename:str):
    """
    Utility function to parse the encoding parameters from the filename. This is only used if topology is encoding
    """
    enctau1 = re.search(r'enctau1_([+-]?([0-9]*[.])?[0-9]+)', filename)
    is_aleaky = re.search(r'aleaky_', filename)
    if enctau1 is not None:
        enctau1 = float(enctau1.group(1))
        enctau2 = float(re.search(r'enctau2_([+-]?([0-9]*[.])?[0-9]+)', filename).group(1))
        enctau3 = float(re.search(r'enctau3_([+-]?([0-9]*[.])?[0-9]+)', filename).group(1))
        df_snn[ENC_TAU_MEM1_COL] = enctau1
        df_snn[ENC_TAU_MEM2_COL] = enctau2
        df_snn[ENC_TAU_MEM3_COL] = enctau3
    else:
        enctau1 = float(re.search(r'enctau_([+-]?([0-9]*[.])?[0-9]+)', filename).group(1))
        df_snn[ENC_TAU_MEM1_COL] = enctau1
    
    encthr1 = re.search(r'encthresh1_([+-]?([0-9]*[.])?[0-9]+)', filename)
    if encthr1 is not None:
        encthr1 = float(encthr1.group(1))
        encthr2 = float(re.search(r'encthresh2_([+-]?([0-9]*[.])?[0-9]+)', filename).group(1))
        encthr3 = float(re.search(r'encthresh3_([+-]?([0-9]*[.])?[0-9]+)', filename).group(1))
        df_snn[ENC_SPK_THRESH1_COL] = encthr1
        df_snn[ENC_SPK_THRESH2_COL] = encthr2
        df_snn[ENC_SPK_THRESH3_COL] = encthr3
    else:
        encthr1 = float(re.search(r'encthresh_([+-]?([0-9]*[.])?[0-9]+)', filename).group(1))
        df_snn[ENC_SPK_THRESH1_COL] = encthr1
    
    if is_aleaky:
        df_snn[USE_ALIF_COL] = True
        threshold_tau_adapt = float(re.search(r'tauadapt_([+-]?([0-9]*[.])?[0-9]+)', filename).group(1))
        threshold_scale_adapt = float(re.search(r'scaleadapt_([+-]?([0-9]*[.])?[0-9]+)', filename).group(1))
        df_snn[ENC_THRESHOLD_TAU_COL] = threshold_tau_adapt
        df_snn[ENC_ALIF_SCALE_COL] = threshold_scale_adapt
    else:
        df_snn[USE_ALIF_COL] = False
        df_snn[ENC_THRESHOLD_TAU_COL] = None
        df_snn[ENC_ALIF_SCALE_COL] = None
    return df_snn
        


# Load all SNN results
def load_and_parse_snn_data_to_df(dt:float, subject_list:List[str],data_type:str,
                                  results_dir:str,network_topology:str,
                                  config:dict[str,str]):
    """
    data_type: 'metrics_df_cv_snn' or 'y_df_cv_snn'
    """
    compiled_df_snn = pd.DataFrame()
    for sub in subject_list:
        data_config = DataConfig(root_dir=config['root_dir'],
                                 root_results_dir=config['root_results_dir'],
                                 subject=fn.reverse_remap(
                                     sub, config['subject_mappings']),
                                 task_type="Trap", finger_type="Individual", day="Day 1",
                                 emg_type="surf", f_samp=10240, subj_map=config['subject_mappings'],
                                 segment_hold=True, verbose=False, common_only=True, images=False,
                                 time_to_cut=1, results_dir=results_dir)
        data_path = os.path.join(data_config.results_path, f'SNN/{network_topology}')
        print(f"Loading data for subject {sub} from {data_path}")
        if not os.path.exists(data_path):
            print(f"Subject {sub} does not exist")
            continue
        for filename in os.listdir(data_path):
            full_path = os.path.join(data_path, filename)
            if data_type in filename and f'dt_{dt}' in filename and MVC_COL in filename:
                if not os.path.exists(full_path):
                    print(f"File {filename} does not exist")
                    continue
                print(f"Loading file: {filename}")
                df_snn = pd.read_pickle(full_path)
                df_snn[MVC_COL] = int(re.search(r'mvc_(\d+)', filename).group(1))
                df_snn[SUBJECT_COL] = sub
                df_snn[EMG_COL] = 'intra'
                df_snn[INPUT_TYPE_COL] = INPUT_TYPE_SP_TIMES
                df_snn[SIGN_MVC_COL] = int(re.search(r'sign_([+-]?\d+)', filename).group(1))
                df_snn[EPOCHS_COL] = int(re.search(r'ep_(\d+)', filename).group(1))
                tau_syn_match = re.search(r'tausyn_([+-]?([0-9]*[.])?[0-9]+)', filename)
                df_snn[TAU_SYN_COL] = float(tau_syn_match.group(1)) if tau_syn_match is not None else 0
                tau_mem_match = re.search(r'taumem_([+-]?([0-9]*[.])?[0-9]+)', filename)
                df_snn[TAU_MEM_COL] = float(tau_mem_match.group(1)) if tau_mem_match is not None else 0
                tau1_mem_filt_match = re.search(r'taufilt1_([+-]?([0-9]*[.])?[0-9]+)', filename)
                tau2_mem_filt_match = re.search(r'taufilt2_([+-]?([0-9]*[.])?[0-9]+)', filename)
                df_snn[TAU1_MEM_FILT_COL] = float(tau1_mem_filt_match.group(1)) if tau1_mem_filt_match is not None else 0
                df_snn[TAU2_MEM_FILT_COL] = float(tau2_mem_filt_match.group(1)) if tau2_mem_filt_match is not None else 0
                df_snn[REG_FING_COL] = [np.arange(1,6,1)] * df_snn.shape[0] if df_snn[MVC_COL].values[0] < 20 else [np.arange(1,4,1)] * df_snn.shape[0]
                
                torch_seed_match = re.search(r'seed_(\d+)', filename)
                df_snn[TORCH_SEED_COL] = int(torch_seed_match.group(1)) if torch_seed_match is not None else None
                
                noise_mode_match = re.search(r'noise_(\w+)_percent', filename)
                noise_seed_match = re.search(r'ns_(\d+)', filename)
                noise_level_match = re.search(r'percent_(\d+)', filename)
                if noise_mode_match is not None:
                    df_snn[NOISE_SEED_COL] = int(noise_seed_match.group(1))
                    df_snn[NOISE_LEVEL_COL] = int(noise_level_match.group(1))

                df_snn[NOISE_MODE_COL] = noise_mode_match.group(1) if noise_mode_match is not None else None
                hold_flag = search_for_pattern_in_filename(filename, r'hold_(True|False)')
                hold_flag = hold_flag == 'True' if hold_flag is not None else None
                df_snn[SEGMENT_HOLD_COL] = hold_flag

                learn_tausyn_match = search_for_pattern_in_filename(filename, r'sinit_([+-]?([0-9]*[.])?[0-9]+)')
                df_snn[TAU_SYN_INIT_COL] = float(learn_tausyn_match) if learn_tausyn_match is not None else 0
                
                learn_taumem_match = search_for_pattern_in_filename(filename, r'minit_([+-]?([0-9]*[.])?[0-9]+)')
                df_snn[TAU_MEM_INIT_COL] = float(learn_taumem_match) if learn_taumem_match is not None else 0

                if 'encoding' in network_topology:
                    df_snn = parse_encoding_params(df_snn, filename)
                # append to the compiled_df_snn
                compiled_df_snn = pd.concat([compiled_df_snn, df_snn], axis=0, ignore_index=True)
    return compiled_df_snn

def search_for_pattern_in_filename(filename:str, pattern:str):
    """
    Search for a pattern in the filename.
    """
    match = re.search(pattern, filename)
    return match.group(1) if match is not None else None

# Load all RNN results
def load_and_parse_rnn_data_to_df(dt:float, subject_list:List[str],data_type:str,
                                  run_dir:str, results_dir:str, config:dict[str,str]):
    """
    data_type: 'metrics_df_cv_rnn' or 'y_df_cv_rnn'
    """
    compiled_df_rnn = pd.DataFrame()

    for sub in subject_list:
        data_config = DataConfig(root_dir=config['root_dir'],
                                 root_results_dir=config['root_results_dir'],
                                 subject=fn.reverse_remap(
                                     sub, config['subject_mappings']),
                                 task_type="Trap", finger_type="Individual", day="Day 1",
                                 emg_type="surf", f_samp=10240, subj_map=config['subject_mappings'],
                                 segment_hold=True, verbose=False, common_only=True, images=False,
                                 time_to_cut=1)
        data_path = os.path.join(data_config.results_path, run_dir)
        print(f"Loading data for subject {sub} from {data_path}")
        if not os.path.exists(data_path):
            print(f"Subject {sub} does not exist")
            continue
        for filename in os.listdir(data_path):
            full_path = os.path.join(data_path, filename)
            if data_type in filename and f'dt_{dt}' in filename and MVC_COL in filename:
                if not os.path.exists(full_path):
                    print(f"File {filename} does not exist")
                    continue
                print(f"Loading file: {filename}")
                df_snn = pd.read_pickle(full_path)
                # add mvc. subject, emg, sign_mvc, input_type
                df_snn[MVC_COL] = int(re.search(r'mvc_(\d+)', filename).group(1))
                df_snn[SUBJECT_COL] = sub
                df_snn[EMG_COL] = 'intra'
                df_snn[INPUT_TYPE_COL] = INPUT_TYPE_SP_TIMES
                df_snn[SIGN_MVC_COL] = int(re.search(r'sign_([+-]?\d+)', filename).group(1))
                df_snn[EPOCHS_COL] = int(re.search(r'ep_(\d+)', filename).group(1))
                df_snn[REG_FING_COL] = [np.arange(1,6,1)] * df_snn.shape[0] if df_snn[MVC_COL].values[0] < 20 else [np.arange(1,4,1)] * df_snn.shape[0]
                torch_seed_match = re.search(r'seed_(\d+)', filename)
                df_snn[TORCH_SEED_COL] = int(torch_seed_match.group(1)) if torch_seed_match is not None else None
                
                noise_seed_match = re.search(r'ns_(\d+)', filename)
                noise_level_match = re.search(r'noise_(\d+)', filename)
                if noise_level_match is not None:
                    df_snn[NOISE_SEED_COL] = int(noise_seed_match.group(1))
                    df_snn[NOISE_LEVEL_COL] = int(noise_level_match.group(1))

                hold_flag = search_for_pattern_in_filename(filename, r'hold_(True|False)')
                hold_flag = hold_flag == 'True' if hold_flag is not None else None
                df_snn[SEGMENT_HOLD_COL] = hold_flag

                project_out = search_for_pattern_in_filename(filename, r'fcout_(True|False)')
                project_out = project_out == 'True' if project_out is not None else None
                df_snn[PROJECT_OUT_COL] = project_out

                normalize_force = search_for_pattern_in_filename(filename, r'normalizeforce_(True|False)')
                normalize_force = normalize_force == 'True' if normalize_force is not None else None
                df_snn[NORMALIZE_FORCE_COL] = normalize_force

                rnn_bias = search_for_pattern_in_filename(filename, r'rnnbias_(True|False)')
                rnn_bias = rnn_bias == 'True' if rnn_bias is not None else None
                df_snn[RNN_BIAS_COL] = rnn_bias

                exp_kernel_size = int(search_for_pattern_in_filename(filename, r'kernelsize_(\d+)'))
                df_snn[KERNEL_SIZE_COL] = exp_kernel_size
                exp_kernel_decay = float(search_for_pattern_in_filename(filename, r'expdecay_([+-]?([0-9]*[.])?[0-9]+)'))
                df_snn[KERNEL_DECAY_COL] = exp_kernel_decay
                hidden_state_size = int(search_for_pattern_in_filename(filename, r'hidsize_(\d+)'))
                df_snn[HIDDEN_STATE_SIZE_COL] = hidden_state_size

                rnn_type = search_for_pattern_in_filename(filename, r'rnn_(\w+)_ep')
                df_snn[RNN_TYPE_COL] = rnn_type
                # append to the compiled_df_snn
                compiled_df_rnn = pd.concat([compiled_df_rnn, df_snn], axis=0, ignore_index=True)
    return compiled_df_rnn


def detect_num_fold_blocks(df:pd.DataFrame):
    """
    Detect the number of blocks of the same value in FOLD_COL.
    """
    # Detect the number of blocks of the same value in 'FOLD_COL'
    fold_blocks = (df[FOLD_COL] != df[FOLD_COL].shift()).cumsum()
    num_blocks = fold_blocks.nunique()
    return num_blocks

def compute_mean_sd_baseline_noise_metric(compiled_df:pd.DataFrame, mvc:int, return_samples=False,
                                            metrics_list=['MAE_test','RMSE_test','R2_test'], aux_cols=None,
                                            metric_type='post'):
    
    """""
    Computes mean and std of the metrics across all folds for each subject, MVC and direction.
    """""
    fing_order = determine_finger_sequence(mvc)
    compiled_df = compiled_df[(compiled_df[MVC_COL] == mvc)]
    metrics_list = [f"{metric}_{metric_type}" for metric in metrics_list]
    # # Compute mean and std
    metrics_map_rename = {f'MAE_test_{metric_type}': 'MAE_test_mean',
                          f'RMSE_test_{metric_type}':'RMSE_test_mean',
                          f'R2_test_{metric_type}': 'R2_test_mean'}
    if aux_cols is None:
        aux_cols = [FING_ID_COL, SUBJECT_COL, MVC_COL, SIGN_MVC_COL]
    baseline_mean_std_df = pd.DataFrame()
    baseline_mean_std_df = compiled_df.groupby(aux_cols, as_index=False)[metrics_list].mean()
    baseline_mean_std_df.rename(columns=metrics_map_rename, inplace=True)
    baseline_mean_std_df[aux_cols + ['MAE_test_std', 'RMSE_test_std', 'R2_test_std']] = compiled_df.groupby(aux_cols, as_index=False)[metrics_list].std()
    baseline_mean_std_df[FING_NAME_COL] = baseline_mean_std_df[FING_ID_COL].map({i: fing_order[i-1] for i in range(len(fing_order)+1)})
    compiled_df[FING_NAME_COL] = compiled_df[FING_ID_COL].map({i: fing_order[i-1] for i in range(len(fing_order)+1)})
    if return_samples:
        return baseline_mean_std_df, compiled_df
    return baseline_mean_std_df


def compute_mean_sd_metric(compiled_df:pd.DataFrame, mvc:int, type_pred_metric='post',
                           return_samples=False,
                           metrics_list=['R2', 'RMSE','MAE'],
                           get_samples_for_metric='RMSE',
                           dataset_type:str='test'):
    mean_metrics_cv = pd.DataFrame()
    metrics_cv_samples = pd.DataFrame()

    fing_order = determine_finger_sequence(mvc)
    for sub in compiled_df[SUBJECT_COL].unique():
        for input_type in compiled_df[INPUT_TYPE_COL].unique():
            for emg in compiled_df[EMG_COL].unique():
                for sign_mvc in compiled_df[SIGN_MVC_COL].unique():
                    for seg_hold in compiled_df[SEGMENT_HOLD_COL].unique():
                        logging.debug(f"Sub {sub} | EMG {emg} | Sign MVC {sign_mvc} | MVC {mvc} | Input {input_type} | Seg Hold {seg_hold}")
                        mean_std_df = pd.DataFrame()
                        for metric in metrics_list:
                            metric_name = f'{metric}_{dataset_type}_{type_pred_metric}'
                            mask = (compiled_df[SUBJECT_COL] == sub) & (compiled_df[EMG_COL] == emg) & (compiled_df[SIGN_MVC_COL]==sign_mvc) & (compiled_df[MVC_COL]==mvc) & (compiled_df[INPUT_TYPE_COL]==input_type) & (compiled_df[SEGMENT_HOLD_COL]==seg_hold)
                            temp_df = compiled_df[mask][[metric_name, FOLD_COL, FING_ID_COL]]#.explode(metric_name)
                            # temp_df.drop_duplicates(inplace=True)
                            # num_blocks = detect_num_fold_blocks(temp_df)
                            # temp_df[FING_ID_COL]= np.tile(np.arange(len(fing_order)),num_blocks)
                            
                            # Get mean across the 2 folds for each finger
                            mean_std_df[f'{metric}_{dataset_type}_mean'] = temp_df.groupby([FING_ID_COL])[metric_name].mean().to_numpy()
                            mean_std_df[f'{metric}_{dataset_type}_std'] = temp_df.groupby([FING_ID_COL])[metric_name].std().to_numpy()
                            fing_id_order =  temp_df.groupby([FING_ID_COL])[metric_name].std().index.to_numpy()

                            if metric == get_samples_for_metric:
                                temp_df[EMG_COL] = emg
                                temp_df[SIGN_MVC_COL] = sign_mvc
                                temp_df[MVC_COL] = mvc
                                temp_df[INPUT_TYPE_COL] = input_type
                                temp_df[SUBJECT_COL] = sub
                                temp_df[SEGMENT_HOLD_COL] = seg_hold
                                metrics_cv_samples = pd.concat([metrics_cv_samples, temp_df])
                        mean_std_df[FING_ID_COL] = fing_id_order
                        mean_std_df[SIGN_MVC_COL] = sign_mvc
                        mean_std_df[SUBJECT_COL] = sub
                        mean_std_df[EMG_COL] = emg
                        mean_std_df[FING_NAME_COL] = mean_std_df[FING_ID_COL].map({i: fing_order[i-1] for i in range(len(fing_order)+1)})#fing_order
                        mean_std_df[INPUT_TYPE_COL] = input_type
                        mean_std_df[MVC_COL] = mvc
                        mean_std_df[SEGMENT_HOLD_COL] = seg_hold
                        mean_metrics_cv = pd.concat([mean_metrics_cv, mean_std_df], axis=0, ignore_index=True)
    if return_samples:
        metrics_cv_samples[FING_NAME_COL] = metrics_cv_samples[FING_ID_COL].map({i: fing_order[i-1] for i in range(len(fing_order)+1)})
        return mean_metrics_cv, metrics_cv_samples
    
    return mean_metrics_cv



def compute_mean_sd_snn_metric(compiled_df_snn:pd.DataFrame, mvc:int, return_samples=False,
                               metrics_list=['MAE_test','RMSE_test','R2_test'], aux_cols=None):
    """""
    Computes mean and std of the metrics across all folds for each subject, MVC and direction.
    """""
    fing_order = determine_finger_sequence(mvc)
    # We care about the inference mode only, so we segment the dataframe to only include inference mode
    compiled_df_snn = compiled_df_snn[(compiled_df_snn['mode'] == 'inference') & (compiled_df_snn[MVC_COL] == mvc)]

    # # Compute mean and std
    metrics_map_rename = {'MAE_test': 'MAE_test_mean',
                          'RMSE_test':'RMSE_test_mean',
                          'R2_test': 'R2_test_mean'}
    if aux_cols is None:
        aux_cols = [FING_ID_COL, SUBJECT_COL, MVC_COL, SIGN_MVC_COL, TAU_SYN_COL,TAU1_MEM_FILT_COL,
                TAU2_MEM_FILT_COL, TAU_MEM_COL, TAU_MEM_INIT_COL, TAU_SYN_INIT_COL, EPOCHS_COL]
    print(f"Computing mean across {compiled_df_snn[TORCH_SEED_COL].nunique()} torch seeds:{compiled_df_snn[TORCH_SEED_COL].unique()}")
    snn_mean_std_df = pd.DataFrame()
    snn_mean_std_df = compiled_df_snn.groupby(aux_cols, as_index=False)[metrics_list].mean()
    snn_mean_std_df.rename(columns=metrics_map_rename, inplace=True)
    snn_mean_std_df[aux_cols + ['MAE_test_std', 'RMSE_test_std', 'R2_test_std']] = compiled_df_snn.groupby(aux_cols, as_index=False)[metrics_list].std()
    snn_mean_std_df[FING_NAME_COL] = snn_mean_std_df[FING_ID_COL].map({i: fing_order[i-1] for i in range(len(fing_order)+1)})
    compiled_df_snn[FING_NAME_COL] = compiled_df_snn[FING_ID_COL].map({i: fing_order[i-1] for i in range(len(fing_order)+1)})
    if return_samples:
        return snn_mean_std_df, compiled_df_snn
    return snn_mean_std_df

def determine_finger_sequence(mvc:int):
    """
    Determines the finger sequence based on the mvc value.
    """
    if mvc in [15, 5, 515]: # 515 is the model for both MVC 5 and 15 togethe
        fing_order = ['thumb', 'index', 'middle', 'ring', 'little']
    elif mvc in [20, 30, 50]:
        fing_order = ['thumb', 'index', 'middle']
    else:
        raise ValueError('Invalid mvc value. It should be 5, 15, 20, 30, or 50')
    return fing_order


def compute_mean_metrics_across_mvcs(compiled_df:pd.DataFrame,
                                     mvcs:List[int],
                                     type_pred_metric='post',
                                     get_samples_for_metric='RMSE'):
    """
    Computes the mean and std of the RMSE across all folds for each subject, MVC and direction. 
    Returns a dataframe with the mean and std of the RMSE across all folds for each subject,
    MVC and direction and all the RSE values for each fold, subject and MVC.

    """
    metric_samples_across_mvc = pd.DataFrame()    # all RMSE values for each fold, subject and MVC
    pooled_metric_across_mvc = pd.DataFrame()     # mean and std of RMSE across folds for both subjects, all MVCS, per direction and finger

    for select_mvc in mvcs:
        # Get mean metric across folds per finger and direction for both subjects for a single mvc
        # metric_samples are all RMSE values per fold, direction and subject for a single mvc
        metric_stat, metric_samples = compute_mean_sd_metric(compiled_df=compiled_df,
                                                            mvc=select_mvc,
                                                            return_samples=True,
                                                            type_pred_metric=type_pred_metric,
                                                            get_samples_for_metric=get_samples_for_metric)
        # Append column for direction and bar_label for easier handling when plotting
        metric_stat[DIR_COL] = metric_stat[SIGN_MVC_COL].map({-1: 'Extension', 1: 'Flexion'})
        metric_stat[MODEL_SHORT_NAME_COL] = metric_stat[EMG_COL] +  " " + metric_stat[INPUT_TYPE_COL].str.lower()
        pooled_metric_across_mvc = pd.concat([pooled_metric_across_mvc, metric_stat], axis=0)
        # Append a column with the bar_label on the metric_samples as well
        metric_samples[MODEL_SHORT_NAME_COL] = metric_samples[EMG_COL] +   " " + metric_samples[INPUT_TYPE_COL].str.lower()
        metric_samples_across_mvc = pd.concat([metric_samples_across_mvc, metric_samples], axis=0)
                
    return pooled_metric_across_mvc, metric_samples_across_mvc


def add_normalized_metric(df:pd.DataFrame,metric:str):
    """
    Add a column with the normalized RMSE values across all models for each finger and direction
    """
    df[f'{metric}_norm'] = df[metric]/df[MVC_COL]

    return df

def compute_pairwise_effect_sizes(samples_df: pd.DataFrame,
                                   model_short_to_full: Dict[str, str],
                                   metric: str = 'RMSE_filtered',
                                   normalized_metric: bool = False,
                                   remove_mvcs: list = [],
                                   seed_col: str = TORCH_SEED_COL) -> pd.DataFrame:
    """
    Compute effect sizes (mean and median differences) for all pairwise model comparisons.
    
    Parameters:
    -----------
    samples_df : pd.DataFrame
        Contains the metric per fold and direction for all mvcs and subjects
    model_short_to_full : Dict[str, str]
        Mapping from short model name to full model name
    metric : str
        Metric column to compute effect sizes on
    normalized_metric : bool
        Whether to use the normalized version of the metric
    remove_mvcs : list
        List of MVCs to exclude from analysis
        
    Returns:
    --------
    pd.DataFrame with columns: model1, model2, pair, subject,
        mean_diff, median_diff, mean_m1, mean_m2, median_m1, median_m2, cohens_d
    """
    models = list(model_short_to_full.keys())
    if len(remove_mvcs) > 0:
        samples_df = samples_df[~samples_df[MVC_COL].isin(remove_mvcs)]
    
    test_on_metric = f'{metric}_norm' if normalized_metric else metric
    if test_on_metric not in samples_df.columns:
        raise ValueError(f"Column {test_on_metric} not found in samples_df!")
    
    # Average across seeds per (model, subject, fold, finger, direction) if seed column exists
    groupby_cols = [MODEL_SHORT_NAME_COL, SUBJECT_COL, FOLD_COL, FING_ID_COL, SIGN_MVC_COL]
    if seed_col in samples_df.columns:
        n_seeds = samples_df[seed_col].nunique()
        print(f"Averaging across {n_seeds} seeds ({samples_df[seed_col].unique()}) per fold/finger/direction")
        samples_df = samples_df.groupby(groupby_cols, as_index=False)[test_on_metric].mean()
    
    effect_size_rows = []
    
    for sub in samples_df[SUBJECT_COL].unique():
        samples_sub = samples_df[samples_df[SUBJECT_COL] == sub]
        
        for pair in itertools.combinations(models, 2):
            # Force 'intra mu spike times' to be model2 (reference) if present
            reference_model = 'intra mu spike times'
            if pair[0] == reference_model and pair[1] != reference_model:
                pair = (pair[1], pair[0])
            elif reference_model in pair[0] and reference_model not in pair[1]:
                pair = (pair[1], pair[0])

            values_m1_df = samples_sub[samples_sub[MODEL_SHORT_NAME_COL] == pair[0]].set_index([FOLD_COL, FING_ID_COL, SIGN_MVC_COL])[test_on_metric].astype(float)
            values_m2_df = samples_sub[samples_sub[MODEL_SHORT_NAME_COL] == pair[1]].set_index([FOLD_COL, FING_ID_COL, SIGN_MVC_COL])[test_on_metric].astype(float)
            
            # Align on shared (fold, finger, direction) pairs
            common_idx = values_m1_df.index.intersection(values_m2_df.index)
            values_m1 = values_m1_df.loc[common_idx].values
            values_m2 = values_m2_df.loc[common_idx].values
            model_i_to_j = f"{model_short_to_full[pair[0]].split(' ')[1]}-{model_short_to_full[pair[1]].split(' ')[1]}"
            
            # Paired differences per fold, finger, direction
            paired_diffs_series = values_m1_df.loc[common_idx] - values_m2_df.loc[common_idx]
            paired_diffs = paired_diffs_series.values
            n = len(paired_diffs)
            
            mean_m1, mean_m2 = np.mean(values_m1), np.mean(values_m2)
            median_m1, median_m2 = np.median(values_m1), np.median(values_m2)
            mean_diff = np.mean(paired_diffs)
            median_diff = np.median(paired_diffs)
            std_diff = np.std(paired_diffs, ddof=1)
            
            # 95% CI on mean difference (t-distribution)
            se_mean = std_diff / np.sqrt(n) if n > 0 else np.nan
            t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else np.nan
            mean_diff_ci_lower = mean_diff - t_crit * se_mean
            mean_diff_ci_upper = mean_diff + t_crit * se_mean
            
            # 95% CI on mean and median differences (bootstrap at the fold level)            
            n_bootstrap = 1000
            rng = np.random.default_rng(seed=42)
            folds = paired_diffs_series.index.get_level_values(FOLD_COL).unique()
            bootstrap_means = np.empty(n_bootstrap)
            bootstrap_medians = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                sampled_folds = rng.choice(folds, size=len(folds), replace=True)
                # Gather all observations from the resampled folds
                boot_diffs = np.concatenate([
                    paired_diffs_series.xs(f, level=FOLD_COL).values for f in sampled_folds
                ])
                bootstrap_means[b] = np.mean(boot_diffs)
                bootstrap_medians[b] = np.median(boot_diffs)
            
            mean_diff_ci_lower_boot = np.percentile(bootstrap_means, 2.5)
            mean_diff_ci_upper_boot = np.percentile(bootstrap_means, 97.5)
            median_diff_ci_lower = np.percentile(bootstrap_medians, 2.5)
            median_diff_ci_upper = np.percentile(bootstrap_medians, 97.5)
            print(f"Check on bootstrap mean:{np.mean(bootstrap_means)} +/- {np.std(bootstrap_means)}")
            # Cohen's d: mean of paired differences / std of paired differences
            cohens_d = mean_diff / std_diff if std_diff > 0 else np.nan
            
            # # Get topology for each model in the pair
            # topology_m1 = samples_sub[samples_sub[MODEL_SHORT_NAME_COL] == pair[0]][TOPOLOGY_COL].iloc[0] \
            #     if TOPOLOGY_COL in samples_sub.columns else pair[0]
            # topology_m2 = samples_sub[samples_sub[MODEL_SHORT_NAME_COL] == pair[1]][TOPOLOGY_COL].iloc[0] \
            #     if TOPOLOGY_COL in samples_sub.columns else pair[1]
            
            effect_size_rows.append({
                'model1': pair[0],
                'model2': pair[1],
                'pair': model_i_to_j,
                'subject': sub,
                'n_pairs': n,
                'n_folds': len(folds),
                'mean_m1': mean_m1,
                'mean_m2': mean_m2,
                'median_m1': median_m1,
                'median_m2': median_m2,
                'mean_diff': mean_diff,
                'mean_diff_ci_lower': mean_diff_ci_lower,
                'mean_diff_ci_upper': mean_diff_ci_upper,
                'mean_diff_ci_lower_boot': mean_diff_ci_lower_boot,
                'mean_diff_ci_upper_boot': mean_diff_ci_upper_boot,
                'median_diff': median_diff,
                'median_diff_ci_lower': median_diff_ci_lower,
                'median_diff_ci_upper': median_diff_ci_upper,
                'std_diff': std_diff,
                'cohens_d': cohens_d
            })


    
    return pd.DataFrame(effect_size_rows)


def conduct_paired_tests(samples_df:pd.DataFrame,
                         model_short_to_full:Dict[str,str],
                         metric= 'RMSE_filtered',
                         normalized_metric:bool=False, remove_mvcs=[], test_type=None):
    """
    Run paired tests across the 4 models
    samples_df: contains the RMSE per fold and direction for all mvcs and subjects
    remove_mvcs: list of mvcs to remove from the analysis
    """
    models = list(model_short_to_full.keys())   # list of models
    mvcs = samples_df[MVC_COL].unique()
    if len(remove_mvcs)>0:
        samples_df = samples_df[~samples_df[MVC_COL].isin(remove_mvcs)]
        mvcs = samples_df[MVC_COL].unique()

    print(f"Running paired tests across {len(models)} models and {len(mvcs)} MVCs ({mvcs})")
    if len(mvcs)==1:
        print("Note: Running paired tests within a single MVC\n")
    else:
        print("Note: Running paired tests across values from all MVCs\n")

    paired_test_df = pd.DataFrame()
    test_on_metric = f'{metric}_norm' if normalized_metric else metric
    if test_on_metric not in samples_df.columns:
        raise ValueError(f"Column {test_on_metric} not found in samples_df!")
    assert test_on_metric in samples_df.columns, f"Column {test_on_metric} not found in samples_df!"
    for sub in samples_df[SUBJECT_COL].unique():
        for mod in models:
            # Step 1: Check if data is normally distributed using Shapiro test
            data = samples_df[(samples_df[MODEL_SHORT_NAME_COL]==mod) &(samples_df[SUBJECT_COL]==sub)]
            _, p_value = stats.shapiro(data[test_on_metric])
            if p_value < 0.05:
                print(f"Data for Model {mod} is not normally distributed.")
        # Step 3: Conduct pairwise comparisons.
        # Apply Wilcoxon signed-rank test if data is not normally distributed
        paired_test_sub = pd.DataFrame(columns=['model1', 'model2',
                                                'p_value','test_performed','levene_p_value',
                                                'subject', 'pair'])
        samples_sub = samples_df[samples_df[SUBJECT_COL]==sub]
        for i, pair in enumerate(itertools.combinations(models, 2)):
            n_points = min(samples_sub[samples_sub[MODEL_SHORT_NAME_COL]==pair[0]][test_on_metric].count(),
                           samples_sub[samples_sub[MODEL_SHORT_NAME_COL]==pair[1]][test_on_metric].count())
            # check is the data is normally distributed for each pair
            _, p1_value = stats.shapiro(samples_sub[samples_sub[MODEL_SHORT_NAME_COL]==pair[0]][test_on_metric].values)
            _, p2_value = stats.shapiro(samples_sub[samples_sub[MODEL_SHORT_NAME_COL]==pair[1]][test_on_metric].values)

            # parse model number
            model_i_to_j = f"{model_short_to_full[pair[0]].split(' ')[1]}-{model_short_to_full[pair[1]].split(' ')[1]}"
            if test_type is None:
                if p1_value < 0.05 and p2_value < 0.05:
                    _, p_value = stats.ttest_rel(samples_sub[samples_sub[MODEL_SHORT_NAME_COL]==pair[0]][test_on_metric].values,
                                                samples_sub[samples_sub[MODEL_SHORT_NAME_COL]==pair[1]][test_on_metric].values)
                    test_perforned = 't-test'
                else:
                    _, p_value = stats.wilcoxon(samples_sub[samples_sub[MODEL_SHORT_NAME_COL]==pair[0]][test_on_metric].values,
                                                samples_sub[samples_sub[MODEL_SHORT_NAME_COL]==pair[1]][test_on_metric].values,
                                                correction=False)
                    test_perforned = 'wilcoxon'
            else:
                if test_type=='mannwhitneyu':
                    _, p_value = stats.mannwhitneyu(samples_sub[samples_sub[MODEL_SHORT_NAME_COL]==pair[0]][test_on_metric].values.astype(float),
                                                samples_sub[samples_sub[MODEL_SHORT_NAME_COL]==pair[1]][test_on_metric].values.astype(float),
                                                alternative='two-sided')
                    print(f"Comparing results of two groups with {n_points} points each")
                    test_perforned = 'mannwhitneyu'
                else:
                    raise ValueError(f"Test type {test_type} not supported")
            _, levene_p_value = stats.levene(samples_sub[samples_sub[MODEL_SHORT_NAME_COL]==pair[0]][test_on_metric].values,
                                    samples_sub[samples_sub[MODEL_SHORT_NAME_COL]==pair[1]][test_on_metric].values)
            paired_test_sub.loc[i,:] = [pair[0],pair[1], p_value, test_perforned, levene_p_value, sub, model_i_to_j]
            # Check on the standard deviation of the difference between the two models using levene test
            
            print(f"Levene test p-value for {pair[0]} and {pair[1]}: {levene_p_value}")
            print(f"Paired test p-value for {pair[0]} and {pair[1]}: {p_value}\n")
        paired_test_df = pd.concat([paired_test_df, paired_test_sub], axis=0)
    return paired_test_df



def plot_snn_pred_with_overlap(snn_pred, snn_true, fig, n_mvcs:int, mvc_samples:int,stride_size:float,
                               dataset, mvc_i, rep_id_one_indexed:int,
                               bin_size:float=0.08, ol:int=50, plot_color:str=None, linestyle=None, alpha=None,
                               y_lims:list=[], sub_sample_pred:bool=True,
                               plot_snn_true_force:bool=False):
    """a
    This function is a smaller version of force_regression.plotting.plot_with_overlap.
    The reason we need this function is that the predictions of the SNN are further discretized by dt (simulation time step)

    """
    true_force_alpha=0.5
    true_linestyle='solid'
    
    bin_size_in_samp = bin_size / dataset.dt
    time_axis = np.arange(0, mvc_samples, bin_size_in_samp)
    total_dur = time_axis.shape[0] * stride_size
    print(time_axis.shape, total_dur)
    start_ar = dataset.segments_start[mvc_i] #compute_windows_start_times(bin_size, ol/100, total_dur)

    seg_nsteps = int(bin_size/dataset.dt)

    stride_in_steps = int(bin_size* (1 - ol/100)/dataset.dt)
    print(f"stride in steps:{stride_in_steps}   seg_nsteps:{seg_nsteps}")
    arr_pointer = 0
    active_fingers_order = dataset.active_fingers_order[rep_id_one_indexed]
    for i, target_finger in enumerate(active_fingers_order):#(range(n_fingers)):
        for row in range(n_mvcs):
            ax = fig.axes[n_mvcs*target_finger-1 + row] # since we plotted the other model first the figure already contains the axes
            sub_sampled_time = []
            sub_sampled_pred = []
            for slice_i, s in enumerate(start_ar):
                # Rely on samples then convert later the x-axis to time
                xaxis = np.arange(slice_i * stride_in_steps, slice_i * stride_in_steps  + seg_nsteps)
                time_ax = xaxis * dataset.dt
                snn_pred_slice = snn_pred[arr_pointer : arr_pointer+ seg_nsteps, target_finger-1]
                snn_true_slice = snn_true[arr_pointer : arr_pointer+ seg_nsteps, target_finger-1]
                if not sub_sample_pred:
                    ax.plot(time_ax, snn_pred_slice, color=plot_color, alpha=alpha, linestyle=linestyle)
                    if plot_snn_true_force:
                        ax.plot(time_ax, snn_true_slice, color=COLORS['midnight_blue'], alpha=true_force_alpha,
                            linestyle=true_linestyle)
                sub_sampled_time.append(time_ax[0])
                sub_sampled_pred.append(snn_pred_slice[0])

                arr_pointer += seg_nsteps
            if sub_sample_pred:
                print(f"original predictions:{snn_pred_slice.shape}")
                ax.plot(sub_sampled_time, sub_sampled_pred, color=plot_color, alpha=alpha, linestyle=linestyle)
            ax.set_ylim(y_lims)
    


def plot_pred_of_models_target_only(data_config:DataConfig, y_compiled_df, subjects:List[str],
                                    emg_types:List[str], segment_hold:bool,
                                    sign_mvcs:List[int], mvcs:List[int], input_types:List[str],
                                    reps:List[int], models_palette_dict:Dict[str, str],
                                    map_label_to_legend:Dict[str,str],
                                    plot_filt_y=False, ylim_list=[], save_figure=False,
                                    pred_force_alpha=0.8, pred_linestyle='dashed', net_params_string:dict=None,
                                    compiled_results_dir:str=None, **kwargs):
    """
    Plots the predicted force of the models on the target finger only.
    The same function can be used with the SNN predictions as well.
    """
    assert len(subjects) == len(emg_types) == len(sign_mvcs) == len(mvcs) == len(input_types) == len(reps), "All lists must have the same length"
    n_models = len(emg_types)
    n_fingers = 3 if mvcs[0]==20 or mvcs[0]==30 or mvcs[0]==50 else data_config.n_ind_fingers
    n_mvcs = 2 if mvcs[0] == 515 else 1
    # figure styling
    fig_length = 4 if n_mvcs == 2 else 3

    fig = plt.figure(figsize=(10, fig_length))
    true_force_alpha=0.5
    true_linestyle='solid'
    plot_var = "y_pred_test_post" if plot_filt_y else "y_pred_test"

    ol = y_compiled_df[OL_COL].dropna().unique() # get the common overlap value

    gs = gridspec.GridSpec(nrows=n_mvcs, ncols=n_fingers)

    model_names = [f"{emg_types[i].lower()} {input_types[i].lower()}" for i in range(n_models)]
    print(f"Comparing models:{model_names}")
    for finger in range(n_fingers):
        for m in range(n_mvcs):
            ax = fig.add_subplot(gs[m, finger])
            for i in range(n_models):
                mask = (y_compiled_df[SUBJECT_COL] == subjects[i]) & (y_compiled_df[EMG_COL] == emg_types[i]) & (y_compiled_df[SIGN_MVC_COL] == sign_mvcs[i]) & (
                    y_compiled_df[MVC_COL] == mvcs[i]) & (y_compiled_df[INPUT_TYPE_COL] == input_types[i]) & (y_compiled_df[TEST_ON_REP] == reps[i]) & (y_compiled_df[SEGMENT_HOLD_COL]==segment_hold)
                temp_df = y_compiled_df[mask].iloc[0]
                current_finger_order = temp_df[FING_ORDER_COL].index(finger+1) # considering that the fingers are shuffled in order
                fing_samples = int(temp_df[plot_var].shape[0]/ n_fingers)
                mvc_samples = int(fing_samples/n_mvcs)
                active_fid = current_finger_order * fing_samples
                mvc_fid = active_fid + mvc_samples * m
                model_color = models_palette_dict[model_names[i]]
                bin_size =  temp_df[INPUT_DUR_COL] if input_types[i] == INPUT_TYPE_SP_TIMES else temp_df[WS_COL]
                # ol = temp_df[OL_COL]
                stride_size = bin_size - (bin_size * ol / 100)
                time_axis = np.arange(0, mvc_samples * stride_size, stride_size)

                if input_types[i] != "MU spike times":
                    ax.plot(time_axis, temp_df[plot_var][mvc_fid: mvc_fid+ mvc_samples, finger],
                            alpha=pred_force_alpha,
                            linestyle=pred_linestyle,
                            label=f"Filtered {fing_order[finger]}",
                            color=model_color)
                    if ylim_list:
                        ax.set_ylim(ylim_list[m])
                    else:
                        ax.set_ylim([-1, temp_df[plot_var].max() + 1])

                    if i==0:
                        # CHECKTHIS: check that the 2 models have exactly the same true force
                        print(f"True force obtained from {model_names[i]}")
                        ax.plot(time_axis, temp_df["y_true_test"][mvc_fid: mvc_fid + mvc_samples, finger],
                                alpha=true_force_alpha,
                                label=f"True {fing_order[finger]}",
                                color=COLORS['midnight_blue'])
             
                plt.box(False)
            if m==0:
                ax.set_title(f"{fing_order[finger].lower()}")
            if finger == 0:
                ax.set_ylabel(FORCE_MVC_LABEL)
        ax.set_xlabel(TIME_LABEL)
        # ax.set_xlim([2,4])

    if input_types[1] == "MU spike times":
        print(kwargs)
        snn_dataset = kwargs['dataset']
        mask = (y_compiled_df[SUBJECT_COL] == subjects[1]) & (y_compiled_df[EMG_COL] == emg_types[1]) & (y_compiled_df[SIGN_MVC_COL] == sign_mvcs[1]) & (
                y_compiled_df[MVC_COL] == mvcs[1]) & (y_compiled_df[INPUT_TYPE_COL] == input_types[1]) & (y_compiled_df[TEST_ON_REP] == reps[1])

        temp_df = y_compiled_df[mask].iloc[0]
        print(temp_df[plot_var].shape)
     
        plot_snn_pred_with_overlap(temp_df[plot_var], temp_df['y_true_test'], fig, n_mvcs,
                                   mvc_samples, stride_size, snn_dataset, mvcs[1], reps[1],
                                   bin_size=bin_size, ol=ol,
                                   plot_color=models_palette_dict[model_names[1]],
                                   linestyle=pred_linestyle,
                                   alpha=pred_force_alpha, y_lims=ylim_list, sub_sample_pred=False)


    handles = [ plt.Line2D([0], [0], color=COLORS['midnight_blue'],  lw=2, alpha=true_force_alpha, linestyle=true_linestyle)] +\
              [plt.Line2D([0], [0], color=models_palette_dict[model_names[i]],  lw=2, alpha=pred_force_alpha, linestyle=pred_linestyle) for i in range(n_models)
              ]
       
    labels = ['Target force'] + [f'Predicted force {map_label_to_legend[model_names[i]]} ' for i in range(n_models)]

    _ = fig.legend(handles=handles,
                     labels=labels,
                     loc='upper center', bbox_to_anchor=(0.5, 1.1),
                     ncols=n_models+1,
                     frameon=False,
                     prop={'size': 8},
                    )
    fig.tight_layout()

    if save_figure:
        # append all model names in a string
        models_string = ""
        for i in range(n_models):
            models_string += f'{model_names[i]}_'
        models_string = models_string[:-1]
        if net_params_string:
            models_string += f'_{net_params_string}'
        filename = f"{subjects[i]}_ypred_mvc_{mvcs[0]}_testrep_{reps[0]}_testsign_{sign_mvcs[0]}_{models_string}_segmenthold_{segment_hold}.png"
        print(f"Saving figure to {filename}")
        plt.savefig(os.path.join(compiled_results_dir, filename), dpi=300, bbox_inches="tight")


def create_filter(df, params, topology):
    """
    Create a filter to select the rows in the dataframe that match the given parameters
    """
    mask = (df[TAU_SYN_COL] == params[TAU_SYN_COL]) & \
           (df[TAU_MEM_COL] == params[TAU_MEM_COL]) & \
           (df[TAU_SYN_INIT_COL] == params[TAU_SYN_INIT_COL]) & \
           (df[TAU_MEM_INIT_COL] == params[TAU_MEM_INIT_COL]) & \
           (df[TAU1_MEM_FILT_COL] == params[TAU1_MEM_FILT_COL]) & \
           (df[TAU2_MEM_FILT_COL] == params[TAU2_MEM_FILT_COL]) & \
           (df[EPOCHS_COL] == params[EPOCHS_COL]) & \
           (df[TOPOLOGY_COL] == topology)
    return mask

def get_true_force_from_conv_model(y_compiled_df:pd.DataFrame, subjects:List[str],
                                    emg_types:List[str], sign_mvcs:List[int], mvcs:List[int],
                                    reps:List[int],
                                    n_mvcs:int, finger:int,n_fingers:int=5, ol:float=50):
    """
    Get the true force from the compiled dataframe for a specific finger and MVC from the conventional model
    since it smoothened the force.
    """
    mask = (y_compiled_df[SUBJECT_COL] == subjects[0]) & (y_compiled_df[EMG_COL] == emg_types[0]) & (y_compiled_df[SIGN_MVC_COL] == sign_mvcs[0]) & (
                    y_compiled_df[MVC_COL] == mvcs[0]) & (y_compiled_df[INPUT_TYPE_COL] == INPUT_TYPE_EMG_GLOB) & (y_compiled_df[TEST_ON_REP] == reps[0])
    temp_df = y_compiled_df[mask].iloc[0]
    current_finger_order = temp_df[FING_ORDER_COL].index(finger+1) # considering that the fingers are shuffled in order
    fing_samples = int(temp_df["y_pred_test_post"].shape[0]/ n_fingers)
    mvc_samples = int(fing_samples/n_mvcs)
    active_fid = current_finger_order * fing_samples

    bin_size = temp_df[WS_COL]
    stride_size = bin_size - (bin_size * ol / 100)
    time_axis = np.arange(0, mvc_samples * stride_size, stride_size)

    return time_axis, temp_df["y_true_test"][active_fid: active_fid + mvc_samples, finger]



def plot_pred_of_models_target_only_different_topologies(data_config: DataConfig, snn_dataset,
                                                         y_compiled_df,
                                                         subjects: List[str],
                                                         emg_types: List[str],
                                                         sign_mvcs: List[int],
                                                         mvcs: List[int],
                                                         input_types: List[str],
                                                         reps: List[int],
                                                         topology_parameters: Dict[str, Dict[str, float]],
                                                         models_palette_dict: Dict[str, str],
                                                         ylim:list,
                                                         xlim:list,
                                                         annotation_duration:int=1, # in seconds
                                                         save_figure=False,
                                                         compiled_results_dir:str=None,
                                                         pred_force_alpha=0.8, pred_linestyle='dashed',
                                                         show_topology=[SHALLOW_LEAKY, SHALLOW_SPIKING_DOUBLE_FILTER],
                                                         show_custom_annotation:bool=False,
                                                         ):
    """
    Plots the predicted force of the models on the target finger only.
    The same function can be used with the SNN predictions as well.
    """
    assert len(subjects) == len(emg_types) == len(sign_mvcs) == len(mvcs) == len(input_types) == len(reps), "All lists must have the same length"
    n_models = len(emg_types)
    n_fingers = 3 if mvcs[0]==20 or mvcs[0]==30 or mvcs[0]==50 else data_config.n_ind_fingers
    n_mvcs = 2 if mvcs[0] == 515 else 1

    ol = y_compiled_df[OL_COL].dropna().unique()

    # figure styling
    fig_length = 4 if n_mvcs == 2 else 3
    fig = plt.figure(figsize=(10, fig_length))
    true_force_alpha=0.5
    true_linestyle='solid'
    plot_var = "y_pred_test_post" #if plot_filt_y else "y_pred"

    gs = gridspec.GridSpec(nrows=n_mvcs, ncols=n_fingers)

    model_names = [f"{emg_types[i].lower()} {input_types[i].lower()}" for i in range(n_models)]
    figure_labels = []
    figure_handles = []
    print(f"Comparing models:{model_names}")
    for finger in range(n_fingers):
        for m in range(n_mvcs):
            ax = fig.add_subplot(gs[m, finger])
            for i in range(n_models):
                mask = (y_compiled_df[SUBJECT_COL] == subjects[i]) & (y_compiled_df[EMG_COL] == emg_types[i]) & (y_compiled_df[SIGN_MVC_COL] == sign_mvcs[i]) & (
                    y_compiled_df[MVC_COL] == mvcs[i]) & (y_compiled_df[INPUT_TYPE_COL] == input_types[i]) & (y_compiled_df[TEST_ON_REP] == reps[i])
                temp_df = y_compiled_df[mask].iloc[0]
                current_finger_order = temp_df[FING_ORDER_COL].index(finger+1) # considering that the fingers are shuffled in order
                fing_samples = int(temp_df[plot_var].shape[0]/ n_fingers)
                mvc_samples = int(fing_samples/n_mvcs)
                active_fid = current_finger_order * fing_samples
                mvc_fid = active_fid + mvc_samples * m
            
                if input_types[i] != INPUT_TYPE_SP_TIMES:
                    model_color = models_palette_dict[model_names[i]]
                    bin_size =  temp_df[INPUT_DUR_COL] if input_types[i] == INPUT_TYPE_SP_TIMES else temp_df[WS_COL]
                    stride_size = bin_size - (bin_size * ol / 100)
                    time_axis = np.arange(0, mvc_samples * stride_size, stride_size)
                    if finger ==0:
                        figure_labels.extend(['Target force', 'Predicted force (baseline)'])
                        figure_handles.extend([plt.Line2D([0], [0], color=COLORS['midnight_blue'],
                                                          lw=2, alpha=0.6,
                                                          linestyle=true_linestyle),
                                                plt.Line2D([0], [0], color=models_palette_dict[model_names[i]],
                                                           lw=2, alpha=LEGEND_ALPHA,
                                                           linestyle=pred_linestyle)])

                    ax.plot(time_axis, temp_df[plot_var][mvc_fid: mvc_fid+ mvc_samples, finger],
                            alpha=pred_force_alpha, linestyle=pred_linestyle,
                            label=f"Filtered {fing_order[finger]}",
                            color=model_color)
                    if ylim:
                        ax.set_ylim(ylim)
                    else:
                        ax.set_ylim([-1, temp_df[plot_var].max() + 1])
                    if i==0:
                        ax.plot(time_axis, temp_df["y_true_test"][mvc_fid: mvc_fid + mvc_samples, finger],
                                alpha=true_force_alpha,
                                label=f"True {fing_order[finger]}",
                                color=COLORS['midnight_blue'])
             
                # add custom vertical lines at 5 and 10
                if show_custom_annotation:
                    ax.vlines(x=5, ymin=0, ymax=22, color=COLORS['midnight_blue'], linestyle='dotted', lw=1, alpha=0.4)
                    ax.vlines(x=10,ymin=0, ymax=22, color=COLORS['midnight_blue'], linestyle='dotted', lw=1, alpha=0.4)
                plt.box(False)
            if m==0:
                ax.set_title(f"{fing_order[finger].lower()}")
            if finger == 0:
                ax.set_ylabel(FORCE_MVC_LABEL)
            else:
                ax.set_yticks([])
        ax.set_xlabel(TIME_LABEL, fontsize=LABEL_FONTSIZE)
        if xlim:
            ax.set_xlim(xlim)
            annotation_xshift = 0.2 * (xlim[-1]- xlim[0])#1 if xlim[-1]-xlim[0] > 4 else 0
            if ylim:
                ax.hlines(ylim[0], xmin=xlim[0]+ annotation_xshift,
                        xmax=xlim[0]+ annotation_xshift+ annotation_duration,
                        colors="black", linestyles="solid", lw=2)
            
                plt.annotate(f"{annotation_duration} sec",
                            xy=(xlim[0]+ annotation_xshift, ylim[0]+0.5),
                            xycoords='data',fontsize=10,ha="left"
                            )

            # annotate time resolution/duration
            ax.set_xticks([])
            ax.set_xlabel('')
      
    if  input_types[1]==INPUT_TYPE_SP_TIMES:
        bin_size =  temp_df[INPUT_DUR_COL]
        stride_size = bin_size - (bin_size * ol / 100)
        topology_color_dict = {topology:models_palette_dict[f'intra {INPUT_TYPE_SP_TIMES.lower()} {topology}'] for topology in topology_parameters.keys()}
        plot_topology_based_snn_predictions(snn_dataset, y_compiled_df, subjects[1], emg_types[1],
                                            sign_mvcs[1], mvcs[1], input_types[1], reps[1],
                                            topology_color_dict, topology_parameters,
                                            n_mvcs, fig, plot_var,
                                            figure_labels, figure_handles, mvc_samples, bin_size,
                                            stride_size,show_topology,
                                            plot_conv_true_force=True)
  
    _ = fig.legend(handles=figure_handles, labels=figure_labels, fontsize=12,
                   loc='upper center', bbox_to_anchor=(0.5, 1.14),
                   ncols=5, frameon=False, prop={'size': 10},
                   )
    fig.tight_layout()

    if save_figure:
        # append all model names in a string
        models_string = ""
        for i in range(n_models):
            models_string += f'{model_names[i]}_'
        # remove the last underscore
        models_string = models_string[:-1]
        topology_string = "" 
        for top in show_topology:
            if 'encoding' in top:
                topology_string += f'encoding_{top.split("_")[2]}_'
            else:
                topology_string += f'{top.split("_")[1]}_'
        print(f"topology string:{topology_string}")
        # if post_process_filt:
        #     topology_string += 'postfilt_'
        if xlim:
            filename = f"{subjects[i]}_ypred_{mvcs[0]}_{topology_string}xlim_{xlim[0]}_{xlim[1]}_testrep_{reps[0]}_testsign_{sign_mvcs[0]}_{models_string}.png"
        else:
            filename = f"{subjects[i]}_ypred_{mvcs[0]}_{topology_string}testrep_{reps[0]}_testsign_{sign_mvcs[0]}_alif.png"
        print(f"Saving figure to {filename}")
        plt.savefig(os.path.join(compiled_results_dir, filename), dpi=300, bbox_inches="tight")


def plot_topology_based_snn_predictions(snn_dataset, y_compiled_df, subject, emg_type,
                                         sign_mvc, mvc, input_type, rep_one_indexed,
                                         topology_color_dict,
                                         topology_parameters,
                                         n_mvc, fig, plot_var, figure_labels,
                                         figure_handles, mvc_samples, bin_size, stride_size,
                                         show_topology:List[str],
                                         plot_conv_true_force:bool=False):
    """
    Plot the SNN predictions based on the topology
    """
    topology_line_alpha = {}
    topology_line_style = {}
    topology_legend_name_map = {SHALLOW_LEAKY:'LI', SHALLOW_SPIKING_DOUBLE_FILTER:'LIF',
                                ENC_SHALLOW_LEAKY:'LI',
                                ENC_SHALLOW_SPIKING:'LIF'}
    for topology in show_topology:
        topology_line_alpha[topology]=0.8
        topology_line_style[topology]='solid'
        if 'encoding' in topology:
            topology_line_alpha[topology] = 0.5
    
    for _, topology_name in enumerate(show_topology):
        if "encoding" in topology_name:
            topology_legend_label = topology_legend_name_map[topology_name]
            figure_labels.append(f'Predicted force (encoding + {topology_legend_label})')
        else:
            topology_legend_label = topology_legend_name_map[topology_name]
            figure_labels.append(f'Predicted force ({topology_legend_label})')
        figure_handles.append(plt.Line2D([0], [0], color=topology_color_dict[topology_name], lw=2,
                                            alpha=topology_line_alpha[topology_name],
                                            linestyle=topology_line_style[topology_name]))
        
        mask = (y_compiled_df[SUBJECT_COL] == subject) & (y_compiled_df[EMG_COL] == emg_type) & (y_compiled_df[SIGN_MVC_COL] == sign_mvc) & (
                y_compiled_df[MVC_COL] == mvc) & (y_compiled_df[INPUT_TYPE_COL] == input_type) & (y_compiled_df[TEST_ON_REP] == rep_one_indexed)
        # if "encoding" in topology_name:
        #     net_mask = create_filter(y_compiled_df, topology_parameters[topology_name][subject], topology_name)
        # else:
        #     net_mask = create_filter(y_compiled_df, topology_parameters[topology_name], topology_name)
        net_mask = y_compiled_df[TOPOLOGY_COL]==topology_name
        temp_df = y_compiled_df[mask & net_mask].iloc[0]
        # temp_df = y_compiled_df[mask].iloc[0]
        print(temp_df[plot_var].shape)

        # invert boolean
        plot_snn_true_force = not plot_conv_true_force
        plot_snn_pred_with_overlap(temp_df[plot_var], temp_df['y_true_test'], fig, n_mvc,
                                mvc_samples,stride_size, snn_dataset,
                                mvc, rep_one_indexed,
                                bin_size=bin_size,
                                plot_color=topology_color_dict[topology_name],
                                linestyle=topology_line_style[topology_name],
                                y_lims= None, sub_sample_pred=False,
                                alpha = topology_line_alpha[topology_name],
                                plot_snn_true_force=plot_snn_true_force)




        