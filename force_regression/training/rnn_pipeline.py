import os
import torch
import pickle as pkl
import pandas as pd
from force_regression.utils.io import create_folder

def create_rnn_results_and_figs_dirs(data_config, topology, percent_omission=0):
    """
    Update the results path in case of load_multi by creating a subfolder. 
    This is similar to SNN case.
    # figs_snn_dir updates the full path automatically figs_snn_path
    # results dir is the parent folder, results_path is the full path adding results_dir + subject
    # here we have to create the folder as the results_path is not updated automatically
    """
    data_config.figs_snn_dir = topology
    data_config.results_path = os.path.join(data_config.results_path, topology)
    create_folder(data_config.results_path)
    if percent_omission is not None and percent_omission > 0:
        data_config.results_path = os.path.join(data_config.results_path, 'noise')
        create_folder(data_config.results_path)

    if len(data_config.load_multi) > 1:
        data_config.results_path = os.path.join(data_config.results_path, 'multi_mvc')
        create_folder(data_config.results_path)


def create_rnn_results_filename_suffix(rnn_config, data_config, mvcs, kfolds_reps,
                                       train_with_cv:bool,
                                       rep_on_rep:str):
    """
    Generate the suffix to be attached to the pkl results file from SNN training.
    """
    file_prefix = f'rnn_{rnn_config.rnn_type}_ep_{rnn_config.num_iter}_seed_{rnn_config.torch_seed}_noise_{rnn_config.percent_omission}_ns_{rnn_config.noise_torch_seed}'
    file_prefix += f'_hidsize_{rnn_config.rnn_hidden_size}_expdecay_{rnn_config.exp_decay_tau}_kernelsize_{rnn_config.exp_kernel_size}_rnnbias_{rnn_config.rnn_bias}_normalizeforce_{rnn_config.normalize_output}_fcout_{rnn_config.project_output_with_fc}'
    file_suffix = f'_dt_{rnn_config.dt}_sign_{data_config.sign_mvc}_mvc_{mvcs}_kfold_{len(kfolds_reps)}'
    file_prefix += f'{file_suffix}_hold_{data_config.segment_hold}'
    if train_with_cv:
        return 'cv_' + file_prefix
    return file_prefix + f'_rep_{rep_on_rep}'

def write_rnn_results_to_files(data_config, metrics_df_cv, y_df_cv, trained_weights_df_cv, file_name):
    """
    Write the results of the SNN training to a pkl file.
    """
    pkl.dump(metrics_df_cv, open(os.path.join(
    data_config.results_path, f'metrics_df_{file_name}.pkl'), 'wb'))
    pkl.dump(y_df_cv, open(os.path.join(
        data_config.results_path, f'y_df_{file_name}.pkl'), 'wb'))
    pkl.dump(trained_weights_df_cv, open(os.path.join(
        data_config.results_path, f'trparams_df_{file_name}.pkl'), 'wb'))


def organize_trained_weights(fold_i:int, input_dur:float, model) -> pd.DataFrame:
    """
    Prepares a dataframe with the trained weights.
    """
    trained_weights_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and name not in trained_weights_dict.keys():
            trained_weights_dict[name] = param.detach().clone()
            print(f"Adding parameter to trained weights dict:{name}")

    trained_weights_df = pd.DataFrame(columns=list(trained_weights_dict.keys()))
    for key in trained_weights_dict.keys():
        trained_weights_df.loc[0, key] = trained_weights_dict[key].cpu().numpy() if isinstance(trained_weights_dict[key], torch.Tensor) else trained_weights_dict[key]
    trained_weights_df.loc[0, 'fold'] = fold_i
    trained_weights_df.loc[0, 'input_dur'] = input_dur

    return trained_weights_df