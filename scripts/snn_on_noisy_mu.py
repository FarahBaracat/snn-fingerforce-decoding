import hydra
import pandas as pd

import force_regression.training.snn_pipeline as snn_prep
import force_regression.utils.configuration as lrc
from force_regression.evaluation.metrics import aggregate_snn_metrics_and_log
from force_regression.models.snn import run_net_single_split
from force_regression.utils.configuration import flatten_hydra_config, load_project_configuration
from force_regression.training.snn_pipeline import SnnMusDatasetPreparation


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="snn")
def run_snn_on_noisy_mu(cfg):
    """
    Main function that trains and reports on the SNN models with noisy MU data.
    """
    data_root_dir, results_root_dir, subject_name_encoding, _ = load_project_configuration(
        'configs/config.json')

    # Create config from hydra (returns wandb.config if use_wandb=True, else cfg)
    snn_config = snn_prep.create_wandb_config_hydra(cfg)
    data_config = lrc.load_input_data_configuration_from_omega(snn_config, data_root_dir,
                                                               results_root_dir,
                                                               subject_name_encoding)

    # Build flat config for functions that expect flat attribute access
    flat_snn_config = flatten_hydra_config(snn_config)
    flat_snn_config = lrc.update_finger_ids(data_config, flat_snn_config)

    mvcs_string = snn_prep.create_mvcs_string(data_config)
    print(f"Config from file: {snn_config}")
    snn_prep.modify_snn_temp_data_path(data_config)

    mu_df_sorted, force_df, data_config = snn_prep.load_mu_data_for_snn(
        data_config,
        snn_config.decoder_type["use_file_data"],
        mvcs_string,
        snn_config)

    # Update directories after loading MUs (data_config is reset when loading from file)
    snn_prep.create_snn_results_and_figs_dirs(data_config,
                                              snn_config.decoder_type["topology"],
                                              snn_config.task["noise_mode"])

    kfolds_reps = snn_prep.get_kfold_reps(flat_snn_config, snn_config.training["train_with_cv"])
    metrics_df_cv = pd.DataFrame()
    y_df_cv = pd.DataFrame()
    trained_params_df_cv = pd.DataFrame()

    for fold_i, rep_on_rep in enumerate(kfolds_reps):
        file_name = snn_prep.create_snn_results_filename_suffix(snn_config,
                                                                data_config,
                                                                mvcs_string,
                                                                kfolds_reps,
                                                                snn_config.training["train_with_cv"],
                                                                rep_on_rep)
        print(f"Fold {fold_i} trainrep_on_testrep: {rep_on_rep}\n-------------------")

        prepared_snndata = SnnMusDatasetPreparation(mu_df_sorted.copy(),
                                                    force_df,
                                                    data_config,
                                                    flat_snn_config,
                                                    snn_config.task["select_dir"],
                                                    'train',
                                                    snn_config.task["shuffle_fingers_seed"])
        offline_dataset = prepared_snndata.create_snn_dataset()
        snn_prep.save_dataset_to_file(offline_dataset, 'offline', data_config=data_config,
                                      percent_omission=snn_config.task["percent_omission"])

        metrics_df, y_df, trained_params_df, _, _ = run_net_single_split(
            snn_config,
            data_config,
            offline_dataset,
            rep_on_rep,
            fold_i,
            use_inference_network=False)

        metrics_df_cv = pd.concat([metrics_df_cv, metrics_df], axis=0)
        y_df_cv = pd.concat([y_df_cv, y_df], axis=0)
        trained_params_df_cv = pd.concat([trained_params_df_cv, trained_params_df], axis=0)

        if snn_config.decoder_type["use_inf_network"]:
            prepared_snndata = SnnMusDatasetPreparation(mu_df_sorted.copy(),
                                                        force_df,
                                                        data_config,
                                                        flat_snn_config,
                                                        snn_config.task["select_dir"],
                                                        'inf',
                                                        snn_config.task["shuffle_fingers_seed"])
            online_dataset = prepared_snndata.create_snn_dataset()
            snn_prep.save_dataset_to_file(online_dataset, 'online', data_config=data_config)

            metrics_df, y_df, _, rec_ts_online, _ = run_net_single_split(
                snn_config,
                data_config,
                online_dataset,
                rep_on_rep,
                fold_i,
                use_inference_network=True)

            metrics_df_cv = pd.concat([metrics_df_cv, metrics_df], axis=0)
            y_df_cv = pd.concat([y_df_cv, y_df], axis=0)
            snn_prep.save_recorded_vars_in_inference(data_config,
                                                     rec_ts_online,
                                                     int(rep_on_rep.split('on')[1]),
                                                     file_name)

    aggregate_snn_metrics_and_log(metrics_df_cv, flat_snn_config)

    if snn_config.logging["save_metrics"]:
        snn_prep.write_results_to_files(
            data_config, metrics_df_cv, y_df_cv, trained_params_df_cv, file_name)


if __name__ == '__main__':
    run_snn_on_noisy_mu()
