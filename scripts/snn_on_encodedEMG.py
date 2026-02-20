import pandas as pd
import hydra

import force_regression.training.snn_pipeline as snn_prep
import force_regression.utils.functions as fnc
import force_regression.utils.configuration as lrc
from force_regression.evaluation.metrics import aggregate_snn_metrics_and_log
from force_regression.evaluation.profiling import (
    aggregate_neurobench_across_folds,
    aggregate_profiling_across_folds,
    create_profiling_log_dir,
    run_snn_fold_profiling,
)
from force_regression.models.snn import run_net_single_split
from force_regression.training.snn_pipeline import prepare_emg_split, build_unsegmented_emg_dataset
from force_regression.utils.configuration import flatten_hydra_config, load_project_configuration


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="snn")
def run_snn_on_encoding(cfg):
    """Train and evaluate an SNN on encoded surface EMG spike trains."""
    data_root_dir, results_root_dir, subject_name_encoding, _ = load_project_configuration(
        'configs/config.json'
    )
    snn_config = snn_prep.create_wandb_config_hydra(cfg)
    data_config = lrc.load_input_data_configuration_from_omega(
        snn_config, data_root_dir, results_root_dir, subject_name_encoding
    )
    flat_snn_config = flatten_hydra_config(snn_config)

    mvcs_string = snn_prep.create_mvcs_string(data_config)
    print(f"Config: {snn_config}")
    print(f"Enc tau mem: {snn_config.decoder_type['enc_tau_mem']}")
    print(f"Enc spk thresh: {snn_config.decoder_type['enc_spk_threshold']}\n")

    data_config.direction = snn_config.task["exp_dir"]
    snn_prep.modify_snn_temp_data_path(data_config)

    emg_force_df = snn_prep.load_emg_data_for_snn(data_config, snn_config.decoder_type["use_file_data"])
    snn_prep.create_snn_results_and_figs_dirs(
        data_config, snn_config.decoder_type["topology"], snn_config.task["noise_mode"]
    )

    kfolds_reps = snn_prep.get_kfold_reps(flat_snn_config, snn_config.training["train_with_cv"])
    enable_profiling = snn_config.logging["enable_profiling"]

    metrics_df_cv = pd.DataFrame()
    y_df_cv = pd.DataFrame()
    trained_params_df_cv = pd.DataFrame()
    all_profiling_results = []
    all_neurobench_results = []

    for fold_i, rep_on_rep in enumerate(kfolds_reps):
        file_name = snn_prep.create_snn_results_filename_suffix(
            snn_config, data_config, mvcs_string, kfolds_reps,
            snn_config.training["train_with_cv"], rep_on_rep,
        )
        print(f"Fold {fold_i} with rep_on_rep: {rep_on_rep}\n-------------------")

        # Offline (training) split
        _, offline_dataset = prepare_emg_split(emg_force_df, data_config, snn_config, 'train')
        snn_prep.save_dataset_to_file(offline_dataset, 'encoding_offline', data_config=data_config)

        metrics_df, y_df, trained_params_df, _, _ = run_net_single_split(
            snn_config, data_config, offline_dataset, rep_on_rep, fold_i,
            use_inference_network=False,
        )
        metrics_df_cv = pd.concat([metrics_df_cv, metrics_df], axis=0)
        y_df_cv = pd.concat([y_df_cv, y_df], axis=0)
        trained_params_df_cv = pd.concat([trained_params_df_cv, trained_params_df], axis=0)

        # Online (inference) split
        if snn_config.decoder_type["use_inf_network"]:
            prepared_online, online_dataset = prepare_emg_split(
                emg_force_df, data_config, snn_config, 'inf'
            )
            snn_prep.save_dataset_to_file(online_dataset, 'encoding_online', data_config=data_config)

            metrics_df, y_df, _, rec_ts_online, _ = run_net_single_split(
                snn_config, data_config, online_dataset, rep_on_rep, fold_i,
                use_inference_network=True,
            )
            metrics_df_cv = pd.concat([metrics_df_cv, metrics_df], axis=0)
            y_df_cv = pd.concat([y_df_cv, y_df], axis=0)
            snn_prep.save_recorded_vars_in_inference(
                data_config, rec_ts_online, int(rep_on_rep.split('on')[1]), file_name
            )

            if enable_profiling:
                subject_id = fnc.remap_subject(data_config.subject, data_config.subj_map)
                unsegmented_dataset = build_unsegmented_emg_dataset(prepared_online, snn_config)
                profiling_results, neurobench_results = run_snn_fold_profiling(
                    snn_config, data_config, unsegmented_dataset, rep_on_rep, fold_i, subject_id
                )
                all_profiling_results.append(profiling_results)
                if neurobench_results is not None:
                    all_neurobench_results.append(neurobench_results)

    # Aggregate profiling across folds
    if enable_profiling and all_profiling_results:
        subject_id = fnc.remap_subject(data_config.subject, data_config.subj_map)
        topology = snn_config.decoder_type["topology"]
        aggregate_profiling_across_folds(
            all_profiling_results,
            log_dir=create_profiling_log_dir(
                base_dir="./logs/snn_profiler",
                subject=subject_id,
                direction=data_config.direction,
                model_type=f"snn_{topology}_summary",
            ),
            model_name="SNN",
        )

    if enable_profiling and all_neurobench_results:
        subject_id = fnc.remap_subject(data_config.subject, data_config.subj_map)
        topology = snn_config.decoder_type["topology"]
        aggregate_neurobench_across_folds(
            all_neurobench_results,
            log_dir=create_profiling_log_dir(
                base_dir="./logs/snn_profiler",
                subject=subject_id,
                direction=data_config.direction,
                model_type=f"snn_{topology}_neurobench_summary",
            ),
            model_name="SNN",
        )

    aggregate_snn_metrics_and_log(metrics_df_cv, flat_snn_config)

    if snn_config.logging["save_metrics"]:
        snn_prep.write_results_to_files(
            data_config, metrics_df_cv, y_df_cv, trained_params_df_cv, file_name
        )


if __name__ == '__main__':
    run_snn_on_encoding()
