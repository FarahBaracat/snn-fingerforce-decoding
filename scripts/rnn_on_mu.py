import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hydra
from torch.utils.data import Subset
from torchsummary import summary
import force_regression.utils.functions as fnc
import force_regression.utils.configuration as lrc
import force_regression.training.rnn_pipeline as rnn_prep
import force_regression.training.snn_pipeline as snn_prep
import wandb
from force_regression.evaluation.metrics import aggregate_snn_metrics_and_log, prepare_snn_metrics_df
from force_regression.evaluation.profiling import (
    aggregate_neurobench_across_folds,
    aggregate_profiling_across_folds,
    create_profiling_log_dir,
    run_rnn_fold_profiling,
)
from force_regression.models.rnn import ExponentialFilterRNN, evaluate_rnn, train_rnn
from force_regression.models.snn import prepare_predictions_df
from force_regression.utils.configuration import flatten_hydra_config, load_project_configuration
from force_regression.training.snn_pipeline import SnnMusDatasetPreparation, convert_predictions_to_np
from force_regression.plotting.time_series_plot import plot_training_loss, plot_predictions_sanity


@hydra.main(version_base=None, config_path="../configs/hydra", config_name="rnn")
def run_rnn_on_mu(cfg):
    """Train and evaluate an RNN on clean motor unit spike trains."""
    data_root_dir, results_root_dir, subject_name_encoding, _ = load_project_configuration(
        'configs/config.json'
    )

    # Create config from hydra
    rnn_config = snn_prep.create_wandb_config_hydra(cfg)
    data_config = lrc.load_input_data_configuration_from_omega(
        rnn_config, data_root_dir, results_root_dir, subject_name_encoding
    )
    mvcs_string = snn_prep.create_mvcs_string(data_config)

    print(f"Config from file: {rnn_config}")
    print(f"RNN type: {rnn_config.decoder_type['rnn_type']}")
    print(f"Hidden size: {rnn_config.decoder_type['rnn_hidden_size']}\n")

    data_config.direction = rnn_config.task["exp_dir"]
    snn_prep.modify_snn_temp_data_path(data_config)

    mu_df_sorted, force_df, data_config = snn_prep.load_mu_data_for_snn(
        data_config, rnn_config.decoder_type["use_file_data"], mvcs_string, rnn_config
    )

    # Create results directories
    rnn_prep.create_rnn_results_and_figs_dirs(
        data_config, rnn_config.decoder_type["topology"], rnn_config.task["percent_omission"]
    )

    # Build flat config before get_kfold_reps: when train_with_cv=False,
    # get_kfold_reps accesses snn_config.rep_on_rep (flat attribute).
    flat_rnn_config = flatten_hydra_config(rnn_config)
    kfolds_reps = snn_prep.get_kfold_reps(flat_rnn_config, rnn_config.training["train_with_cv"])
    enable_profiling = rnn_config.logging["enable_profiling"]

    # Prepare dataset
    prepared_snndata = SnnMusDatasetPreparation(
        mu_df_sorted.copy(), force_df, data_config, flat_rnn_config,
        rnn_config.task["select_dir"], 'train', rnn_config.task["shuffle_fingers_seed"],
    )
    dataset = prepared_snndata.create_rnn_dataset()

    # Normalize output if configured
    if rnn_config.decoder_type["normalize_output"]:
        dataset.labels = dataset.labels / 100  # normalize to [0,1] range

    dataset_rep_1 = Subset(dataset, np.arange(5))
    dataset_rep_2 = Subset(dataset, np.arange(5, 10))

    metrics_df_cv = pd.DataFrame()
    y_df_cv = pd.DataFrame()
    trained_weights_df_cv = pd.DataFrame()
    all_profiling_results = []
    all_neurobench_results = []

    for fold_i, rep_on_rep in enumerate(kfolds_reps):
        file_name = rnn_prep.create_rnn_results_filename_suffix(
            flat_rnn_config, data_config, mvcs_string, kfolds_reps,
            rnn_config.training["train_with_cv"], rep_on_rep,
        )
        test_rep = int(rep_on_rep.split('on')[1])
        print(f"Fold {fold_i} trainrep_on_testrep: {rep_on_rep}\n-------------------")

        test_dataset, train_dataset = snn_prep.assign_datasets_for_train_test(
            dataset_rep_1, dataset_rep_2, rep_on_rep
        )

        num_inputs = dataset.input.size(-1)
        num_outputs = dataset.labels.size(-1)
        timesteps = dataset.input.size(1)

        if rnn_config.decoder_type["topology"] == 'RNN':
            rnn_reg = ExponentialFilterRNN(
                input_size=num_inputs,
                output_size=num_outputs,
                hidden_size=rnn_config.decoder_type["rnn_hidden_size"],
                rnn_config=flat_rnn_config,
                dt=rnn_config.training["dt"],
                decay_tau=rnn_config.decoder_type["exp_decay_tau"],
                kernel_size=rnn_config.decoder_type["exp_kernel_size"],
                project_out=rnn_config.decoder_type["project_output_with_fc"],
                rnn_bias=rnn_config.decoder_type["rnn_bias"],
            )

            if fold_i == 0:
                summary(rnn_reg, (timesteps, num_inputs), verbose=2)

            tr_loss_per_epoch, _, ts_loss_every_nepochs, _ = train_rnn(
                rnn_reg, flat_rnn_config, train_dataset, test_dataset
            )
            print("Training complete!")

            _, train_predicted_values_cache, train_true_values_cache, _ = evaluate_rnn(rnn_reg,
                                                                                       flat_rnn_config,
                                                                                       train_dataset)

            if enable_profiling:
                subject_id = fnc.remap_subject(data_config.subject, data_config.subj_map)
                (_, test_predicted_values_cache, test_true_values_cache,
                 _, profiling_results, neurobench_results) = run_rnn_fold_profiling(rnn_reg,
                                                                                    flat_rnn_config,
                                                                                    test_dataset,
                                                                                    data_config,
                                                                                    rnn_config,
                                                                                    fold_i,
                                                                                    subject_id,
                                                                                    timesteps,
                                                                                    num_inputs,
                                                                                    )
                all_profiling_results.append(profiling_results)
                all_neurobench_results.append(neurobench_results)

            else:
                _, test_predicted_values_cache, test_true_values_cache, _ = evaluate_rnn(
                    rnn_reg, flat_rnn_config, test_dataset
                )

            plot_training_loss(
                tr_loss_per_epoch, ts_loss_every_nepochs,
                num_iter=rnn_config.training["num_iter"],
                log_ts_freq=rnn_config.logging["log_ts_freq"],
                use_wandb=rnn_config.logging["wandb"]["use_wandb"],
                wandb_run=wandb if rnn_config.logging["wandb"]["use_wandb"] else None,
            )
        else:
            raise ValueError(f"Topology {rnn_config.decoder_type['topology']} not recognized.")

        y_pred_test, y_true_test, y_pred_train, y_true_train = convert_predictions_to_np(
            train_predicted_values_cache, train_true_values_cache,
            test_predicted_values_cache, test_true_values_cache,
        )

        # Scale back to %MVC if normalized
        if rnn_config.decoder_type["normalize_output"]:
            y_pred_test = fnc.convert_to_mvc(y_pred_test)
            y_true_test = fnc.convert_to_mvc(y_true_test)
            y_pred_train = fnc.convert_to_mvc(y_pred_train)
            y_true_train = fnc.convert_to_mvc(y_true_train)

        metrics_df = prepare_snn_metrics_df(
            y_pred_train, y_true_train, y_pred_test, y_true_test, test_rep, fold_i,
            mode='inference',
            input_dur=rnn_config.training["inf_rep_binwidth"],
            fing_list=dataset.active_fingers_order[test_rep],
        )
        y_df = prepare_predictions_df(
            y_pred_test, y_true_test, test_rep, fold_i, mode='inference',
            input_dur=rnn_config.training["inf_rep_binwidth"],
            fing_list=dataset.active_fingers_order[test_rep],
        )

        metrics_df_cv = pd.concat([metrics_df_cv, metrics_df], axis=0)
        y_df_cv = pd.concat([y_df_cv, y_df], axis=0)

        trained_weights_df = rnn_prep.organize_trained_weights(
            fold_i, rnn_config.training["train_rep_binwidth"], rnn_reg
        )
        trained_weights_df_cv = pd.concat([trained_weights_df_cv, trained_weights_df], axis=0)

    # Aggregate profiling results across folds
    if enable_profiling and all_profiling_results:
        subject_id = fnc.remap_subject(data_config.subject, data_config.subj_map)
        aggregate_profiling_across_folds(
            all_profiling_results,
            log_dir=create_profiling_log_dir(
                base_dir="./logs/rnn_profiler",
                subject=subject_id,
                direction=data_config.direction,
                model_type=f"rnn_{rnn_config.decoder_type['rnn_type']}_summary",
            ),
            model_name="RNN",
        )

    if enable_profiling and all_neurobench_results:
        subject_id = fnc.remap_subject(data_config.subject, data_config.subj_map)
        aggregate_neurobench_across_folds(
            all_neurobench_results,
            log_dir=create_profiling_log_dir(
                base_dir="./logs/rnn_profiler",
                subject=subject_id,
                direction=data_config.direction,
                model_type=f"rnn_{rnn_config.decoder_type['rnn_type']}_neurobench_summary",
            ),
            model_name="RNN",
        )

    aggregate_snn_metrics_and_log(metrics_df_cv, flat_rnn_config, show_training=True)

    if rnn_config.logging["save_metrics"]:
        rnn_prep.write_rnn_results_to_files(
            data_config, metrics_df_cv, y_df_cv, trained_weights_df_cv, file_name
        )

    plot_predictions_sanity(y_df_cv)
    if not rnn_config.logging["wandb"]["use_wandb"]:
        plt.show()


if __name__ == '__main__':
    run_rnn_on_mu()
