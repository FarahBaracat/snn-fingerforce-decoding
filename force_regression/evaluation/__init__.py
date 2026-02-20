"""
Evaluation module for force regression models.

This module contains:
- metrics: Metric computation functions (MAE, MSE, RMSE, R2, etc.)
- compile_results: Result loading, parsing, and statistical analysis
- profiling: TensorBoard logging utilities for profiling results
"""

from force_regression.evaluation.metrics import (
    compute_metric,
    aggregate_snn_metrics_and_log,
    print_test_metric,
    prepare_snn_metrics_df,
    prepare_convreg_metrics_df,
)

from force_regression.evaluation.compile_results import (
    parse_params_from_filename,
    load_and_parse_df_data,
    load_and_parse_snn_data_to_df,
    load_and_parse_rnn_data_to_df,
    compute_mean_sd_metric,
    compute_mean_sd_snn_metric,
    compute_mean_metrics_across_mvcs,
    conduct_paired_tests,
    compute_pairwise_effect_sizes,
)

from force_regression.evaluation.profiling import (
    log_profiling_to_tensorboard,
    log_profiling_comparison_to_tensorboard,
    create_profiling_log_dir,
    aggregate_profiling_across_folds,
    log_neurobench_to_tensorboard,
    aggregate_neurobench_across_folds,
    run_snn_fold_profiling,
    run_rnn_neurobench,
    run_rnn_fold_profiling,
)
