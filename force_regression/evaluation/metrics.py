from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #root_mean_squared_error
import numpy as np
from scipy.stats import pearsonr
from tabulate import tabulate
import pandas as pd
import wandb
from configs.constants import *


def compute_metric(y_true, y_pred, metric_name, multi_output: str = 'raw_values'):

    """
    Computes the metric given the true and predicted values of a trained model.
    """
    if metric_name == MAE:
        return mean_absolute_error(y_true, y_pred, multioutput=multi_output)
    
    elif metric_name == MSE:
        return mean_squared_error(y_true, y_pred, multioutput=multi_output)

    elif metric_name == RMSE:
        # return root_mean_squared_error(y_true, y_pred, multioutput=multi_output)
        return mean_squared_error(y_true, y_pred, multioutput=multi_output, squared=False)

    elif metric_name == R2:
        return r2_score(y_true, y_pred, multioutput=multi_output)

    elif metric_name == CORR:
        # to return a list same as for the other scores
        return [pearsonr(y_true, y_pred)[0]]


def aggregate_snn_metrics_and_log(metrics_df_cv, snn_config, show_training: bool = False):
    """
    Aggregate metrics across folds and fingers and log to wandb
    """
    # organize metrics table
    aux_cols = [MODE_COL, INPUT_DUR_COL, FOLD_COL, TEST_ON_REP, FING_NAME_COL]
    m_cols = metrics_df_cv.columns[~metrics_df_cv.columns.isin(aux_cols)]
    # using as index=False because wandb doesn't log correctly multi-level df
    mean_df_across = metrics_df_cv.groupby([MODE_COL,INPUT_DUR_COL], as_index=False)[m_cols].mean()
    std_df_across = metrics_df_cv.groupby([MODE_COL,INPUT_DUR_COL], as_index=False)[m_cols].std()
    print(f"Mean metrics across folds and fingers\n{tabulate(mean_df_across,headers='keys', tablefmt='psql')}")

    # compute mean rmse on the test rep across folds and fingers
    mean_rmse_ts_across = mean_df_across[mean_df_across[MODE_COL]
                                         == 'inference'][RMSE_TEST].iloc[0]
    std_rmse_ts_across = std_df_across[std_df_across[MODE_COL]
                                       == 'inference'][RMSE_TEST].iloc[0]
    if show_training:
        mean_rmse_tr_across = mean_df_across[mean_df_across[MODE_COL]
                                         == 'inference'][RMSE_TRAIN].iloc[0]
        std_rmse_tr_across = std_df_across[std_df_across[MODE_COL]
                                       == 'inference'][RMSE_TRAIN].iloc[0]
        print(f"\nMean Train RMSE across folds and fingers:{np.round(mean_rmse_tr_across,3)}   +/-  {np.round(std_rmse_tr_across,3)}  % MVC")
    print(f"\nMean RMSE across folds and fingers:{np.round(mean_rmse_ts_across,3)}   +/-  {np.round(std_rmse_ts_across,3)}  % MVC")
    use_wandb = snn_config.use_wandb if hasattr(snn_config, 'use_wandb') else snn_config.logging["wandb"]["use_wandb"]
    if use_wandb:
        wtab = wandb.Table(data=metrics_df_cv,
                           columns=metrics_df_cv.columns.tolist())
        wandb.log({'metrics table': wtab})
        wandb.log({'mean metrics per finger': metrics_df_cv.groupby(
            [MODE_COL, INPUT_DUR_COL, FING_ID_COL])[m_cols].mean()})
        wandb.log({'mean metrics across finger': mean_df_across})
        wandb.log({'mean_RMSE_ts': mean_rmse_ts_across,
                  'std_RMSE_ts': std_rmse_ts_across})
        if show_training:
            wandb.log({'mean_RMSE_tr': mean_rmse_tr_across,
                      'std_RMSE_tr': std_rmse_tr_across})


def print_test_metric(post_process:bool, metrics_df:pd.DataFrame, metric=str):
    """
    Prints the mean metric on the test data.
    """
    if post_process:
        metric_column  = [f'{metric}_test_post']
    else:
        metric_column = [f'{metric}_test']
    
    fold_mean_test_post = metrics_df.groupby([FOLD_COL, TEST_ON_REP],
                                        as_index=False)[metric_column].mean()
    fold_std_test_post = metrics_df.groupby([FOLD_COL, TEST_ON_REP],
                                        as_index=False)[metric_column].std()
    overall_mean_test_post = fold_mean_test_post[metric_column].mean()
    overall_std_test_post = fold_std_test_post[metric_column].mean()
    print(f"Mean {metric} on test data\n"
            f"{tabulate(fold_mean_test_post,headers='keys', tablefmt='psql')}")
    print(f"Overall mean {metric} on test data: {overall_mean_test_post.values[0]:.2f}"\
            f" +/- {overall_std_test_post.values[0]:.2f} % MVC")


def prepare_snn_metrics_df(y_pred_tr: np.ndarray, y_true_tr: np.ndarray, y_pred_ts: np.ndarray,
                       y_true_ts: np.ndarray, test_rep: int, fold_i:int, mode:str, input_dur:float,
                       fing_list:list) -> pd.DataFrame:
    """
    Prepares a dataframe with the metrics for the predictions.
    """
    metrics_list = [MAE,MSE, RMSE, R2]
    # Organize metrics_df
    metrics_col = [f'{metric}_test' for metric in metrics_list]
    if y_true_tr is not None: # add the training metrics as well
        metrics_col += [f'{metric}_train' for metric in metrics_list]

    metrics_df = pd.DataFrame(columns=metrics_col)

    for metric in metrics_list:
        metrics_df.loc[:, f'{metric}_test'] = compute_metric(y_true_ts, y_pred_ts, metric)
        if y_true_tr is not None:
            metrics_df.loc[:, f'{metric}_train'] = compute_metric(y_true_tr, y_pred_tr, metric)
    metrics_df[TEST_ON_REP] = test_rep
    metrics_df[FOLD_COL] = fold_i
    mean_metrics = metrics_df.mean()
    print(f"Mean metrics across fingers df\n{tabulate(pd.DataFrame(mean_metrics), headers='keys', tablefmt='psql')}")
    metrics_df[MODE_COL] = mode
    metrics_df[INPUT_DUR_COL] = input_dur
    metrics_df[FING_ID_COL] = fing_list   # tile with the finger name
    print(f"Metrics df\n{tabulate(metrics_df, headers='keys', tablefmt='psql')}")
    return metrics_df


def prepare_convreg_metrics_df(y_pred_train, target_train, y_pred_test, target_test,
                               one_indexed_test_rep, fold_index, input_dur,
                               fing_list, metrics_list,
                               y_pred_train_post=None,
                               y_pred_test_post=None)-> pd.DataFrame:
    """
    Prepares a dataframe with the metrics for the predictions of the conventional regression model.
    """
    # fold_metrics_df = pd.DataFrame(columns=[f'{m}_{suffix}' for m in metrics_list for suffix in ['unfiltered', 'filtered']] + [FOLD_COL, TEST_ON_REP])
    fold_metrics_df = pd.DataFrame()
    for metric in metrics_list:
        fold_metrics_df.loc[:, f'{metric}_test'] = compute_metric(target_test, y_pred_test, metric)
        fold_metrics_df.loc[:, f'{metric}_train'] = compute_metric(target_train, y_pred_train, metric)

        if y_pred_train_post is not None:
            fold_metrics_df.loc[:, f'{metric}_train_post'] = compute_metric(target_train, y_pred_train_post, metric)
        if y_pred_test_post is not None:
            fold_metrics_df.loc[:, f'{metric}_test_post'] = compute_metric(target_test, y_pred_test_post, metric)
    mean_metrics = fold_metrics_df.mean()
    print(f"Mean metrics across fingers df\n{tabulate(pd.DataFrame(mean_metrics), headers='keys', tablefmt='psql')}")

    fold_metrics_df[TEST_ON_REP] = one_indexed_test_rep
    fold_metrics_df[FOLD_COL] = fold_index
    fold_metrics_df[INPUT_DUR_COL] = input_dur
    fold_metrics_df[FING_ID_COL] = fing_list   # tile with the finger name

    return fold_metrics_df
