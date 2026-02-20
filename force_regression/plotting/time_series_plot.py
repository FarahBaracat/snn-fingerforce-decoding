import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import gridspec

import seaborn as sns
from scipy.ndimage import gaussian_filter

from force_regression.data.preprocessing.force import downsample_signal


def plot_force(force_dictionary, config):
    plt.figure(figsize=(5, 1))

    # Initialize a variable to keep track of the time offset for each repetition
    time_offset = 0

    for rep_key in force_dictionary:
        if force_dictionary[rep_key] is not None:
            signal, trapezoid_points = force_dictionary[rep_key]['target'], force_dictionary[rep_key]['points']

            # Convert trapezoid points from samples to seconds and adjust for the time offset
            trapezoid_points_sec = [(i / config.f_samp) + time_offset for i in trapezoid_points]

            # Create a time axis for the signal with the time offset
            time_ax = np.arange(time_offset, (len(signal) / config.f_samp) + time_offset, 1 / config.f_samp)
            plt.plot(time_ax, signal, label=f'Force' if rep_key == 'rep1' else '', color=config.color_dict['green_sea'])

            if config.task_type == 'Trap':
                trapezoid_values = [signal[int((i - time_offset) * config.f_samp)] for i in trapezoid_points_sec]
                plt.scatter(trapezoid_points_sec, trapezoid_values, color=config.color_dict['pumpkin'],
                            label=f'Trapezoid Points' if rep_key == 'rep1' else '', s=40, marker='x')

            time_offset += len(signal) / config.f_samp

    plt.xlabel('Time (s)')
    plt.ylabel('% MVC')
    plt.legend()
    plt.show()


def plot_force_before_after_cut(config):
    """
    Plots the force profiles before and after cutting the region of interest.
    The region of interest was defined earlier in load_force, some time_to_cut prior of the 
    start of ramp and after end of ramp is considered (see load_force for more details)
    """
           
    _, axs = plt.subplots(1, 2, figsize=(10, 2)) 

    force = config.both_forces_uncut
    target = config.target_force_uncut
    trapezoid_points = config.points_uncut 

    ax1 = axs[0]
    ax1_force = ax1.twinx()
    time_ax = np.arange(0, force.shape[0] / config.f_samp, 1 / config.f_samp)
    ax1_force.plot(time_ax, force, color=config.color_dict['blueberry'], linewidth=1, alpha=0.4)
    ax1_force.plot(time_ax, target, 'k', alpha=1, linewidth=1)

    trapezoid_points_sec = [(i / config.f_samp) for i in trapezoid_points]
    trapezoid_values = [target[int(i*config.f_samp)] for i in trapezoid_points_sec]
    ax1_force.scatter(trapezoid_points_sec, trapezoid_values, color='r', s=30, marker='x')
    ax1.set_xticks(np.arange(0, force.shape[0] / config.f_samp, 2))
    ax1.set_xticklabels(np.arange(0, force.shape[0] / config.f_samp, 2).astype(int))
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Force (% MVC)')
    ax1.set_title('Force Before Cutting')

    force = config.both_forces_cut
    target = config.target_force_cut
    trapezoid_points = config.points_cut
    ax2 = axs[1]
    ax2_force = ax2.twinx()
    time_ax = np.arange(0, force.shape[0] / config.f_samp, 1 / config.f_samp)
    ax2_force.plot(time_ax, force, color=config.color_dict['blueberry'], linewidth=1, alpha=0.4)
    ax2_force.plot(time_ax, target, 'k', alpha=1, linewidth=1)

    trapezoid_points_sec = [(i / config.f_samp) for i in trapezoid_points]
    trapezoid_values = [target[int(i*config.f_samp)] for i in trapezoid_points_sec]
    ax2_force.scatter(trapezoid_points_sec, trapezoid_values, color='r', s=30, marker='x')
    ax2.set_xticks(np.arange(0, force.shape[0] / config.f_samp, 2))
    ax2.set_xticklabels(np.arange(0, force.shape[0] / config.f_samp, 2).astype(int))
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Force (% MVC)')
    ax2.set_title('Force After Cutting')
    plt.tight_layout()
    # plt.show()

def plot_forces(reg_df, dt, decay_tau, rep_id, config):
    fig = plt.figure(figsize=(8, 2))
    ax = fig.add_subplot(111)
    ax.plot(reg_df['time'], reg_df[config.force_cols_list])
    ax.legend(config.force_cols_list, ncol=1, bbox_to_anchor=(1.2, 1), loc='upper right')
    plt.title('Performed Path')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (% MVC)')
    plt.box(False)
    plt.xticks(np.arange(0, reg_df['time'].iloc[-1], 10))
    # plt.show()
    fig.savefig(os.path.join(config.figs_lr_path,
                             f"concatenate_target_force_repid_{rep_id}_{config.task_name()}_kernel_dt_{dt}_decay_{decay_tau}.png"),
                dpi=300, bbox_inches='tight')


def plot_emg_ch(signal: np.ndarray, title: str, start_time: float,
                end_time: float, f_samp: int, n: int = 2, ax=None):
    """
    Args:
        :param ax:
        :param n:
        :param signal: The emg to display
        :param title: A string containing the title of the plot
        :param f_samp:
        :param end_time:
        :param start_time:
    """

    if signal.shape[0] < signal.shape[1]:
        signal = np.transpose(signal)

    if end_time == -1:
        end_time = signal.shape[0] / f_samp
    emg_to_plot = np.array(signal[int(start_time * f_samp): int(end_time * f_samp), :])
    y_offset = np.amax(abs(emg_to_plot))

    time_ax = np.arange(start_time, end_time, 1 / f_samp)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3))

    y_ticks = []
    for i in range(signal.shape[1]):
        ax.plot(time_ax, emg_to_plot[:, i] + y_offset * i, linewidth=0.4)
        if (i + 1) % n == 0:
            y_ticks.append(y_offset * i)

    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_title(title)
    ax.set_xticks(np.arange(start_time, end_time + 1, 5))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'Ch {i + 1}' for i in range(signal.shape[1]) if (i + 1) % n == 0])


def plot_force_data(ax, df, force_columns, duration, labels, y_limit=(0, 17), xticks_step=30, direction: str = 'flex'):
    ax.plot(df['time'], df[force_columns], label=labels)
    ax.set_ylim(y_limit)
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Measured Force [% MVC]')
    ax.set_title(f'Force Data Visualization in {direction}')
    ax.set_xticks(np.arange(0, duration, xticks_step))



def plot_heatmaps_three_electrodes(input_data, segments, use_same_lims, ws_plot, sigma, overall_min, overall_max,
                                   config, average: bool = False):

    input_data = abs(input_data)
    _, axs = plt.subplots(1, 3, figsize=(10, 2)) 
    for i, (start, end) in enumerate(segments):

        if average:
            data = input_data[start:end, :]
        else:
            data, x_axis_downsampled = downsample_signal(input_data[start:end, :], ws_plot)

        smoothed_data = gaussian_filter(abs(data), sigma=sigma)

        if average:
            smoothed_data = np.mean(smoothed_data, axis=1)
            smoothed_data = smoothed_data.reshape(-1, 1)

        if use_same_lims:
            sns.heatmap(np.flipud(smoothed_data), cmap='jet', xticklabels=10, yticklabels=1, 
                        robust=True, ax=axs[i], vmin=overall_min, vmax=overall_max)
        else:
            sns.heatmap(np.flipud(smoothed_data), cmap='jet', xticklabels=10, yticklabels=1, 
                        robust=True, ax=axs[i])
        
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Channels')
        channels = smoothed_data.shape[0]
        channel_step = 5
        y_labels = [f'Ch {channels - i}' for i in range(0, channels, channel_step)]
        y_ticks = np.linspace(0, channels - 1, len(y_labels))
        axs[i].set_yticks(y_ticks)
        axs[i].set_yticklabels(y_labels)
        axs[i].set_title(f'{config.electrodes[i]}')

        if not average:
            x_tick_step = 20 
            x_ticks = np.arange(0, len(x_axis_downsampled), x_tick_step)
            x_labels = [f'{x_axis_downsampled[j]/config.f_samp:.2f}' for j in x_ticks]
            axs[i].set_xticks(x_ticks)
            axs[i].set_xticklabels(x_labels, rotation=45)
    plt.tight_layout()
    plt.show()


def plot_training_loss(tr_loss_per_epoch, ts_loss_every_nepochs, num_iter, log_ts_freq,
                       use_wandb=False, wandb_run=None):
    """
    Plot training and test loss curves vs epochs.

    Args:
        tr_loss_per_epoch: List of per-epoch training losses.
        ts_loss_every_nepochs: List of test losses sampled every log_ts_freq epochs.
        num_iter: Total number of training epochs.
        log_ts_freq: Epoch interval at which test loss was recorded.
        use_wandb: If True, log the figure to W&B.
        wandb_run: Active wandb run (only used when use_wandb=True).
    """
    fig = plt.figure(figsize=(5, 3))
    plt.plot(np.arange(num_iter), tr_loss_per_epoch)
    plt.plot(np.arange(0, len(ts_loss_every_nepochs) * log_ts_freq, log_ts_freq),
             ts_loss_every_nepochs, 'o-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss")
    plt.legend(["Train Loss", "Test Loss"])
    fig.tight_layout()
    if use_wandb and wandb_run is not None:
        wandb_run.log({"Train and Test Loss": wandb_run.Image(fig)})


def plot_predictions_sanity(y_df_cv):
    """
    Quick diagnostic plot of true vs predicted force for each fold.

    Args:
        y_df_cv: DataFrame with columns 'fold', 'y_true_test', 'y_pred_test'.
    """
    fig = plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(ncols=2, nrows=1)
    for fold_i in range(y_df_cv['fold'].nunique()):
        ax = fig.add_subplot(gs[fold_i])
        ax.plot(y_df_cv.iloc[fold_i]['y_true_test'])
        ax.plot(y_df_cv.iloc[fold_i]['y_pred_test'], alpha=0.4)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Force (%MVC)")
        ax.set_title(f"True vs Predicted Force on Test Set Fold {fold_i}")
    fig.tight_layout()