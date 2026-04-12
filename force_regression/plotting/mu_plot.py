import numpy as np
import os
import logging
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

from force_regression.data.loaders.force import get_repetitions_for_task
import force_regression.data.preprocessing.spikes as spk
import force_regression.utils.functions as fn
import force_regression.utils.io as io
from force_regression.config.dataconfig import DataConfig
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

from configs.constants import *


def raster(data, f_samp):
    plt.eventplot(spk.convert_to_s(data, f_samp), color='k', linewidths=0.5, linelengths=0.5, alpha=0.5)


def plot_raster_with_force(ax, ts, force, target, title, colors, f_samp, trapezoid_points_sec=None, trapezoid_values=None):
    time_ax = np.arange(0, force.shape[0] / f_samp, 1 / f_samp)

    ax.set_title(title)
    raster(ts, f_samp)

    ax_force = ax.twinx()
    ax_force.plot(time_ax, force, color=colors['blueberry'], linewidth=1, alpha=0.4)
    ax_force.plot(time_ax, target, color='k', alpha=1, linewidth=1)
    ax_force.set_ylabel('Force', color=colors['blueberry'])
    ax_force.tick_params(axis='y', labelcolor=colors['blueberry'])

    ax.set_xticks(np.arange(0, force.shape[0] / f_samp, 2))
    ax.set_xticklabels(np.arange(0, force.shape[0] / f_samp, 2).astype(int))

    if trapezoid_points_sec is not None and trapezoid_values is not None:
        ax_force.scatter(trapezoid_points_sec, trapezoid_values, color='r', s=30, marker='x')

def raster_before_after(ts_before, ts_after, config_raster, cut: bool = False):
    """
    Plots the spike times before and after cleaning +/- cutting
    """
    colors, f_samp = config_raster.color_dict, config_raster.f_samp

    if cut:
        force = config_raster.both_forces_cut
        target = config_raster.target_force_cut
        points = config_raster.points_cut
        cut_title = "Cut"
    else:
        force = config_raster.both_forces_uncut
        target = config_raster.target_force_uncut
        points = config_raster.points_uncut
        cut_title = "Uncut"

    trapezoid_points_sec = [(i / f_samp) for i in points]
    trapezoid_values = [target[int(i * f_samp)] for i in trapezoid_points_sec]

    plt.figure(figsize=(10, 2))
    ax1 = plt.subplot(1, 2, 1)
    plot_raster_with_force(ax1, spk.cuda_to_numpy(ts_before), force, target, "Before", colors, f_samp, trapezoid_points_sec, trapezoid_values)

    ax2 = plt.subplot(1, 2, 2)
    plot_raster_with_force(ax2, spk.cuda_to_numpy(ts_after), force, target, "After", colors, f_samp, trapezoid_points_sec, trapezoid_values)
    plt.suptitle(f"Subject {fn.remap_subject(config_raster.subject, config_raster.subj_map)}: {config_raster.task_name()}. electrode: {config_raster.electrode_name}. "+cut_title)
    plt.tight_layout()
    plt.show()



def grid_raster_plot_bothreps(mu_df_sorted, config, force_per_finger,
                              plot_direction=['flex', 'ext']):
    # plotting the cut units
    alphas = [0.8, 0.5]
    colors = [config.color_dict['belize'], config.color_dict['pumpkin']]

    rep_ids = get_repetitions_for_task(config.task_type)

    for config.direction in plot_direction:
        fig, axs = plt.subplots(len(config.finger_label_map.keys()), 3, figsize=(8, 5))
        for i, rep_id in enumerate(range(1, len(rep_ids) + 1)):
            for row, fing_name in enumerate(list(config.finger_label_map.keys())):
                for col, e in enumerate(config.electrodes):
                    ax = axs[row, col]

                    mask = (mu_df_sorted[FING_NAME_COL] == fing_name) & (
                            mu_df_sorted[ELEC_NAME] == e) & (mu_df_sorted[REP_ID] == rep_id - 1) & \
                           (mu_df_sorted[FING_DIR] == config.direction)

                    # retrieve sp times and number of MUs from mu_df and mu_summary_df
                    sp_times = mu_df_sorted.loc[mask, SP_TIME].values
                    n_mu = mu_df_sorted[mask].shape[0]

                    for mu in range(n_mu):
                        mu_sp_time = sp_times[mu] / config.f_samp 
                        if rep_id == 2:
                            mu_sp_time = mu_sp_time - mu_df_sorted[mask][START_TIME].values[0] 
                        ax.eventplot(mu_sp_time,
                                     lineoffsets=mu,
                                     color=colors[i] if config.task_type == 'Trap' else 'k',
                                     alpha=alphas[i] if config.task_type == 'Trap' else 0.4, linewidths=0.5)

                    if col == 0:
                        ax.set_ylabel(f'{fing_name}', fontsize=8)

                    if row == 0:
                        ax.set_title(f"{e}")

                    if n_mu > 0:
                        # select the force for the given finger and direction
                        temp_df = force_per_finger[fing_name]
                        force = temp_df[temp_df[FING_DIR] == config.direction][rep_id - 1].values.reshape(-1, 1)
                        time_ax = np.arange(0, force.shape[0] / config.f_samp, 1 / config.f_samp)

                        # plot target force on a secondary axis
                        ax2 = ax.twinx()
                        ax2.plot(time_ax, force, label='Force',
                                 color=colors[i] if config.task_type == 'Trap' else config.color_dict['green_sea'],
                                 alpha=0.8)
                        if rep_id == 2:
                            ax2.set_ylabel('Force (%MVC)', fontsize=6, color='red')
                            ax2.tick_params(axis='y', labelcolor='red')
                        else:
                            ax2.set_yticks([])

        fig.suptitle(f"Subject {fn.remap_subject(config.subject, config.subj_map)}: Both reps. {config.task_name()}",
                     fontsize=8)

        for ax in axs[-1, :]:
            ax.set_xlabel('Time (s)', fontsize=6)

        fig.legend(handles=[plt.Line2D([0], [0], color=c, alpha=a, lw=2) for c, a in zip(colors, alphas)],
                   labels=['Rep 1', 'Rep 2'],
                   loc='upper right')

        filename = os.path.join(config.figs_decomp_path, f"raster_plot_electrodes_reps_"
                                                         f"{fn.remap_subject(config.subject, config.subj_map)}_{config.task_name()}.png")
        logging.info("Saving figure to %s", filename)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        plt.show()


def raster_plot_sep_rep_inset(mu_df_sorted, config, force_df,
                               plot_direction=['flex', 'ext'],
                               plot_common_only=False, save_fig=True,
                               rep_list=[1,2],
                               max_ylim=80,
                               plot_all_forces:bool=False,
                               same_mus_color_per_task:bool=False,
                               output_figures_dir:str=None,
                               source_data_path:str=None,
                               sheet_name:str=f'sample_decomposition_data'):
    """
    Plot each repetiton raster plot in a separate subplot with inset on hold
    either append MU ids to have blocks or remove the id from the axis
    """
    alphas = [0.1, 0.1]
    for config.direction in plot_direction:
        fig = plt.figure(figsize=(10, 3))
        gs = gridspec.GridSpec(nrows=len(rep_list), ncols=len(config.finger_label_map.keys()))
        mu_df_sorted_dir = prepare_mu_df_sorted_dir(mu_df_sorted, config.direction, plot_common_only)
        for col, fing_name in enumerate(config.finger_label_map.keys()):
            if same_mus_color_per_task:
                color_rep_list = [COLORS['midnight_blue'], COLORS['midnight_blue']]
            else:
                color_rep_list = [config.finger_color_list[col], config.finger_color_list[col]]
            start_hold_reps = mu_df_sorted_dir[mu_df_sorted_dir[FING_NAME_COL] == fing_name][START_HOLD].unique()
            end_holds_reps =  mu_df_sorted_dir[mu_df_sorted_dir[FING_NAME_COL] == fing_name][END_HOLD].unique()
            
            # check if start_hold_reps is empty. This can happen if no MUs for the finger are found. There would be no entry for this finger
            if start_hold_reps.shape[0]== 0:
                start_hold_reps = mu_df_sorted_dir[START_HOLD].unique()
                end_holds_reps = mu_df_sorted_dir[END_HOLD].unique()

            for row, rep_id in enumerate(rep_list):
                if rep_id == 2:
                    start_hold_reps = start_hold_reps - mu_df_sorted_dir[mu_df_sorted_dir[FING_NAME_COL] == fing_name][START_TIME].unique()
                    end_holds_reps= end_holds_reps - mu_df_sorted_dir[mu_df_sorted_dir[FING_NAME_COL] == fing_name][START_TIME].unique()

                ax = fig.add_subplot(gs[row, col])
                plot_finger_data(ax, mu_df_sorted_dir, fing_name, config,
                                 alphas, color_rep_list, rep_id_one_indexed=rep_id, max_ylim=max_ylim)
                if col ==0:
                    ax.set_ylabel(MU_LABEL, fontsize=LABEL_FONTSIZE, labelpad=YLAB_PAD)
                else:
                    ax.set_yticks([])
                ax.set_xlabel(TIME_LABEL,fontsize=LABEL_FONTSIZE, labelpad=5)
                plot_force_data_for_rep(ax, force_df, fing_name,  config.direction, config,
                                        rep_id, plot_color=config.color_dict['midnight_blue'], fig_col=col,
                                        zoom_start=start_hold_reps[row], zoom_end = end_holds_reps[row],
                                        plot_all_forces=plot_all_forces)
                ax.set_title(f"{fing_name}", fontsize=TITLE_FONTSIZE)

        # add figure title
        fig.tight_layout()
        if save_fig:
            filename = get_filename(config, plot_common_only, filename_prefix=f'mu_force_plot_rep_{rep_id}_allforces_{plot_all_forces}')
            fig.savefig(os.path.join(output_figures_dir, filename) , dpi=300, bbox_inches = 'tight', format='pdf')
            print(f"Saving figure to {filename}")

            if source_data_path is not None:
                force_cols = [c for c in force_df.columns if c.startswith('force_')]
                spike_blocks, force_blocks = [], []
                for fing_name in config.finger_label_map.keys():
                    for rep_id_s in rep_list:
                        # Spike data for this finger/rep
                        mu_cols = [FING_NAME_COL, ELEC_NAME, CONS_MU_ID, SP_TIME]
                        smask = (mu_df_sorted_dir[FING_NAME_COL] == fing_name) & (mu_df_sorted_dir[REP_ID] == rep_id_s - 1)
                        spikes = mu_df_sorted_dir[smask][mu_cols].copy().reset_index(drop=True)
                        spikes = spikes.explode(SP_TIME)
                        spikes[SP_TIME] = spikes[SP_TIME].astype(float) / config.f_samp
                        if rep_id_s == 2:
                            start_time = mu_df_sorted_dir[smask][START_TIME].values[0]  # already in seconds
                            spikes[SP_TIME] = spikes[SP_TIME] - start_time
                        spikes.rename(columns={SP_TIME: 'spike_time_s'}, inplace=True)

                        # Compute relative_plot_mu_id: replicate plot sorting (sort MUs by first spike time)
                        first_spike = spikes.groupby(CONS_MU_ID)['spike_time_s'].min().sort_values()
                        plot_id_map = {mu_id: i for i, mu_id in enumerate(first_spike.index)}
                        spikes['relative_plot_mu_id'] = spikes[CONS_MU_ID].map(plot_id_map)
                        spike_blocks.append(spikes)

                io.save_to_source_data(source_data_path, pd.concat(spike_blocks, ignore_index=True),
                                       sheet_name=sheet_name)

                # Force: one wide table per rep (time_s | force_finger1 | force_finger2 | ...)
                for rep_id_s in rep_list:
                    rep_force = None
                    for fing_name in config.finger_label_map.keys():
                        fmask = ((force_df[FING_NAME_COL] == fing_name) &
                                 (force_df[FING_DIR] == config.direction) &
                                 (force_df[REP_ID] == rep_id_s - 1))
                        fdf = force_df[fmask][[f'force_{fing_name}']].reset_index(drop=True)
                        rep_force = fdf if rep_force is None else pd.concat([rep_force, fdf], axis=1)
                    rep_force.insert(0, 'time_s', np.arange(len(rep_force)) / config.f_samp)
                    io.save_to_source_data(source_data_path, rep_force,
                                           sheet_name=f'{sheet_name}_force_rep{rep_id_s}')




def prepare_mu_df_sorted_dir(mu_df_sorted, direction, plot_common_only):
    mu_df_sorted_dir = mu_df_sorted[mu_df_sorted[FING_DIR] == direction].copy()
    mu_df_sorted_dir[CONS_MU_ID] = mu_df_sorted_dir[MU_ID].map(
        {k: v for v, k in enumerate(mu_df_sorted_dir[MU_ID].unique())})
    if plot_common_only:
        counter_cons_id = mu_df_sorted_dir.groupby([CONS_MU_ID])[FIRST_SP_TIME].count()
        # extract the mu_ids that are common to both reps
        common_mu_ids = counter_cons_id[counter_cons_id == 2].index.values
        mu_df_sorted_dir = mu_df_sorted_dir[mu_df_sorted_dir[CONS_MU_ID].isin(common_mu_ids)]

    return mu_df_sorted_dir


def remap_muid_to_consecutive_ids_for_plotting(original_ids:list[int]):
    """
    Remaps the MU IDs to consecutive integers starting from 0.
    """
    remap_dict = {original_id: new_id for new_id, original_id in enumerate(original_ids)}
    remapped_ids = [remap_dict[id] for id in original_ids]
    return remapped_ids


def plot_finger_data(ax, mu_df_sorted_dir, fing_name, config, alphas, color_rep_list,
                     rep_id_one_indexed, max_ylim=80):
    combined_timestamps, combined_ids, labels, alphas_list, color_list = [], [], [], [], []
    mu_df_sorted_dir = mu_df_sorted_dir[mu_df_sorted_dir[REP_ID]==rep_id_one_indexed - 1]
    for e in config.electrodes:
        mask = (mu_df_sorted_dir[FING_NAME_COL] == fing_name) & (mu_df_sorted_dir[ELEC_NAME] == e)
        sp_times, sp_ids = get_sp_times_ids(mu_df_sorted_dir, mask, config, rep_id_one_indexed)
        if len(sp_times) > 0:
            combined_timestamps.extend(sp_times)
            combined_ids.extend(sp_ids)
            labels.extend([f'{e}'] * len(sp_times))
            alpha_index = (rep_id_one_indexed - 1) % len(alphas)
            color_index = (rep_id_one_indexed - 1) % len(color_rep_list)
            alphas_list.extend([alphas[alpha_index]] * len(sp_times))
            color_list.extend([color_rep_list[color_index] for _ in range(len(sp_times))])
    logging.info(f"Finger {fing_name} rep {rep_id_one_indexed}: plotting {len(np.unique(combined_ids))} MUs: sptimes:{len(combined_timestamps[0])} first {combined_timestamps[0][:10]}")
    plot_mu_raster_sorted_for_rep(ax, combined_timestamps, combined_ids, labels, alphas_list, color_list, max_ylim)


def get_sp_times_ids(mu_df_sorted_dir, mask, config, rep_id):
    if mu_df_sorted_dir.loc[mask, SP_TIME].values.size > 0:
        sp_times = mu_df_sorted_dir.loc[mask, SP_TIME].values / config.f_samp
        if rep_id == 2:
            sp_times = sp_times - mu_df_sorted_dir[mask][START_TIME].values[0]
    else:
        sp_times = []
    sp_ids = mu_df_sorted_dir.loc[mask, CONS_MU_ID].values
    return sp_times, sp_ids


def plot_mu_raster_sorted_for_rep(ax, combined_timestamps, combined_ids, labels, alphas_list, color_list,
                                  max_ylim=80):
    """
    Plot the raster plot for the given MUs.
    """
    sorted_stamps = sorted([(stamp, id, label, alpha, color) for stamp, id, label, alpha, color in
                            zip(combined_timestamps, combined_ids, labels, alphas_list, color_list) if
                            len(stamp) > 1], key=lambda x: x[0][0])
    # replace the ids with consecutive ids in the tuple
    sorted_ids = [sp_id for _, sp_id, _, _, _ in sorted_stamps]
    consecutive_sorted_ids = remap_muid_to_consecutive_ids_for_plotting(sorted_ids)

    # sorted_time_stamps = [(stamp, id, label, alpha, color) for stamp, id, label, alpha, color in           
    for sample_index, (stamp, _, _, alpha, color) in enumerate(sorted_stamps):
        ax.plot(stamp, np.repeat(consecutive_sorted_ids[sample_index], len(stamp)), "|", ms=2, color=color, alpha=alpha)
    # ax.set_ylim([-1, max(consecutive_sorted_ids) + 30] )
    ax.set_ylim([-1, max_ylim])

def plot_force_data(ax, temp_df, direction, config, color_reps):
    force_rep1 = temp_df[temp_df[FING_DIR] == direction][0].values.reshape(-1, 1)

    time_ax = np.arange(0, force_rep1.shape[0] / config.f_samp, 1 / config.f_samp)
    ax2 = ax.twinx()
    plot_color = color_reps[0]
    plot_alpha = 0.5
    if config.task_type == 'Pseudo':
        plot_color = config.color_dict['green_sea']
        plot_alpha = 0.8
    
    ax2.plot(time_ax, force_rep1, label='Force', color=plot_color, alpha=plot_alpha)
    if len(get_repetitions_for_task(config.task_type)) > 1:
        force_rep2 = temp_df[temp_df[FING_DIR] == direction][1].values.reshape(-1, 1)
        ax2.plot(time_ax, force_rep2, label='Force',color=color_reps[1], alpha=0.5)
    ax2.set_ylabel(FORCE_MVC_LABEL, fontsize=6, labelpad=YLAB_PAD)


def plot_force_data_for_rep(ax:plt.axis, force_df:pd.DataFrame, fing_name:str, direction:str,
                            config:DataConfig,
                            rep_id_one_indexed: int, plot_color:str,
                            is_inset:bool=False, zoom_start:float=None,
                            zoom_end:float=None, fig_col:int=0,
                            annotate_plateau:bool=False,
                            plot_all_forces:bool=False):
    """
    plot a single rep force data
    """
    active_finger_force_df = force_df[(force_df[FING_NAME_COL]==fing_name) &(force_df[FING_DIR] == direction) & (force_df[REP_ID]==rep_id_one_indexed-1)]
    force_rep = active_finger_force_df[f'force_{fing_name}'].values.reshape(-1, 1)
    if plot_all_forces:
        non_active_finger_cols = [col for col in force_df.columns if col.startswith('force_') and col != f'force_{fing_name}']
        other_finger_forces_rep = active_finger_force_df[non_active_finger_cols].values.reshape(-1, len(non_active_finger_cols))
    time_ax = np.arange(0, force_rep.shape[0] / config.f_samp, 1 / config.f_samp)
    ax.set_xticks(np.arange(0, force_rep.shape[0] / config.f_samp, 10))

    plot_alpha = 0.8
    if not is_inset:
        ax2 = ax.twinx()
        ax2.plot(time_ax ,force_rep,label='Force',color=plot_color, alpha=plot_alpha)
        if plot_all_forces:
            ax2.plot(time_ax, other_finger_forces_rep,color=COLORS['midnight_blue'], alpha=0.2,
                     linewidth=0.3)
        ax2.tick_params(axis='y')
        # set axis color
        if fig_col ==4:
            ax2.set_ylabel(FORCE_MVC_LABEL, fontsize=LABEL_FONTSIZE, color= plot_color)
            y_ticks_labels = np.arange(0, config.mvc+10, 5)
            ax2.set_yticks(y_ticks_labels, labels=np.round(y_ticks_labels,2))
        else:
            ax2.set_yticks([])

        # annotate force fluctuations on ax2
        if annotate_plateau:
            hold_shift = 1 if config.mvc==15 else 2 # shifting the start of the hold because of the delay in execution
            mask_hold = (time_ax>zoom_start + hold_shift) & (time_ax<=zoom_end -0.5)  # zoom_end - 0.5 added for case of 5%MVC
            fluct_max = force_rep[mask_hold].max()
            fluct_min = force_rep[mask_hold].min()
            
            prop_xshift = 2.5 if config.mvc==15 else 2
            text_xshift = 4
            arrow_style = mpatches.ArrowStyle.BarAB(widthA=0.26, widthB=0.26) #|-| style
            ax2.annotate('', xy=(zoom_start - prop_xshift , fluct_min),
                        xytext=(zoom_start - prop_xshift , fluct_max),
                        xycoords='data', textcoords='data',
                        arrowprops={'arrowstyle': arrow_style, 'lw': .7, 'color':plot_color,})
            ax2.annotate(rf'$\Delta$ force = {np.round(fluct_max - fluct_min,2)} %',
                        xy=(zoom_start - prop_xshift , fluct_min),
                        xytext=(zoom_start - text_xshift , fluct_max+0.5),
                        color=plot_color,
                        xycoords='data', textcoords='data',fontsize=8)
        ax2.set_ylim([0, force_rep.max()])
        # remove spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    else:
        inset_mask = (time_ax >= zoom_start) & (time_ax<=zoom_end)
        ax.plot(time_ax[inset_mask], force_rep[inset_mask], color=plot_color, alpha=0.5)
        ax.set_yticks(np.arange(force_rep[inset_mask].min(), force_rep[inset_mask].max(), 5),
                      labels = np.round(np.arange(force_rep[inset_mask].min(), force_rep[inset_mask].max(), 5),1))
        ax.set_xticks(np.arange(zoom_start, zoom_end, 1))
        ax.tick_params(axis='both', which='major', pad=0.5, labelsize=8)
        #remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        

def get_figure_title(config):
    return f"Subject {fn.remap_subject(config.subject, config.subj_map)}: Both reps. {config.task_type}_{config.wrist_pos}_{config.sign_mvc * config.mvc}_{config.rest_dur}_{config.ramp_dur}_{config.hold_dur}"


def get_filename(config, plot_common_only, filename_prefix="raster_plot_reps"):
    if plot_common_only:
        filename_suffix = "common"
    else:
        filename_suffix = "finger"
    return os.path.join(config.figs_decomp_path, f"{filename_prefix}_{filename_suffix}_"
                                                 f"{fn.remap_subject(config.subject, config.subj_map)}_{config.task_name()}.pdf")


def plot_heatmap(matrix_in, axis, lw=0, min_pulse_rate=None, annot=False, cbar=False, vmin=None, vmax=None,
                 cbar_label='', ylabel='', xlabel='',
                 cmap="rocket_r", title=''):
    heatmap = sns.heatmap(matrix_in, linewidths=lw, cmap=cmap, cbar=cbar, ax=axis,
                          annot=annot, cbar_kws={'label': f'{cbar_label}'}, vmin=vmin, vmax=vmax)

    if annot:
        for t in heatmap.texts:
            if float(t.get_text()) >= min_pulse_rate:
                t.set_text(t.get_text())  # if the value is greater than 0.4 then I set the text
            else:
                t.set_text("")  # if not it sets an empty text
    # heatmap.invert_yaxis()
    axis.set_xlabel(xlabel, fontsize=6)
    axis.set_ylabel(ylabel, fontsize=6)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    plt.title(title, fontsize=8)
    return heatmap
