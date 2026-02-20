from __future__ import annotations
import os
from typing import List
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib.patches import Patch
import wandb
import force_regression.utils.functions as fn
from force_regression.training.snn_pipeline import get_repetition_duration_from_segments, map_decay_to_tau
from force_regression.config.dataconfig import DataConfig
from force_regression.config.snnconfig import SNNConfig
from configs.constants import *

#MANPLOT: SNN training loss for each fold
def plot_epochs_loss(snn_config:SNNConfig, data_config:DataConfig,
                    tr_loss_per_epoch:List[float], ts_loss_per_epoch:List[float],
                    all_tr_loss_hist:List[float], fold_i:int):
    """
    Plots the training and test loss per epoch.
    """

    n_epochs = snn_config.num_iter if hasattr(snn_config, 'num_iter') else snn_config.training['num_iter']
    batch_size = snn_config.test_batch_size if hasattr(snn_config, 'test_batch_size') else snn_config.training['test_batch_size']
    use_wandb = snn_config.use_wandb if hasattr(snn_config, 'use_wandb') else snn_config.logging['wandb']['use_wandb']
    bin_width = snn_config.train_rep_binwidth if hasattr(snn_config, 'train_rep_binwidth') else snn_config.training['train_rep_binwidth']
    log_ts_freq = snn_config.log_ts_freq if hasattr(snn_config, 'log_ts_freq') else snn_config.logging['log_ts_freq']

    exp_dir = snn_config.exp_dir if hasattr(snn_config, 'exp_dir') else snn_config.task['exp_dir']
    first_tau_filt = snn_config.first_filter_tau if hasattr(snn_config, 'first_filter_tau') else snn_config.decoder_type['first_filter_tau']
    second_tau_filt = snn_config.second_filter_tau if hasattr(snn_config, 'second_filter_tau') else snn_config.decoder_type['second_filter_tau']
    tau_syn = snn_config.tau_syn if hasattr(snn_config, 'tau_syn') else snn_config.decoder_type['tau_syn']
    learn_tau_mem = snn_config.learn_tau_mem if hasattr(snn_config, 'learn_tau_mem') else snn_config.decoder_type['learn_tau_mem']
    save_fig = snn_config.save_fig if hasattr(snn_config, 'save_fig') else snn_config.logging['save_fig']
    # figure styling
    tr_alpha = 0.6
    ts_alpha = 0.9
    tr_samples_alpha =0.15
    palette = sns.color_palette(palette='Accent')
    snn_color = palette[0]


    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    epochs_range = np.arange(0, len(all_tr_loss_hist), len(all_tr_loss_hist)/n_epochs)

    ax.plot(epochs_range, np.sqrt(tr_loss_per_epoch), 'o-', label='Train loss', alpha=tr_alpha,
        color= data_config.color_dict['midnight_blue'], markersize=4)
 
    ax.plot(epochs_range[::log_ts_freq], np.sqrt(ts_loss_per_epoch), 'o-',
            markersize=4, color= snn_color, label='Test Loss', alpha=ts_alpha)

    ax.plot(np.arange(len(all_tr_loss_hist)), np.sqrt(all_tr_loss_hist), alpha=tr_samples_alpha,
            color=data_config.color_dict['midnight_blue'], label='Train Batch Loss')

    # Plotting styling
    tot_nticks = 15
    x_ticks = np.arange(0, len(all_tr_loss_hist), len(all_tr_loss_hist)/n_epochs)
    step_size = int(n_epochs/tot_nticks)  if n_epochs > tot_nticks else 1 # show 15 ticks total

    ax.set_xticks(x_ticks[::step_size], np.arange(n_epochs)[::step_size])
    ax.set_xlabel('Iteration', labelpad=XLAB_PAD)
    ax.set_ylabel(RMSE_LABEL, labelpad=YLAB_PAD)
    sns.despine(ax=ax, offset=0, trim=False)

    fig.legend(loc='upper center', bbox_to_anchor=(0.52, 1.1), ncol=3, frameon=False)
    fig.tight_layout()

    if SHOW_TITLE:
        plt.title(f'Loss history - Batch size = {batch_size}')
    if use_wandb:
        wandb.log({f"Tr Loss": wandb.Image(fig)})

    if save_fig:
        filename = f"{data_config.mvc}_{exp_dir}_ep_{n_epochs}_fold_{fold_i}_trloss_binwidth_{bin_width}_batchsize_{batch_size}_taufilt1_{first_tau_filt}_taufilt2_{second_tau_filt}_tau_syn_{tau_syn}_learn_beta_{learn_tau_mem}"
        file_path = os.path.join(data_config.figs_snn_path, f"{filename}.png")
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saving figure to {file_path}")


#MANPLOT: SNN learnt weight distribution for each fold
def plot_learnt_wdist(snn_config:SNNConfig, data_config:DataConfig, w_learnt:np.ndarray, w_init:np.ndarray):
    """
    Plot heatmap of the learnt weights after training.
    """
    n_epochs = snn_config.num_iter if hasattr(snn_config, 'num_iter') else snn_config.training['num_iter']
    use_wandb = snn_config.use_wandb if hasattr(snn_config, 'use_wandb') else snn_config.logging['wandb']['use_wandb']
    exp_dir = snn_config.exp_dir if hasattr(snn_config, 'exp_dir') else snn_config.task['exp_dir']
    first_tau_filt = snn_config.first_filter_tau if hasattr(snn_config, 'first_filter_tau') else snn_config.decoder_type['first_filter_tau']
    second_tau_filt = snn_config.second_filter_tau if hasattr(snn_config, 'second_filter_tau') else snn_config.decoder_type['second_filter_tau']
    tau_syn = snn_config.tau_syn if hasattr(snn_config, 'tau_syn') else snn_config.decoder_type['tau_syn']
    learn_tau_mem = snn_config.learn_tau_mem if hasattr(snn_config, 'learn_tau_mem') else snn_config.decoder_type['learn_tau_mem']
    save_fig = snn_config.save_fig if hasattr(snn_config, 'save_fig') else snn_config.logging['save_fig']
    # Figure styling
    trained_alpha = 0.6
    init_alpha = 0.4
    snn_color = sns.color_palette(palette='Accent')[0]

    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.hist(w_learnt.flatten(), bins=100, alpha=trained_alpha, label='Trained',
            color=snn_color)
    if w_init is not None:
        ax.hist(w_init.flatten(), bins=100, alpha=init_alpha, label='Initial',
                color=data_config.color_dict['midnight_blue'])

    ax.set_xlabel('Weight value (a.u.)', labelpad=XLAB_PAD)
    ax.set_ylabel('Frequency', labelpad=YLAB_PAD)
    sns.despine(ax=ax, offset=0, trim=False)
    # fig.legend(loc='upper left', ncol=1, frameon=False)
    plt.legend(frameon=False)
    fig.tight_layout()

    if SHOW_TITLE:
        plt.title('weight distribution')

    if use_wandb:
        wandb.log({f"W init": wandb.Image(fig)})

    if save_fig:
        filename = f"{data_config.mvc}_{exp_dir}_ep_{n_epochs}_winitdist_taufilt1_{first_tau_filt}_taufilt2_{second_tau_filt}_tau_syn_{tau_syn}_learn_beta_{learn_tau_mem}"
        file_path = os.path.join(data_config.figs_snn_path, f"{filename}.png") 
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saving figure to {file_path}")


        

def plot_with_overlap(var_numpy:np.ndarray, snnreg_model, snn_config:SNNConfig, dataset:EMGDataset, ax,
                      mvc_i:int,
                      fing_color_list:List[str], alpha:float=0.4,
                      linestyle=None, label:str=None, fig=None, gs=None, ylim=None, mode='training'):
    """
    If the figure is provided, then create subplots using the given gs; adding axes to the figure.
    Otherwise, plot the var_numpy in the provided axis.
    """
    if linestyle is None:
        linestyle = DEFAULT_LINESTYLE
    if mode == 'training':
        bin_width = snn_config.train_rep_binwidth if hasattr(snn_config, 'train_rep_binwidth') else snn_config.training['train_rep_binwidth']
    else:
        bin_width = snn_config.inf_rep_binwidth if hasattr(snn_config, 'inf_rep_binwidth') else snn_config.training['inf_rep_binwidth']
    overlap_perc = snn_config.overlap_perc if hasattr(snn_config, 'overlap_perc') else snn_config.task['overlap_perc']
    train_fing_id = snn_config.train_fing_id if hasattr(snn_config, 'train_fing_id') else snn_config.task['train_fing_id']
    tot_dur = get_repetition_duration_from_segments(dataset.num_segments, bin_width, overlap_perc)
    start_ar = dataset.segments_start[mvc_i]#fn.compute_windows_start_times(bin_width, snn_config.overlap_perc, dataset.rep_dur)
    seg_nsteps = int(bin_width/snnreg_model.dt)
    
    use_subplots = False
    n_mvcs = 1  #default to 1 MVC
    if fig:
        use_subplots = True
        if dataset.n_samples==20: # if the dataset has 20 samples, then we are plotting for both MVCs. #FIXME: later rely on the load_multi for instance to retrieve the exact number of samples
            n_mvcs = 2    #TODO: get the n_mvc from data_config
     
    else:
        ax.set_prop_cycle(color=fing_color_list)

    shift_start = 0 # shift the start of the plot in case of multiple fingers in the same plot
    buffer_time = 1 # in seconds, to avoid attaching the different finger plots together
    buffer_samples = int(buffer_time/snnreg_model.dt)
    stride_in_steps = int(bin_width* (1 - overlap_perc)/snnreg_model.dt)
    arr_pointer = 0
    for i, _ in enumerate(train_fing_id):
        for row in range(n_mvcs):
    
            if use_subplots:
                # if the figure is provided, each finger is a subplot in this figure
                # check if the figure has enough axes else create a new one
                ax = fig.add_subplot(gs[row, i]) if len(fig.axes) < snnreg_model.num_outputs *n_mvcs else fig.axes[n_mvcs*i + row]
                ax.set_prop_cycle(color=fing_color_list)
                ax.set_xlabel(TIME_LABEL, labelpad=XLAB_PAD)


            else: # this shift is needed in case of plotting multiple fingers in the same plot (e.g. debugging plots)
                shift_start = i * (tot_dur/snnreg_model.dt+ buffer_samples * (len(train_fing_id)>1 and i>0))

            for slice_i, s in enumerate(start_ar):
                # Rely on samples then convert later the x-axis to time
                xaxis = np.arange(slice_i * stride_in_steps+shift_start, slice_i * stride_in_steps + shift_start + seg_nsteps)
                time_ax = xaxis * snnreg_model.dt
                ax.plot(time_ax, var_numpy[arr_pointer : arr_pointer+ seg_nsteps], alpha=alpha, linestyle=linestyle, label=label)
                arr_pointer += seg_nsteps
            
            # Figure styling
            if ylim is not None:
                ax.set_ylim(ylim[row][0], ylim[row][1])

            if i == 0:
                if ylim is not None:
                    ax.set_yticks(np.linspace(0, ylim[row][-1], N_YTICKS), np.linspace(0, ylim[row][-1], N_YTICKS).astype(int))

                ax.set_ylabel(FORCE_MVC_LABEL, labelpad=YLAB_PAD)
            if use_subplots:
                ax.set_xticks(np.arange(0, tot_dur, XTICKS_STEP), np.arange(0, tot_dur, XTICKS_STEP).astype(int))
    
   

#MANPLOT: SNN learnt tau_mem/ tau_syn in a separate figure
def plot_learnt_parameter(trained_parameter:np.ndarray, initial_parameter:np.ndarray, n_neurons:int, dt:float,
                          data_config:DataConfig, snn_config:SNNConfig,
                          fold_i:int, param_type:str='taumem', ax:plt.Axes=None):
    """
    Plots the learnt tau_mem or tau_syn or threshold for each finger in a bar plot showing the initial and trained values..
    """
    n_epochs = snn_config.num_iter if hasattr(snn_config, 'num_iter') else snn_config.training['num_iter']
    use_wandb = snn_config.use_wandb if hasattr(snn_config, 'use_wandb') else snn_config.logging['wandb']['use_wandb']
    exp_dir = snn_config.exp_dir if hasattr(snn_config, 'exp_dir') else snn_config.task['exp_dir']
    first_tau_filt = snn_config.first_filter_tau if hasattr(snn_config, 'first_filter_tau') else snn_config.decoder_type['first_filter_tau']
    second_tau_filt = snn_config.second_filter_tau if hasattr(snn_config, 'second_filter_tau') else snn_config.decoder_type['second_filter_tau']
    tau_syn = snn_config.tau_syn if hasattr(snn_config, 'tau_syn') else snn_config.decoder_type['tau_syn']
    tau_mem = snn_config.tau_mem if hasattr(snn_config, 'tau_mem') else snn_config.decoder_type['tau_mem']
    learn_tau_mem = snn_config.learn_tau_mem if hasattr(snn_config, 'learn_tau_mem') else snn_config.decoder_type['learn_tau_mem']
    learn_tau_syn = snn_config.learn_tau_syn if hasattr(snn_config, 'learn_tau_syn') else snn_config.decoder_type['learn_tau_syn']
    save_fig = snn_config.save_fig if hasattr(snn_config, 'save_fig') else snn_config.logging['save_fig']

    if len(trained_parameter.shape) < 1:
        trained_parameter = np.repeat(trained_parameter, n_neurons)
        initial_parameter = np.repeat(initial_parameter, n_neurons)

    if param_type=='taumem' or param_type=='tausyn':
        # map from constant back to tau (in ms)
        plot_learned_parameter = map_decay_to_tau(trained_parameter, dt=dt) * 1000
        plot_initial_parameter = map_decay_to_tau(initial_parameter, dt=dt) * 1000
        padding = 0.5 # padding for the annotation
    elif param_type=='threshold':
        plot_learned_parameter = trained_parameter
        plot_initial_parameter = initial_parameter
        padding = 0.005
    else:
        raise ValueError(f"Invalid parameter type {param_type}")
    # Figure styling
    # set y limit based on the parameter max value
    width = 0.45
    snn_color = sns.color_palette(palette='Accent') [0]
    ylim = np.max(plot_learned_parameter) * 1.5
    alpha_trained = 0.6
    alpha_init = 0.4

    is_new_figure = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        is_new_figure= True
    ax.bar(np.arange(n_neurons), plot_learned_parameter, width=width, alpha=alpha_trained,
           color= snn_color, label='Trained')
    ax.bar(np.arange(n_neurons), plot_initial_parameter, width=width, alpha=alpha_init,
            color= data_config.color_dict['midnight_blue'], label='Initial')
    ax.set_xlim(-0.5, n_neurons-0.5)
    ax.set_xlabel("Finger Neuron", labelpad=XLAB_PAD)
    ax.set_xticks(np.arange(0, n_neurons, 1), list(data_config.finger_label_map.keys()))

    # ax.set_ylim(0, ylim)
    sns.despine(ax=ax, offset=0, trim=False)

    if is_new_figure:
        plt.legend(frameon=False)

    # select the maximum value of learnt_tau and init_tau to use for annotation
    annot_tip = np.maximum.reduce([plot_learned_parameter, plot_initial_parameter])
    annotate_tip_bar(ax, annot_tip, annot_tip, data_config.color_dict['midnight_blue'], padding=padding)

    if param_type == 'taumem':
        ax.set_ylabel(r"$\tau_{m}$ (ms)", labelpad=YLAB_PAD)
    elif param_type == 'tausyn':
        ax.set_ylabel(r"$\tau_{s}$ (ms)", labelpad=YLAB_PAD)
    elif param_type == 'threshold':
        ax.set_ylabel("Threshold (a.u.)", labelpad=YLAB_PAD)

    if is_new_figure:
        if use_wandb:
            if param_type == 'taumem':
                wandb.log({f"Tau mem": wandb.Image(fig)})
            elif param_type == 'tausyn':
                wandb.log({f"Tau syn": wandb.Image(fig)})
        if  save_fig:
            filename = f"{data_config.mvc}_{exp_dir}_ep_{n_epochs}_fold_{fold_i}_taufilt1_{first_tau_filt}_taufilt2_{second_tau_filt}_trained{param_type}"
            if not learn_tau_syn:
                filename += f"_tausyn_{tau_syn}"  #tausyn_{snn_config.tau_syn}_learn_beta_{snn_config.learn_tau_mem}
            if not learn_tau_mem:
                filename += f"_taumem_{tau_mem}"
        
            # title = f"ep_{snn_config.num_iter}_learnt_taumem_fold_{fold_i}"
            file_path = os.path.join(data_config.figs_snn_path, f"{filename}.png")
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saving figure to {file_path}")



# def barplot_learnt_tau_and_decay_rate(ax, learnt_param, layer, param_name, dt, color_dict):
#     """
#     Bar plot for the learnt tau_mem or tau_syn for each neuron in the layer
#     """
#     if param_name == 'Beta':
#         tau = 'tau_mem'
#         y_lim = 100
#     elif param_name == 'Alpha':
#         tau = 'tau_syn'
#         y_lim = 40
#     width = 0.45
#     n_neurons = len(learnt_param) 
#     ax.bar(np.arange(n_neurons), learnt_param,
#            label=f"Learnt {param_name} {layer}", width=width,
#            color=color_dict['midnight_blue'], alpha=0.6)
#     ax.set_xlim(-0.5, n_neurons-0.5)

#     # map from constant back to tau (in ms)
#     mapped_tau_in_ms = map_decay_to_tau(learnt_param, dt=dt) * 1000
#     # add value of tau to the tip of bar
#     annotate_tip_bar(ax, learnt_param,mapped_tau_in_ms, color_dict['wisteria'])

#     ax2 = ax.twinx()
#     ax2.bar(np.arange(n_neurons), mapped_tau_in_ms, width=width, color=color_dict['wisteria'], alpha=0.6)
#     ax.set_xlabel("Neuron Id")
#     ax.set_ylabel(f"Learnt {param_name}")
#     ax.set_ylim(0, 1)
#     ax2.set_ylabel("tau [ms]", color=color_dict['wisteria'])
#     ax2.tick_params(axis='y', labelcolor=color_dict['wisteria'], labelsize=6)
#     plt.title(f'{tau}')
#     ax2.set_ylim(-1, y_lim)


def annotate_tip_bar(ax, y_value:np.ndarray, mapped_tau_in_ms:np.ndarray, color:str, padding:float=None):
    """
    Adds the value of tau to the tip of the bar
    """
    x_locs = [rect.get_x() for rect in ax.patches]
    for i, tau in enumerate(mapped_tau_in_ms):
        if padding:
            y_value[i] += padding
        plt.text(x_locs[i],y_value[i], f"{tau:.2f}", fontsize=8,
                 color=color)
        




#MANPLOT: SNN raster input spikes
def plot_network_variable(net_var:np.ndarray, dataset:EMGDataset, dt, snn_config:SNNConfig,
                         data_config:DataConfig, fold_i:int, var_name:str,ax:plt.Axes=None,
                         linestyle='solid', solid_alpha=0.4):
    """
    Plots the input spikes, membrane potential and synaptic current  for a given fold considering the overlap and the bin width.
    It sequentially plots the data for each finger and each MVC level.
        net_var : numpy array of shape (1, num_samples, num_neurons)

    """

    bin_width = snn_config.train_rep_binwidth if hasattr(snn_config, 'train_rep_binwidth') else snn_config.training['train_rep_binwidth']
    overlap_perc = snn_config.overlap_perc if hasattr(snn_config, 'overlap_perc') else snn_config.task['overlap_perc']
    train_fing_id = snn_config.train_fing_id if hasattr(snn_config, 'train_fing_id') else snn_config.task['train_fing_id']
    dt = snn_config.dt if hasattr(snn_config, 'dt') else snn_config.training['dt']
    single_rep_dur = get_repetition_duration_from_segments(dataset.num_segments, bin_width, overlap_perc)
    save_fig = snn_config.save_fig if hasattr(snn_config, 'save_fig') else snn_config.logging['save_fig']
    num_iter = snn_config.num_iter if hasattr(snn_config, 'num_iter') else snn_config.training['num_iter']
    exp_dir = snn_config.exp_dir if hasattr(snn_config, 'exp_dir') else snn_config.task['exp_dir']

    start_ar = dataset.segments_start[data_config.mvc]
    seg_nsteps = int(bin_width/dt)

    # figure styling
    mvc_marker = ['.', '|']
    mvc_marker_size = [.7, 4]
    scatter_alpha = 0.6

    shift_start = 0  # shift the start of the plot in case of multiple fingers in the same plot
    buffer_samples = int(BUFFER_TIME_PLOT/dt)
    stride_in_steps = int(bin_width * (1 - overlap_perc)/dt)
    arr_pointer = 0
    is_new_figure = False

    if ax is None:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        is_new_figure=True
    for fing_i, _ in enumerate(train_fing_id):
        for mvc_j in range(len(data_config.load_multi)):
            # this shift is needed in case of plotting multiple fingers in the same plot (e.g. debugging plots)
            shift_start = (2*fing_i+mvc_j) * (single_rep_dur/dt +
                                              buffer_samples * (len(train_fing_id) > 1 and fing_i > 0))
            for slice_i, s in enumerate(start_ar):
                # Rely on samples then convert later the x-axis to time
                xaxis = np.arange(slice_i * stride_in_steps+shift_start,
                                  slice_i * stride_in_steps + shift_start + seg_nsteps)
                if var_name == 'spk_in' or var_name == 'spk_out':
                    slice_input = net_var[0, arr_pointer: arr_pointer + seg_nsteps, :]
                    slice_times = np.where(slice_input > 0)[0] * dt + xaxis[0] * dt
                    slice_ids = np.where(slice_input > 0)[1]
                    ax.plot(slice_times, slice_ids,
                            mvc_marker[mvc_j], markersize=mvc_marker_size[mvc_j], alpha=scatter_alpha,
                            color=data_config.finger_color_list[fing_i])
                if var_name == 'syn' or var_name == 'mem' or var_name=='cur' or var_name=='emg_in':
                    time_axis = xaxis * dt
                    ax.plot(time_axis, net_var[0, arr_pointer: arr_pointer + seg_nsteps, :], alpha=solid_alpha,
                            linestyle=linestyle)

                arr_pointer += seg_nsteps
    ax.set_xlabel(xlabel=TIME_LABEL, labelpad=XLAB_PAD)
    ax.set_xticks(np.arange(0, xaxis[-1]*dt, single_rep_dur))
    ax.set_xlim([-1, xaxis[-1]*dt + 2])

    if var_name == 'spk_in':
        nticks = 5
        ax.set_ylabel(ylabel=NEURON_LABEL, labelpad=YLAB_PAD)
        ax.set_yticks(np.arange(0, dataset.num_inputs,  dataset.num_inputs/nticks))

    elif var_name == 'syn':
        ax.set_xlabel(TIME_LABEL, labelpad=XLAB_PAD)
        ax.set_ylabel("Syn. Cur. (a.u.)", labelpad=YLAB_PAD)
    elif var_name == 'cur':
        ax.set_xlabel(TIME_LABEL, labelpad=XLAB_PAD)
        ax.set_ylabel("Neu. Cur. (a.u.)", labelpad=YLAB_PAD)
    elif var_name == 'mem':
        ax.set_xlabel(TIME_LABEL, labelpad=XLAB_PAD)
        ax.set_ylabel("Mem. Pot. (a.u.)", labelpad=YLAB_PAD)

    sns.despine(ax=ax, offset=0, trim=False)

    if is_new_figure:
        handles = [Patch(facecolor=c, edgecolor=data_config.color_dict['midnight_blue'], alpha=LEGEND_ALPHA) for c in data_config.finger_color_list] + \
            [plt.Line2D([], [], alpha=0),
             plt.Line2D([], [], color=data_config.color_dict['midnight_blue'], alpha=LEGEND_ALPHA,
                        marker=mvc_marker[0],
                        markersize=10, linestyle='None'),

             plt.Line2D([], [], color=data_config.color_dict['midnight_blue'], alpha=LEGEND_ALPHA,
                        marker=mvc_marker[1],
                        markersize=10, linestyle='None')]

        labels = list(data_config.finger_label_map.keys())
        labels = [f'{l.capitalize()}' for l in labels] + \
            ['',  '15 % MVC', '5   % MVC']
        fig.legend(handles=handles, labels=labels, loc='center left',
                   bbox_to_anchor=(-0.22, 0.65),
                   ncols=1, frameon=False)
        fig.tight_layout()

        if save_fig:
            filename = f"input_{exp_dir}_ep_{num_iter}_fold_{fold_i}"
            file_path = os.path.join(data_config.figs_snn_path, f"{filename}.png")
            fig.savefig(file_path, dpi=300,bbox_inches='tight')
            logging.info(f"Saving figure to {file_path}")



#MANPLOT: SNN heatmap trained weights
def plot_weight_heatmap(weights:np.ndarray, snn_config:SNNConfig, data_config:DataConfig, fold_i:int):
    hmap_xlabel = "Input Neurons"
    hmap_ylabel = "Output Neurons"
    n_epochs = snn_config.num_iter if hasattr(snn_config, 'num_iter') else snn_config.training['num_iter']
    exp_dir = snn_config.exp_dir if hasattr(snn_config, 'exp_dir') else snn_config.task['exp_dir']
    first_tau_filt = snn_config.first_filter_tau if hasattr(snn_config, 'first_filter_tau') else snn_config.decoder_type['first_filter_tau']
    second_tau_filt = snn_config.second_filter_tau if hasattr(snn_config, 'second_filter_tau') else snn_config.decoder_type['second_filter_tau']
    tau_syn = snn_config.tau_syn if hasattr(snn_config, 'tau_syn') else snn_config.decoder_type['tau_syn']
    learn_tau_mem = snn_config.learn_tau_mem if hasattr(snn_config, 'learn_tau_mem') else snn_config.decoder_type['learn_tau_mem']
    save_fig = snn_config.save_fig if hasattr(snn_config, 'save_fig') else snn_config.logging['save_fig']
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111)

    sns.heatmap(weights, ax=ax, cmap='YlGnBu',cbar_kws={'label': 'Synaptic Weight (a.u.)'} )
    ax.set_xlabel(hmap_xlabel)
    ax.set_ylabel(hmap_ylabel)
    #TODO: set yticks to name of the finger and set a number of x ticks to show. Even Better to save the weights + biass and plot
    # in compile notebook
    ax.set_yticks(np.arange(0, weights.shape[0], 1), list(data_config.finger_label_map.keys()))
    fig.tight_layout()

    if save_fig:
        filename = f"{data_config.mvc}_{exp_dir}_ep_{n_epochs}_whmap_fold{fold_i}_taufilt1_{first_tau_filt}_taufilt2_{second_tau_filt}_tau_syn_{tau_syn}_learn_beta_{learn_tau_mem}"
        file_path = os.path.join(data_config.figs_snn_path, f"{filename}.png")
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saving figure to {file_path}")


def detach_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def print_filters_time_constants(snn_network):
    """
    Prints the time constants of the filters in the network"""
    print("Filters time constants:")
    for filter_layer in snn_network.filter_layers:
        print(map_decay_to_tau(filter_layer.beta.detach().clone().cpu().numpy(), snn_network.dt))


#MANPLOT: SNN debugging plots
def network_variables_plot(snn_network, snn_config, data_config, record_dict, y_true, dataset,
                           fold_i, beta_init, alpha_init, threshold_init,
                           fing_id=None):

    """
    rec: dict of tensors shape (num_timesteps, batch_size, num_neurons)
    """
    if len(snn_network.filter_layers) > 0:
        print_filters_time_constants(snn_network)

    split_type = snn_config.split_type if hasattr(snn_config, 'split_type') else snn_config.task['split_type']
    split_rep = snn_config.split_rep if hasattr(snn_config, 'split_rep') else snn_config.task['split_rep']
    train_fing_id = snn_config.train_fing_id if hasattr(snn_config, 'train_fing_id') else snn_config.task['train_fing_id']
    exp_dir = snn_config.exp_dir if hasattr(snn_config, 'exp_dir') else snn_config.task['exp_dir']
    num_iter = snn_config.num_iter if hasattr(snn_config, 'num_iter') else snn_config.training['num_iter']
    save_fig = snn_config.save_fig if hasattr(snn_config, 'save_fig') else snn_config.logging['save_fig']
    use_wandb = snn_config.use_wandb if hasattr(snn_config, 'use_wandb') else snn_config.logging['wandb']['use_wandb']
    tau_syn = snn_config.tau_syn if hasattr(snn_config, 'tau_syn') else snn_config.decoder_type['tau_syn']
    tau_mem = snn_config.tau_mem if hasattr(snn_config, 'tau_mem') else snn_config.decoder_type['tau_mem']
    learn_tau_syn = snn_config.learn_tau_syn if hasattr(snn_config, 'learn_tau_syn') else snn_config.decoder_type['learn_tau_syn']
    learn_tau_mem = snn_config.learn_tau_mem if hasattr(snn_config, 'learn_tau_mem') else snn_config.decoder_type['learn_tau_mem']


    # Extracting the variables from the record_dict to plot
    spk_in_or_enc = detach_to_numpy(record_dict[ENC_SPK_OUT]) if 'encoding' in snn_network.topology  else detach_to_numpy(record_dict[SPK_IN])
    predicted_output = detach_to_numpy(record_dict[snn_network.prediction_var])
    spk_out = detach_to_numpy(record_dict[SPK_OUT])

    syn_current = detach_to_numpy(record_dict[snn_network.plot_syn_cur])
    neuron_current = detach_to_numpy(record_dict[snn_network.plot_neu_cur])
    neuron_membrane = detach_to_numpy(record_dict[snn_network.plot_neu_mem])

    learnt_threshold = detach_to_numpy(snn_network.lif1.threshold)
    learnt_weights = detach_to_numpy(snn_network.fc1.weight)
    learnt_beta = detach_to_numpy(snn_network.lif1.beta)
    learnt_alpha = detach_to_numpy(snn_network.lif1.alpha)

    dt = snn_network.dt

    # figure styling
    title = f'tau_syn:{tau_syn*1000} ms'
    hmap_xlabel = "Input Neurons"
    hmap_ylabel = "Output Neurons"
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(4, 2)
    time_ax = np.arange(0, spk_in_or_enc.shape[1]*dt, dt)

    # 0.0 Input spikes
    ax = fig.add_subplot(gs[0, 0])
    plot_network_variable(spk_in_or_enc, dataset, dt, snn_config, data_config, fold_i, ax=ax, var_name='spk_in')
    
    # 0.1 Trained weights
    ax = fig.add_subplot(gs[0, 1])
    sns.heatmap(learnt_weights, ax=ax, cmap='YlGnBu',cbar_kws={'label': 'Syn. Weight (a.u.)'} )
    ax.set_xlabel(hmap_xlabel)
    ax.set_ylabel(hmap_ylabel)

    # 1.0 Spiking layer synaptic current
    ax = fig.add_subplot(gs[1, 0])
    ax.set_prop_cycle(color=data_config.finger_color_list)
    if split_type == 'with_overlap' and split_rep:
        plot_network_variable(syn_current, dataset,dt, snn_config, data_config, fold_i, ax=ax, var_name='syn')
    else:
        ax.plot(time_ax, syn_current[0, :, :], label="Syn current", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(TIME_LABEL, labelpad=XLAB_PAD)
        ax.set_ylabel("Syn. Current (a.u.)", labelpad=YLAB_PAD)
        sns.despine(ax=ax, offset=0, trim=False)
    
    # 1.1 Spiking layer neuron's input current
    ax = fig.add_subplot(gs[1, 1])
    ax.set_prop_cycle(color=data_config.finger_color_list)
    if split_type == 'with_overlap' and split_rep:
        plot_network_variable(neuron_current, dataset, dt, snn_config, data_config, fold_i, ax=ax, var_name='cur')

    else:
        ax.plot(time_ax, neuron_current[0, :, :], alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel(TIME_LABEL, labelpad=XLAB_PAD)
        ax.set_ylabel("Input Current", labelpad=YLAB_PAD)
        sns.despine(ax=ax, offset=0, trim=False)

    # 2.0 Plot prediction
    ax = fig.add_subplot(gs[2, 0])
    ax.set_prop_cycle(color=data_config.finger_color_list)

    if split_type == 'with_overlap' and split_rep:
        plot_network_variable(predicted_output, dataset, dt, snn_config, data_config, fold_i, ax=ax, var_name='mem',
                                linestyle='dotted', solid_alpha=0.2)
        plot_network_variable(np.expand_dims(y_true,axis=0), dataset, dt, snn_config, data_config, fold_i, ax=ax, var_name='mem',
                                solid_alpha=0.4)
        
        # add output spikes
        ax2 = ax.twinx()
        plot_network_variable(spk_out, dataset, dt, snn_config, data_config, fold_i, ax=ax2, var_name='spk_out')
        sns.despine(ax=ax2, offset=0, trim=False, right=True, left=False)
        ax2.set_yticks([])
        ax2.set_ylim([-1,len(train_fing_id) + 2])
    else:
        ax.set_prop_cycle(color=data_config.finger_color_list)
        ax.plot(time_ax, predicted_output[0, :, :], label="vmem filt", alpha=0.5, linestyle='dashed')
        ax.plot(time_ax, y_true , label="Target", alpha=0.7)
        ax.set_xlabel(TIME_LABEL, labelpad=XLAB_PAD)
        ax.set_ylabel("Mem. Potential (a.u.)", labelpad=YLAB_PAD)
        sns.despine(ax=ax, offset=0, trim=False)

    # 2.1 Initial vs trained threshold of spiking neurons
    ax = fig.add_subplot(gs[2, 1])
    print(f"learnt threshold:{learnt_threshold}")
    plot_learnt_parameter(learnt_threshold, threshold_init, snn_network.num_outputs,dt, data_config,
                          snn_config, fold_i, param_type='threshold', ax=ax)

    # 3.0/3.1 Initial vs trained tau_mem and tau_syn of the spiking layer
    ax = fig.add_subplot(gs[3, 0])
    print(f"learnt beta:{learnt_beta}")
    plot_learnt_parameter(learnt_beta, beta_init, snn_network.num_outputs, dt, data_config , snn_config, fold_i,
                        param_type='taumem', ax=ax)

    ax = fig.add_subplot(gs[3, 1])
    print(f"learnt alpha:{learnt_alpha}")
    plot_learnt_parameter(learnt_alpha, alpha_init,snn_network.num_outputs, dt, data_config , snn_config, fold_i,
                    param_type='tausyn', ax=ax)

    # add a common title
    fig.tight_layout()
    if use_wandb:
        wandb.log({"Network variables": wandb.Image(fig)})
    if save_fig:
        filename = f"{data_config.mvc}_{exp_dir}_ep_{num_iter}_fold_{fold_i}_taumem_{tau_mem}_tau_syn_{tau_syn}_learn_alpha_{learn_tau_syn}_learn_beta_{learn_tau_mem}"
        if fing_id is not None:
            filename = f"{filename}_fing_{fing_id}"
        file_path = os.path.join(data_config.figs_snn_path, f"{filename}.png")
        fig.savefig(file_path, dpi=300)
        logging.info(f"Saving figure to {file_path}")

    # Plot learnt tau_mem and tau_syn in a separate figure
    plot_learnt_parameter(learnt_beta, beta_init, snn_network.num_outputs, dt, data_config , snn_config, fold_i, param_type='taumem')
    plot_learnt_parameter(learnt_alpha, alpha_init, snn_network.num_outputs, dt, data_config , snn_config, fold_i, param_type='tausyn')


#MANPLOT: SNN predictions vs true
def plot_pred_vs_true(snn_network, snn_config:SNNConfig, data_config:DataConfig, dataset:EMGDataset,
                      y_pred:np.ndarray, y_true:np.ndarray, test_rep:int, fold_i:int, mode:str='training'):
    """
    Plots the predicted and true output for the given fold /test repetition.
    """
    train_rep_binwidth = snn_config.train_rep_binwidth if hasattr(snn_config, 'train_rep_binwidth') else snn_config.training['train_rep_binwidth']
    inf_rep_binwidth = snn_config.inf_rep_binwidth if hasattr(snn_config, 'inf_rep_binwidth') else snn_config.training['inf_rep_binwidth']
    split_rep = snn_config.split_rep if hasattr(snn_config, 'split_rep') else snn_config.task['split_rep']
    split_type = snn_config.split_type if hasattr(snn_config, 'split_type') else snn_config.task['split_type']
    exp_dir = snn_config.exp_dir if hasattr(snn_config, 'exp_dir') else snn_config.task['exp_dir']
    num_iter = snn_config.num_iter if hasattr(snn_config, 'num_iter') else snn_config.training['num_iter']
    save_fig = snn_config.save_fig if hasattr(snn_config, 'save_fig') else snn_config.logging['save_fig']
    use_wandb = snn_config.use_wandb if hasattr(snn_config, 'use_wandb') else snn_config.logging['wandb']['use_wandb']
    tau_syn = snn_config.tau_syn if hasattr(snn_config, 'tau_syn') else snn_config.decoder_type['tau_syn']
    learn_tau_mem = snn_config.learn_tau_mem if hasattr(snn_config, 'learn_tau_mem') else snn_config.decoder_type['learn_tau_mem']
    first_filter_tau = snn_config.first_filter_tau if hasattr(snn_config, 'first_filter_tau') else snn_config.decoder_type['first_filter_tau']
    second_filter_tau = snn_config.second_filter_tau if hasattr(snn_config, 'second_filter_tau') else snn_config.decoder_type['second_filter_tau']

    fig  = plt.figure(figsize=(13,4))
    gs = gridspec.GridSpec(nrows=len(data_config.load_multi), ncols = snn_network.num_outputs)
  
    bin_width = train_rep_binwidth if mode == 'training' else inf_rep_binwidth


    if mode == 'training':
        alpha_pred_plot  = 0.3
        alpha_true_plot = 0.8
        lstyle_true = (0, (1, 3))

    if mode == 'inference':
        alpha_pred_plot  = 0.15
        alpha_true_plot = 1
        lstyle_true = (0.5, (1, 30))   # dotted line the larger the second number the more spaced the dots


    if split_type == 'with_overlap' and split_rep:
        ax = None  # dont use an axis, the axis is created in the plot_with_overlap function in case of using subplots
        if y_pred is not None:
            plot_with_overlap(y_pred, snn_network, snn_config, dataset, ax, data_config.mvc,
                              data_config.finger_color_list,
                              alpha=alpha_pred_plot, fig=fig, gs=gs, linestyle=LSTYLE_PRED, ylim=None, mode=mode)
            
        plot_with_overlap(y_true, snn_network, snn_config, dataset, ax, data_config.mvc,
                          data_config.finger_color_list,
                          alpha=alpha_true_plot, fig=fig, gs=gs, linestyle=lstyle_true,ylim=None,  mode=mode)

        if SHOW_TITLE:
            fig.suptitle(
                f'{data_config.mvc} % MVC, {data_config.direction}  Fold {fold_i}  Test Rep {test_rep}  -  Mode:{mode.capitalize()}  [{bin_width} seconds]')        
        fig.tight_layout()

        # tricking legend to show all patches on one column
        handles = [Patch(facecolor=c, edgecolor=data_config.color_dict['midnight_blue'], alpha=LEGEND_ALPHA) for  c in data_config.finger_color_list] + \
            [plt.Line2D([0], [0], color=data_config.color_dict['midnight_blue'], lw=2, alpha=LEGEND_ALPHA, linestyle='dotted'),
             plt.Line2D([0], [0], color=data_config.color_dict['midnight_blue'], lw=2, alpha=LEGEND_ALPHA, linestyle='solid'),
            plt.Line2D([], [], alpha=0), plt.Line2D([], [], alpha=0), ]

        labels = list(data_config.finger_label_map.keys())
        labels = [f'{l.capitalize()}' for l in labels] + ['True', 'Predicted','', '', ]
        _ = fig.legend(handles=handles,
                         labels=labels,
                         loc='upper left', bbox_to_anchor=(-0.18, 1.01),
                         ncols=2, frameon=False)
    
    if use_wandb:
        wandb.log({f"Test Pred_vs_true Fold {fold_i}": wandb.Image(fig)})
    

    if save_fig:
        filename = f"{data_config.mvc}_{exp_dir}_ep_{num_iter}_fold_{fold_i}_test_rep_{test_rep}_mode_{mode}_binwidth_{bin_width}_taufilt1_{first_filter_tau}_taufilt2_{second_filter_tau}_tau_syn_{tau_syn}_learn_beta_{learn_tau_mem}"
        file_path = os.path.join(data_config.figs_snn_path, f"{filename}.png")
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saving figure to {file_path}")