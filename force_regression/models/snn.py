from __future__ import annotations
from typing import Dict, Tuple
import logging
import os
import pickle as pkl
import random
import time
import warnings

from neurobench.benchmarks import Benchmark
from neurobench.metrics.static import (
        Footprint,
        ConnectionSparsity,
    )
from neurobench.metrics.workload import (
        ActivationSparsity,
        SynapticOperations,
        ClassificationAccuracy
    )
from neurobench.models import SNNTorchModel
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from snntorch import surrogate
from tabulate import tabulate
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchsummary import summary
import neurobench
import numpy as np
import pandas as pd
import snntorch._neurons as snn
import torch
import torch.nn as nn
import tqdm
import wandb

from configs.constants import *
from force_regression.config.dataconfig import DataConfig
from force_regression.config.snnconfig import SNNConfig
from force_regression.data.preprocessing.emg import apply_butter_lowpass
from force_regression.evaluation.metrics import prepare_snn_metrics_df
from force_regression.training.snn_pipeline import assign_datasets_for_train_test, conc_var_tracker, target_tracker_to_np, create_subsets, reconstruct_rep_from_bins
from force_regression.training.snn_pipeline import map_tau_to_decay
from force_regression.training.snn_pipeline import split_tensors_for_rep
import force_regression.plotting.snn_plot as snnplot
import force_regression.data.preprocessing.spikes as spk
import force_regression.utils.functions as fn

warnings.simplefilter("ignore")


DEVICE = 'cpu'
DTYPE = torch.float




# --- SNN Topologies ---

class SnnTopology(torch.nn.Module):
    """
    Base class for SNN topologies
    """
    def __init__(self, timesteps, num_inputs, num_outputs, snn_config, use_wandb, dt):
        super(SnnTopology, self).__init__()
        self.use_wandb = use_wandb
        self.dt = dt
        self.timesteps = timesteps
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.n_electrodes = 3
        self.n_channels_per_electrode = 40
        self.topology = snn_config["topology"]
        self.snn_config = snn_config
        self.spike_grad = surrogate.fast_sigmoid()
        self.param_std = 0.05  # standard deviation for parameter initialization
        self.leaky_integrator_threshold = 300 # spiking threshold for the leaky integrator
        self.loss_function = torch.nn.MSELoss()
        self.state_vars = self._topology_state_variables()
        self.plot_syn_cur, self.plot_neu_cur, self.plot_neu_mem = self._plot_network_variables()
        self.empty_record_dict = self.create_record_dict_for_vars()

    def _topology_state_variables(self):
        """
        Returns the state variables of the network
        """
        state_vars = []
        if self.topology == SHALLOW_SPIKING_DOUBLE_FILTER:
            state_vars = [SPK_IN, SPK_SYN_CUR, SPK_NEU_CUR,
                          SPK_NEU_MEM, SPK_OUT, FILT_NEU_CUR,
                          FILT_NEU_MEM, FILT_SPK_OUT,FILT2_NEU_MEM
                          ]

        if self.topology == SHALLOW_LEAKY:
            state_vars = [SPK_IN, LEAK_SYN_CUR, LEAK_NEU_CUR, SPK_OUT,
                          LEAK_NEU_MEM]
            if not self.snn_config["post_process_filt"]:
                state_vars.extend([FILT_NEU_MEM, FILT_SPK_OUT])

        if self.topology== ENC_SHALLOW_SPIKING:
            state_vars = [EMG_IN, ENC_NEU_CUR, ENC_NEU_MEM, ENC_SPK_OUT,
                          SPK_SYN_CUR, SPK_NEU_CUR, SPK_NEU_MEM, SPK_OUT,
                          FILT_NEU_CUR, FILT_NEU_MEM, FILT_SPK_OUT,
                          FILT2_NEU_MEM]
                
        if self.topology == ENC_SHALLOW_LEAKY:
            state_vars = [EMG_IN, ENC_NEU_CUR, ENC_NEU_MEM, ENC_SPK_OUT,
                          LEAK_SYN_CUR, LEAK_NEU_CUR, SPK_OUT,
                          LEAK_NEU_MEM]
            if self.snn_config["first_filter_tau"] > 0:
                state_vars.extend([FILT_NEU_MEM, FILT_SPK_OUT])
            if self.snn_config["use_aleaky"]:
                state_vars.extend([ENC_THRESHOLD_ADAPT, ENC_ALIF_THRESHOLD])

        return state_vars
    

    def _set_parameter_per_electrode(self, parameter:float, parameter_name:str):
        """
        Sets the decay and threshold parameter for the encoding layer
        """
        if self.set_electrode_specific_parameters:
            # assert that self.enc_tau_mem is a list of length num_inputs
            assert isinstance(parameter, list), "parameter should be a list"
            parameter_per_neuron = np.repeat(parameter , self.n_channels_per_electrode)
            assert len(parameter_per_neuron) == self.num_inputs, f"parameter should be a list of length {self.num_inputs}"
            if parameter_name == 'tau_mem':
                parameter_per_neuron = torch.stack([map_tau_to_decay(parameter_per_neuron[i], self.dt) for i in range(self.num_inputs)],dim=0)
        else:
            parameter_per_neuron = parameter
            if parameter_name == 'tau_mem':
                parameter_per_neuron = map_tau_to_decay(parameter,self.dt)
        return parameter_per_neuron

    def _plot_network_variables(self):
        """
        Defines which network variables to plot in the network_variables_plot
        """
        plot_syn_cur = None
        plot_neu_cur = None
        plot_neu_mem = None
        if self.topology == SHALLOW_SPIKING_DOUBLE_FILTER:
            plot_syn_cur = SPK_SYN_CUR
            plot_neu_cur = SPK_NEU_CUR
            plot_neu_mem = SPK_NEU_MEM
        
        if self.topology == SHALLOW_LEAKY:
            plot_syn_cur = LEAK_SYN_CUR
            plot_neu_cur = LEAK_NEU_CUR
            plot_neu_mem = LEAK_NEU_MEM
        
        if self.topology == ENC_SHALLOW_SPIKING:
            plot_syn_cur = SPK_SYN_CUR
            plot_neu_cur = ENC_NEU_CUR
            plot_neu_mem = ENC_NEU_MEM
        if self.topology == ENC_SHALLOW_LEAKY:
            plot_syn_cur = LEAK_SYN_CUR
            plot_neu_cur = ENC_NEU_CUR
            plot_neu_mem = ENC_NEU_MEM
        
        return plot_syn_cur, plot_neu_cur, plot_neu_mem
    
    def create_record_dict_for_vars(self):
        """
        Creates a dictionary to store the variables of the network
        """
        record_dict = {f'{state_var}':[] for state_var in self.state_vars}
        return record_dict

    def describe(self):
        """
        Describes the SNN topology
        """
        descriptions = {
            SHALLOW_LEAKY: 'a single layer with a leaky integrate and fire neuron',
            'shallow_spiking_single_filter': 'a single layer with a spiking neuron and a single filter. Filter is a leaky integrator',
            SHALLOW_SPIKING_DOUBLE_FILTER: 'a single layer with a spiking neuron and two filters. Two layers of leaky integrators'
        }
        return descriptions.get(self.topology, 'Unknown topology')

    def forward(self):
        """
        Forward pass for the network
        """
        print("Forward is implemented in the child class")

class ShallowSpikingDoubleFilter(SnnTopology):
    def __init__(self, timesteps, num_inputs, num_outputs, snn_config, use_wandb, dt):
        SnnTopology.__init__(self, timesteps, num_inputs, num_outputs, snn_config, use_wandb, dt)
        self._init_network_parameters()
        self._create_network_layers()
        self.prediction_var = FILT2_NEU_MEM if self.snn_config["second_filter_tau"] > 0 else FILT_NEU_MEM
        self.input_var = SPK_IN
        self.leaky_integrators_mem = [FILT2_NEU_MEM, FILT2_NEU_MEM]
        self.leaky_integrators_spk= [FILT_SPK_OUT]

        self.filter_layers = [self.first_filter, self.second_filter] #if not self.post_process_filt else [self.first_filter]
       
    def _create_network_layers(self):
        self.fc1 = torch.nn.Linear(in_features=self.num_inputs,
                                out_features=self.num_outputs)

        self.lif1 = snn.Synaptic(beta=self.neurons_betas,
                                alpha=self.synapses_alphas,
                                spike_grad=self.spike_grad,
                                reset_mechanism="zero",
                                threshold=self.neurons_thresholds,
                                learn_threshold=self.learn_threshold,
                                learn_beta=self.learn_beta,
                                learn_alpha=self.learn_alpha)

        self.fc2 = torch.nn.Linear(in_features=self.num_outputs,
                                out_features=self.num_outputs,
                                bias=False)
        self.first_filter = snn.Leaky(beta=self.first_filter_beta,
                                    threshold=self.leaky_integrator_threshold,
                                    learn_beta=False)
        # if not self.post_process_filt:
        self.second_filter = snn.Leaky(beta=self.second_filter_beta,
                                threshold=self.leaky_integrator_threshold,
                                learn_beta=False)

        if self.snn_config["w_init_dist"] == 'uniform':
            nn.init.uniform_(self.fc1.weight,a=self.snn_config["w_init"][0],b=self.snn_config["w_init"][1])
        if self.snn_config["w_init_dist"] == 'normal':
            nn.init.normal_(self.fc1.weight, mean=self.snn_config["w_init_mean"],
                            std=self.snn_config["w_init_std"])
        nn.init.uniform(self.fc1.bias,a=self.snn_config["bias_init"][0],b=self.snn_config["bias_init"][1])
        # enforce a one-to-one mapping between the spiking layer and the filtering layer
        self.fc2.weight = nn.Parameter(torch.eye(self.num_outputs) * self.snn_config["w_fixed_filt"])
        self.fc2.bias = nn.Parameter(torch.zeros(self.num_outputs))
        self.fc2.requires_grad_(False)

    def _init_network_parameters(self):
        # Input to spiking neuron connected via alpha synapse with parameters tau_syn
        # Spiking neuron has parameters tau_mem and threshold
        self.tau_syn = self.snn_config["tau_syn"]
        self.tau_mem = self.snn_config["tau_mem"]
        self.spk_threshold = self.snn_config["spk_threshold"]
        self.alpha = map_tau_to_decay(self.tau_syn, self.dt)
        self.beta = map_tau_to_decay(self.tau_mem, self.dt)

        self.learn_alpha = self.snn_config["learn_tau_syn"]
        self.learn_beta = self.snn_config["learn_tau_mem"]
        self.learn_threshold = self.snn_config["learn_threshold"]
        self.post_process_filt = self.snn_config["post_process_filt"]
        self.post_process_filt_cutoff = self.snn_config["post_filt_cutoff"]
        # In case parameters are learned, initialize them from a distribution
        if self.learn_beta:
            neurons_betas = torch.rand(self.num_outputs)
            nn.init.normal_(neurons_betas, mean=self.beta,
                            std=self.param_std * self.beta)
            print(f"Init beta:{neurons_betas}\n")
        else:
            neurons_betas = self.beta

        if self.learn_alpha:
            synapses_alphas = torch.rand(self.num_outputs)
            nn.init.normal_(synapses_alphas, mean=self.alpha,
                            std=self.param_std * self.alpha)
        else:
            synapses_alphas = self.alpha

        if self.learn_threshold:
            neurons_thresholds = torch.rand(self.num_outputs)
            nn.init.normal_(neurons_thresholds, mean=self.spk_threshold,
                            std=self.param_std * self.spk_threshold)
        else:
            neurons_thresholds = self.spk_threshold

        self.neurons_betas = neurons_betas
        self.synapses_alphas = synapses_alphas
        self.neurons_thresholds = neurons_thresholds

        self.first_filter_beta = map_tau_to_decay(self.snn_config["first_filter_tau"], self.dt)
        self.second_filter_beta = map_tau_to_decay(self.snn_config["second_filter_tau"], self.dt)


    def forward(self, spk_in, state_dict=None):
        """Forward pass for each time step"""
        empty_record_dict = self.create_record_dict_for_vars()
        if state_dict is None:
            # Initalize membrane potential as empty tensor
            spk_syn_cur, spk_neu_mem = self.lif1.init_synaptic()
            filt_neu_mem =  self.first_filter.init_leaky()
            # if not self.post_process_filt:
            filt2_neu_mem = self.second_filter.init_leaky()

        else: # if not initializing as empty rely on last timestep
            spk_syn_cur = state_dict[SPK_SYN_CUR]
            spk_neu_cur = state_dict[SPK_NEU_CUR]
            spk_neu_mem = state_dict[SPK_NEU_MEM]
            spk_out = state_dict[SPK_OUT]

            filt_neu_mem = state_dict[FILT_NEU_MEM]
            filt_neu_cur = state_dict[FILT_NEU_CUR]
            filt_spk_out = state_dict[FILT_SPK_OUT]
            # if not self.post_process_filt:
            filt2_neu_mem = state_dict[FILT2_NEU_MEM]

        for step in range(self.timesteps):
            if state_dict is None or step > 0:
                spk_neu_cur = self.fc1(spk_in[:, step])
                spk_out, spk_syn_cur, spk_neu_mem = self.lif1(spk_neu_cur, spk_syn_cur, spk_neu_mem)
                filt_neu_cur = self.fc2(spk_out)
                filt_spk_out, filt_neu_mem = self.first_filter(filt_neu_cur, filt_neu_mem)
                # if not self.post_process_filt:
                _, filt2_neu_mem = self.second_filter(filt_neu_mem, filt2_neu_mem)

            # record the variables for layer 1
            empty_record_dict[SPK_IN].append(spk_in[:, step])
            empty_record_dict[SPK_SYN_CUR].append(spk_syn_cur)
            empty_record_dict[SPK_NEU_CUR].append(spk_neu_cur)
            empty_record_dict[SPK_NEU_MEM].append(spk_neu_mem)
            empty_record_dict[SPK_OUT].append(spk_out)
            empty_record_dict[FILT_NEU_CUR].append(filt_neu_cur)
            empty_record_dict[FILT_NEU_MEM].append(filt_neu_mem)
            empty_record_dict[FILT_SPK_OUT].append(filt_spk_out)
            # if not self.post_process_filt:
            empty_record_dict[FILT2_NEU_MEM].append(filt2_neu_mem)

        # stack the recorded variables
        for var in empty_record_dict:
            empty_record_dict[var] = torch.stack(empty_record_dict[var], dim=1)
        filled_record_dict = empty_record_dict
        return filled_record_dict


class ShallowLeaky(SnnTopology):
    """
    A shallow network with lekay integrator neurons followed by a filtering neuron
    """
    def __init__(self, timesteps, num_inputs, num_outputs, snn_config, use_wandb, dt):
        SnnTopology.__init__(self, timesteps, num_inputs, num_outputs, snn_config, use_wandb, dt)
        self._init_network_parameters()
        self._create_network_layers()

        self.prediction_var = FILT_NEU_MEM if not self.post_process_filt else LEAK_NEU_MEM
        self.input_var = SPK_IN
        self.leaky_integrators_mem = [FILT_NEU_MEM]
        self.leaky_integrators_spk= [SPK_OUT, FILT_SPK_OUT] if not self.post_process_filt else [SPK_OUT]
        self.filter_layers = [self.first_filter] if not self.post_process_filt else []

    def _init_network_parameters(self):
        # Input to leaky neuron connected via alpha synapse with parameters tau_syn
        # Leaky neuron has parameters tau_mem
        self.tau_syn = self.snn_config["tau_syn"]
        self.tau_mem = self.snn_config["tau_mem"]
        self.alpha = map_tau_to_decay(self.tau_syn, self.dt)
        self.beta = map_tau_to_decay(self.tau_mem, self.dt)

        self.learn_alpha = self.snn_config["learn_tau_syn"]
        self.learn_beta = self.snn_config["learn_tau_mem"]
        self.post_process_filt = self.snn_config["post_process_filt"]
        self.post_process_filt_cutoff = self.snn_config["post_filt_cutoff"]
        # In case parameters are learned, initialize them from a distribution
        if self.learn_beta:
            neurons_betas = torch.rand(self.num_outputs)
            nn.init.normal_(neurons_betas, mean=self.beta,
                            std=self.param_std * self.beta)
            print(f"Init beta:{neurons_betas}\n")
        else:
            neurons_betas = self.beta

        if self.learn_alpha:
            synapses_alphas = torch.rand(self.num_outputs)
            nn.init.normal_(synapses_alphas, mean=self.alpha,
                            std=self.param_std * self.alpha)
        else:
            synapses_alphas = self.alpha


        self.neurons_betas = neurons_betas
        self.synapses_alphas = synapses_alphas

        self.first_filter_beta = map_tau_to_decay(self.snn_config["first_filter_tau"], self.dt)

    def _create_network_layers(self):
        self.fc1 = torch.nn.Linear(in_features=self.num_inputs,
                                   out_features=self.num_outputs)

        self.lif1 = snn.Synaptic(beta=self.neurons_betas,
                                alpha=self.synapses_alphas,
                                spike_grad=self.spike_grad,
                                reset_mechanism="zero",
                                threshold=self.leaky_integrator_threshold,
                                learn_threshold=False,
                                learn_beta=self.learn_beta,
                                learn_alpha=self.learn_alpha)
        if not self.post_process_filt:
            self.first_filter = snn.Leaky(beta=self.first_filter_beta,
                                    threshold=self.leaky_integrator_threshold,
                                    learn_beta=False)
      

        if self.snn_config["w_init_dist"] == 'uniform':
            nn.init.uniform_(self.fc1.weight,a=self.snn_config["w_init"][0],b=self.snn_config["w_init"][1])
        if self.snn_config["w_init_dist"] == 'normal':
            nn.init.normal_(self.fc1.weight, mean=self.snn_config["w_init_mean"],
                            std=self.snn_config["w_init_std"])
        nn.init.uniform(self.fc1.bias,a=self.snn_config["bias_init"][0],b=self.snn_config["bias_init"][1])


    def forward(self, spk_in, state_dict=None):
        empty_record_dict = self.create_record_dict_for_vars()
        if state_dict is None:
            # Initalize membrane potential as empty tensor
            leaky_syn_cur, leaky_neu_mem = self.lif1.init_synaptic()
            if not self.post_process_filt:
                filt_neu_mem =  self.first_filter.init_leaky()

        else: # if not initializing as empty rely on last timestep values, restroring the state
            leaky_syn_cur = state_dict[LEAK_SYN_CUR]
            leaky_neu_cur = state_dict[LEAK_NEU_CUR]
            leaky_neu_mem = state_dict[LEAK_NEU_MEM]
            spk_out = state_dict[SPK_OUT]
            if not self.post_process_filt:
                filt_neu_mem = state_dict[FILT_NEU_MEM]
                filt_spk_out = state_dict[FILT_SPK_OUT]

        for step in range(self.timesteps):
            if state_dict is None or step > 0:
                leaky_neu_cur = self.fc1(spk_in[:, step])
                spk_out, leaky_syn_cur, leaky_neu_mem = self.lif1(leaky_neu_cur, leaky_syn_cur, leaky_neu_mem)

                # clip membrane potential to 0
                leaky_neu_mem = torch.clamp(leaky_neu_mem, min=0)
                if not self.post_process_filt:
                    filt_spk_out, filt_neu_mem = self.first_filter(leaky_neu_mem, filt_neu_mem)
    

            # record the variables for layer 1
            empty_record_dict[SPK_IN].append(spk_in[:, step])
            empty_record_dict[LEAK_SYN_CUR].append(leaky_syn_cur)
            empty_record_dict[LEAK_NEU_CUR].append(leaky_neu_cur)
            empty_record_dict[LEAK_NEU_MEM].append(leaky_neu_mem)
            empty_record_dict[SPK_OUT].append(spk_out)
            if not self.post_process_filt:
                empty_record_dict[FILT_NEU_MEM].append(filt_neu_mem)
                empty_record_dict[FILT_SPK_OUT].append(filt_spk_out)

        # stack the recorded variables
        for var in empty_record_dict:
            empty_record_dict[var] = torch.stack(empty_record_dict[var], dim=1)
        filled_record_dict = empty_record_dict
        return filled_record_dict
    




class EncodingShallowSpikingDoubleFilter(SnnTopology):
    """
    A shallow SNN encoding iEMG into spikes then decoding forces
    """
    def __init__(self, timesteps, num_inputs, num_outputs, snn_config, use_wandb, dt):
        SnnTopology.__init__(self, timesteps, num_inputs, num_outputs, snn_config, use_wandb, dt)
        self._init_network_parameters()
        self._create_network_layers()

        self.prediction_var = FILT2_NEU_MEM if self.snn_config["second_filter_tau"] > 0 else FILT_NEU_MEM
        self.input_var = EMG_IN
        self.leaky_integrators_mem = [FILT_NEU_MEM, FILT2_NEU_MEM]
        self.leaky_integrators_spk= [FILT_SPK_OUT]

        self.filter_layers = [self.first_filter, self.second_filter]
       
    def _create_network_layers(self):
        self.enc_fc = torch.nn.Linear(in_features=self.num_inputs,
                                        out_features=self.num_inputs,
                                        bias=False)
        # set weights of encoding layer to identity
        self.enc_fc.weight = nn.Parameter(torch.eye(self.num_inputs) * self.snn_config["w_fixed_enc"])
        # self.enc_fc.bias = nn.Parameter(torch.zeros(self.num_inputs))
        self.enc_fc.requires_grad_(False)

        self.encoding_layer = snn.Leaky(beta=self.enc_beta,
                                        threshold=self.enc_neurons_threshold,
                                        learn_beta=False,
                                        learn_threshold=self.enc_learn_threshold,
                                        reset_mechanism="zero")

        self.fc1 = torch.nn.Linear(in_features=self.num_inputs,
                                   out_features=self.num_outputs)

        self.lif1 = snn.Synaptic(beta=self.neurons_betas,
                                alpha=self.synapses_alphas,
                                spike_grad=self.spike_grad,
                                reset_mechanism="subtract",
                                threshold=self.neurons_thresholds,
                                learn_threshold=self.learn_threshold,
                                learn_beta=self.learn_beta,
                                learn_alpha=self.learn_alpha)

        self.fc2 = torch.nn.Linear(in_features=self.num_outputs,
                                out_features=self.num_outputs,
                                bias=False)
        self.first_filter = snn.Leaky(beta=self.first_filter_beta,
                                    threshold=self.leaky_integrator_threshold,
                                    learn_beta=False)
        self.second_filter = snn.Leaky(beta=self.second_filter_beta,
                                    threshold=self.leaky_integrator_threshold,
                                    learn_beta=False)

        if self.snn_config["w_init_dist"] == 'uniform':
            nn.init.uniform_(self.fc1.weight,a=self.snn_config["w_init"][0],b=self.snn_config["w_init"][1])
        if self.snn_config["w_init_dist"] == 'normal':
            nn.init.normal_(self.fc1.weight, mean=self.snn_config["w_init_mean"],
                            std=self.snn_config["w_init_std"])
        nn.init.uniform(self.fc1.bias,a=self.snn_config["bias_init"][0],b=self.snn_config["bias_init"][1])
        # enforce a one-to-one mapping between the spiking layer and the filtering layer
        self.fc2.weight = nn.Parameter(torch.eye(self.num_outputs) * self.snn_config["w_fixed_filt"])
        self.fc2.bias = nn.Parameter(torch.zeros(self.num_outputs))
        self.fc2.requires_grad_(False)

    def _init_network_parameters(self):
        # Input to spiking neuron connected via alpha synapse with parameters tau_syn
        # Spiking neuron has parameters tau_mem and threshold
        self.set_electrode_specific_parameters = self.snn_config["set_electrode_specific_parameters"]
        self.enc_tau_mem = self.snn_config["enc_tau_mem"]
        
        self.enc_beta = self._set_parameter_per_electrode(self.enc_tau_mem, 'tau_mem') #map_tau_to_decay(self.enc_tau_mem, self.dt)
        self.enc_learn_threshold = self.snn_config["learn_enc_threshold"]
            
        self.tau_syn = self.snn_config["tau_syn"]
        self.tau_mem = self.snn_config["tau_mem"]
        self.spk_threshold = self.snn_config["spk_threshold"]
        self.enc_threshold= self._set_parameter_per_electrode(self.snn_config["enc_spk_threshold"], 'threshold') #self.snn_config.enc_spk_threshold

        self.alpha = map_tau_to_decay(self.tau_syn, self.dt)
        self.beta = map_tau_to_decay(self.tau_mem, self.dt)

        self.learn_alpha = self.snn_config["learn_tau_syn"]
        self.learn_beta = self.snn_config["learn_tau_mem"]
        self.learn_threshold = self.snn_config["learn_threshold"]

        # In case parameters are learned, initialize them from a distribution
        if self.enc_learn_threshold:
            enc_neurons_thresholds = torch.rand(self.num_inputs)
            nn.init.normal_(enc_neurons_thresholds, mean=self.enc_threshold,
                            std=self.param_std * self.enc_threshold)
        else:
            enc_neurons_thresholds = self.enc_threshold
        if self.learn_beta:
            neurons_betas = torch.rand(self.num_outputs)
            nn.init.normal_(neurons_betas, mean=self.beta,
                            std=self.param_std * self.beta)
            print(f"Init beta:{neurons_betas}\n")
        else:
            neurons_betas = self.beta

        if self.learn_alpha:
            synapses_alphas = torch.rand(self.num_outputs)
            nn.init.normal_(synapses_alphas, mean=self.alpha,
                            std=self.param_std * self.alpha)
        else:
            synapses_alphas = self.alpha

        if self.learn_threshold:
            neurons_thresholds = torch.rand(self.num_outputs)
            nn.init.normal_(neurons_thresholds, mean=self.spk_threshold,
                            std=self.param_std * self.spk_threshold)
        else:
            neurons_thresholds = self.spk_threshold

        self.neurons_betas = neurons_betas
        self.synapses_alphas = synapses_alphas
        self.neurons_thresholds = neurons_thresholds
        self.enc_neurons_threshold = enc_neurons_thresholds

        self.first_filter_beta = map_tau_to_decay(self.snn_config["first_filter_tau"], self.dt)
        self.second_filter_beta = map_tau_to_decay(self.snn_config["second_filter_tau"], self.dt)


    def forward(self, emg_in, state_dict):
        """Forward pass for each time step"""
        empty_record_dict = self.create_record_dict_for_vars()
        if state_dict is None:
            # Initalize membrane potential as empty tensor
            enc_neu_mem = self.encoding_layer.init_leaky()
            spk_syn_cur, spk_neu_mem = self.lif1.init_synaptic()
            filt_neu_mem =  self.first_filter.init_leaky()
            filt2_neu_mem = self.second_filter.init_leaky()

        else: # if not initializing as empty rely on last timestep
            enc_neu_mem = state_dict[ENC_NEU_MEM]
            enc_neu_cur = state_dict[ENC_NEU_CUR]
            enc_spk_out = state_dict[ENC_SPK_OUT]

            spk_syn_cur = state_dict[SPK_SYN_CUR]
            spk_neu_cur = state_dict[SPK_NEU_CUR]
            spk_neu_mem = state_dict[SPK_NEU_MEM]
            spk_out = state_dict[SPK_OUT]

            filt_neu_mem = state_dict[FILT_NEU_MEM]
            filt_neu_cur = state_dict[FILT_NEU_CUR]
            filt_spk_out = state_dict[FILT_SPK_OUT]

            filt2_neu_mem = state_dict[FILT2_NEU_MEM]

        for step in range(self.timesteps):
            if state_dict is None or step > 0:
                enc_neu_cur = self.enc_fc(emg_in[:, step])
                enc_spk_out, enc_neu_mem = self.encoding_layer(enc_neu_cur, enc_neu_mem)

                spk_neu_cur = self.fc1(enc_spk_out)
                spk_out, spk_syn_cur, spk_neu_mem = self.lif1(spk_neu_cur, spk_syn_cur, spk_neu_mem)
                filt_neu_cur = self.fc2(spk_out)
                filt_spk_out, filt_neu_mem = self.first_filter(filt_neu_cur, filt_neu_mem)
                _, filt2_neu_mem = self.second_filter(filt_neu_mem, filt2_neu_mem)

            # record the variables for layer 1
            empty_record_dict[EMG_IN].append(emg_in[:, step])
            empty_record_dict[ENC_NEU_CUR].append(enc_neu_cur)
            empty_record_dict[ENC_NEU_MEM].append(enc_neu_mem)
            empty_record_dict[ENC_SPK_OUT].append(enc_spk_out)
        
            empty_record_dict[SPK_SYN_CUR].append(spk_syn_cur)
            empty_record_dict[SPK_NEU_CUR].append(spk_neu_cur)
            empty_record_dict[SPK_NEU_MEM].append(spk_neu_mem)
            empty_record_dict[SPK_OUT].append(spk_out)
            empty_record_dict[FILT_NEU_CUR].append(filt_neu_cur)
            empty_record_dict[FILT_NEU_MEM].append(filt_neu_mem)
            empty_record_dict[FILT_SPK_OUT].append(filt_spk_out)
            empty_record_dict[FILT2_NEU_MEM].append(filt2_neu_mem)

        # stack the recorded variables
        for var in empty_record_dict:
            empty_record_dict[var] = torch.stack(empty_record_dict[var], dim=1)
        filled_record_dict = empty_record_dict
        return filled_record_dict

class EncodingShallowLeaky(SnnTopology):
    """
    A shallow network with lekay integrator neurons followed by a filtering neuron
    """
    def __init__(self, timesteps, num_inputs, num_outputs, snn_config, use_wandb, dt):
        SnnTopology.__init__(self, timesteps, num_inputs, num_outputs, snn_config, use_wandb, dt)

        self._init_network_parameters()
        self._create_network_layers()
        self.prediction_var = FILT_NEU_MEM   if self.snn_config["first_filter_tau"] >0 else LEAK_NEU_MEM

        self.input_var = EMG_IN
        self.leaky_integrators_mem = [FILT_NEU_MEM]
        self.leaky_integrators_spk= [SPK_OUT, FILT_SPK_OUT] if self.snn_config["first_filter_tau"] >0 else [SPK_OUT]
        self.filter_layers = [self.first_filter] if self.snn_config["first_filter_tau"] >0 else []

    def _init_network_parameters(self):
        # Input to leaky neuron connected via alpha synapse with parameters tau_syn
        # Leaky neuron has parameters tau_mem
        self.set_electrode_specific_parameters = self.snn_config["set_electrode_specific_parameters"]
        self.enc_tau_mem = self.snn_config["enc_tau_mem"]
        
        self.enc_beta = self._set_parameter_per_electrode(self.enc_tau_mem, 'tau_mem') #map_tau_to_decay(self.enc_tau_mem, self.dt)
        self.enc_learn_threshold = self.snn_config["learn_enc_threshold"]
        self.tau_syn = self.snn_config["tau_syn"]
        self.tau_mem = self.snn_config["tau_mem"]
        self.w_recurrent = self.snn_config["w_recurrent"]
        self.alpha = map_tau_to_decay(self.tau_syn, self.dt)
        self.beta = map_tau_to_decay(self.tau_mem, self.dt)
        self.enc_threshold= self._set_parameter_per_electrode(self.snn_config["enc_spk_threshold"], 'threshold') #self.snn_config.enc_spk_threshold
        self.learn_alpha = self.snn_config["learn_tau_syn"]
        self.learn_beta = self.snn_config["learn_tau_mem"]
        self.post_process_filt = self.snn_config["post_process_filt"]
        self.post_process_filt_cutoff = self.snn_config["post_filt_cutoff"]

        enc_neurons_thresholds = self.enc_threshold

        if self.learn_beta:
            neurons_betas = torch.rand(self.num_outputs)
            nn.init.normal_(neurons_betas, mean=self.beta,
                            std=self.param_std * self.beta)
            print(f"Init beta:{neurons_betas}\n")
        else:
            neurons_betas = self.beta

        if self.learn_alpha:
            synapses_alphas = torch.rand(self.num_outputs)
            nn.init.normal_(synapses_alphas, mean=self.alpha,
                            std=self.param_std * self.alpha)
        else:
            synapses_alphas = self.alpha

        self.neurons_betas = neurons_betas
        self.synapses_alphas = synapses_alphas
        self.enc_neurons_threshold = enc_neurons_thresholds
        if self.snn_config["first_filter_tau"] > 0:
            self.first_filter_beta = map_tau_to_decay(self.snn_config["first_filter_tau"], self.dt)

    def _create_network_layers(self):
        self.enc_fc = torch.nn.Linear(in_features=self.num_inputs,
                                        out_features=self.num_inputs,
                                        bias=False)
        # set weights of encoding layer to identity
        self.enc_fc.weight = nn.Parameter(torch.eye(self.num_inputs) * self.snn_config["w_fixed_enc"])
        self.enc_fc.requires_grad_(False)

        if self.snn_config["add_recurrent"]:
            self.encoding_layer = snn.RLeaky(beta=self.enc_beta,
                                        threshold=self.enc_neurons_threshold,
                                        learn_beta=False,
                                        learn_threshold=self.enc_learn_threshold,
                                        reset_mechanism="subtract",
                                        all_to_all=False,
                                        V=self.w_recurrent,
                                        learn_recurrent=False)
        elif self.snn_config["use_aleaky"]:
            self.encoding_layer = snn.ALeaky(beta=self.enc_beta,
                                        threshold=self.enc_neurons_threshold,
                                        learn_beta=False,
                                        learn_threshold=self.enc_learn_threshold,
                                        reset_mechanism="zero",
                                        tau_adapt = self.snn_config["threshold_tau_adapt"],
                                        threshold_weight = self.snn_config["threshold_scale_adapt"],
                                        dt = self.dt)

        else:
            self.encoding_layer = snn.Leaky(beta=self.enc_beta,
                                        threshold=self.enc_neurons_threshold,
                                        learn_beta=False,
                                        learn_threshold=self.enc_learn_threshold,
                                        reset_mechanism="zero")
            
        self.fc1 = torch.nn.Linear(in_features=self.num_inputs,
                                   out_features=self.num_outputs,
                                  )

        self.lif1 = snn.Synaptic(beta=self.neurons_betas,
                                alpha=self.synapses_alphas,
                                spike_grad=self.spike_grad,
                                reset_mechanism="zero",
                                threshold=self.leaky_integrator_threshold,
                                learn_threshold=False,
                                learn_beta=self.learn_beta,
                                learn_alpha=self.learn_alpha)

        if self.snn_config["first_filter_tau"] > 0:
            self.first_filter = snn.Leaky(beta=self.first_filter_beta,
                                    threshold=self.leaky_integrator_threshold,
                                    learn_beta=False)

        if self.snn_config["w_init_dist"] == 'uniform':
            nn.init.uniform_(self.fc1.weight,a=self.snn_config["w_init"][0],b=self.snn_config["w_init"][1])
        if self.snn_config["w_init_dist"] == 'normal':
            nn.init.normal_(self.fc1.weight, mean=self.snn_config["w_init_mean"],
                            std=self.snn_config["w_init_std"])
        # keep bias fixed
        # self.fc1.bias = nn.Parameter(torch.zeros(self.num_outputs))
        # self.fc1.bias.requires_grad = False
        nn.init.uniform(self.fc1.bias,a=self.snn_config["bias_init"][0],b=self.snn_config["bias_init"][1])

    def forward(self, emg_in, state_dict=None):
        empty_record_dict = self.create_record_dict_for_vars()
        if state_dict is None:
            # Initalize membrane potential as empty tensor
            if self.snn_config["add_recurrent"]:
                enc_spk_out, enc_neu_mem = self.encoding_layer.init_rleaky()
            elif self.snn_config["use_aleaky"]:
                enc_neu_mem, threshold_adapt = self.encoding_layer.init_aleaky()
            else:
                enc_neu_mem = self.encoding_layer.init_leaky()
                
            leaky_syn_cur, leaky_neu_mem = self.lif1.init_synaptic()
            if self.snn_config["first_filter_tau"] > 0:
                filt_neu_mem =  self.first_filter.init_leaky()

        else: # if not initializing as empty rely on last timestep values, restroring the state
            enc_neu_mem = state_dict[ENC_NEU_MEM]
            enc_neu_cur = state_dict[ENC_NEU_CUR]
            enc_spk_out = state_dict[ENC_SPK_OUT]

            leaky_syn_cur = state_dict[LEAK_SYN_CUR]
            leaky_neu_cur = state_dict[LEAK_NEU_CUR]
            leaky_neu_mem = state_dict[LEAK_NEU_MEM]
            spk_out = state_dict[SPK_OUT]
            if self.snn_config["first_filter_tau"] > 0:
                filt_neu_mem = state_dict[FILT_NEU_MEM]
                filt_spk_out = state_dict[FILT_SPK_OUT]
            if self.snn_config["use_aleaky"]:
                threshold_adapt = state_dict[ENC_THRESHOLD_ADAPT]
                threshold = state_dict[ENC_ALIF_THRESHOLD]
        for step in range(self.timesteps):
            if state_dict is None or step > 0:
                enc_neu_cur = self.enc_fc(emg_in[:, step])
                if self.snn_config["add_recurrent"]:
                    enc_spk_out, enc_neu_mem = self.encoding_layer(enc_neu_cur,enc_spk_out, enc_neu_mem)
                elif self.snn_config["use_aleaky"]:
                    enc_spk_out, enc_neu_mem, threshold_adapt, threshold = self.encoding_layer(enc_neu_cur, enc_neu_mem, threshold_adapt)

                else:
                    enc_spk_out, enc_neu_mem = self.encoding_layer(enc_neu_cur, enc_neu_mem)

                leaky_neu_cur = self.fc1(enc_spk_out)
                spk_out, leaky_syn_cur, leaky_neu_mem = self.lif1(leaky_neu_cur, leaky_syn_cur, leaky_neu_mem)

                # clip membrane potential to 0
                leaky_neu_mem = torch.clamp(leaky_neu_mem, min=0)
                if self.snn_config["first_filter_tau"] > 0:
                    filt_spk_out, filt_neu_mem = self.first_filter(leaky_neu_mem, filt_neu_mem)
    

            # record the variables for layer 1
            empty_record_dict[EMG_IN].append(emg_in[:, step])
            empty_record_dict[ENC_NEU_CUR].append(enc_neu_cur)
            empty_record_dict[ENC_NEU_MEM].append(enc_neu_mem)
            empty_record_dict[ENC_SPK_OUT].append(enc_spk_out)

            empty_record_dict[LEAK_SYN_CUR].append(leaky_syn_cur)
            empty_record_dict[LEAK_NEU_CUR].append(leaky_neu_cur)
            empty_record_dict[LEAK_NEU_MEM].append(leaky_neu_mem)
            empty_record_dict[SPK_OUT].append(spk_out)
            if self.snn_config["first_filter_tau"] > 0:
                empty_record_dict[FILT_NEU_MEM].append(filt_neu_mem)
                empty_record_dict[FILT_SPK_OUT].append(filt_spk_out)
            if self.snn_config["use_aleaky"]:
                empty_record_dict[ENC_THRESHOLD_ADAPT].append(threshold_adapt)
                empty_record_dict[ENC_ALIF_THRESHOLD].append(threshold)

        # stack the recorded variables
        for var in empty_record_dict:
            empty_record_dict[var] = torch.stack(empty_record_dict[var], dim=1)
        filled_record_dict = empty_record_dict
        return filled_record_dict



# --- SNN Regression ---

class SnnReg():
    """
    Single-layer spiking neural network in snntorch to regress finger forces
    """
    def __init__(self, timesteps:int, num_inputs:int, num_outputs:int, snn_config:SNNConfig, train_dataset, test_dataset,
                 ):

        self.snn_config = snn_config
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        torch_seed = snn_config.training["torch_seed"]
        use_wandb = snn_config.logging["wandb"]["use_wandb"]
        dt = snn_config.training["dt"]
       
        self._set_random_seeds(torch_seed)
    
        passed_net_config = snn_config.decoder_type
        
        if passed_net_config["topology"] == SHALLOW_SPIKING_DOUBLE_FILTER:
            self.network = ShallowSpikingDoubleFilter(timesteps,num_inputs,num_outputs, passed_net_config,
                                                      use_wandb=use_wandb, dt=dt)
        if passed_net_config["topology"] == SHALLOW_LEAKY:
            self.network = ShallowLeaky(timesteps,num_inputs,num_outputs, passed_net_config,
                                       use_wandb=use_wandb, dt=dt)
        if passed_net_config["topology"] == ENC_SHALLOW_SPIKING:
            self.network = EncodingShallowSpikingDoubleFilter(timesteps,num_inputs,num_outputs, passed_net_config,
                                                              use_wandb=use_wandb, dt=dt)
        if passed_net_config["topology"] == ENC_SHALLOW_LEAKY:
            self.network = EncodingShallowLeaky(timesteps,num_inputs,num_outputs, passed_net_config, 
                                                use_wandb=use_wandb, dt=dt)
    
    def _set_random_seeds(self,seed):
        """
        Sets seed for random shuffle of the fingers
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        print(f"\nnetwork initialization seed:{torch.random.initial_seed()}\n")
        

    def _check_leaky_integrators_not_firing(self, record_dict:dict):
        """
        Checks that the filtering layer is not emitting any spikes.
        """
        for var in self.network.leaky_integrators_spk:
            if torch.sum(record_dict[var]) > 0:
                print(f"Careful!! Spikes in the filtering layer: {torch.sum(record_dict[var])}...\n")

    def _save_network_state_dict(self, previous_record_dict:dict):
        """
        Saves the state of the network. 
        Variables saved are the membrane potential, synaptic current, synaptic conductance, 
        and the input spikes.
        """
        split_type = self.snn_config.task["split_type"]
        overlap_perc = self.snn_config.task["overlap_perc"]
        if split_type == 'without_overlap':
            t_id = -1  # saved time step should be the last of the previous sample
        else:
            t_id = int(overlap_perc * self.network.timesteps)
            # in case the consecutive samples are non-overlapping, use the last time step
        state_dict = {}
        for var in self.network.state_vars:
            state_dict[var] = previous_record_dict[var].detach()[:, t_id, :]
        return state_dict

    def _initialize_state_dict(self):
        """
        Initializes the state dictionary of the network.
        """
        state_dict = {}
        for var in self.network.state_vars:
            state_dict[var] = torch.zeros((1, self.num_outputs)) if 'enc' not in var else torch.zeros((1, self.num_inputs))
        return state_dict

    def _train(self):
        """
        Trains the network using Adam optimizer and logs losses to wandb.
        """
        lr = self.snn_config.decoder_type["lr"]
        num_iter = self.snn_config.training["num_iter"]
        batch_size = self.snn_config.training["train_batch_size"]
        use_wandb = self.snn_config.logging["wandb"]["use_wandb"]
        log_ts_freq = self.snn_config.logging["log_ts_freq"]
        snn_network = self.network
        optimizer = torch.optim.Adam(params=snn_network.parameters(), lr=lr)
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)

        record_dict_cache = []   # records variables from all samples
        targets_cache = []
        tr_loss_per_epoch = []
        ts_loss_every_nepochs = []
        tr_batch_loss_hist = []  # record loss for all iterations and each batch


        # test print torch summary
        # summary(snn_network, (1000, 120), None)


        with tqdm.trange(num_iter) as pbar:
            for i in pbar:
                train_batch = iter(train_loader)
                ep_running_loss = 0   # the current epoch loss averaged over the batches and samples
                mse_epoch = 0
                r2_epoch = 0
                state_dict = None
                for sample_id, (data, label, _) in enumerate(train_batch):
                    spk_or_emg_in = data.to(DEVICE)
                    targets = label.to(DEVICE)
                    if sample_id > 0 and batch_size==1: # save the state of vmem to use with the next sample
                        state_dict = self._save_network_state_dict(record_dict)
                    # print(f"Sample {sample_id} of {len(train_loader)}")
                    record_dict = snn_network.forward(spk_or_emg_in, state_dict)
                    self._check_leaky_integrators_not_firing(record_dict)

                    if i == num_iter - 1:
                        record_dict_cache.append(record_dict)
                        targets_cache.append(targets)

                    prediction_var = snn_network.prediction_var
                    sample_loss_value = snn_network.loss_function(record_dict[prediction_var], targets)

                    predicted_value = record_dict[prediction_var].cpu().detach()
                    predicted_value = predicted_value.numpy().reshape(-1, predicted_value.shape[-1])
                    targets_reshaped = targets.cpu().detach().numpy().reshape(-1, targets.shape[-1])

                    r2_epoch += r2_score(predicted_value, targets_reshaped)
                    mse_epoch += mean_squared_error(predicted_value, targets_reshaped)

                    optimizer.zero_grad()
                    sample_loss_value.backward()    # calculate gradients
                    optimizer.step()

                    # store losses: loss_hist is for all iterations,
                    # hist_loss_epoch is for the current epoch
                    tr_batch_loss_hist.append(sample_loss_value.item())
                    ep_running_loss += sample_loss_value.item()

                    n_samples = (sample_id+1)
                    mean_batch_loss = ep_running_loss / n_samples
                    pbar.set_postfix(loss="%.3e" % mean_batch_loss)

                if use_wandb:
                    wandb.log({'mse_tr_per_epoch': mse_epoch / n_samples,'epoch': i})
                if i % log_ts_freq == 0:
                    test_eloss,_, _, _= self.evaluate(self.test_dataset)
                    ts_loss_every_nepochs.append(test_eloss)
                tr_loss_per_epoch.append(mean_batch_loss)

        print("Training complete!")
        return tr_loss_per_epoch, tr_batch_loss_hist, ts_loss_every_nepochs, record_dict_cache, targets_cache

    def post_process_predictions(self,y_pred:torch.Tensor):
        """
        Post-processes the predictions by applying a low-pass filter.
        """
        post_filt_cutoff = self.snn_config.decoder_type["post_filt_cutoff"]
        post_filt_order = self.snn_config.decoder_type["post_filt_order"]
        y_pred_post = torch.zeros_like(y_pred)
        for batch in range(y_pred.shape[0]):
            for output in range(y_pred.shape[-1]):
                filtered_y_pred = apply_butter_lowpass(y_pred[batch,:, output], cutoff=post_filt_cutoff,
                                                                fs=1/self.network.dt, order=post_filt_order)
                y_pred_post[batch,:, output] = torch.tensor(filtered_y_pred.copy(), dtype=DTYPE, device=DEVICE)

        return y_pred_post



    def evaluate(self, eval_dataset, enable_profiling: bool = False,
                 num_profile_samples: int = 10):
        """
        Gets the predictions for the test set.
        Concatenates the predictions made over the time segments of the test repetition.
        The assumption that there is a single test repetition, and it is segmented into smaller windows.

        Args:
            eval_dataset: Dataset to evaluate on
            enable_profiling: If True, profile CPU time and memory for first N samples
            num_profile_samples: Number of samples to profile (default: 10)

        Returns:
            If enable_profiling=False:
                (average_loss, record_dict_cache, predicted_values_cache, true_values_cache)
            If enable_profiling=True:
                (average_loss, record_dict_cache, predicted_values_cache, true_values_cache, profiling_results)
        """
        batch_size = self.snn_config.training["test_batch_size"]
        percent_omission = self.snn_config.task["percent_omission"]
        percent_addition = self.snn_config.task["percent_addition"]
        percent_misattribution = self.snn_config.task["percent_misattribution"]
        jitter_std = self.snn_config.task["jitter_std"]
        noise_mode = self.snn_config.task["noise_mode"]
        inf_rep_binwidth = self.snn_config.training["inf_rep_binwidth"]
        post_process_filt = self.snn_config.decoder_type["post_process_filt"]
        if noise_mode=='omission':
            percent_noise = percent_omission
        elif noise_mode=='addition':
            percent_noise = percent_addition
        elif noise_mode=='location':
            percent_noise = jitter_std
        elif noise_mode=='misattribution':
            percent_noise = percent_misattribution
        else:
            percent_noise = 0
        snn_network = self.network
        test_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        # Profiling storage
        profiling_samples = []

        with torch.no_grad():
            snn_network.eval()
            loss_per_batch = []
            record_dict_cache = []
            predicted_values_cache = []
            true_values_cache = []
            if batch_size ==1:
                state_dict = self._initialize_state_dict()
            else:
                state_dict = None
            no_input_counter = 0  # counter for the number of samples with no input spikes
            no_input_id = None
            for i, (data, label, noisy_data) in enumerate(test_loader):
                if noise_mode is None or percent_noise == 0 :
                    spk_or_emg_in  = data.to(DEVICE)
                else:
                    spk_or_emg_in = noisy_data.to(DEVICE)
                target_values = label.to(DEVICE)
                if torch.sum(spk_or_emg_in) == 0 :
                    if no_input_id is None or no_input_id==i-1:  # check on first all zeros sample
                        no_input_counter += 1
                        no_input_id = i
                else:
                    no_input_counter = 0
                    no_input_id = None

                # forward pass
                if i > 0 and batch_size==1:  # save the last time step
                    state_dict = self._save_network_state_dict(record_dict)

                # if it stays silent for more than 0.5 seconds, reset the state dict
                if no_input_counter > int(40/inf_rep_binwidth):
                    print("No input spikes for more than 10 samples..Resetting the state dict and counter")
                    state_dict = None
                    no_input_counter = 0
                    no_input_id = None

                # Profile this sample if enabled and we haven't reached the limit
                if enable_profiling and len(profiling_samples) < num_profile_samples:
                    profile_result = self.profile_snn_inference(spk_or_emg_in, state_dict)
                    profiling_samples.append(profile_result)
                    record_dict = profile_result['record_dict']
                else:
                    record_dict = snn_network.forward(spk_or_emg_in, state_dict)

                prediction_var = snn_network.prediction_var
                predicted_values = record_dict[prediction_var].cpu().detach()

                if post_process_filt:
                    predicted_values = self.post_process_predictions(predicted_values)
                loss_value = snn_network.loss_function(predicted_values, target_values)

                loss_per_batch.append(loss_value.item())
                predicted_values_cache.append(predicted_values)
                true_values_cache.append(target_values)
                record_dict_cache.append(record_dict)
            average_loss = np.sum(loss_per_batch) / len(loss_per_batch)

            # Aggregate and return profiling results
            if enable_profiling and profiling_samples:
                profiling_results = aggregate_snn_profiling_results(
                    profiling_samples, effective_ops=None, model=snn_network, verbose=True
                )
                return average_loss, record_dict_cache, predicted_values_cache, true_values_cache, profiling_results

            return average_loss, record_dict_cache, predicted_values_cache, true_values_cache

    def train_and_evaluate_on_training_set(self):
        """
        Trains the network on the training dataset. The train function also returns
        some test losses these are the loss on the test every n epochs
        During offline mode, the batch size is greater than 1.
        """
            # return tr_loss_per_epoch, tr_batch_loss_hist, ts_loss_every_nepochs, record_dict_cache, targets_cache

        tr_loss_per_epoch, tr_batch_loss_hist, ts_loss_every_nepochs, _, _ = self._train()
        e_loss_tr, rec_tr, y_pred_tr, y_true_tr = self.evaluate(self.train_dataset)
        print(f"Train MSE loss: {e_loss_tr:.4f}")
        rec_tr, _ = conc_var_tracker(tracker=rec_tr, is_target_tracker=False)
        y_pred_tr = target_tracker_to_np(y_pred_tr)
        y_true_tr = target_tracker_to_np(y_true_tr)
        return y_pred_tr,y_true_tr,tr_loss_per_epoch,tr_batch_loss_hist,ts_loss_every_nepochs,rec_tr


    def get_initial_network_params(self):
        """
        Gets the initial values of the network parameters to be used for plotting."""
        snn_network = self.network
        beta_init = snn_network.lif1.beta.detach().clone().cpu().numpy()
        alpha_init = snn_network.lif1.alpha.detach().clone().cpu().numpy()
        initial_weights = snn_network.fc1.weight.detach().clone().cpu().numpy()
        threshold_init = snn_network.lif1.threshold.detach().clone().cpu().numpy()
        return beta_init,alpha_init,threshold_init,initial_weights
    
    def evaluate_and_log_on_test_set(self, data_config, num_segments_per_rep:int,
                                     repetition_dur:float,
                                     fold_i:int, mode:str, binwidth,
                                     enable_profiling: bool = False,
                                     num_profile_samples: int = 10):
        """
        Evaluates the model after training is complete using the test set.
        This test set can have a batch size =1 to mimic online inference or
        a larger batch size to evaluate the model.

        Args:
            data_config: Data configuration
            num_segments_per_rep: Number of segments per repetition
            repetition_dur: Duration of repetition
            fold_i: Fold index
            mode: 'training' or 'inference'
            binwidth: Bin width for reconstruction
            enable_profiling: If True, profile CPU time and memory
            num_profile_samples: Number of samples to profile

        Returns:
            If enable_profiling=False:
                (y_pred_ts, y_true_ts, rec_ts)
            If enable_profiling=True:
                (y_pred_ts, y_true_ts, rec_ts, profiling_results)
        """
        profiling_results = None
        if enable_profiling:
            _, rec_ts, y_pred_ts, y_true_ts, profiling_results = self.evaluate(
                self.test_dataset, enable_profiling=True, num_profile_samples=num_profile_samples
            )
        else:
            _, rec_ts, y_pred_ts, y_true_ts = self.evaluate(self.test_dataset)
        rec_ts, _ = conc_var_tracker(tracker=rec_ts, is_target_tracker=False)

        #Log the output spikes: here I assume that given the overlap percentage,ol, there is ol% of the spikes from the previous sample
        overlap_perc = self.snn_config.task["overlap_perc"]
        output_spike_count_df = sum_output_spikes_per_active_finger(overlap_perc, data_config,
                                                                    num_segments_per_rep,
                                                                    repetition_dur,
                                                                    fold_i, binwidth, self.network,
                                                                    mode, rec_ts)
        print(f"Output spike count df\n{tabulate(output_spike_count_df, headers='keys', tablefmt='psql')}")
        use_wandb = self.snn_config.logging["wandb"]["use_wandb"]
        if use_wandb:
            log_spike_count_df(output_spike_count_df, fold_i, binwidth)

        # Compute metrics and create y_df
        y_pred_ts = target_tracker_to_np(y_pred_ts)
        y_true_ts = target_tracker_to_np(y_true_ts)

        if enable_profiling and profiling_results is not None:
            return y_pred_ts, y_true_ts, rec_ts, profiling_results
        return y_pred_ts, y_true_ts, rec_ts

    def calculate_effective_ops(self, record_dict: dict, verbose: bool = False) -> Dict[str, float]:
        """
        Calculate effective operations for SNN based on actual spike activity.
        This accounts for the sparse nature of spiking neural networks.

        Operation Types:
        ----------------
        1. **SynOps (Synaptic Operations)**: Spike-dependent operations at synapses
           - For encoding layer: Treated as MAC (weight × spike_value) since encoding
             may produce non-binary spike representations
           - For non-encoding layer: Treated as AC (accumulate) only since spikes are
             binary (just add the weight when spike=1, no multiplication needed)

        2. **MAC Operations**: Multiply-accumulate for membrane/synapse dynamics
           - Membrane decay: beta × mem
           - Synaptic decay: alpha × syn_cur
           - These run every timestep (dense operations)

        3. **Add Operations**: Simple additions
           - Adding synaptic current to membrane potential
           - Dense operations that run every timestep

        Total Effective FLOPs = SynOps_FLOPs + MAC_ops × 2 + Add_ops
        Where:
           - SynOps_FLOPs = SynOps × 2 (for encoding) or SynOps × 1 (for non-encoding)
           - MAC × 2 because each MAC = 1 multiply + 1 add

        Args:
            record_dict: Dictionary containing recorded network variables from forward pass
            verbose: If True, print detailed breakdown of operations

        Returns:
            Dictionary containing various operation counts and metrics
        """
        ops = {
            'mac_ops': 0,           # Multiply-accumulate operations (dense, membrane/synapse dynamics)
            'add_ops': 0,           # Addition operations (dense)
            'comparison_ops': 0,    # Threshold comparisons
            'synops_mac': 0,        # SynOps counted as MAC (encoding layer) - fc1
            'synops_ac': 0,         # SynOps counted as AC (non-encoding layer) - fc1
            'synops_ac_fc2': 0,     # SynOps for fc2 layer (output spikes -> filter neurons)
            'total_input_spikes': 0,
            'total_output_spikes': 0,
            'total_encoding_spikes': 0,
        }

        # Get batch size and timesteps from recorded data
        input_var = self.network.input_var
        input_data = record_dict[input_var]
        batch_size = input_data.shape[0]
        timesteps = input_data.shape[1]

        topology = self.network.topology

        # Count spikes based on topology
        if topology in [ENC_SHALLOW_SPIKING, ENC_SHALLOW_LEAKY]:
            # For encoding topologies, count encoding layer spikes
            enc_spikes = record_dict[ENC_SPK_OUT]
            ops['total_encoding_spikes'] = torch.sum(enc_spikes).item()

            # Encoding layer operations (EMG to spikes) - DENSE operations
            # Linear layer: enc_fc (identity matrix, but still processes each input)
            ops['mac_ops'] += batch_size * timesteps * self.num_inputs

            # Leaky integrator update for encoding layer
            # beta * mem + current (1 MAC + 1 add per neuron per timestep)
            ops['mac_ops'] += batch_size * timesteps * self.num_inputs
            ops['add_ops'] += batch_size * timesteps * self.num_inputs

            # Threshold comparison for encoding layer
            ops['comparison_ops'] += batch_size * timesteps * self.num_inputs

            # Decoding layer: fc1 operations (only for active spikes due to sparsity)
            # For encoding layer output, spikes may have continuous values, so treat as MAC
            ops['synops_mac'] = ops['total_encoding_spikes'] * self.num_outputs

        else:
            # For non-encoding topologies, count input spikes
            input_spikes = record_dict[SPK_IN]
            ops['total_input_spikes'] = torch.sum(input_spikes).item()

            # Linear layer operations (only for active input spikes)
            # Since spikes are binary (0 or 1), weight × spike = weight when spike=1
            # So this is just an accumulate operation (AC), not multiply-accumulate (MAC)
            ops['synops_ac'] = ops['total_input_spikes'] * self.num_outputs

        # Output layer spikes
        output_spikes = record_dict[SPK_OUT]
        ops['total_output_spikes'] = torch.sum(output_spikes).item()

        # fc2 synaptic operations: output spikes -> filter neurons
        # In SHALLOW_SPIKING_DOUBLE_FILTER, fc2 processes binary output spikes (spk_out)
        # to produce input current for the filter neurons
        # Since spk_out contains binary spikes, this is an AC operation (just accumulate weight)
        if topology == SHALLOW_SPIKING_DOUBLE_FILTER:
            # fc2: output_spikes × num_outputs (filter neurons have same size as output)
            ops['synops_ac_fc2'] = ops['total_output_spikes'] * self.num_outputs
        elif topology == ENC_SHALLOW_SPIKING:
            # For encoding topology, output spikes also feed fc2 to filter neurons
            ops['synops_ac_fc2'] = ops['total_output_spikes'] * self.num_outputs

        # Synaptic and membrane dynamics for output neurons - DENSE operations
        if topology in [SHALLOW_SPIKING_DOUBLE_FILTER, ENC_SHALLOW_SPIKING]:
            # Synaptic current update: alpha * syn_cur (1 MAC per neuron per timestep)
            ops['mac_ops'] += batch_size * timesteps * self.num_outputs
            # Membrane potential update: beta * mem (1 MAC per neuron per timestep)
            ops['mac_ops'] += batch_size * timesteps * self.num_outputs
            # Adding synaptic current to membrane (2 adds per neuron per timestep)
            ops['add_ops'] += batch_size * timesteps * self.num_outputs * 2

        elif topology in [SHALLOW_LEAKY, ENC_SHALLOW_LEAKY]:
            # Leaky neuron: alpha * syn_cur + beta * mem (2 MACs + 2 adds)
            ops['mac_ops'] += batch_size * timesteps * self.num_outputs * 2
            ops['add_ops'] += batch_size * timesteps * self.num_outputs * 2

        # Threshold comparison for output neurons
        ops['comparison_ops'] += batch_size * timesteps * self.num_outputs

        # Filter layers (always dense, not spike-based)
        if hasattr(self.network, 'filter_layers'):
            for _ in self.network.filter_layers:
                # Each filter: beta * mem + input (1 MAC + 1 add per output per timestep)
                ops['mac_ops'] += batch_size * timesteps * self.num_outputs
                ops['add_ops'] += batch_size * timesteps * self.num_outputs

        # Calculate total synaptic operations (SynOps)
        # This is the spike-dependent operation count reported in SNN literature
        # Includes fc1 SynOps (input/encoding -> output) + fc2 SynOps (output -> filter)
        ops['synaptic_ops'] = ops['synops_mac'] + ops['synops_ac'] + ops['synops_ac_fc2']

        # Calculate total FLOPs with proper accounting:
        # - MAC operations = 2 FLOPs each (1 multiply + 1 accumulate)
        # - Add operations = 1 FLOP each
        # - SynOps as MAC (encoding) = 2 FLOPs each
        # - SynOps as AC (non-encoding) = 1 FLOP each (just accumulate)
        # - SynOps fc2 as AC = 1 FLOP each (binary output spikes)
        ops['synops_flops'] = ops['synops_mac'] * 2 + ops['synops_ac'] * 1 + ops['synops_ac_fc2'] * 1
        ops['membrane_synapse_flops'] = ops['mac_ops'] * 2 + ops['add_ops']
        ops['total_flops'] = ops['synops_flops'] + ops['membrane_synapse_flops']
        ops['total_ops'] = ops['total_flops'] + ops['comparison_ops']

        # Calculate sparsity metrics
        total_possible_input_spikes = batch_size * timesteps * self.num_inputs
        total_possible_output_spikes = batch_size * timesteps * self.num_outputs

        if topology in [ENC_SHALLOW_SPIKING, ENC_SHALLOW_LEAKY]:
            ops['encoding_spike_rate'] = ops['total_encoding_spikes'] / total_possible_input_spikes
        else:
            ops['input_spike_rate'] = ops['total_input_spikes'] / total_possible_input_spikes

        ops['output_spike_rate'] = ops['total_output_spikes'] / total_possible_output_spikes

        # Calculate what percentage of total FLOPs are synaptic operations
        ops['synaptic_percentage'] = (ops['synops_flops']) / ops['total_flops'] * 100 if ops['total_flops'] > 0 else 0

        # Store dimensions for reference
        ops['batch_size'] = batch_size
        ops['timesteps'] = timesteps
        ops['num_inputs'] = self.num_inputs
        ops['num_outputs'] = self.num_outputs

        # Calculate per-sample averages (useful for reporting and comparison)
        ops['total_flops_per_sample'] = ops['total_flops'] / batch_size
        ops['synaptic_ops_per_sample'] = ops['synaptic_ops'] / batch_size
        ops['synops_flops_per_sample'] = ops['synops_flops'] / batch_size
        ops['total_ops_per_sample'] = ops['total_ops'] / batch_size

        if verbose:
            print("\n" + "="*70)
            print("SNN Effective Operations Analysis")
            print("="*70)
            print(f"Topology: {topology}")
            print(f"Batch size: {batch_size}, Timesteps: {timesteps}")
            print(f"Inputs: {self.num_inputs}, Outputs: {self.num_outputs}")
            print("-"*70)
            if topology in [ENC_SHALLOW_SPIKING, ENC_SHALLOW_LEAKY]:
                print(f"Encoding spikes: {ops['total_encoding_spikes']:,} "
                      f"(rate: {ops['encoding_spike_rate']:.4f})")
            else:
                print(f"Input spikes: {ops['total_input_spikes']:,} "
                      f"(rate: {ops['input_spike_rate']:.4f})")
            print(f"Output spikes: {ops['total_output_spikes']:,} "
                  f"(rate: {ops['output_spike_rate']:.4f})")
            print("-"*70)
            print("SPIKE-DEPENDENT (Sparse) Operations:")
            print(f"  Synaptic operations (SynOps): {ops['synaptic_ops']:,}")
            if ops['synops_mac'] > 0:
                print(f"    - fc1 SynOps as MAC (encoding): {ops['synops_mac']:,} -> {ops['synops_mac']*2:,} FLOPs")
            if ops['synops_ac'] > 0:
                print(f"    - fc1 SynOps as AC (non-enc): {ops['synops_ac']:,} -> {ops['synops_ac']:,} FLOPs")
            if ops['synops_ac_fc2'] > 0:
                print(f"    - fc2 SynOps as AC (out->filt): {ops['synops_ac_fc2']:,} -> {ops['synops_ac_fc2']:,} FLOPs")
            print(f"  SynOps FLOPs: {ops['synops_flops']:,}")
            print(f"  Per sample: {ops['synaptic_ops_per_sample']:,.0f} SynOps")
            print("-"*70)
            print("DENSE Operations (membrane/synapse dynamics):")
            print(f"  MAC operations: {ops['mac_ops']:,} -> {ops['mac_ops']*2:,} FLOPs")
            print(f"  Add operations: {ops['add_ops']:,} -> {ops['add_ops']:,} FLOPs")
            print(f"  Dense FLOPs: {ops['membrane_synapse_flops']:,}")
            print("-"*70)
            print(f"Comparison operations: {ops['comparison_ops']:,}")
            print("-"*70)
            print(f"TOTAL FLOPs: {ops['total_flops']:,}")
            print(f"  = SynOps FLOPs ({ops['synops_flops']:,}) + Dense FLOPs ({ops['membrane_synapse_flops']:,})")
            print(f"  Per sample: {ops['total_flops_per_sample']:,.0f} FLOPs")
            print(f"  SynOps as % of total FLOPs: {ops['synaptic_percentage']:.1f}%")
            print(f"Total operations (incl. comparisons): {ops['total_ops']:,}")
            print("="*70 + "\n")

        return ops

    def profile_snn_inference(self, input_tensor: torch.Tensor,
                               state_dict: dict = None) -> Dict:
        """
        Profile a single forward pass of the SNN model.

        Args:
            input_tensor: Input tensor for forward pass (batch, timesteps, inputs)
            state_dict: Optional state dictionary for maintaining membrane potentials

        Returns:
            Dictionary with profiling metrics for this sample:
            - cpu_memory: CPU memory usage in bytes
            - cpu_time: CPU time in microseconds
            - flops: FLOPs from torch.profiler
            - record_dict: Network output dictionary with spikes and membrane potentials
        """
        result = {
            'cpu_memory': 0,
            'cpu_time': 0,
            'flops': 0,
            'record_dict': None
        }

        # Profile using torch.profiler
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True
        ) as prof:
            record_dict = self.network.forward(input_tensor, state_dict)

        result['record_dict'] = record_dict

        # Extract metrics from profiler
        for event in prof.key_averages():
            if event.flops is not None:
                result['flops'] += event.flops
            result['cpu_time'] += event.cpu_time_total
            result['cpu_memory'] += event.cpu_memory_usage

        return result

    def calculate_theoretical_ops(self, batch_size: int, timesteps: int) -> Dict[str, float]:
        """
        Calculate theoretical maximum operations assuming dense (non-sparse) computation.
        This represents the worst-case scenario where all neurons spike at every timestep (100% spike rate).

        Args:
            batch_size: Batch size for the calculation
            timesteps: Number of timesteps

        Returns:
            Dictionary containing theoretical operation counts
        """
        ops = {
            'mac_ops': 0,
            'add_ops': 0,
            'comparison_ops': 0,
        }

        topology = self.network.topology

        # Input layer operations
        if topology in [ENC_SHALLOW_SPIKING, ENC_SHALLOW_LEAKY]:
            # Encoding layer: EMG to spikes
            # Linear layer (identity matrix): num_inputs multiplications
            ops['mac_ops'] += batch_size * timesteps * self.num_inputs
            # Leaky integrator: beta * mem + current
            ops['mac_ops'] += batch_size * timesteps * self.num_inputs
            ops['add_ops'] += batch_size * timesteps * self.num_inputs
            ops['comparison_ops'] += batch_size * timesteps * self.num_inputs

            # Decoding layer: assuming every neuron spikes at every timestep (worst case)
            ops['mac_ops'] += batch_size * timesteps * self.num_inputs * self.num_outputs
        else:
            # Standard input to fc1: assuming all inputs are active at every timestep
            ops['mac_ops'] += batch_size * timesteps * self.num_inputs * self.num_outputs

        # Output neuron dynamics
        if topology in [SHALLOW_SPIKING_DOUBLE_FILTER, ENC_SHALLOW_SPIKING]:
            # Synaptic current: alpha * syn_cur
            ops['mac_ops'] += batch_size * timesteps * self.num_outputs
            # Membrane potential: beta * mem
            ops['mac_ops'] += batch_size * timesteps * self.num_outputs
            # Additions (syn_cur + input, mem + syn)
            ops['add_ops'] += batch_size * timesteps * self.num_outputs * 2

        elif topology in [SHALLOW_LEAKY, ENC_SHALLOW_LEAKY]:
            # Leaky dynamics: alpha * syn_cur + beta * mem
            ops['mac_ops'] += batch_size * timesteps * self.num_outputs * 2
            ops['add_ops'] += batch_size * timesteps * self.num_outputs * 2

        # Threshold comparisons
        ops['comparison_ops'] += batch_size * timesteps * self.num_outputs

        # Filter layers
        if hasattr(self.network, 'filter_layers'):
            num_filters = len(self.network.filter_layers)
            for _ in range(num_filters):
                ops['mac_ops'] += batch_size * timesteps * self.num_outputs
                ops['add_ops'] += batch_size * timesteps * self.num_outputs

        # Total FLOPs
        ops['total_flops'] = ops['mac_ops'] * 2 + ops['add_ops']
        ops['total_ops'] = ops['total_flops'] + ops['comparison_ops']

        ops['batch_size'] = batch_size
        ops['timesteps'] = timesteps
        ops['assumption'] = 'full_density'

        return ops

    def compare_ops_with_baseline(self, effective_ops,
                                  baseline_type: str = 'dense_ann') -> Dict[str, float]:
        """
        Compare SNN operations with baseline architectures.

        Args:
            record_dict: Dictionary containing recorded network variables
            baseline_type: Type of baseline ('dense_ann', 'theoretical_snn')

        Returns:
            Dictionary with comparison metrics
        """
        # effective_ops = self.calculate_effective_ops(record_dict, verbose=False)

        batch_size = effective_ops['batch_size']
        timesteps = effective_ops['timesteps']

        comparison = {
            'effective_ops': effective_ops,
        }

        if baseline_type == 'dense_ann':
            # Dense ANN: all neurons active at every timestep
            dense_ops = batch_size * timesteps * self.num_inputs * self.num_outputs * 2  # fc1
            dense_ops += batch_size * timesteps * self.num_outputs * 2  # bias and activation

            comparison['dense_ann_flops'] = dense_ops
            comparison['reduction_vs_dense'] = dense_ops / effective_ops['total_flops']

        elif baseline_type == 'theoretical_snn':
            theoretical_ops = self.calculate_theoretical_ops(batch_size, timesteps)
            comparison['theoretical_snn_ops'] = theoretical_ops
            comparison['reduction_vs_theoretical'] = (theoretical_ops['total_flops'] /
                                                     effective_ops['total_flops'])

        return comparison


def create_inference_net(snn_config:SNNConfig, trained_params_file:str, timesteps:int,
                         num_inputs:int, num_outputs:int,test_dataset:EMGDataset):
    """
    Creates a new network object using the trained parameters.
    """
    trained_params = load_params_from_pickle(trained_params_file)
    snnreg_model = SnnReg(timesteps,num_inputs,num_outputs, snn_config, None, test_dataset)
    net = snnreg_model.network
    net.fc1.weight = nn.Parameter(trained_params['fc1.weight'])
    net.fc1.bias = nn.Parameter(trained_params['fc1.bias'])
    net.lif1.alpha = nn.Parameter(trained_params['lif1.alpha'])
    net.lif1.beta = nn.Parameter(trained_params['lif1.beta'])
    # net.filter.beta = nn.Parameter(trained_params['filter.beta'])
    return snnreg_model


def load_params_from_pickle(trained_params_file):
    """
    Loads the trained parameters from a pickle file.
    """
    with open(trained_params_file, 'rb') as f:
        trained_params = pkl.load(f)
    return trained_params


def log_spike_count_df(output_spike_count_df:pd.DataFrame,fold_i:int, binwidth:float)-> None:
    """
    Logs table of output spike counts to wandb.
    """
    wandb.log({f'Spike Count table Fold {fold_i} {binwidth}': wandb.Table(data=output_spike_count_df,
                                                                           columns=output_spike_count_df.columns.tolist())})


def sum_output_spikes_per_active_finger(overlap_perc:float, data_config:DataConfig,
                                        num_segments_per_rep:int,
                                        repetition_dur:float,
                                        fold_i:int, binwidth:float, snn_network,
                                        mode:str,
                                        rec_ts:dict) -> pd.DataFrame:
    """
    Sums the output spikes for each active finger by first reconstructing the whole repetition using the overlapping windows.
    """
    reconst_spikes = reconstruct_rep_from_bins(rec_ts[SPK_OUT], num_segments_per_rep,
                                               binwidth, overlap_perc,
                                               snn_network.dt,repetition_dur)
    output_spike_count = torch.sum(reconst_spikes,dim=0)
    output_spike_count_df = pd.DataFrame(output_spike_count, columns=[f'neu_{i}' for i in range(output_spike_count.shape[1])])
    output_spike_count_df[MODE_COL] = mode
    output_spike_count_df[REP_DUR_COL] = reconst_spikes.shape[0]* snn_network.dt
    output_spike_count_df[FOLD_COL] = fold_i
    output_spike_count_df['active_fing_name'] = list(data_config.finger_label_map.keys())
    return output_spike_count_df


def run_net_single_split(snn_config:SNNConfig, data_config:DataConfig, dataset:EMGDataset,
                         rep_on_rep:str, fold_i:int, use_inference_network:bool=False,
                         num_profile_samples:int=10,
                         enable_profiling:bool=False):
    """
    Runs the network for a single split of the dataset. 
    This function is used for both training (or offline) and inference modes (mimicks online inference with batch size=1).
    In training mode, the dataset is split into train and test sets, and the network is trained on the train set then evaluated
    on both train and test sets. 
    In inference mode, a new network is created using the trained weights and and other parameters saved in a pickle file is used to make predictions on the test set.
    This new network is created to run on the test set using a smaller binwidth than the one used for training to mimic online inference.
    """
    dataset_rep_1, dataset_rep_2 = create_subsets(data_config, snn_config, dataset)
    test_rep = int(rep_on_rep.split('on')[1])

    neuron_beta_before_training = None   # inital tau_mem for the network prior training
    synapse_alpha_before_training = None  # inital tau_syn for the network prior training
    neuron_thr_before_training = None
    weights_before_training = None
    y_pred_tr, y_true_tr = None, None

    trained_params_file = generate_trained_params_filename(snn_config, data_config, fold_i)
    trained_params_df = None

    if use_inference_network:
        mode = 'inference'
        test_dataset, _ = assign_datasets_for_train_test(dataset_rep_1, dataset_rep_2, rep_on_rep)
        snnreg_model = create_inference_net(snn_config, trained_params_file,
                                           dataset.num_steps,dataset.num_inputs,dataset.num_outputs,
                                           test_dataset)
        snn_network = snnreg_model.network
        binwidth = snn_config.training["inf_rep_binwidth"]
    else:
        mode = 'training'
        test_dataset, train_dataset = assign_datasets_for_train_test(dataset_rep_1, dataset_rep_2, rep_on_rep)
        snnreg_model = SnnReg(dataset.num_steps,dataset.num_inputs,dataset.num_outputs,
                              snn_config, train_dataset, test_dataset)
        snn_network = snnreg_model.network

        save_network_statedict_to_file(snn_config, data_config, snn_network)
        neuron_beta_before_training, synapse_alpha_before_training, neuron_thr_before_training, weights_before_training =snnreg_model.get_initial_network_params()
        binwidth = snn_config.training["train_rep_binwidth"]

        y_pred_tr, y_true_tr, tr_loss_hist, loss_per_epoch, ts_loss_hist, rec_tr = snnreg_model.train_and_evaluate_on_training_set()

        trained_params_df = prepare_and_save_trained_params(fold_i, trained_params_file, snn_network,
                                                            neuron_beta_before_training, synapse_alpha_before_training,
                                                            neuron_thr_before_training, weights_before_training,
                                                            binwidth)
        trained_weights = snn_network.fc1.weight.detach().clone().numpy()
        plot_network_summary(snn_config, data_config, dataset, fold_i, neuron_beta_before_training, synapse_alpha_before_training,
                             neuron_thr_before_training, weights_before_training, y_true_tr, snn_network, tr_loss_hist,
                             loss_per_epoch, ts_loss_hist, rec_tr, trained_weights)
        
    profiling_results = None
    if enable_profiling:
        eval_result = snnreg_model.evaluate_and_log_on_test_set(data_config, dataset.num_segments,
                                                                dataset.rep_dur, fold_i, mode, binwidth,
                                                                enable_profiling=True,
                                                                num_profile_samples=num_profile_samples)
        if len(eval_result) == 4:
            y_pred_ts, y_true_ts, rec_ts, profiling_results = eval_result
        else:
            y_pred_ts, y_true_ts, rec_ts = eval_result
    else:
        y_pred_ts, y_true_ts, rec_ts = snnreg_model.evaluate_and_log_on_test_set(data_config, dataset.num_segments,
                                                                                  dataset.rep_dur, fold_i, mode, binwidth)

    # Note: Effective SynOps calculation has been moved to profile_snn_on_unsegmented_data()
    # to only run when profiling is enabled, not on every inference run.

    metrics_df = prepare_snn_metrics_df(y_pred_tr, y_true_tr, y_pred_ts, y_true_ts, test_rep, fold_i, mode, binwidth,
                                    fing_list=dataset.active_fingers_order[test_rep])
    y_df = prepare_predictions_df(y_pred_ts, y_true_ts, test_rep, fold_i, mode, binwidth,
                                  dataset.active_fingers_order[test_rep])
    snnplot.plot_pred_vs_true(snn_network, snn_config, data_config, dataset,
                              y_pred_ts, y_true_ts, test_rep,
                              fold_i, mode=mode)
    return metrics_df, y_df, trained_params_df, rec_ts, profiling_results


def profile_snn_on_unsegmented_data(snn_config: SNNConfig, data_config: DataConfig,
                                     unsegmented_dataset: EMGDataset, rep_on_rep: str,
                                     fold_i: int, num_profile_samples: int = 5,
                                     ) -> Dict:
    """
    Profile SNN inference on unsegmented data for accurate operation counting.

    This function profiles the SNN on full-length repetitions (unsegmented data)
    to get accurate FLOP counts that match the RNN profiling approach. This avoids
    the overlap-induced redundancy from windowed segmentation.

    Args:
        snn_config: SNN configuration
        data_config: Data configuration
        unsegmented_dataset: Dataset with unsegmented inputs (full repetition)
        rep_on_rep: Training/test split string (e.g., '2on1')
        fold_i: Fold index
        num_profile_samples: Number of samples to profile (default: 5)

    Returns:
        Dictionary with profiling results including:
        - CPU memory and time statistics
        - Effective operations (SynOps, MACs, etc.)
        - Spike counts and rates
        - Comparison with dense ANN baseline
    """

    # Load trained parameters
    trained_params_file = generate_trained_params_filename(snn_config, data_config, fold_i)

    # For unsegmented data, samples are arranged as:
    # [finger0_rep1, finger0_rep2, finger1_rep1, finger1_rep2, ...]
    # So even indices (0,2,4,6,8) are rep1, odd indices (1,3,5,7,9) are rep2
    test_rep = int(rep_on_rep.split('on')[1])

    # Create test subset based on rep_on_rep
    # For unsegmented data with 10 samples (5 fingers x 2 reps):
    # rep1 indices: 0, 2, 4, 6, 8 (even)
    # rep2 indices: 1, 3, 5, 7, 9 (odd)
    n_fingers = data_config.n_ind_fingers
    if test_rep == 1:
        test_indices = np.arange(0, n_fingers * 2, 2)  # [0, 2, 4, 6, 8]
    else:
        test_indices = np.arange(1, n_fingers * 2, 2)  # [1, 3, 5, 7, 9]

    test_dataset = Subset(unsegmented_dataset, test_indices)

    snnreg_model = create_inference_net(snn_config, trained_params_file,
                                        unsegmented_dataset.num_steps,
                                        unsegmented_dataset.num_inputs,
                                        unsegmented_dataset.num_outputs,
                                        test_dataset)
    snn_network = snnreg_model.network

    # Profile on unsegmented samples
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    profiling_samples = []
    record_dicts = []

    print(f"\n[Unsegmented Profiling] Test dataset size: {len(test_dataset)}")
    print(f"[Unsegmented Profiling] Timesteps per sample: {unsegmented_dataset.num_steps}")

    # import the NeuroBench wrapper to wrap the snnTorch model for usage in the NeuroBench framework
    # import the benchmark class


    neuro_model = SNNTorchModel(snn_network, custom_forward=True, prediction_var=snn_network.prediction_var)
    static_metrics = [Footprint, ConnectionSparsity]
    workload_metrics = [ActivationSparsity, SynapticOperations]
    benchmark = Benchmark(neuro_model, test_loader, [], [], [static_metrics, workload_metrics])
    results = benchmark.run(device=DEVICE)
    # results = {}
    # Add parameter and buffer breakdown to NeuroBench results
    results['parameters'] = [
        (name, list(p.shape), p.requires_grad, p.numel(), p.dtype)
        for name, p in snn_network.named_parameters()
    ]
    results['buffers'] = [
        (name, list(b.shape), b.numel(), b.dtype)
        for name, b in snn_network.named_buffers()
    ]

    with torch.no_grad():
        snn_network.eval()
        for i, (data, label, noisy_data) in enumerate(test_loader):
            if i >= num_profile_samples:
                break

            spk_or_emg_in = data.to(DEVICE)

            # Profile this sample
            profile_result = {
                'cpu_memory': 0,
                'cpu_time': 0,
                'flops': 0,
                'wall_clock_time': 0,
                'record_dict': None
            }

            # Measure wall-clock time separately (without profiler overhead)
            wall_start = time.perf_counter()
            _ = snn_network.forward(spk_or_emg_in, None)
            wall_end = time.perf_counter()
            profile_result['wall_clock_time'] = (wall_end - wall_start) * 1e6  # Convert to microseconds

            # Now run with profiler for FLOPs and memory (separate pass)
            with profile(
                activities=[ProfilerActivity.CPU],
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
                with_modules=True
            ) as prof:
                record_dict = snn_network.forward(spk_or_emg_in, None)

            profile_result['record_dict'] = record_dict
            record_dicts.append(record_dict)

            for event in prof.key_averages():
                if event.flops is not None:
                    profile_result['flops'] += event.flops
                profile_result['cpu_time'] += event.cpu_time_total
                profile_result['cpu_memory'] += event.cpu_memory_usage

            profiling_samples.append(profile_result)
            print(f"  Profiled sample {i+1}/{min(num_profile_samples, len(test_dataset))}")

    # Calculate effective ops on one sample (full repetition)
    if record_dicts:
        # Use first sample for effective ops calculation
        sample_record_dict = record_dicts[0]
        test_ops = snnreg_model.calculate_effective_ops(sample_record_dict, verbose=True)
        comparison = snnreg_model.compare_ops_with_baseline(test_ops, baseline_type='dense_ann')
    else:
        print("[Warning] No samples were profiled - record_dicts is empty")
        test_ops = {}
        comparison = {'dense_ann_flops': 0, 'reduction_vs_dense': 0}

    # Check if we have profiling samples before aggregation
    if not profiling_samples:
        print("[Warning] No profiling samples collected - returning empty results")
        return {
            'peak_cpu_memory_bytes': 0,
            'avg_cpu_memory_bytes': 0,
            'std_cpu_memory_bytes': 0,
            'peak_cpu_time_us': 0,
            'avg_cpu_time_us': 0,
            'std_cpu_time_us': 0,
            'avg_flops': 0,
            'avg_flops_torch_profiler': 0,
            'num_samples_profiled': 0,
            'data_type': 'unsegmented',
            'timesteps_per_sample': unsegmented_dataset.num_steps,
            'dense_ann_flops': 0,
            'reduction_vs_dense': 0
        }

    # Aggregate profiling results
    profiling_results = aggregate_snn_profiling_results(
        profiling_samples, effective_ops=test_ops, model=snn_network, verbose=True
    )

    # Add comparison metrics
    profiling_results['dense_ann_flops'] = comparison['dense_ann_flops']
    profiling_results['reduction_vs_dense'] = comparison['reduction_vs_dense']

    # Add metadata
    profiling_results['data_type'] = 'unsegmented'
    profiling_results['timesteps_per_sample'] = unsegmented_dataset.num_steps

    # Add NeuroBench benchmark results
    profiling_results['neurobench_results'] = results

    print(f"[Unsegmented Profiling] FLOP reduction vs dense ANN: {comparison['reduction_vs_dense']:.2f}x")

    return profiling_results


def plot_network_summary(snn_config, data_config, dataset, fold_i, beta_init, alpha_init, threshold_init,
                         initial_weights, y_true_tr, snn_network, tr_loss_hist, loss_per_epoch,
                         ts_loss_hist, rec_tr, trained_weights):
    """
    Plots the training an test losses per epoch, learnt vs initial weights, input spikes, 
    network intermediary variables, and the learnt vs initial weights.
    """
    snnplot.plot_epochs_loss(snn_config, data_config, tr_loss_hist, ts_loss_hist, loss_per_epoch,   fold_i)
    snnplot.plot_learnt_wdist(snn_config, data_config, trained_weights, initial_weights)

    # Plot the fold input
    input_var = rec_tr[snn_network.input_var].cpu().detach().numpy()
    snnplot.plot_network_variable(input_var, dataset, snn_network.dt, snn_config,data_config, fold_i, var_name=snn_network.input_var)
    snnplot.network_variables_plot(snn_network, snn_config, data_config, rec_tr, y_true_tr,
                                   dataset, fold_i, beta_init, alpha_init, threshold_init)
    snnplot.plot_weight_heatmap(snn_network.fc1.weight.detach().cpu().numpy(),snn_config, data_config, fold_i)


def prepare_and_save_trained_params(fold_i, trained_params_file, snn_network,
                                    neuron_beta_before_training, synapse_alpha_before_training,
                                    neuron_thr_before_training, weights_before_training,
                                    binwidth):
    """
    Prepares the trained parameters dataframe and saves it to a pickle file."""
    trained_params_dict = {
        'fc1.weight.before': weights_before_training,
        'lif1.alpha.before': synapse_alpha_before_training,
        'lif1.beta.before': neuron_beta_before_training,
        'lif1.threshold.before': neuron_thr_before_training,
        'fc1.weight': snn_network.fc1.weight.detach().clone(),
        'fc1.bias': snn_network.fc1.bias.detach().clone(),
        'lif1.alpha': snn_network.lif1.alpha.detach().clone(),
        'lif1.beta': snn_network.lif1.beta.detach().clone(),
        'lif1.threshold': snn_network.lif1.threshold.detach().clone(),
    }
    for name, param in snn_network.named_parameters():
        if param.requires_grad and name not in trained_params_dict.keys():
            trained_params_dict[name] = param.detach().clone()
            print(f"Adding parameter to trained params dict:{name} with values:\n{param.detach().clone()}")
    
    trained_params_df = prepare_trained_params_df(trained_params_dict, fold_i, binwidth)
    pkl.dump(trained_params_dict, open(trained_params_file, 'wb'))
    return trained_params_df



def generate_trained_params_filename(snn_config, data_config, fold_i):
    """
    Generates the file path for the trained parameters."""
    batch_size = snn_config.training["train_batch_size"]
    num_iter = snn_config.training["num_iter"]
    learn_tau_mem = snn_config.decoder_type["learn_tau_mem"]
    first_filter_tau = snn_config.decoder_type["first_filter_tau"]
    second_filter_tau = snn_config.decoder_type["second_filter_tau"]
    tau_syn = snn_config.decoder_type["tau_syn"]
    exp_dir = snn_config.task["exp_dir"]
    train_rep_binwidth = snn_config.training["train_rep_binwidth"]

    temp_filename = f"{data_config.mvc}_{exp_dir}_ep_{num_iter}_trained_params_fold_{fold_i}_binwidth_{train_rep_binwidth}_batchsize_{batch_size}_taufilt1_{first_filter_tau}_taufilt2_{second_filter_tau}_tau_syn_{tau_syn}_learn_beta_{learn_tau_mem}.pkl"
    trained_params_file = os.path.join(data_config.tr_weights_data_path, temp_filename)
    return trained_params_file


def save_network_statedict_to_file(snn_config, data_config, snn_network):
    """
    Saves the model parameters to a file.
    """
    exp_dir = snn_config.task["exp_dir"]
    topology = snn_config.decoder_type["topology"]
    subject = snn_config.task["subject"]
    filename = f"{subject}_mvc_{data_config.mvc}_{exp_dir}_networkparam_topology_{topology}.pt"
    common_path_across_subjects = os.path.dirname(data_config.snn_temp_data_path)
    torch.save(snn_network.state_dict(), os.path.join(common_path_across_subjects, filename))

def prepare_predictions_df(y_pred_test:np.ndarray, y_true_test:np.ndarray, test_rep:int,
                           fold_index:int, mode:str,
                           input_dur:float,fing_list:list[str]) -> pd.DataFrame:
    """
    Prepares a dataframe with the predictions and the true values.
    """
    df_columns = [Y_PRED_TEST, Y_TRUE_TEST,TEST_ON_REP, FOLD_COL,
                  INPUT_DUR_COL, FING_ORDER_COL, MODE_COL]
    y_pred_df = pd.DataFrame(columns=df_columns)

    y_pred_df.loc[0, Y_PRED_TEST] = y_pred_test
    y_pred_df.loc[0, Y_TRUE_TEST] = y_true_test
    y_pred_df.loc[0, TEST_ON_REP] = test_rep
    y_pred_df.loc[0, FOLD_COL] = fold_index
    y_pred_df.loc[0, MODE_COL] = mode
    y_pred_df.loc[0, INPUT_DUR_COL] = input_dur
    y_pred_df.loc[0, FING_ORDER_COL] = fing_list

    return y_pred_df

def prepare_trained_params_df(trained_params:dict, fold_i:int, input_dur:float) -> pd.DataFrame:
    """
    Prepares a dataframe with the trained parameters.
    """
    trained_params_df = pd.DataFrame(columns=list(trained_params.keys()))
    for key in trained_params.keys():

        trained_params_df.loc[0, key] = trained_params[key].cpu().numpy() if isinstance(trained_params[key], torch.Tensor) else trained_params[key]
    trained_params_df.loc[0, 'fold'] = fold_i
    trained_params_df.loc[0, 'input_dur'] = input_dur

    return trained_params_df


def calculate_ops_for_dataset(snn_model: SnnReg, dataset: EMGDataset,
                               verbose: bool = True) -> Tuple[Dict, pd.DataFrame]:
    """
    Calculate operations for an entire dataset by running inference and aggregating results.

    Args:
        snn_model: Trained SNN regression model
        dataset: Dataset to evaluate
        verbose: If True, print summary statistics

    Returns:
        Tuple of (aggregated_ops_dict, ops_dataframe)
    """

    batch_size = 1  # Process one sample at a time for accurate spike counting
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_ops = []

    with torch.no_grad():
        snn_model.network.eval()
        state_dict = None

        for i, (data, label, _) in enumerate(data_loader):
            spk_or_emg_in = data.to(DEVICE)

            # Forward pass
            if i > 0 and batch_size == 1:
                state_dict = snn_model._save_network_state_dict(record_dict)

            record_dict = snn_model.network.forward(spk_or_emg_in, state_dict)

            # Calculate ops for this sample
            ops = snn_model.calculate_effective_ops(record_dict, verbose=False)
            ops['sample_id'] = i
            all_ops.append(ops)

    # Convert to DataFrame
    ops_df = pd.DataFrame(all_ops)

    # Calculate aggregated statistics
    aggregated = {
        'total_samples': len(all_ops),
        'mean_flops_per_sample': ops_df['total_flops'].mean(),
        'std_flops_per_sample': ops_df['total_flops'].std(),
        'total_flops': ops_df['total_flops'].sum(),
        'mean_spike_rate': ops_df.get('encoding_spike_rate', ops_df.get('input_spike_rate', 0)).mean(),
        'mean_output_spike_rate': ops_df['output_spike_rate'].mean(),
    }

    if verbose:
        print("\n" + "="*70)
        print("Dataset-wide Operations Summary")
        print("="*70)
        print(f"Total samples processed: {aggregated['total_samples']}")
        print(f"Total FLOPs: {aggregated['total_flops']:,.0f}")
        print(f"Mean FLOPs per sample: {aggregated['mean_flops_per_sample']:,.0f} ± {aggregated['std_flops_per_sample']:,.0f}")
        print(f"Mean spike rate: {aggregated['mean_spike_rate']:.4f}")
        print(f"Mean output spike rate: {aggregated['mean_output_spike_rate']:.4f}")
        print("="*70 + "\n")

    return aggregated, ops_df


def log_ops_to_wandb(ops_dict: Dict, prefix: str = 'snn', fold_i: int = None):
    """
    Log operation counts to Weights & Biases.

    Args:
        ops_dict: Dictionary of operation counts from calculate_effective_ops
        prefix: Prefix for wandb logging keys (e.g., 'snn_train', 'snn_test')
        fold_i: Optional fold index for cross-validation
    """
    log_dict = {
        # Total operations
        f'{prefix}/total_flops': ops_dict['total_flops'],
        f'{prefix}/synaptic_ops': ops_dict['synaptic_ops'],
        f'{prefix}/synaptic_percentage': ops_dict['synaptic_percentage'],
        f'{prefix}/mac_ops': ops_dict['mac_ops'],
        f'{prefix}/add_ops': ops_dict['add_ops'],
        f'{prefix}/comparison_ops': ops_dict['comparison_ops'],

        # Per-sample averages (important for fair comparison)
        f'{prefix}/total_flops_per_sample': ops_dict['total_flops_per_sample'],
        f'{prefix}/synaptic_ops_per_sample': ops_dict['synaptic_ops_per_sample'],

        # Spike rates
        f'{prefix}/output_spike_rate': ops_dict['output_spike_rate'],
    }

    # Add spike rate (encoding or input depending on topology)
    if 'encoding_spike_rate' in ops_dict:
        log_dict[f'{prefix}/encoding_spike_rate'] = ops_dict['encoding_spike_rate']
    elif 'input_spike_rate' in ops_dict:
        log_dict[f'{prefix}/input_spike_rate'] = ops_dict['input_spike_rate']

    if fold_i is not None:
        log_dict['fold'] = fold_i

    wandb.log(log_dict)


def aggregate_snn_profiling_results(profiling_samples: list, effective_ops: Dict = None,
                                     model: torch.nn.Module = None, verbose: bool = True) -> Dict:
    """
    Aggregate profiling results across multiple SNN samples.

    Args:
        profiling_samples: List of profiling results from profile_snn_inference
        effective_ops: Dictionary from calculate_effective_ops (optional, for SynOps metrics)
        model: The SNN model for parameter counting (optional)
        verbose: If True, print summary

    Returns:
        Dictionary with aggregated statistics:
        - peak/avg/std for cpu_memory, cpu_time
        - total_params, trainable_params, counted_params (if model provided)
        - effective_ops metrics if provided
    """
    cpu_mem = np.array([s['cpu_memory'] for s in profiling_samples])
    cpu_time = np.array([s['cpu_time'] for s in profiling_samples])
    flops = np.array([s['flops'] for s in profiling_samples])
    wall_clock_time = np.array([s.get('wall_clock_time', 0) for s in profiling_samples])

    results = {
        # Per-sample arrays
        'cpu_memory_per_sample': cpu_mem.tolist(),
        'cpu_time_per_sample': cpu_time.tolist(),
        'flops_per_sample': flops.tolist(),
        'wall_clock_time_per_sample': wall_clock_time.tolist(),

        # Aggregated CPU memory stats
        'peak_cpu_memory_bytes': int(np.max(cpu_mem)),
        'avg_cpu_memory_bytes': float(np.mean(cpu_mem)),
        'std_cpu_memory_bytes': float(np.std(cpu_mem)),

        # Aggregated CPU time stats (from profiler - includes overhead)
        'peak_cpu_time_us': int(np.max(cpu_time)),
        'avg_cpu_time_us': float(np.mean(cpu_time)),
        'std_cpu_time_us': float(np.std(cpu_time)),

        # Wall-clock time stats (actual inference time without profiler overhead)
        'peak_wall_clock_time_us': float(np.max(wall_clock_time)),
        'avg_wall_clock_time_us': float(np.mean(wall_clock_time)),
        'std_wall_clock_time_us': float(np.std(wall_clock_time)),

        # Aggregated FLOPs stats (from torch.profiler)
        'peak_flops': int(np.max(flops)),
        'avg_flops': float(np.mean(flops)),
        'avg_flops_torch_profiler': float(np.mean(flops)),

        'num_samples_profiled': len(profiling_samples)
    }

    # Add model parameter counts if model is provided
    if model is not None:
        results['total_params'] = sum(p.numel() for p in model.parameters())
        results['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results['counted_params'] = [
            (name, list(param.data.size()), param.requires_grad, param.numel(), param.dtype)
            for name, param in model.named_parameters()
        ]

    # Add effective ops metrics if provided
    if effective_ops is not None:
        # Core operation counts
        results['synaptic_ops'] = effective_ops.get('synaptic_ops', 0)
        results['synaptic_ops_per_sample'] = effective_ops.get('synaptic_ops_per_sample', 0)
        results['synops_mac'] = effective_ops.get('synops_mac', 0)  # SynOps as MAC (encoding) - fc1
        results['synops_ac'] = effective_ops.get('synops_ac', 0)    # SynOps as AC (non-encoding) - fc1
        results['synops_ac_fc2'] = effective_ops.get('synops_ac_fc2', 0)  # SynOps as AC for fc2 (out->filt)
        results['synops_flops'] = effective_ops.get('synops_flops', 0)
        results['synops_flops_per_sample'] = effective_ops.get('synops_flops_per_sample', 0)
        results['mac_ops'] = effective_ops.get('mac_ops', 0)        # Dense MAC ops
        results['add_ops'] = effective_ops.get('add_ops', 0)        # Dense add ops
        results['membrane_synapse_flops'] = effective_ops.get('membrane_synapse_flops', 0)
        results['comparison_ops'] = effective_ops.get('comparison_ops', 0)
        results['total_effective_flops'] = effective_ops.get('total_flops', 0)
        results['total_effective_flops_per_sample'] = effective_ops.get('total_flops_per_sample', 0)
        results['total_ops'] = effective_ops.get('total_ops', 0)
        results['total_ops_per_sample'] = effective_ops.get('total_ops_per_sample', 0)
        results['synaptic_percentage'] = effective_ops.get('synaptic_percentage', 0)

        # Spike counts and rates
        results['total_input_spikes'] = effective_ops.get('total_input_spikes', 0)
        results['total_output_spikes'] = effective_ops.get('total_output_spikes', 0)
        results['total_encoding_spikes'] = effective_ops.get('total_encoding_spikes', 0)
        if 'encoding_spike_rate' in effective_ops:
            results['encoding_spike_rate'] = effective_ops['encoding_spike_rate']
        if 'input_spike_rate' in effective_ops:
            results['input_spike_rate'] = effective_ops['input_spike_rate']
        results['output_spike_rate'] = effective_ops.get('output_spike_rate', 0)

        # Dimensions
        results['timesteps'] = effective_ops.get('timesteps', 0)
        results['batch_size'] = effective_ops.get('batch_size', 0)
        results['num_inputs'] = effective_ops.get('num_inputs', 0)
        results['num_outputs'] = effective_ops.get('num_outputs', 0)

        # Comparison metrics (if available)
        if 'dense_ann_flops' in effective_ops:
            results['dense_ann_flops'] = effective_ops['dense_ann_flops']
        if 'reduction_vs_dense' in effective_ops:
            results['reduction_vs_dense'] = effective_ops['reduction_vs_dense']

    if verbose:
        print("\n" + "=" * 70)
        print("SNN Profiling Results (Aggregated)")
        print("=" * 70)
        print(f"Samples profiled: {results['num_samples_profiled']}")
        if model is not None:
            print(f"\n[Model Parameters]")
            print(f"  Total:     {results['total_params']:,}")
            print(f"  Trainable: {results['trainable_params']:,}")
        print(f"\n[CPU Memory]")
        print(f"  Peak:    {results['peak_cpu_memory_bytes'] / 1024**2:.2f} MiB")
        print(f"  Average: {results['avg_cpu_memory_bytes'] / 1024**2:.2f} MiB")
        print(f"  Std:     {results['std_cpu_memory_bytes'] / 1024**2:.2f} MiB")
        print(f"\n[Wall-Clock Time (actual inference)]")
        print(f"  Peak:    {results['peak_wall_clock_time_us'] / 1000:.3f} ms")
        print(f"  Average: {results['avg_wall_clock_time_us'] / 1000:.3f} ms")
        print(f"  Std:     {results['std_wall_clock_time_us'] / 1000:.3f} ms")
        print(f"\n[FLOPs - torch.profiler (dense ops only)]")
        print(f"  Average: {results['avg_flops_torch_profiler']:,.0f}")
        if effective_ops is not None:
            print(f"\n[Effective Ops - SNN-aware FLOP Breakdown]")
            print(f"  --- Spike-Dependent (Sparse) ---")
            print(f"  SynOps: {results['synaptic_ops']:,}")
            if results.get('synops_mac', 0) > 0:
                print(f"    - As MAC (encoding): {results['synops_mac']:,} -> {results['synops_mac']*2:,} FLOPs")
            if results.get('synops_ac', 0) > 0:
                print(f"    - As AC (non-enc):   {results['synops_ac']:,} -> {results['synops_ac']:,} FLOPs")
            print(f"  SynOps FLOPs: {results.get('synops_flops', 0):,}")
            print(f"  --- Dense (membrane/synapse) ---")
            print(f"  MAC ops: {results['mac_ops']:,} -> {results['mac_ops']*2:,} FLOPs")
            print(f"  Add ops: {results['add_ops']:,} -> {results['add_ops']:,} FLOPs")
            print(f"  Dense FLOPs: {results.get('membrane_synapse_flops', 0):,}")
            print(f"  --- Totals ---")
            print(f"  Total Effective FLOPs: {results['total_effective_flops']:,}")
            print(f"    Per sample: {results['total_effective_flops_per_sample']:,.0f}")
            print(f"  SynOps as % of FLOPs: {results['synaptic_percentage']:.1f}%")
            print(f"\n[Spike Rates]")
            if 'encoding_spike_rate' in results:
                print(f"  Encoding spike rate: {results['encoding_spike_rate']:.4f}")
            elif 'input_spike_rate' in results:
                print(f"  Input spike rate: {results['input_spike_rate']:.4f}")
            print(f"  Output spike rate: {results['output_spike_rate']:.4f}")
        print("=" * 70 + "\n")

    return results
