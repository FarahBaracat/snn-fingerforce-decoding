from dataclasses import dataclass, field


@dataclass
class SNNConfig:
    """
    Spiking Neural Network configuration class.
    """
    exp_dir: str = 'ext' # ext | flex
    subject:str = 'S1' # S1 | S2
    mvc: list[int] = field(default_factory=lambda: [15])
    num_iter: int = 2

    noise_mode: str = None # None| 'omission' | 'addition' | 'location'
    percent_omission: float = 0  # percentage of spikes to omit at random during inference, set to 0 for no omission
    percent_addition: float = 0  # percentage of spikes to add at random during inference, set to 0 for no addition
    jitter_std: float = 0  # std of the normal distribution in seconds for the jitter
    rand_indices_ratio: float = 0.1  # ratio of indices to apply jitter to
    percent_misattribution: float = 0  # percentage of spikes to misattribute at random during inference, set to 0 for no misattribution
    noise_torch_seed: int = 100
    use_aleaky: bool = False
    threshold_tau_adapt: float = 100e-3
    threshold_scale_adapt: float = 1

    lr: float = 1e-3
    learn_tau_syn: bool = False
    learn_tau_mem: bool = True
    learn_threshold: bool = True
    # normalize: bool = True
    # encoding layer parameters
    set_electrode_specific_parameters: bool = False # flag to set the encoding parameters for each electrode independently
    enc_tau_mem: float= 40e-3
    enc_spk_threshold: float = 0.5
    w_fixed_enc: float = 1e2   # this is the gain to applied to the input iEMG
    learn_enc_threshold:bool = False
    rect_type:str = 'full'      # full | half. The type of rectifier to be used in the encoding layer
    clip_channels:bool = True   # boolean to clip the channels prior to encoding
    add_recurrent: bool = False
    w_recurrent: float = 1

    # Decoding layers parameters
    tau_mem: float = 100e-3
    tau_syn: float = 10e-3
    spk_threshold:float = 2.0
    # filt_tau: float = 25e-3   # time constant for filter kernel applied on the output spikes
    first_filter_tau: float = 40e-3
    second_filter_tau: float = 100e-3

    filt_size_in_sec: float = 100e-3
    post_process_filt:bool=False
    post_filt_cutoff: float = 2
    post_filt_order: int = 4

    torch_seed: int = 90
    train_batch_size: int = 1
    test_batch_size: int = 1
    split_rep: bool = True
    train_rep_binwidth: float = 5
    inf_rep_binwidth: float = 5
    use_inf_network: bool = False
    log_ts_freq: int = 2   # evaluation frequency on the test set in terms of number of iterations

    dt: float = 10e-3
    train_fing_id: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    save_fig: bool = False
    save_weights: bool = True
    w_init: list[float] = field(default_factory=lambda: [-0.4, 0.4])
    w_init_dist: str = 'uniform'
    w_init_mean: float = 0
    w_init_std: float = 0.05
    w_fixed_filt:float= 1.0
    bias_init:list[float] = field(default_factory=lambda: [-0.001, 0.001])
    rep_on_rep: str = '1on1'  # 1on2: train on rep1, test on rep2 | 2on1: train on rep2, test on rep1
    overlap_perc: float = 0.3
    split_type: str = "with_overlap"  # with_overlap | without_overlap
    extract_hold_only: bool = True
    use_wandb: bool = False

    # network topoogy string
    topology: str = 'shallow_spiking_double_filter'
