"""
Module containing constants for variable and column names.
"""

# force_regression generated-dataframe column names
SP_TIME = 'sp_times'
MU_ID = 'mu_id'
MU_RATE = 'mu_rate'
ELEC_NAME = 'elec_name'
FING_NAME_COL = 'fing_name'
REP_ID = 'rep_id'
SP_COUNT = 'sp_count'
FING_DIR = 'fing_dir'

FIRST_SP_TIME = 'first_spike_time'
GLOB_MU_ID = 'glob_mu_id'
START_TIME = 'start_time'
END_TIME = 'end_time'
START_RAMP = 'start_ramp'
START_HOLD = 'start_hold'
END_HOLD = 'end_hold'
END_RAMP = 'end_ramp'
CONS_MU_ID = 'cons_mu_ids'
FORCE_COL = 'force'
TIME = 'time'
TOTAL_MUS = 'total_mus'
TOTAL_MUS_REP1 = 'total_mus_rep1'
TOTAL_MUS_REP2 = 'total_mus_rep2'
ISI = 'isi'
FING_ELEC = 'finger_elec'
FILTERS = 'filters'
MVC_LVL = 'mvc_lvl'

# Compile notebooks generated-dataframes column names
SUBJECT_COL = 'subject'
WS_COL = 'window_size'
OL_COL = 'overlap'   # percentage overlap between windows
SIGN_MVC_COL = 'sign_mvc'
MVC_COL = 'mvc'
EMG_COL = 'emg_type'
FING_NAME_COL = 'fing_name'
FING_ORDER_COL = 'fing_order'
FING_ID_COL = 'fing_id'
SEGMENT_HOLD_COL = 'segment_hold'

# Added columns for RNN
PROJECT_OUT_COL = 'project_to_out'
NORMALIZE_FORCE_COL = 'normalize_force'
RNN_BIAS_COL = 'rnn_bias'
KERNEL_SIZE_COL = 'exp_kernel_size'
KERNEL_DECAY_COL = 'exp_kernel_decay'
HIDDEN_STATE_SIZE_COL = 'rnn_hidden_size'
RNN_TYPE_COL = 'rnn_type'

# Other constants
TEST_ON_REP = 'test_on_rep'
Y_PRED_TRAIN = 'y_pred_train'
Y_TRUE_TRAIN = 'y_true_train'
Y_PRED_TEST = 'y_pred_test'
Y_TRUE_TEST = 'y_true_test'
Y_PRED_TRAIN_POST = 'y_pred_train_post'
Y_PRED_TEST_POST = 'y_pred_test_post'
INPUT_TYPE_COL = (
    'input_type'  # type of input used with the regression model: can be MUs|global features
)
DIR_COL = 'direction'  # direction column: flexion | extension
REG_FING_COL = 'reg_fingers'  # finger list over which the regression was fit.

INPUT_DUR_COL = (
    'input_dur'  # duration of the input presented to the SNN: eg 80ms (inference mode)| 5s (training mode)
)
REP_DUR_COL = 'rep_dur'
EPOCHS_COL = 'n_epochs'  # number of epochs columns
FOLD_COL = 'fold'  # fold column
MODE_COL = 'mode'  # mode column: training | inference
TOPOLOGY_COL = 'topology'  # topology column
TORCH_SEED_COL = 'torch_seed'  # torch seed column
NOISE_SEED_COL = 'noise_seed'  # noise seed column
NOISE_LEVEL_COL = 'noise_level'  # omission level column
NOISE_MODE_COL = 'noise_mode'  # noise mode column: omission | addition | misattribution

TAU_SYN_COL = 'tau_syn'  # synaptical time constant
TAU_MEM_COL = 'tau_mem'  # membrane time constant
TAU_MEM_FILT_COL = (
    'tau_mem_filt'  # filtering membrane time constant in case of a single tau membrane
)
TAU1_MEM_FILT_COL = (
    'tau1_mem_filt'  # filtering membrane time constant in case of a double tau membrane
)
TAU2_MEM_FILT_COL = (
    'tau2_mem_filt'  # filtering membrane time constant in case of a double tau membrane
)
ENC_TAU_MEM1_COL = 'enc_tau_mem1' # encoding neuron time constant for first electrode
ENC_TAU_MEM2_COL = 'enc_tau_mem2' # encoding neuron time constant for second electrode
ENC_TAU_MEM3_COL = 'enc_tau_mem3' # encoding neuron time constant for third electrode
ENC_SPK_THRESH1_COL = 'enc_spk_threshold1'  # encoding neuron spike threshold for first electrode
ENC_SPK_THRESH2_COL = 'enc_spk_threshold2'  # encoding neuron spike threshold for second electrode
ENC_SPK_THRESH3_COL = 'enc_spk_threshold3'  # encoding neuron spike threshold for third electrode
ENC_THRESHOLD_TAU_COL = 'enc_threshold_tau'  # for ALIF neurons, the adaptive threshold variable
ENC_ALIF_SCALE_COL = 'enc_alif_scale'  # for ALIF neurons, the effective threshold variable after adaptation
USE_ALIF_COL = 'use_alif'  # whether ALIF neurons were used

LEARN_TAU_SYN_COL = 'learn_tau_syn'
LEARN_TAU_MEM_COL = 'learn_tau_mem'
TAU_SYN_INIT_COL = 'tau_syn_init'
TAU_MEM_INIT_COL = 'tau_mem_init'
fing_order = ['thumb', 'index', 'middle', 'ring', 'little']

MODEL_SHORT_NAME_COL = 'model_short_name'

INPUT_TYPE_SP_TIMES = 'MU spike times'  # type of inputs/features that the models receive
INPUT_TYPE_SP_COUNT = 'MU spike count'  # changed to spike count
INPUT_TYPE_EMG_GLOB = 'global features'
RESULTS_FILE_ID_EMG_GLOB = 'cv_on_emg'
RESULTS_FILE_ID_SP_COUNT = 'cv_on_mu'

# Metrics columns
RMSE = 'RMSE'
R2 = 'R2'
MAE = 'MAE'
MSE = 'MSE'
CORR = 'Corr'
RMSE_TRAIN = 'RMSE_train'
RMSE_TEST = 'RMSE_test'
R2_TRAIN = 'R2_train'
R2_TEST = 'R2_test'
MAE_TRAIN = 'MAE_train'
MAE_TEST = 'MAE_test'
MSE_TRAIN = 'MSE_train'
MSE_TEST = 'MSE_test'

#### Figure styling
SHOW_TITLE = False  # display title on figure

XLAB_PAD = 2  # pad distance for x-axis label
YLAB_PAD = 3  # pad distance for y-axis label

BUFFER_TIME_PLOT = 1  # buffer time in seconds to avoid attaching different finger plots together

# plot x and y labels
FORCE_MVC_LABEL = 'Force level (% MVC)'
TIME_LABEL = 'time (s)'
MU_LABEL = 'Motor Unit ID'
FINGER_LABEL = 'Finger'

RMSE_LABEL = 'RMSE (% MVC)'
NORM_RMSE_LABEL = 'Normalized RMSE (%)'
R2_LABEL = 'R2'
MAE_LABEL = 'MAE (% MVC)'
NEURON_LABEL = 'Neuron ID'
# legend parameters
FING_LTITLE = 'Target Finger'  # legend title for target finger
LEGEND_ALPHA = 0.7  # legend alpha value

# line styles
DEFAULT_LINESTYLE = 'solid'  # default line style
LSTYLE_PRED = 'solid'  # predictions line style

# x,y -axis parameters
N_YTICKS = 5  # number of ticks to show on y-axis
XTICKS_STEP = 10  # step for x-axis ticks in seconds
LABEL_FONTSIZE = 12  # font size for x and y axis labels
TITLE_FONTSIZE = 14  # font size for title
# color palette
COLORS = {
    'burgundy': '#610933',
    'pumpkin': '#d35400',
    'midnight_blue': '#2c3e50',
    'pomgrenate': '#c0392b',
    'green_sea': '#16a085',
    'wisteria': '#8e44ad',
    'orange': '#f39c12',
    'clouds': '#7f8c8d',
    'naval': '#40739e',
    'purple': '#8c7ae6',
    'viz_orange': '#fa8775',
    'belize': '#2980b9',
    'cherry': '#99004c',
    'blueberry': '#0066cc',
    # from https://personal.sron.nl/~pault/#sec:qualitative
    'cyan': '#66CCEE',
    'forest_green': '#228833',
    'pink_red': '#BB5566',
    'mustard': '#DDAA33',
    'blue': '#4477AA',
    'cherry_red': '#B90047',
    'orange_red': '#FF5E45',
    'bright_yellow': '#FBED99',
    'turquoise': '#7BC2A6',
    'sky_blue': '#2290D0',
}

# SNN topology names
SHALLOW_SPIKING_DOUBLE_FILTER = 'shallow_spiking_double_filter'
SHALLOW_LEAKY = 'shallow_leaky'
ENC_SHALLOW_SPIKING = 'encoding_shallow_spiking'
ENC_SHALLOW_LEAKY = 'encoding_shallow_leaky'

# SNN network state variable names
SPK_IN = 'spk_in'
EMG_IN = 'emg_in'
ENC_NEU_CUR = 'enc_neu_cur'  # for encoding neuron
ENC_NEU_MEM = 'enc_neu_mem'  # for encoding neuron
ENC_SPK_OUT = 'enc_out'
ENC_THRESHOLD_ADAPT = 'enc_threshold_adapt'  # for adaptive leaky IF neurons, the adaptive threshold variable
ENC_ALIF_THRESHOLD = 'enc_alif_threshold'  # for adaptive leaky IF neurons, the effective threshold variable after adaptation

SPK_SYN_CUR = 'spk_syn_cur'  # for spiking neuron
SPK_NEU_CUR = 'spk_neu_cur'
SPK_NEU_MEM = 'spk_neu_mem'
SPK_OUT = 'spk_out'
LEAK_SYN_CUR = 'leak_syn_cur' # for leaky neuron
LEAK_NEU_CUR = 'leak_neu_cur'
LEAK_NEU_MEM = 'leak_neu_mem'

FILT_NEU_CUR = 'filt_neu_cur'
FILT_NEU_MEM = 'filt_neu_mem'
FILT_SPK_OUT = 'filt_spk_out'
FILT2_NEU_CUR = 'filt2_neu_cur'
FILT2_NEU_MEM = 'filt2_neu_mem'
FILT2_SPK_OUT = 'filt2_spk_out'