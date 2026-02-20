from dataclasses import dataclass, field
import os
import logging
import matplotlib.pyplot as plt
import force_regression.utils.functions as fn



def default_finger_map() -> dict:
    """
    Default mapping of fingers to their IDs.
    """
    return {'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'little': 5}


def default_electrodes() -> list:
    """
    Default order for intramuscular electrodes.
    """
    return ['ext_digitorum', 'flx_digitorum', 'ext_pollicis']


def default_color_dict() -> dict:
    return {
        'pumpkin': '#d35400', 'midnight_blue': '#2c3e50', 'pomgrenate': '#c0392b',
        'green_sea': '#16a085', 'wisteria': '#8e44ad', 'orange': '#f39c12',
        'clouds': '#7f8c8d', 'naval': '#40739e', 'purple': '#8c7ae6',
        'viz_orange': '#fa8775', 'belize': '#2980b9', 'cherry': '#99004c', 'blueberry': '#0066cc',
        # from https://personal.sron.nl/~pault/#sec:qualitative
        'cyan': '#66CCEE', 'forest_green': '#228833', 'pink_red': '#BB5566', 'mustard': '#DDAA33',
        'blue': '#4477AA'
    }

@dataclass
class DataConfig:
    """
    Data configuration class.
    """
    def __init__(self,
                 root_dir: str = 'data',
                 root_results_dir: str = 'results',
                 # Constants
                 data_input_dir: str = r"emg_data",
                 data_decomp_dir: str = r"decomposition", #r"data/output/decomposition",
                 decomp_dir: str = field(init=False),
                 input_dir: str = field(init=False),
                 subject: str = 'AK',
                 emg_type: str = 'intra',
                 decomp_subfolder: str = 'classification_bothreps',  # old decomp folder: 'both_reps_unique', new folder: 'classification_bothreps'
                 figs_dir: str = r'figures',
                 results_dir: str = r'output_files',  # set default directory for results
                 figs_lr_dir: str = 'LR',
                 figs_decomp_dir: str = 'mu_analysis',
                 figs_snn_dir: str = 'SNN',  # set the default directory for SNN figures

                 tr_weights_data_dir: str ='tr_weights',
                 snn_temp_data_dir: str = 'snn_data',
                 convreg_temp_data_dir: str= 'regression_data',

                 subj_map: dict = field(default_factory=lambda: {}),
                
                # Task
                 f_samp: int = 10240,
                 task_type: str = 'Trap',
                 finger_type: str = 'Individual',
                 day: str = 'Day 1',
                 wrist_pos: int = 0,
                 mvc: int = 15,
                 load_multi: list = [],
                
                 _sign_mvc = 1,
                 _direction = 'flex',
                 ramp_mvc: int = 5,
                 rest_dur: int = 1,
                 ramp_dur: int = 3,
                 hold_dur: int = 20,
                 tot_dur: int = field(init=False),
                 reps: int = 2,
                 rep_id: int = 0,
                 start_time: int = 0,
                 end_time: int = -1,
                 freq1: str = '01',
                 freq2: str = '02',

                # Decomposition
                 n_iter: int = 100,
                 sil: float = 0.85,
                 extension_factor: int = 50,
                 segment_hold: bool = False,

                # Electrode
                 electrodes: list = default_electrodes(),
                 electrode_name: str = 'ext_digitorum',
                 n_electrodes: int = field(init=False),

                # Finger
                 n_ind_fingers:int=field(init=False),
                 finger_label_map: dict = field(init=False),
                 fing_id: int = 1,

                # Force
                 time_to_cut: int = 1,   # seconds that precede the start of the ramp

                # Processing
                 window_size_in_seconds: float = 0.5,
                 max_fr: float = 50.0,
                 min_fr: float = 4.0,
                 cov_cut_off: float = 50.0,

                # Plots
                 dpi: int = 150,
                 font_size: int = 10,
                 axisbelow: bool = True,
                 color_dict: dict = default_color_dict(),
                 figsize: tuple = (15, 20),
                 finger_color_list: list =field(init=False),

                # Directories
                 figs_path: str = field(init=False),
                 results_path: str = field(init=False),
                 figs_lr_path: str = field(init=False),
                 figs_decomp_path: str = field(init=False),
                 figs_snn_path: str = field(init=False),
                 snn_temp_data_path: str = field(init=False),
                 tr_weights_data_path: str = field(init=False),

                # Others
                 verbose: bool = False,
                 common_only: bool = False,
                 images: bool = True,
    ):
        
        # Private
        self._root_dir = root_dir
        self._root_results_dir = root_results_dir
        self._subject = subject
        self._day = day
        self._data_input_dir = data_input_dir
        self._task_type = task_type
        self._finger_type = finger_type
        self._mvc = mvc
        self._load_multi = load_multi if len(load_multi)>0 else [self.mvc]
        self._decomp_subfolder = decomp_subfolder #f'{decomp_type}' if emg_type == "intra" else f'{decomp_type}_surf'#'classification_bothreps' if emg_type == "intra" else 'classification_bothreps_surf'#'both_reps_unique' if emg_type == "intra" else 'both_reps_unique_surf'
        self._decomp_type = f'{decomp_subfolder}' if emg_type == "intra" else f'{decomp_subfolder}_surf'
        self._input_dir = input_dir
        self._emg_type = emg_type

        self._figs_snn_dir = figs_snn_dir
        self._results_dir = results_dir

        # Constants
        self.data_decomp_dir = data_decomp_dir
        self.decomp_dir = decomp_dir
        self.figs_dir =  figs_dir
        self.figs_lr_dir = figs_lr_dir
        self.figs_decomp_dir = figs_decomp_dir

        self.tr_weights_data_dir = tr_weights_data_dir
        self.snn_temp_data_dir = snn_temp_data_dir
        self.convreg_temp_data_dir = convreg_temp_data_dir

        self.subj_map = subj_map
        
        # Task
        self.f_samp = f_samp
        self.wrist_pos = wrist_pos
        self._sign_mvc = _sign_mvc
        self._direction = _direction
        self.ramp_mvc = ramp_mvc
        self.rest_dur = rest_dur
        self.ramp_dur = ramp_dur
        self.hold_dur = hold_dur
        self.tot_dur = tot_dur
        self.reps = reps
        self.rep_id = rep_id
        self.start_time = start_time
        self.end_time = end_time
        self.freq1 = freq1
        self.freq2 = freq2

        # Decomposition
        self.n_iter = n_iter
        self.sil = sil
        self.extension_factor = extension_factor
        self.segment_hold = segment_hold

        # Electrode
        self.electrodes = electrodes
        self.electrode_name = electrode_name
        self.n_electrodes = n_electrodes

        # Force sensors order
        self.first_extension_sensor_id = 0 # 0th sensor is the first extension sensor
        self.first_flexion_sensor_id = 5   # 5th sensor is the first flexion sensor
        self.n_extension_sensors = 5
        self.n_flexion_sensors = 5

        # Finger
        self.n_ind_fingers = n_ind_fingers
        self.finger_label_map = finger_label_map
        self.fing_id = fing_id

        # Force
        self.rep1_s: int = field(init=False)
        self.rep1_e: int = field(init=False)
        self.rep2_s_cut: int = field(init=False)
        self.rep2_e_cut: int = field(init=False)

        self.rep1_s_ramp_cut: int = field(init=False)
        self.rep1_e_ramp_cut: int = field(init=False)
        self.rep1_s_hold_cut: int = field(init=False)
        self.rep1_e_hold_cut: int = field(init=False)

        self.rep2_s_ramp_cut: int = field(init=False)
        self.rep2_e_ramp_cut: int = field(init=False)
        self.rep2_s_hold_cut: int = field(init=False)
        self.rep2_e_hold_cut: int = field(init=False)
        self.rep2_s_decomp: int = field(init=False)
        self.rep2_e_decomp: int = field(init=False)
        self.both_forces_cut: list = field(init=False)
        self.both_forces_uncut: list = field(init=False)
        self.target_force: list = field(init=False)
        self.time_removed: int = field(init=False)

        self.time_to_cut = time_to_cut   # seconds that precede the start of the ramp

        # Processing
        self.window_size_in_seconds = window_size_in_seconds
        self.max_fr = max_fr
        self.min_fr = min_fr
        self.cov_cut_off = cov_cut_off

        # Plots
        self.dpi = dpi
        self.font_size = font_size
        self.axisbelow = axisbelow
        self.color_dict = color_dict
        self.figsize = figsize
        self.finger_color_list = finger_color_list

        # Directories
        self.figs_path = figs_path
        self.figs_lr_path = figs_lr_path
        self.figs_decomp_path = figs_decomp_path
        self.figs_snn_path = figs_snn_path
        self.snn_temp_data_path = snn_temp_data_path
        self.tr_weights_data_path = tr_weights_data_path
        self.results_path = results_path

        # Others
        self.verbose = verbose
        self.common_only = common_only
        self.images = images

        self.__post_init__()

    def __post_init__(self):
        self.define_paths()
        self.initialize_file_directories()
        self.initialize_directories()
        self.initialize_task_duration()
        self.initialize_electrode_count()
        self.initialize_finger_labels()
        self.initialize_plots_config()
    

    @property
    def emg_type(self):
        return self._emg_type
    
    @emg_type.setter
    def emg_type(self, value):
        self._emg_type = value
        self._decomp_type =  f'{self._decomp_subfolder}' if value=='intra'  else f'{self._decomp_subfolder}_surf'#'classification_bothreps' if value == "intra" else 'classification_bothreps_surf' #'both_reps_unique' if value == "intra" else "both_reps_unique_surf"
        self._update_input_dir()
        self._update_decomp_dir()
        self._update_paths()
    
    @property
    def sign_mvc(self):
        return self._sign_mvc
    
    @sign_mvc.setter
    def sign_mvc(self, value):
        self._sign_mvc = value
        self._direction = 'ext' if value == -1 else 'flex'

        self._update_input_dir()
        self._update_decomp_dir()
        self._update_paths()

    @property
    def direction(self):
        return self._direction
    
    @direction.setter
    def direction(self, value):
        self._direction = value
        self._sign_mvc = -1 if value == 'ext' else 1
        self._update_input_dir()
        self._update_paths()

    @property
    def load_multi(self):
        return self._load_multi
    
    @load_multi.setter
    def load_multi(self, value):
        self._load_multi = value
        # self._update_load_multi()
    
    @property
    def root_results_dir(self):
        return self._root_results_dir

    @root_results_dir.setter
    def root_results_dir(self, value):
        self.root_results_dir = value
        self._update_paths()

    @property
    def root_dir(self):
        return self._root_dir

    @root_dir.setter
    def root_dir(self, value):
        self._root_dir = value
        self._update_input_dir()
        self._update_decomp_dir()
        self._update_paths()


    @property
    def subject(self):
        return self._subject

    @subject.setter
    def subject(self, value):
        self._subject = value
        self._update_input_dir()
        self._update_decomp_dir()
        self._update_paths()

    @property
    def day(self):
        return self._day

    @day.setter
    def day(self, value):
        self._day = value
        self._update_input_dir()
        self._update_decomp_dir()
        self._update_paths()

    @property
    def data_input_dir(self):
        return self._data_input_dir

    @data_input_dir.setter
    def data_input_dir(self, value):
        self._data_input_dir = value
        self._update_input_dir()
        self._update_decomp_dir()
        self._update_paths()

    @property
    def figs_snn_dir(self):
        return self._figs_snn_dir
    
    @figs_snn_dir.setter
    def figs_snn_dir(self, value):
        self._figs_snn_dir = value
        self.figs_snn_path = os.path.join(self.figs_path, self.figs_snn_dir).replace("\\", "/")
        self.create_directory(self.figs_snn_path)

    @property
    def results_dir(self):
        return self._results_dir
    
    @results_dir.setter
    def results_dir(self, value):
        self._results_dir = value
        self.results_path = os.path.join(self.results_dir, fn.remap_subject(self.subject, self.subj_map)).replace("\\", "/")
        self.create_directory(self.results_path)

    @property
    def task_type(self):
        return self._task_type

    @task_type.setter
    def task_type(self, value):
        self._task_type = value
        self._update_input_dir()
        self._update_decomp_dir()
        self._update_paths()

    @property
    def finger_type(self):
        return self._finger_type

    @finger_type.setter
    def finger_type(self, value):
        self._finger_type = value
        self._update_input_dir()
        self._update_decomp_dir()
        self._update_paths()

    @property
    def mvc(self):
        return self._mvc

    @mvc.setter
    def mvc(self, value):
        self._mvc = value
        self.initialize_task_duration()
        self._update_input_dir()
        self._update_decomp_dir()
        self._update_paths()

    @property
    def input_dir(self):
        return self._input_dir

    def _update_input_dir(self):
        # Update the input_dir based on the current values of the attributes
        self._input_dir = os.path.join(
            self.root_dir, self.subject, self.day, self.data_input_dir,
            self.task_type,  str(self.mvc)
        ).replace("\\", "/")

    def _update_decomp_dir(self):
        # Update the input_dir based on the current values of the attributes
        self.decomp_dir = os.path.join(self.root_dir, self.subject, self.day, self.data_decomp_dir,
                                           self.task_type, str(self.mvc),).replace(
                "\\", "/")

    def _update_paths(self):
        self.figs_path = os.path.join(self.root_results_dir, self.figs_dir,
                                      fn.remap_subject(self.subject, self.subj_map)).replace("\\", "/")
        self.figs_lr_path = os.path.join(self.figs_path, self.figs_lr_dir).replace("\\", "/")
        self.figs_decomp_path = os.path.join(self.figs_path, self.figs_decomp_dir).replace("\\", "/")
        self.figs_snn_path = os.path.join(self.figs_path, self.figs_snn_dir).replace("\\", "/")

        self.results_path = os.path.join(self.root_results_dir,self.results_dir, fn.remap_subject(self.subject, self.subj_map)).replace("\\", "/")
        self.snn_temp_data_path = os.path.join(self.root_results_dir,self.snn_temp_data_dir,fn.remap_subject(self.subject, self.subj_map))
        self.convreg_temp_data_path = os.path.join(self.root_results_dir, self.convreg_temp_data_dir,fn.remap_subject(self.subject, self.subj_map))
        self.tr_weights_data_path  = os.path.join(self.snn_temp_data_path, self.tr_weights_data_dir)
    
    def _update_load_multi(self):
        self.load_multi = [self.mvc]

  
    def initialize_file_directories(self):
        """Create necessary directories based on configuration."""
        self._update_input_dir()
        self._update_decomp_dir()
        

    def define_paths(self):
        """Define the file paths used in the configuration."""
        self._update_paths()

    def initialize_directories(self):
        """Create necessary directories based on configuration."""
        directories = [self.figs_path, self.figs_lr_path, self.results_path, self.figs_snn_path, self.figs_decomp_path, self.snn_temp_data_path,self.convreg_temp_data_path, self.tr_weights_data_path]
        for directory in directories:
            self.create_directory(directory)

    def initialize_task_duration(self):
        """Calculate total duration based on task configuration."""
        self.ramp_dur = self.mvc // 5
        self.hold_dur = self.calculate_hold_duration()
        self.tot_dur = self.ramp_dur * 2 + self.hold_dur + 2 * self.rest_dur

    def initialize_electrode_count(self):
        """Initialize the count of electrodes."""
        self.n_electrodes = len(self.electrodes)

    def initialize_finger_labels(self):
        """Initialize finger label mapping."""
        self.finger_label_map = self.get_fingers()
        self.n_ind_fingers = len(self.finger_label_map)
        self.force_cols_list = [f'force_{fing_name}'for fing_name in list(self.finger_label_map.keys())]

        self.finger_color_list = ['#EAB543',self.color_dict['naval'],  '#ff793f',
                                 self.color_dict['wisteria'], self.color_dict['green_sea']][:self.n_ind_fingers]
                                          


    def initialize_plots_config(self):
        """Set up matplotlib plot configurations."""
        plt.rcParams.update({
            "figure.dpi": self.dpi, 'font.size': self.font_size,
            'figure.figsize': self.figsize, 'axes.axisbelow': self.axisbelow,
            'axes.edgecolor': self.color_dict['clouds'], 'axes.linewidth': 0.4
        })

    def task_name(self):
        if self.task_type == 'Trap':
            task_name = (f"{self.task_type}_{self.wrist_pos}_{self.sign_mvc * int(self.mvc)}_"
                         f"{self.fing_id}_{self.rest_dur}_{self.ramp_dur}_{self.hold_dur}")
        elif self.task_type == 'Pseudo':
            task_name = (
                f"Pseudorandom_{self.wrist_pos}_{self.sign_mvc * int(self.mvc)}_{self.freq1}_"
                f"{self.freq2}_{self.fing_id}_{self.hold_dur}")
        else:
            raise ValueError(f'Invalid task type {self.task_type}')
        return task_name

    @staticmethod
    def create_directory(path: str):
        """Create a directory if it does not exist."""
        if not os.path.exists(path):
            os.makedirs(path)
            logging.info('Output directory %s created', path)

        else:
            logging.warning('Output directory %s already exists', path)

    def get_fingers(self) -> dict:
        """Return a map of fingers based on task type and MVC."""
        if self.task_type == 'Trap' and self.mvc in [20, 30, 50]:
            return {k: v for k, v in default_finger_map().items() if v <= 3}
        return default_finger_map()

    def calculate_hold_duration(self) -> int:
        """Calculate hold duration based on task and finger type."""
        if self.task_type == 'Trap':
            if self.finger_type == 'Individual':
                return 20 if self.mvc in [15, 5] else 10 if self.mvc in [20, 30, 50] else self.raise_invalid_mvc()
            elif self.finger_type == 'Combinations':
                return 25 if self.mvc == 15 else self.raise_invalid_mvc()
        elif self.task_type == 'Pseudo':
            return 40
        else:
            self.raise_invalid_mvc()

    def raise_invalid_mvc(self):
        """Raise an error for invalid MVC values."""
        raise ValueError(f'Invalid MVC value {self.mvc}')
    
    def reset_sign_emg_type(self, sign_mvc: int, emg_type:str):
        """
        Resets the mvc sign and emg type in the config object.
        """
        self.sign_mvc = sign_mvc
        if emg_type:
            self.emg_type = emg_type
