import os
import logging
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from force_regression.data.loaders.emg import load_matfile
import force_regression.utils.functions as fn
from force_regression.config.dataconfig import DataConfig
from force_regression.config.nomenclature import *


@dataclass
class TaskLandmarks():
    """
    Defines 4 points of interest in the force profile.
    """
    start_rise: int
    end_rise: int
    start_fall: int
    end_fall: int


    def to_list(self):
        """
        Returns the points of interest as a list.
        """
        return [self.start_rise, self.end_rise, self.start_fall, self.end_fall]



def identify_landmarks(target, task_type: str = 'Trap',
                       threshold_ratio: float = 0.7)->Dict[str, TaskLandmarks]|Dict[str, list]:
    """
    Use the target force profile to identify 4 points of interest in samples:
        start_rise, end_rise (i.e start plateau), start_fall (i.e end plateau), end_fall
    Those points are extracted relying on the derivative of the target trace.
    """

    # Initialize an empty dictionary to store points of interest for all trapezoids
    all_trapezoids = {}

    if task_type == TRAP:
        # Calculate the derivative of the target
        derivative = np.diff(target)

        # Determine a threshold for significant changes
        max_derivative = np.max(np.abs(derivative))
        threshold = max_derivative * threshold_ratio

        # Start index for searching new trapezoid
        search_start_index = 0
        trapezoid_count = 0

        while search_start_index < len(derivative):
            # Find where the derivative is significantly positive (rising edge)
            rising_points = np.where(derivative[search_start_index:] > threshold)[0]
            if len(rising_points) == 0:
                break  # No more trapezoids found
            start_rise = rising_points[0] + search_start_index

            # Find the end of the rise (start of the plateau)
            # Maximum point in the target as the start of the plateau
            end_rise = np.argmax(target[start_rise:]) + start_rise

            # Find the start of the fall (end of the plateau)
            # Point where derivative becomes significantly negative after the plateau
            falling_points = np.where(derivative[end_rise:] < -threshold)[0]
            if len(falling_points) == 0:
                break  # No falling edge found for the current trapezoid
            start_fall = falling_points[0] + end_rise

            # Find the end of the fall (where the force goes to zero)
            end_fall = np.where(target[start_fall:] <= np.finfo(float).eps)[0]
            if len(end_fall) == 0:
                break  # No end of fall found
            end_fall = end_fall[0] + start_fall

            trapezoid_count += 1
            all_trapezoids[f'rep{trapezoid_count}'] = TaskLandmarks(start_rise, end_rise, start_fall, end_fall)
            search_start_index = end_fall
            assert target[all_trapezoids[f'rep{trapezoid_count}'].start_rise] == 0, "Start of trapezoid is not zero"
            assert target[all_trapezoids[f'rep{trapezoid_count}'].end_fall] == 0, "End of trapezoid is not zero"
            assert target[all_trapezoids[f'rep{trapezoid_count}'].end_rise] == np.max(target), "Start of plateau is not max"
            assert target[all_trapezoids[f'rep{trapezoid_count}'].start_fall] == np.max(target), "End of plateau is not max"
        return all_trapezoids

    all_trapezoids['rep1'] = [0, len(target) - 1]
    return all_trapezoids


def is_extension_task(filename:str):
    """
    Checks if the task is an extension task based on the filename.
    if the task is an extension task, the MVC value is negative.
    Otherwise, it is a flexion task.
    """
    mvc = filename.split('_')[2]
    if int(mvc) < 0:
        return True
    else:
        return False


def get_repetitions_for_task(task_type:str):
    """
    Returns list of repetitions based on the task type.
    """
    if task_type == TRAP:
        rep_ids = [FIRST_REP, SECOND_REP]
    elif task_type == PSEUDO:
        rep_ids = [FIRST_REP]
    else:
        raise ValueError(f'Invalid task type {task_type}')

    return rep_ids

def remove_force_baseline(config:DataConfig, forces:np.ndarray, ):
    """
    Removes the baseline from the force trajectory.
    """
    baseline_samples = 500
    if config.task_type == TRAP: # define baseline based on the first 500 samples of the force trajectory
        baseline = forces[:, :baseline_samples]
        baseline_mean = np.mean(baseline, axis=1).reshape(-1, 1)
    else:
        baseline_mean = np.array(fn.get_baselines_trap_for_pseudo(config)).reshape(-1, 1)
    return forces - baseline_mean

def normalize_force_to_percentage_mvc(config:DataConfig, emg_force_dict:Dict[str, Any]):
    """
    Scales the forces to the MVC level.
    """
    forces = emg_force_dict[EMG]['forces']
    forces = remove_force_baseline(config, forces)
    mvcs = fn.recorded_mvc_per_sensor(config)
    forces_in_mvc = forces / mvcs * 100

    return forces_in_mvc

    # baseline = forces[:, :500]  # taking few samples to calculate the baseline activity
    # baseline_mean = np.mean(baseline, axis=1).reshape(-1, 1)
    # mvcs = fn.recorded_mvc_per_sensor(config)
    # scaled_forces = (forces - baseline_mean) / mvcs * 100


def extract_force_target(rep, mat, force_id):
    force_rep = np.array(mat['file'][f'rep{rep}']['data']['force']['scaled_force'][force_id - 1, :])
    target_rep = np.array(mat['file'][f'rep{rep}']['data']['force']['target_force'])
    return force_rep, target_rep

def samples_to_time(samples:int, config:DataConfig):
    """
    Converts samples tp time
    """
    return samples/config.f_samp

def load_data(config:DataConfig, load_emg:bool=False):
    """
    Loads force sensor data: both the target and recorded force
    """
    file_directory = construct_filepath_for_task(config)
    data_dict = load_matfile(file_directory)
    forces_in_perc_mvc = normalize_force_to_percentage_mvc(config, data_dict)
    target_path = data_dict[EMG][TARGET]
    electrode_columns = [f'intra_{name}' for name in config.electrodes] + [f'surf_{name}' for name in config.electrodes]
    emg_dict = None
    if load_emg:
        emg_dict = {electrode: data_dict[EMG][electrode] for electrode in electrode_columns}
    return target_path, forces_in_perc_mvc, emg_dict


def load_data_and_segment_force(config:DataConfig,
                                return_force_for_single_sensor:bool=False):
    """
    Loads the force data.
    Note that the target path is not the same for all files
    """
    target_path, forces_in_perc_mvc, emg_dict = load_data(config,load_emg=True)
    one_indexed_force_id = get_force_sensor_id(config)
    logging.info("Loading sensor: %s in %s (%s samples)", one_indexed_force_id, config.direction, forces_in_perc_mvc.shape[1])
    recorded_force_for_sensor = forces_in_perc_mvc[one_indexed_force_id - 1, :]
    force_dict_per_rep = segment_and_identify_force_landmarks(config,
                                                              target_path,
                                                              recorded_force_for_sensor)
    force_for_sensor_reshaped = recorded_force_for_sensor.reshape(1, -1)
    if return_force_for_single_sensor:
        return force_dict_per_rep, force_for_sensor_reshaped, emg_dict
    return force_dict_per_rep, forces_in_perc_mvc, emg_dict


def segment_and_identify_force_landmarks(config:DataConfig,
                                   target_path:np.ndarray,
                                   actual_performed_path:np.ndarray)->Dict[str, Dict[str,Any]]:
    """
    Identifies landmarks of the trapezoid force profile and segment the force profile create a force_dict with each repetition cut.
    to only include time_to_cut around each repetition:
        - stage 1: crop from rep1_s to rep1_e and rep2_s to rep2_e. This are on the global time axis
        - stage 2: stitch these two reps back together such that now rep2 (1 sec prior to its rising) starts at rep1_e
            stage 2 is crucial since the spike times
    Careful! this function is not pure and updates the config object simultaneously.
    """

    force_dict = {FIRST_REP: {}, SECOND_REP: {}}
    reps_global_landmarks = identify_landmarks(target_path, config.task_type)
    n_repetitions = len(reps_global_landmarks.keys())
    init_reps_start_end(config, actual_performed_path,reps_global_landmarks)
    # print(f" Start rise:{reps_global_landmarks[FIRST_REP].start_rise/config.f_samp}, end fall: {reps_global_landmarks[FIRST_REP].end_fall/config.f_samp}")
    # print(f" Initialize start and end for reps: {config.rep1_s}, {config.rep1_e}, {config.rep2_s}, {config.rep2_e}")
    config.rep1_force_uncut = create_force_target_landmark_dict(actual_performed_path,
                                                                target_path,
                                                                reps_global_landmarks[FIRST_REP])
    
    # update the points such that the new zero is the first sample in the cut performed_path
    rep1_landmarks_relative_to_rep1 = shift_landmark(reps_global_landmarks[FIRST_REP],
                                                     config.rep1_s,
                                                     shift_sign='negative')
    config.rep1_force_cut = create_force_target_landmark_dict(actual_performed_path[config.rep1_s:config.rep1_e],
                                                              target_path[config.rep1_s:config.rep1_e],
                                                              rep1_landmarks_relative_to_rep1)
    force_dict[FIRST_REP] = config.rep1_force_cut
    config.rep1_force_samples = len(force_dict[FIRST_REP][FORCE])
    initialize_force_config_for_both_reps(config, config.rep1_force_uncut, config.rep1_force_cut)
    if config.task_type == TRAP:
        rep2_landmarks_relative_to_rep1 = None
        if n_repetitions > 1:
            config.rep2_force_uncut = create_force_target_landmark_dict(actual_performed_path,
                                                                        target_path,
                                                                        reps_global_landmarks[SECOND_REP])
        
            rep2_landmarks_relative_to_rep2 = shift_landmark(reps_global_landmarks[SECOND_REP],
                                                             config.rep2_s,
                                                             shift_sign='negative')
            config.rep2_force_cut = create_force_target_landmark_dict(actual_performed_path[config.rep2_s:config.rep2_e],
                                                                      target_path[config.rep2_s:config.rep2_e],
                                                                      rep2_landmarks_relative_to_rep2)

            force_dict[SECOND_REP] = config.rep2_force_cut
            config.rep2_force_samples = len(force_dict[SECOND_REP][FORCE])
            rep2_landmarks_relative_to_rep1 = shift_landmark(config.rep2_force_cut[LANDMARK],
                                                             len(config.rep1_force_cut[FORCE]),
                                                             shift_sign='positive')
            stitch_reps_force_target_landmark(config, rep2_landmarks_relative_to_rep1)
            set_total_time_cut(config)
        set_ramp_hold_cut_landmarks(config, rep1_landmarks_relative_to_rep1, rep2_landmarks_relative_to_rep1)
    else:
        config.rep1_s_ramp_cut, config.rep1_e_ramp_cut = 0, 0
        config.rep1_s_hold_cut, config.rep1_e_hold_cut = 0, 0
        config.rep2_force_cut = {FORCE: actual_performed_path[config.rep2_s:config.rep2_e],
                                 TARGET: target_path[config.rep2_s:config.rep2_e],
                                 LANDMARK: []}

        force_dict[SECOND_REP] = config.rep2_force_cut
        config.rep2_force_samples = len(force_dict[SECOND_REP][FORCE])
        config.both_forces_cut = concatenate_rep1_rep2_data(config.rep1_force_cut[FORCE],
                                                            config.rep2_force_cut[FORCE])
        config.target_force_cut = concatenate_rep1_rep2_data(config.rep1_force_cut[TARGET],
                                                             config.rep2_force_cut[TARGET])
    return force_dict


def stitch_reps_force_target_landmark(config:DataConfig,
                                      rep2_landmarks_relative_to_rep1:TaskLandmarks):
    """
    Stitches the two repetitions back together after cutting
    """
    config.points_uncut  = concatenate_rep1_rep2_data(config.rep1_force_uncut[LANDMARK].to_list(),
                                                              config.rep2_force_uncut[LANDMARK].to_list())
    config.both_forces_cut = concatenate_rep1_rep2_data(config.rep1_force_cut[FORCE],
                                                                config.rep2_force_cut[FORCE])
    config.target_force_cut = concatenate_rep1_rep2_data(config.rep1_force_cut[TARGET],
                                                                 config.rep2_force_cut[TARGET])
            # Now that we are stitching both reps back together after cutting, we consider the new end of rep1
    config.points_cut = concatenate_rep1_rep2_data(config.rep1_force_cut[LANDMARK].to_list(),
                                                           rep2_landmarks_relative_to_rep1.to_list())

def set_ramp_hold_cut_landmarks(config:DataConfig, rep1_landmarks:TaskLandmarks,
                                rep2_landmarks:TaskLandmarks):
    """
    Sets the ramp and hold points for the first and second reps.
    """
    config.rep1_s_ramp_cut = rep1_landmarks.start_rise
    config.rep1_e_ramp_cut = rep1_landmarks.end_fall
    config.rep1_s_hold_cut = rep1_landmarks.end_rise
    config.rep1_e_hold_cut = rep1_landmarks.start_fall
    if rep2_landmarks is not None:
        config.rep2_s_ramp_cut = rep2_landmarks.start_rise
        config.rep2_e_ramp_cut = rep2_landmarks.end_fall
        config.rep2_s_hold_cut = rep2_landmarks.end_rise
        config.rep2_e_hold_cut = rep2_landmarks.start_fall


def initialize_force_config_for_both_reps(config:DataConfig,
                                          uncut_dict:Dict[str, Any],
                                          cut_dict:Dict[str, Any]):
    """
    Initializes both_forces_cut, target_force_cut, points_cut,
    both_forces_uncut, target_force_uncut, points_uncut in the config object.
    """
    config.both_forces_cut = cut_dict[FORCE]#config.rep1_force_cut[FORCE]
    config.target_force_cut = cut_dict[TARGET]#config.rep1_force_cut[TARGET]
    config.points_cut = cut_dict[LANDMARK]#config.rep1_force_cut[LANDMARK]

    config.both_forces_uncut = uncut_dict[FORCE]#config.rep1_force_uncut[FORCE]
    config.target_force_uncut = uncut_dict[TARGET] #config.rep1_force_uncut[TARGET]
    config.points_uncut = uncut_dict[LANDMARK]#config.rep1_force_uncut[LANDMARK]

def set_total_time_cut(config:DataConfig):
    """
    Calculates the total time removed from the task. 
    There are two parts removed: the initial time and the intra repetition time.
    """
    time_between_cut_reps = config.rep2_s - config.rep1_e
    config.time_removed = time_between_cut_reps + config.rep1_s
    assert (config.rep1_e - config.rep1_s) == config.rep2_s-config.time_removed , "Time removed is not correct"

def concatenate_rep1_rep2_data(rep1_data:np.ndarray, rep2_data:np.ndarray)->np.ndarray:
    """
    Concatenates the data from the two repetitions.
    """
    return np.concatenate((rep1_data, rep2_data))


def create_force_target_landmark_dict(performed_path:np.ndarray,
                                      target_path:np.ndarray,
                                      task_landmarks:TaskLandmarks)->Dict[str, Any]:
    """
    Creates a dictionary bundling the measured force (performed path), the 
    target trajectory to follow, and the landmarks of the task.
    The landmarks are relative to the start of the performed path.
    """
    return {FORCE: performed_path, TARGET: target_path, LANDMARK: task_landmarks}


def shift_landmark(points:TaskLandmarks, shift_samples:int, shift_sign:str)->TaskLandmarks:
    """
    Shifts the points of interest in the force profile by a given number of samples.
    """
    if shift_sign == 'positive':
        return TaskLandmarks(points.start_rise + shift_samples,
                             points.end_rise + shift_samples,
                             points.start_fall + shift_samples,
                             points.end_fall + shift_samples)
    if shift_sign == 'negative':
        return TaskLandmarks(points.start_rise - shift_samples,
                            points.end_rise - shift_samples,
                            points.start_fall - shift_samples,
                            points.end_fall - shift_samples)

def init_reps_start_end(config:DataConfig, performed_path:np.ndarray,
                        task_landmarks:dict[str,TaskLandmarks]):
    """
    Updates the start and end of repetition(s) for the task based on the task type.
    """
    buffer_samples = int(config.time_to_cut * config.f_samp)       # 1s in samples

    if config.task_type == TRAP:
        config.rep1_s = int(task_landmarks[FIRST_REP].start_rise - buffer_samples)
        config.rep1_e = int(task_landmarks[FIRST_REP].end_fall + buffer_samples)
    else:
        config.rep1_s = 0
        first_half_performed_path = int(performed_path.shape[0]/2)
        config.rep1_e = first_half_performed_path

    if SECOND_REP in task_landmarks:
        config.rep2_s = int(task_landmarks[SECOND_REP].start_rise - buffer_samples)
        config.rep2_e = int(task_landmarks[SECOND_REP].end_fall + buffer_samples)
    else:
        config.rep2_s = config.rep1_e
        config.rep2_e = performed_path.shape[0]


def get_force_sensor_id(config:DataConfig):
    """
    Returns the force sensor id based on the task name. Sensor ids are ordered as follows:
    1 - 5 for extension tasks and 6 - 10 for flexion tasks.
    Note that the returned force_id is 1-indexed
    """
    if is_extension_task(config.task_name()):
        one_indexed_force_id = config.fing_id + config.first_extension_sensor_id
    else:
        one_indexed_force_id = config.fing_id + config.first_flexion_sensor_id
    return one_indexed_force_id

def construct_filepath_for_task(config:DataConfig):
    """
    Constructs the file path for the task based on the task name.
    """
    # file_path = os.path.join(os.path.abspath(os.path.join(config.input_dir, os.pardir)),
    #                         config.task_name() + '.mat')
    rel_file_path = os.path.join(config.input_dir, config.task_name() + '.mat')
    file_path = os.path.abspath(rel_file_path)

    return file_path