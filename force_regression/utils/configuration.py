import logging
import json
import types
from pathlib import Path

from force_regression.config.dataconfig import DataConfig

import force_regression.utils.functions as fn
# Project root is three levels up from this file:
# force_regression/utils/configuration.py -> force_regression/utils/ -> force_regression/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_project_configuration(config_file):
    """
    Load the project configuration from a JSON file.
    Relative paths are resolved from the project root, not the CWD.
    """
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    with open(config_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
        data_root_dir = str(_PROJECT_ROOT / config['root_dir'])
        results_root_dir = str(_PROJECT_ROOT / config['root_results_dir'])
        subject_name_encoding = config['subject_mappings']
        wandb_entity = config['entity']
        return data_root_dir, results_root_dir, subject_name_encoding, wandb_entity


def load_input_data_configuration_from_omega(default_params, root_dir: str,
                                  results_root_dir: str,
                                  subject_mappings: dict[str, str],
                                  ):
    """Loads and updates the data configuration."""
    load_multi = default_params.task["mvc"]
    logging.info(
        "Loading data for subject %s with MVCs %r %r",
        default_params.task["subject"],
        load_multi,
        default_params.task["mvc"][0],
    )
    data_config = DataConfig(root_dir=root_dir,
                             root_results_dir=results_root_dir,
                             figs_dir="figures",
                             subject=fn.reverse_remap(default_params.task["subject"],
                                                      subject_mappings),
                             task_type="Trap",
                             finger_type="Individual",
                             day="Day 1",
                             mvc=default_params.task["mvc"][0],
                             emg_type="intra",
                             f_samp=10240,
                             subj_map=subject_mappings,
                             decomp_subfolder='classification_bothreps',
                             segment_hold=default_params.task["segment_hold"],
                             verbose=True,
                             common_only=True,
                             images=False,
                             time_to_cut=1,
                             load_multi=load_multi,
                             )
    # need to overwrite the direction in data_config
    data_config.direction = default_params.task["exp_dir"]
    return data_config

def load_input_data_configuration(default_params, root_dir: str,
                                  results_root_dir: str,
                                  segment_hold: bool,
                                  subject_mappings: dict[str, str],
                                  ):
    """Loads and updates the data configuration."""
    load_multi = default_params.mvc
    logging.info(
        "Loading data for subject %s with MVCs %r %r",
        default_params.subject,
        load_multi,
        default_params.mvc[0],
    )
    data_config = DataConfig(root_dir=root_dir,
                             root_results_dir=results_root_dir,
                             figs_dir="figures",
                             subject=fn.reverse_remap(default_params.subject,
                                                      subject_mappings),
                             task_type="Trap",
                             finger_type="Individual",
                             day="Day 1",
                             mvc=default_params.mvc[0],
                             emg_type="intra",
                             f_samp=10240,
                             subj_map=subject_mappings,
                             decomp_subfolder='classification_bothreps',
                             segment_hold=segment_hold,  # default is True
                             verbose=True,
                             common_only=True,
                             images=False,
                             time_to_cut=1,
                             load_multi=load_multi,
                             )
    # need to overwrite the direction in data_config
    data_config.direction = default_params.exp_dir
    return data_config

def update_finger_ids(data_config, snn_config):
    """
    Updates the finger ids in snn_config to match the finger ids in data_config.
    """
    snn_config.train_fing_id = [
        value - 1 for value in data_config.finger_label_map.values()]
    return snn_config


def flatten_hydra_config(cfg):
    """
    Flatten a nested Hydra DictConfig into a SimpleNamespace with flat attribute access.

    Required by functions that were designed for the older flat config style
    (e.g. SnnMusDatasetPreparation, get_kfold_reps).  Works transparently
    whether cfg comes from Hydra (DictConfig) or W&B (plain dict).

    Merge order: task → training → decoder_type → logging (excluding the
    nested wandb sub-dict).  training takes priority over logging for any
    duplicate keys (e.g. log_ts_freq appears in both).
    """
    flat = types.SimpleNamespace()
    for group in ('task', 'training', 'decoder_type'):
        for k, v in cfg[group].items():
            setattr(flat, k, v)
    already_set = vars(flat).keys()
    for k, v in cfg.logging.items():
        if k != 'wandb' and k not in already_set:
            setattr(flat, k, v)
    flat.use_wandb = cfg.logging['wandb']['use_wandb']
    return flat