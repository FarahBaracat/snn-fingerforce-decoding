from typing import Dict, Sequence
import torch
from force_regression.plotting.mu_plot import raster_before_after
from force_regression.data.preprocessing.duplicates import filter_abnormal_mus_data
from force_regression.config.dataconfig import DataConfig
from force_regression.config.nomenclature import *


def calculate_mean_fr(chunk_spike_times,
                      total_duration_start:float,
                      total_duration_end:float,
                      config:DataConfig):
    """
    Calculates mean firing rate based on the spike times
    """
    # Create window start and end times
    window_start_times = torch.arange(total_duration_start, total_duration_end,
                                      config.window_size_in_seconds).to(chunk_spike_times.device)
    window_end_times = (window_start_times + config.window_size_in_seconds).to(chunk_spike_times.device)

    # If there are no spikes, return a tensor of zeros for firing rates in each window
    if len(chunk_spike_times) == 0:
        firing_rates_in_hz = torch.zeros_like(window_start_times, dtype=torch.float)
    else:
        # Calculate spikes in each window
        spikes_in_windows = (chunk_spike_times[:, None] >= window_start_times) & (
                chunk_spike_times[:, None] < window_end_times)
        spikes_count_per_window = spikes_in_windows.sum(dim=0)
        firing_rates_in_hz = spikes_count_per_window.float() / config.window_size_in_seconds
    return firing_rates_in_hz.mean().round(decimals=2).cpu().item()

def mask_sptimes_for_roi(spiketimes_in_samples:list[torch.tensor],
                         config:DataConfig):
    """
    Masking all MUs into the region of interest defined by config.repi_s
    marking the beginning of the task already segment to buffer_time before start ramp
    """
    masked_sptimes = []
    if config.task_type == TRAP:
        first_rep_start = config.rep1_s / config.f_samp
        first_rep_end = config.rep1_e / config.f_samp
        second_rep_start = config.rep2_s / config.f_samp
        second_rep_end = config.rep2_e / config.f_samp

        for mu_sptimes_in_samples in spiketimes_in_samples:
            sp_times = mu_sptimes_in_samples.float() / config.f_samp
            rep1_sptimes, rep2_sptimes = segment_reps_sptimes_for_mu(first_rep_start, first_rep_end,
                                                                     second_rep_start, second_rep_end,
                                                                     sp_times)
            if len(rep1_sptimes) > 0 or len(rep2_sptimes) > 0:
                combined_sptimes = torch.cat(((rep1_sptimes * config.f_samp).int(),
                                              (rep2_sptimes * config.f_samp).int()))
                masked_sptimes.append(combined_sptimes)
            else:
                masked_sptimes.append(torch.tensor([], dtype=torch.int32))
    else:
        masked_sptimes = mu_sptimes_in_samples
    return masked_sptimes

def is_mu_abnormal(config:DataConfig,
                      rep1_mean_fr:float, rep1_sptimes:list[torch.tensor],
                      rep2_mean_fr:float, rep2_sptimes:list[torch.tensor])->bool:
    """
    Checks whether the MU in question is abnormal. This is identified 
    by checking on the mean firing rates of the MU for both repetitions
    """
    # Check for abnormal firing rates in each chunk.
    is_mu_abnormal_in_rep1 = rep1_mean_fr > 0 and (
                rep1_mean_fr > config.max_fr or rep1_mean_fr < config.min_fr)
    is_mu_abnormal_in_rep2 = rep2_mean_fr > 0 and (
                rep2_mean_fr > config.max_fr or rep2_mean_fr < config.min_fr)

    is_abnormal = False
    if is_mu_abnormal_in_rep1 and is_mu_abnormal_in_rep2:
        is_abnormal = True
    else:
        if is_mu_abnormal_in_rep1:
            rep1_sptimes = delete_sptimes_in_abnormal_rep()
            if len(rep2_sptimes) == 0:
                is_abnormal = True
        if is_mu_abnormal_in_rep2:
            rep2_sptimes = delete_sptimes_in_abnormal_rep()
            if len(rep1_sptimes) == 0:
                is_abnormal = True
    return is_abnormal, rep1_sptimes, rep2_sptimes


def delete_sptimes_in_abnormal_rep():
    """
    Deletes all sp times in rep
    """
    return torch.tensor([])

def convert_sptimes_to_samples(config:DataConfig, rep_sptimes:list[torch.tensor]):
    """
    Converts the list of spike times in second to a list in samples
    """
    rep_sptimes_in_samples = rep_sptimes
    if len(rep_sptimes)>0:
        rep_sptimes_in_samples = (rep_sptimes * config.f_samp).int()
    return rep_sptimes_in_samples

def identify_abdnormal_units_and_filter_sptimes(spiketimes_in_samples:list[torch.tensor],
                                     config:DataConfig):
    """
    In this function, we calculate the mean firing rate for each unit in two separate time chunks and identify the units 
    that have abnormal firing rates in either of the chunks.
    
    valid_mu_sptimes:are the filtered spike times deleting spike times of MUs for one or both repetitions in case
    the MU had abnormal activity.
    """
    fr_for_reps = []
    valid_mus_sptimes = []
    abnormal_mu_ids = []

    if config.task_type == TRAP:
        # config.rep1_s is the beginning of the task already segment to buffer_time before start ramp
        first_rep_start = config.rep1_s / config.f_samp
        first_rep_end = config.rep1_e / config.f_samp

        second_rep_start = config.rep2_s / config.f_samp
        second_rep_end = config.rep2_e / config.f_samp

        for mu_id, mu_sptimes_in_samples in enumerate(spiketimes_in_samples):
            sp_times = mu_sptimes_in_samples.float() / config.f_samp
            rep1_sptimes, rep2_sptimes = segment_reps_sptimes_for_mu(first_rep_start, first_rep_end,
                                                                  second_rep_start, second_rep_end,
                                                                  sp_times)
            rep1_mean_fr = calculate_mean_fr(rep1_sptimes,
                                             first_rep_start, first_rep_end,
                                             config=config)
            rep2_mean_fr = calculate_mean_fr(rep2_sptimes,
                                             second_rep_start, second_rep_end,
                                             config=config)

            fr_for_reps.append((rep1_mean_fr, rep2_mean_fr))
            is_abnormal, rep1_sptimes, rep2_sptimes = is_mu_abnormal(config,
                                                                     rep1_mean_fr,
                                                                     rep1_sptimes,
                                                                     rep2_mean_fr,
                                                                     rep2_sptimes)
            if is_abnormal:
                abnormal_mu_ids.append(mu_id)
            rep1_sptimes_in_samples = convert_sptimes_to_samples(config,rep1_sptimes)
            rep2_sptimes_in_samples = convert_sptimes_to_samples(config, rep2_sptimes)
            valid_mus_sptimes.append(torch.cat((rep1_sptimes_in_samples,
                                               rep2_sptimes_in_samples)))

    else:
        config.window_size_in_seconds = 2
        for mu_id, mu_sptimes_in_samples in enumerate(spiketimes_in_samples):
            sp_times = mu_sptimes_in_samples.float() / config.f_samp
            mean_fr_whole = calculate_mean_fr(sp_times,
                                              total_duration_start=0,
                                              total_duration_end=config.rep1_force_samples,
                                              config=config)
            fr_for_reps.append(mean_fr_whole)
            valid_mus_sptimes.append(mu_sptimes_in_samples)
    return abnormal_mu_ids, fr_for_reps, valid_mus_sptimes

def segment_reps_sptimes_for_mu(first_rep_start:float, first_rep_end:float,
                             second_rep_start:float, second_rep_end:float,
                             sp_times:list[torch.tensor]):
    """
    Segments the spike times into the spike times for each rep.
    """
    first_rep_mask = (first_rep_start < sp_times) & (sp_times <= first_rep_end)
    second_rep_mask = (second_rep_start < sp_times) & (sp_times <= second_rep_end)

    first_rep_sptimes = sp_times[first_rep_mask]
    second_rep_sptimes = sp_times[second_rep_mask]
    return first_rep_sptimes, second_rep_sptimes


def post_process_units(mu_dict: Dict, config:DataConfig):
    """
    Post-process MUs by:
        1. removing units with very high (greater than config.max_fr) or very low
    (less than config.min_fr) firing rates.
        2. Masking the timestamps of the cleaned MUs given region of interest and stitch the repetitions back together.
        This process shifts the time axis to this new time axis where 0 corresponds to config.rep1_s
    """
    if MU_TIMESTAMP not in mu_dict.keys():
        mu_dict[MU_TIMESTAMP] = []
    spiketimes_in_samples = mu_dict[MU_TIMESTAMP]
    masked_sptimes = mask_sptimes_for_roi(spiketimes_in_samples, config)
    abnormal_mu_ids, firing_rates, partially_filt_spiketimes = identify_abdnormal_units_and_filter_sptimes(mu_dict[MU_TIMESTAMP],
                                                                                       config)
    mu_dict[MU_TIMESTAMP] = partially_filt_spiketimes
    filtered_mu_dict = filter_abnormal_mus_data(mu_dict, abnormal_mu_ids)

    if config.verbose:
        print("---------------------------------------------")
        print(f"Found {len(abnormal_mu_ids)} units with abnormal firing rates.")

    if abnormal_mu_ids:
        for idx in abnormal_mu_ids:
            fr = firing_rates[idx]
            fr_str = ", ".join(f"fr rep {i + 1}: {rate:.2f} Hz" for i, rate in enumerate(fr))
            if config.verbose:
                print(f"Unit {idx} {fr_str}. Removing...")
        print("---------------------------------------------")
    if config.task_type == TRAP: #CHECKTHIS: why do we need to remove abnormal as well
        filtered_mu_dict[FR_FIRST_REP] = [fr[0] for j, fr in enumerate(firing_rates) if j not in abnormal_mu_ids]
        filtered_mu_dict[FR_SECOND_REP] = [fr[1] for j, fr in enumerate(firing_rates) if j not in abnormal_mu_ids]
    else:
        # FIXME: added the same values across the two reps for now for the Pseudo case
        filtered_mu_dict[FR_FIRST_REP] = [fr for j, fr in enumerate(firing_rates) if j not in abnormal_mu_ids]
        filtered_mu_dict[FR_SECOND_REP] = [fr for j, fr in enumerate(firing_rates) if j not in abnormal_mu_ids]
        
    filtered_mus_and_spiketimes = filtered_mu_dict[MU_TIMESTAMP]
    if config.images:
        raster_before_after(masked_sptimes, filtered_mus_and_spiketimes, config, cut=False)
    #CHECKTHIS: masked_sptimes and combined_reps_unshifted are the same
    # _, _, _, combined_reps_unshifted = shift_timestamps_reference(masked_sptimes, config)
    rep1_units_shifted, rep2_units_shifted, shifted_unit_for_reps = shift_time_reference_for_mu(filtered_mus_and_spiketimes,
                                                                                                config)
    if config.images:
        if config.task_type == TRAP:
            raster_before_after(masked_sptimes, filtered_mus_and_spiketimes, config, cut=True)
    return filtered_mu_dict, rep1_units_shifted, rep2_units_shifted, shifted_unit_for_reps


def shift_time_reference_for_mu(spiketimes_in_samples:list[torch.tensor],
                                config:DataConfig):
    """
    Apply mask to retreive spike times in the region of interest (between repi_s and repi_e). 
    Then shift the timestamps such that they start with respect to the new axis (ie. the new zero is now rep1_s)
    """
    if config.task_type == TRAP: #and config._decomp_subfolder in config._decomp_type:#CHECKTHIS 'both_reps_unique' in configuration.decomp_type:
        #CHECKTHIS: this part seems redundant the MUs were already masked in
        # segment_reps_sptimes_for_mu
        rep1_units = [segment_reps_sptimes_for_mu(config.rep1_s, config.rep1_e,
                                                             config.rep2_s, config.rep2_e,
                                                             unit)[0] for unit in spiketimes_in_samples]
        rep2_units = [segment_reps_sptimes_for_mu(config.rep1_s, config.rep1_e,
                                                             config.rep2_s, config.rep2_e,
                                                             unit)[1] for unit in spiketimes_in_samples]
        # rep1_units = [unit[(unit >= config.rep1_s) & (unit <= config.rep1_e)].cpu() for unit in
        #               spiketimes_in_samples]
        rep1_units_shifted = [unit - config.rep1_s for unit in rep1_units]  # shifting the unit times so that the cut point starts at 0
        # rep2_units = [unit[(unit > config.rep2_s) & (unit <= config.rep2_e)].cpu() for unit in spiketimes_in_samples ]

        # rep2_s is given wrt. to the global time. Since we cut some region in between rep1_e and rep2_s, we want
        # to consider this time to correctly shift the spike times of rep2
        inter_rep_time_in_samples = config.rep2_s - config.rep1_e
        rep2_units_shifted = [unit - inter_rep_time_in_samples - config.rep1_s  for unit in rep2_units]   # removing the time in between the two reps and shifting the unit times by the same amount as rep1
        # reps_concatenated_unshifted =  [torch.cat([rep1_unit, rep2_unit], dim=0) for rep1_unit, rep2_unit in zip(rep1_units, rep2_units)]
        shifted_units_for_rep = [rep1_units_shifted, rep2_units_shifted]

    elif config.task_type == 'Pseudo':

        length_recording = int(config.both_forces_uncut.shape[0] / 2)
        rep1_units = [unit[(unit <= length_recording)].cpu() for unit in spiketimes_in_samples]

        rep1_units_shifted = [unit for unit in rep1_units]
        rep2_units = [unit[(unit > length_recording)].cpu() for unit in spiketimes_in_samples]
        rep2_units_shifted = [unit - length_recording for unit in rep2_units]
        #TODO: check the us e rep1_shifted in the following line
        # reps_concatenated_unshifted =  [torch.cat([unit_rep1, unit_rep2], dim=0) for unit_rep1, unit_rep2 in zip(rep1_units_shifted, rep2_units_shifted)]
        shifted_units_for_rep = [rep1_units_shifted, rep2_units_shifted]

    else:
        raise ValueError(f"Unknown task type: {config.task_type}")
    return rep1_units_shifted, rep2_units_shifted, shifted_units_for_rep


def post_process_units_tracking(data_rep1: Dict, data_rep2: Dict, configuration: Dict):

    # Rep1
    if 'timestamps' not in data_rep1.keys():
        data_rep1['timestamps'] = []

    abnormal_fr_indices_rep1, firing_rates_rep1, clean_ts_rep1, ts_before_clean_rep1 = identify_abdnormal_units_and_filter_sptimes(data_rep1['timestamps'],
                                                                                            config=configuration)

    data_rep1['timestamps'] = clean_ts_rep1
    data_rep1 = filter_abnormal_mus_data(data_rep1, abnormal_fr_indices_rep1)

    if configuration.verbose:
        print(f"Found {len(abnormal_fr_indices_rep1)} units with abnormal firing rates.")

    if abnormal_fr_indices_rep1:
        for idx in abnormal_fr_indices_rep1:
            fr = firing_rates_rep1[idx]
            fr_str = ", ".join(f"fr rep {i + 1}: {rate:.2f}" for i, rate in enumerate(fr))

            if configuration.verbose:
                print(f"Unit {idx} {fr_str}. Removing...")

    # Rep2 
    if 'timestamps' not in data_rep2.keys():
        data_rep2['timestamps'] = []
    
    abnormal_fr_indices_rep2, firing_rates_rep2, clean_ts_rep2, ts_before_clean_rep2 = identify_abdnormal_units_and_filter_sptimes(data_rep2['timestamps'],
                                                                                            config=configuration)

    data_rep2['timestamps'] = clean_ts_rep2
    data_rep2 = filter_abnormal_mus_data(data_rep2, abnormal_fr_indices_rep2)

    if configuration.verbose:
        print("---------------------------------------------")
        print(f"Found {len(abnormal_fr_indices_rep2)} units with abnormal firing rates.\n")

    if abnormal_fr_indices_rep2:
        for idx in abnormal_fr_indices_rep2:
            fr = firing_rates_rep2[idx]
            fr_str = ", ".join(f"fr rep {i + 1}: {rate:.2f}" for i, rate in enumerate(fr))

            if configuration.verbose:
                print(f"Unit {idx} {fr_str}. Removing...")
        print("---------------------------------------------")

    data_rep1['fr rep 1'] = [fr[0] for j, fr in enumerate(firing_rates_rep1) if j not in abnormal_fr_indices_rep1]
    data_rep2['fr rep 2'] = [fr[1] for j, fr in enumerate(firing_rates_rep2) if j not in abnormal_fr_indices_rep2]

    ts_after_clean_rep1 = data_rep1['timestamps']
    ts_after_clean_rep2 = data_rep2['timestamps']

    if configuration.images:
        raster_before_after(ts_before_clean_rep1, ts_after_clean_rep1, configuration, cut=False)
    
    # Cut the units
    units_rep1 = data_rep1.get('timestamps', []) 
    units_rep2 = data_rep2.get('timestamps', []) 

    # move units to cpu
    units_rep1 = [unit.cpu() for unit in units_rep1]
    units_rep2 = [unit.cpu() for unit in units_rep2]
    return data_rep1, data_rep2, units_rep1, units_rep2, [units_rep1, units_rep2]