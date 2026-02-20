import logging
import torch
import pandas as pd
from collections import defaultdict


def create_spike_omission_mask(unsegmented_input_flat:torch.Tensor, noise_level_in_percent:int):
    """
    Create a mask with the same shape as the unsegmented input tensor (n_samples, n_mus * n_time_steps)
    where the values are True for the spikes to be omitted and False for the spikes to be kept.
    The mask is created by counting the number of spikes in each sample and then randomly selecting
    the specified percentage of spikes to be omitted.
    The number of spikes to be omitted is calculated as a percentage of the total number of spikes
    """
    spike_count_df_flat = pd.DataFrame(torch.sum(unsegmented_input_flat == 1, dim=1))
    spike_count_df_flat['n_omitted_spikes'] = (spike_count_df_flat[0] * noise_level_in_percent / 100).astype(int)

    # Create a mask for the spikes to be omitted in each sample
    mask = torch.zeros_like(unsegmented_input_flat, dtype=torch.bool)
    for i, count in enumerate(spike_count_df_flat['n_omitted_spikes']):
        indices = torch.nonzero(unsegmented_input_flat[i] == 1).squeeze()
        if len(indices) > count:
            omit_indices = indices[torch.randperm(len(indices))[:count]]
            mask[i, omit_indices] = True
        else:
            mask[i, indices] = True
    return mask


def create_spike_addition_mask(unsegmented_input:torch.Tensor, noise_level_in_percent:int):
    """
    Create a mask with the same shape as the unsegmented input tensor (n_samples, n_mus * n_time_steps)
    where the values are True for the locations of spikes to be added and False otherwise.
    The mask is created by counting the number of spike time steps in each sample, for the active, identified
    MU and then randomly selecting the specified percentage of spike time steps to add spikes to.
    The number of spikes to be added is calculated as a percentage of the total number of spike time steps
    """
    # use the unsemented input (shape: n_sample, n_timesteps, n_mus) to identify the active MUs
    spike_count_df = pd.DataFrame(torch.zeros(unsegmented_input.size(0), unsegmented_input.size(2)))
        # get the active MU ids per saample
    for i, sample in enumerate(unsegmented_input):
        mu_ids = torch.unique(torch.where(sample==1)[1])
        spike_count_per_mu = torch.sum(sample[:, mu_ids]==1,dim=0)
        spike_count_df.loc[i, mu_ids.numpy()] = spike_count_per_mu.numpy()

    spike_added_df = spike_count_df * (noise_level_in_percent / 100)

    # Create a mask for the spikes to be added in each sample and for the active MUs only
    mask = torch.zeros_like(unsegmented_input, dtype=torch.bool)
    for i in range(spike_added_df.shape[0]):
        mu_ids = spike_added_df.columns[spike_added_df.loc[i] > 0]
        for mu_id in mu_ids:
            count = int(spike_added_df.loc[i, mu_id])
            mu_input_flat = unsegmented_input[i, :, mu_id].view(-1)
            indices = torch.nonzero(mu_input_flat == 0).squeeze()
            if len(indices) > count:
                add_indices = indices[torch.randperm(len(indices))[:count]]
                mask[i, add_indices, mu_id] = True
            else:
                mask[i, indices, mu_id] = True
    return mask

def flatten_input_along_mus(unsegmented_input:torch.Tensor):
    """
    Flatten the unsegmented input tensor along the second and third dimensions. 
    That is to combine all the spike times of all the MUs into a single dimension.
    """
    # flatten the tensor along the second and third dimensions
    unsegmented_input_flat = unsegmented_input.view(unsegmented_input.size(0), -1)
    print(f"unsegmented_input_flat size: {unsegmented_input_flat.size()}")
    return unsegmented_input_flat


def omit_spikes(unsegmented_input:torch.Tensor, noise_level_in_percent:int):
    """
    Omit spikes from the unsegmented input tensor based on the specified noise level.
    The noise level is specified as a percentage of the total number of spikes in each sample.
    """
    n_samples = unsegmented_input.size(0)
    n_timesteps = unsegmented_input.size(1)

    unsegmented_input_flat = flatten_input_along_mus(unsegmented_input)
    # Create a mask for the spikes to be omitted
    ommission_mask = create_spike_omission_mask(unsegmented_input_flat, noise_level_in_percent)
    logging.debug("Omission mask size: %s", ommission_mask.size())

    # Omit the spikes
    unsegmented_input_flat_omitted = unsegmented_input_flat.clone()
    unsegmented_input_flat_omitted[ommission_mask] = 0

    # get the MU IDs of the omitted spikes
    ommission_mask_unflattened = ommission_mask.view(n_samples, n_timesteps, -1)
    logging.debug("ommission_mask_unflattened size: %s", ommission_mask_unflattened.size())
    # revert the unsegmented input to its original shape
    unsegmented_input_omitted = unsegmented_input_flat_omitted.view(n_samples, n_timesteps, -1)
    logging.debug("unsegmented_input_omitted size: %s", unsegmented_input_omitted.size())
    return unsegmented_input_omitted, ommission_mask_unflattened

def add_jitter(unsegmented_input:torch.Tensor, jitter_std:float, sim_dt:float, rand_indices_ratio:float=0.5):
    """
    Add Gaussian jitter noise to randomly selected spikes in the unsegmented input tensor.

    This function simulates temporal jitter by shifting spike times according to a Gaussian distribution.
    The jitter is applied to a random subset of spikes, and the shifted spike times are clipped to
    remain within the valid time step range.

    Args:
        unsegmented_input (torch.Tensor): Input tensor of shape (n_samples, n_timesteps, n_mus)
            containing binary spike data (1 for spike, 0 for no spike).
        jitter_std (float): Standard deviation of the Gaussian jitter noise in seconds.
            The mean is 0. A warning is issued if jitter_std < sim_dt.
        sim_dt (float): Simulation time step in seconds, used to convert jitter from
            time to discrete time steps.
        rand_indices_ratio (float, optional): Ratio of spikes to apply jitter to (0.0 to 1.0).
            Defaults to 0.1 (10% of spikes).

    Returns:
        tuple: A tuple containing:
            - unsegmented_input_jittered (torch.Tensor): The jittered input tensor with the same
              shape as the input (n_samples, n_timesteps, n_mus).
            - org_to_jitter_indices (dict): A nested dictionary mapping sample indices to
              lists of tuples containing the original and jittered spike coordinates.
              Structure: {sample_i: [(org_timestep, org_mu, new_timestep, new_mu), ...]}.

    Notes:
        - The Gaussian noise has mean=0 and std=jitter_std.
        - Jitter time is converted to discrete time steps by dividing by sim_dt and rounding.
        - If a jittered spike time falls outside [0, n_timesteps - 1], it is clipped
          to the nearest valid boundary.
        - The MU ID remains constant for each spike - only the timestep is jittered.
    """
    n_samples = unsegmented_input.size(0)
    n_timesteps = unsegmented_input.size(1)
    n_mus = unsegmented_input.size(2)

    unsegmented_input_jittered = unsegmented_input.clone()

    # check on the jitter_std vs sim_dt, if much smaller then we wouldn't see any difference with jitter
    if jitter_std <= sim_dt:
        print(f"\nWarning! the jitter applied is smaller than the simulation time step ({jitter_std}<{sim_dt} ) and might be ineffective!\n")

    org_to_jitter_indices = {}
    for sample_i in range(n_samples):
        # Find all spikes in this sample (returns indices as [timestep, mu] pairs)
        spike_coords = torch.nonzero(unsegmented_input[sample_i] == 1)
        # select random indices to jitter
        jitter_count = int(rand_indices_ratio * len(spike_coords))
        rand_indices = torch.randperm(len(spike_coords))[:jitter_count]
        rand_spike_coords = spike_coords[rand_indices]

        # Initialize mapping for this sample as a list of tuples
        org_to_jitter_indices[sample_i] = []

        for spike_coord in rand_spike_coords:
            org_timestep = spike_coord[0].item()
            mu_id = spike_coord[1].item()

            # Generate Gaussian noise
            jitter_in_time = torch.randn(1).item() * jitter_std
            jitter_in_timesteps = int(jitter_in_time / sim_dt)

            if jitter_in_timesteps != 0:
                new_timestep = org_timestep + jitter_in_timesteps
                # Clip the new timestep to be within valid range (keep same MU)
                new_timestep = max(0, min(n_timesteps - 1, new_timestep))

                # Store the mapping as (org_timestep, mu_id, new_timestep, mu_id)
                org_to_jitter_indices[sample_i].append((org_timestep, mu_id, new_timestep, mu_id))

                # Move the spike to the new timestep (same MU)
                unsegmented_input_jittered[sample_i, org_timestep, mu_id] = 0
                unsegmented_input_jittered[sample_i, new_timestep, mu_id] = 1

    # logging
    for sample_i in range(n_samples):
        n_spikes_jittered = len(org_to_jitter_indices[sample_i])
        logging.info("For sample %s, n_spikes jittered: %s", sample_i, n_spikes_jittered)
    return unsegmented_input_jittered, org_to_jitter_indices

def add_spikes(unsegmented_input:torch.Tensor, noise_level_in_percent:int):
    """
    Add spikes to the unsegmented input tensor based on the specified noise level.
    This mode simulates false positive spikes by adding spikes at random time steps where there were no spikes.
    The noise level is specified as a percentage of the total number of spikes for the active MUs in each sample.
    This implementation is designed to:
        1. avoid adding false positives for inactive MUs and
        2. ensure that the number of false positive spikes added is proportional to the number of spike time steps.
    """
    addition_mask = create_spike_addition_mask(unsegmented_input, noise_level_in_percent)
    logging.debug("Addition mask size: %s", addition_mask.size())

    # Add the spikes
    unsegmented_input_added = unsegmented_input.clone()
    unsegmented_input_added[addition_mask] = 1
    logging.debug("unsegmented_input_added size: %s", unsegmented_input_added.size())
    return unsegmented_input_added, addition_mask



def misattribute_spikes(unsegmented_input:torch.Tensor, noise_level_in_percent:int, verbose:bool=True):
    """
    Misattribute spikes to incorrect motor units (MUs) based on the specified noise level.

    This function simulates scenarios where spikes are incorrectly assigned to neighboring MUs,
    representing cross-talk or classification error in the spike data. For each sample,
    a percentage of spikes from active MUs are randomly reassigned to neighboring active MUs
    (consecutive neighbors in the sorted list of active MUs) at the same time step.

    Args:
        unsegmented_input (torch.Tensor): Input tensor of shape (n_samples, n_timesteps, n_mus)
            containing binary spike data (1 for spike, 0 for no spike).
        noise_level_in_percent (int): Percentage of spikes to misattribute to neighboring MUs.

    Returns:
        tuple: A tuple containing:
            - unsegmented_input_misattributed (torch.Tensor): The modified input tensor with
              misattributed spikes, same shape as input (n_samples, n_timesteps, n_mus).
            - org_to_misattribute_indices (dict): A nested dictionary mapping sample indices to
              lists of tuples containing the misattribution details.
              Structure: {sample_i: [(timestep, source_mu_id, target_mu_id), ...]}.

    Notes:
        - Only spikes from active MUs (MUs with at least one spike) are considered for misattribution.
        - Spikes are reassigned only to neighboring active MUs (consecutive neighbors in sorted order).
        - The original spike is removed and a new spike is added to a neighboring MU.
        - If a MU has no active neighbors, its spikes cannot be misattributed.
    """
    n_samples = unsegmented_input.size(0)
    n_mus = unsegmented_input.size(2)

    unsegmented_input_misattributed = unsegmented_input.clone()
    org_to_misattribute_indices = {}

    # Count spikes per active MU for each sample
    spike_count_df = pd.DataFrame(torch.zeros(n_samples, n_mus))
    for i, sample in enumerate(unsegmented_input):
        mu_ids = torch.unique(torch.where(sample == 1)[1])
        spike_count_per_mu = torch.sum(sample[:, mu_ids] == 1, dim=0)
        spike_count_df.loc[i, mu_ids.numpy()] = spike_count_per_mu.numpy()

    # Calculate number of spikes to misattribute per MU
    spike_misattribute_df = spike_count_df * (noise_level_in_percent / 100)

    # Process each sample
    for i in range(n_samples):
        # Initialize mapping for this sample as a list of tuples
        org_to_misattribute_indices[i] = []

        # Get active MU IDs for this sample
        active_mu_ids = spike_count_df.columns[spike_count_df.loc[i] > 0].tolist()

        # Need at least 2 active MUs to misattribute
        if len(active_mu_ids) < 2:
            logging.debug("Sample %s: Only %s active MU(s), skipping misattribution", i, len(active_mu_ids))
            continue

        # Sort active MUs by recruitment order (time of first spike)
        mu_first_spike_time = {}
        for mu_id in active_mu_ids:
            first_spike_timestep = torch.nonzero(unsegmented_input[i, :, mu_id] == 1)[0].item()
            mu_first_spike_time[mu_id] = first_spike_timestep

        # Sort MUs by recruitment order (first spike time)
        active_mu_ids_sorted_by_recruitment = sorted(active_mu_ids, key=lambda mu: mu_first_spike_time[mu])

        # Map MU IDs to their recruitment order index
        mu_id_to_recruitment_idx = {mu_id: idx for idx, mu_id in enumerate(active_mu_ids_sorted_by_recruitment)}

        # For each active MU, misattribute the specified percentage of spikes
        for source_mu_id in active_mu_ids:
            count = int(spike_misattribute_df.loc[i, source_mu_id])
            if count == 0:
                continue

            # Find all spike timesteps for this MU
            spike_timesteps = torch.nonzero(unsegmented_input[i, :, source_mu_id] == 1).squeeze()
            if spike_timesteps.dim() == 0:
                spike_timesteps = spike_timesteps.unsqueeze(0)

            # Randomly select spikes to misattribute
            if len(spike_timesteps) > count:
                selected_indices = torch.randperm(len(spike_timesteps))[:count]
                misattribute_timesteps = spike_timesteps[selected_indices]
            else:
                misattribute_timesteps = spike_timesteps

            source_recruitment_idx = mu_id_to_recruitment_idx[source_mu_id]

            # Find neighboring active MUs (consecutive neighbors in recruitment order)
            neighboring_active_mus = [
                mu for mu in active_mu_ids
                if mu != source_mu_id and abs(mu_id_to_recruitment_idx[mu] - source_recruitment_idx) == 1
            ]

            # Skip if no neighboring active MUs
            if len(neighboring_active_mus) == 0:
                logging.debug("Sample %s, MU %s: No neighboring active MUs, skipping", i, source_mu_id)
                continue

            # For each selected spike, reassign to a neighboring active MU
            for timestep in misattribute_timesteps:
                timestep = timestep.item()

                # Remove spike from source MU
                unsegmented_input_misattributed[i, timestep, source_mu_id] = 0

                # Choose a neighboring active MU to reassign to
                target_mu_id = neighboring_active_mus[torch.randint(len(neighboring_active_mus), (1,)).item()]

                # Add spike to target MU
                unsegmented_input_misattributed[i, timestep, target_mu_id] = 1

                # Store the mapping as (timestep, source_mu_id, target_mu_id)
                org_to_misattribute_indices[i].append((timestep, source_mu_id, target_mu_id))

        logging.info("Sample %s: Misattributed spikes across %s active MUs", i, len(active_mu_ids))

    # Print summary of misattributions
    if verbose:
        print_misattribution_summary(org_to_misattribute_indices)
    return unsegmented_input_misattributed, org_to_misattribute_indices


def print_misattribution_summary(org_to_misattribute_indices):
    """
    Print a summary of spike misattributions for each sample.

    This function provides a readable summary showing which motor units had spikes
    misattributed to which target MUs, and how many spikes were affected for each
    source->target MU pair.

    Args:
        org_to_misattribute_indices (dict): Dictionary mapping sample indices to
            lists of tuples containing the misattribution details.
            Structure: {sample_i: [(timestep, source_mu_id, target_mu_id), ...]}

    Example output:
        Sample 0: Total 10 spikes misattributed
        --------------------------------------------------------------------------------
          MU 5 → MU 12: 3 spikes
          MU 12 → MU 13: 7 spikes
    """
    print("=" * 80)
    print("SPIKE MISATTRIBUTION SUMMARY")
    print("=" * 80)

    for sample_i, misattributions in org_to_misattribute_indices.items():
        if len(misattributions) == 0:
            print(f"\nSample {sample_i}: No spikes misattributed")
            continue
        print(f"\nSample {sample_i}: Total {len(misattributions)} spikes misattributed")
        print("-" * 80)

        # Group misattributions by source->target MU pairs
        mu_pair_counts = defaultdict(int)
        for timestep, source_mu, target_mu in misattributions:
            mu_pair_counts[(source_mu, target_mu)] += 1

        # Print summary by MU pairs
        for (source_mu, target_mu), count in sorted(mu_pair_counts.items()):
            print(f"  MU {source_mu} → MU {target_mu}: {count} spikes")

    print("\n" + "=" * 80)
