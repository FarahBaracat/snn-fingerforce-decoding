import torch
from force_regression.data.loaders.decomposition import load_pickle


def filter_abnormal_mus_data(data:dict, idx_list:list):
    """
    Deletes bad MU ids from the data dict
    """
    for key in data.keys():
        if key == 'source':
            if len(data[key]) > 0:
                data[key] = [i for j, i in enumerate(data[key]) if j not in idx_list]
        else:
            data[key] = [i for j, i in enumerate(data[key]) if j not in idx_list]
    return data


def bootstrapped_coeff_var(timestamps: torch.Tensor, n_iterations: int = 1000):
    """Runs a bootstrapped version of the interspike coefficient of variation
    and returns the 75th quantile"""

    isi = timestamps.diff().float()

    # Remove big gaps in ISI (likely to be real derecruitment-recruitments)
    isi = isi[isi < 5 * isi.median()]
    n_isi = isi.shape[0]
    assert n_isi > 2, "Not enough timestamps for ISI"

    # Generate a matrix of random indices for bootstrapping
    indices = torch.randint(0, n_isi, (n_iterations, n_isi)).to(
        timestamps.device
    )

    # Using the indices to get bootstrapped ISIs
    resampled_isi = torch.index_select(isi, 0, indices.view(-1)).view(
        n_iterations, n_isi
    )

    resampled_coeff_vars = resampled_isi.std(1) / resampled_isi.mean(1)

    return resampled_coeff_vars.quantile(0.75)


def rate_of_agreement(
        timestamps_1: torch.Tensor,
        timestamps_2: torch.Tensor,
        tolerance: int = 5,
        max_shift: int = 100,
) -> torch.Tensor:
    """Matrix operation based implementation of rate of agreement finding
    between two timestamp sets"""

    assert tolerance >= 0, "tolerance must be non-negative"
    assert max_shift >= 0, "max_shift must be non-negative"

    # Make sure timestamps on same device
    timestamps_2 = timestamps_2.type_as(timestamps_1)

    # Set tolerance aranges
    tol_range = torch.arange(-tolerance, tolerance).type_as(timestamps_1)

    # Create shifted versions of timestamp_1 with tolerance bands
    expand_1 = timestamps_1.unsqueeze(0)
    expand_1 = expand_1 - tol_range.unsqueeze(1)
    expand_1 = expand_1.unsqueeze(-1).tile([1, 1, timestamps_2.shape[0]])

    # Tile up timestamp_2
    expand_2 = (
        timestamps_2.unsqueeze(0)
        .unsqueeze(0)
        .tile([expand_1.shape[0], expand_1.shape[1], 1])
    )

    # Iterate through shift amounts
    matches = []
    for shift in range(-max_shift, max_shift):
        matches.append((expand_1 + shift == expand_2).any(2).any(0).sum())
    matches = torch.stack(matches).max()

    # Return RoA as percentage
    return 100 * matches.divide(timestamps_1.shape[0] + timestamps_2.shape[0] - matches)


def compute_roa_repeats(data, device: str = 'cuda'):
    pred_timestamps = data['timestamps']
    true_firings = data['timestamps']
    results = []
    roa_dict = {}
    for pred_idx, pred in enumerate(pred_timestamps):
        roa_values = torch.stack([
            rate_of_agreement(pred.to(device), ground.to(device), max_shift=100, tolerance=5)
            if ground_idx != pred_idx else torch.tensor(0).to(device)
            for ground_idx, ground in enumerate(true_firings)
        ])
        roa_dict[pred_idx] = roa_values.max().item()
        print(roa_values.max().item(), roa_values.argmax().item())
        results.append({
            'pred_idx': pred_idx,
            'roa': roa_values.max().item(),
            'gt': roa_values.argmax().item()
        })
    return roa_dict, results


def load_data_clean(path_pkl, cov_cutoff: int = 30):
    data = load_pickle(path_pkl)

    none_indices = [i for i, x in enumerate(data['timestamps']) if x is None]
    for key in data.keys():
        data[key] = [i for j, i in enumerate(data[key]) if j not in none_indices]

    clean_data = {'timestamps': [], 'RoA': [], 'num firings': [], 'truth index': [], 'silhouettes': [], 'cv': [],
                  'best exp': []}

    for i in range(len(data['timestamps'])):
        t = data['timestamps'][i]
        cov = bootstrapped_coeff_var(t).item() * 100
        num_firings = len(t)

        if 1 < cov <= cov_cutoff:
            clean_data['timestamps'].append(t)
            clean_data['num firings'].append(num_firings)
            clean_data['silhouettes'].append(data['silhouettes'][i])
            clean_data['cv'].append(cov)
            # clean_data['best exp'].append(data['best_exp'][i])
        else:
            print(f"Unit {i} CoV: {cov}. Removed.")

    return clean_data

