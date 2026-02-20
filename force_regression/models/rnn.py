import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import tqdm
import wandb
from torch.profiler import profile, ProfilerActivity
from typing import Dict, Optional
from force_regression.data.preprocessing.emg import apply_butter_lowpass

# Optional fvcore import for Conv1d FLOP counting
try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False


DEVICE = torch.device('cpu')
DTYPE = torch.float
class ExponentialFilterRNN(nn.Module):
    """
    RNN model with an exponential filter applied to the input data.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size:int,
                 rnn_config, decay_tau: float, kernel_size: int, dt: float,
                 project_out:bool=False, rnn_bias:bool=False
                ):
        super().__init__()
        self.rnn_config = rnn_config
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.project_out = project_out # whether to project the rnn hidden to out layer with fc layer
        self.rnn_bias = rnn_bias
        # Create exponential kernel
        t = torch.arange(self.kernel_size, dtype=torch.float32) * dt
        kernel = torch.exp(-t / decay_tau)  # e^{-t/tau}
        kernel = kernel / kernel.sum()  # normalize area

        self.n_inputs = input_size
        # Each input channel gets its own filter (groups=input_size)
        self.exp_filter = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,  # causal padding
            groups=input_size,  # depthwise convolution
            bias=False
        )

        # Initialize weights with the exponential kernel
        with torch.no_grad():
            # Set all filters to the same exponential kernel
            for i in range(input_size):
                self.exp_filter.weight[i, 0, :] = kernel

        # Freeze the filter weights (make non-trainable)
        self.exp_filter.weight.requires_grad = False

        self._set_random_seeds(rnn_config.torch_seed)
        # RNN and output layer
        if rnn_config.rnn_type=='vanilla':
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bias=self.rnn_bias,
            )
        elif rnn_config.rnn_type=='lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bias=self.rnn_bias,
            )
        elif rnn_config.rnn_type=='gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bias=self.rnn_bias,
                dropout=0
            )
        else:
            raise ValueError("rnn_type must be either 'vanilla', 'lstm' or 'gru'")
        if project_out:
            self.fc = nn.Linear(hidden_size, self.output_size, bias=False)

    def _set_random_seeds(self,seed):
        """
        Sets seed for random shuffle of the fingers
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    def forward(self, x):
        x = x.transpose(1, 2)  # -> (B, C, T) where C=input_size

        # # OLD: Apply the exponential kernel to each channel independently
        # # Using groups = n_inputs to apply same kernel per channel
        # filtered = F.conv1d(x,
        #                     self.kernel.repeat(self.n_inputs, 1, 1),  # (n_inputs,1,K)
        #                     padding=self.kernel_size - 1,  # causal-style padding
        #                     groups=self.n_inputs
        #                 )
        # # Trim to match original length (causal convolution)
        # filtered = filtered[:, :, :x.shape[2]]  # (B, 196, T)

        # NEW: Apply exponential filter using Conv1d layer
        filtered = self.exp_filter(x)  # (B, C, T+padding)
        # Trim to match original length (causal convolution)
        filtered = filtered[:, :, :x.shape[2]]  # (B, C, T)

        # Pass through RNN
        filtered = filtered.transpose(1, 2)  # -> (B, T, C)
        out, _ = self.rnn(filtered)      # (B, T, hidden)
        if self.project_out:
            out = self.fc(out)               # (B, T, output_size)
        return out
    
    def post_process_predictions(self,y_pred:torch.Tensor):
        """
        Post-processes the predictions by applying a low-pass filter.
        """
        y_pred_post = torch.zeros_like(y_pred)
        for batch in range(y_pred.shape[0]):
            for output in range(y_pred.shape[-1]):
                filtered_y_pred = apply_butter_lowpass(y_pred[batch,:, output],
                                                       cutoff=self.rnn_config.post_filt_cutoff,
                                                       fs=1/self.rnn_config.dt,
                                                       order=self.rnn_config.post_filt_order)
                y_pred_post[batch,:, output] = torch.tensor(filtered_y_pred.copy(), dtype=DTYPE, device=DEVICE)

        return y_pred_post



def train_rnn(model, model_config, train_dataset, test_dataset):
    """
    Train the RNN model
    """
    model.train()
    lr = model_config.lr
    num_iter = model_config.num_iter
    batch_size = model_config.train_batch_size

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    targets_cache = []
    tr_loss_per_epoch = []
    ts_loss_every_nepochs = []
    tr_batch_loss_hist = []  # record loss for all iterations and each batch

    with tqdm.trange(num_iter) as pbar:
        for i in pbar:
            train_batch = iter(train_loader)
            ep_running_loss = 0   # the current epoch loss averaged over the batches and samples
            mse_epoch = 0
            r2_epoch = 0
            for sample_id, (inputs, labels, _) in enumerate(train_batch):
                inputs, targets = inputs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)


                loss = loss_function(outputs, targets)
                # Compute metrics R2 and MSE metrics
                predicted_value = outputs.cpu().detach()
                predicted_value = predicted_value.numpy().reshape(-1, predicted_value.shape[-1])
                targets_reshaped = targets.cpu().detach().numpy().reshape(-1, targets.shape[-1])
                r2_epoch += r2_score(predicted_value, targets_reshaped)
                mse_epoch += mean_squared_error(predicted_value, targets_reshaped)

                # Backpropagation and optimization step
                loss.backward()
                optimizer.step()

                if i == num_iter - 1:
                    targets_cache.append(targets)

                tr_batch_loss_hist.append(loss.item())
                ep_running_loss += loss.item()

                n_samples = (sample_id+1)
                mean_batch_loss = ep_running_loss / n_samples
                pbar.set_postfix(loss=f"{mean_batch_loss:.3e}")
            if model_config.use_wandb:
                wandb.log({'mse_tr_per_epoch': mse_epoch / n_samples,'epoch': i})

            if i % model_config.log_ts_freq == 0:
                test_eloss,_, _, _= evaluate_rnn(model, model_config, test_dataset)
                ts_loss_every_nepochs.append(test_eloss)
            tr_loss_per_epoch.append(mean_batch_loss)
    return tr_loss_per_epoch, tr_batch_loss_hist, ts_loss_every_nepochs, targets_cache


def profile_rnn_inference(model, input_tensor: torch.Tensor,
                          get_conv1d_flops: bool = False) -> Dict:
    """
    Profile a single forward pass of the RNN model.

    Args:
        model: RNN model to profile
        input_tensor: Input tensor for forward pass
        get_conv1d_flops: If True, use fvcore to get Conv1d FLOPs

    Returns:
        Dictionary with profiling metrics for this sample:
        - cpu_memory: CPU memory usage in bytes
        - cpu_time: CPU time in microseconds (from profiler, includes overhead)
        - wall_clock_time: Actual wall-clock time in microseconds (without profiler overhead)
        - flops: FLOPs from torch.profiler
        - conv1d_flops: Conv1d FLOPs from fvcore (if requested)
        - output: Model output tensor
    """
    import time

    result = {
        'cpu_memory': 0,
        'cpu_time': 0,
        'wall_clock_time': 0,
        'flops': 0,
        'conv1d_flops': 0,
        'output': None
    }

    # Measure wall-clock time separately (without profiler overhead)
    wall_start = time.perf_counter()
    _ = model(input_tensor)
    wall_end = time.perf_counter()
    result['wall_clock_time'] = (wall_end - wall_start) * 1e6  # Convert to microseconds

    # Profile using torch.profiler (separate pass for FLOPs and memory)
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_modules=True
    ) as prof:
        output = model(input_tensor)

    result['output'] = output

    # Extract metrics from profiler
    for event in prof.key_averages():
        if event.flops is not None:
            result['flops'] += event.flops
        result['cpu_time'] += event.cpu_time_total
        result['cpu_memory'] += event.cpu_memory_usage

    # Get Conv1d FLOPs from fvcore if requested
    if get_conv1d_flops and FVCORE_AVAILABLE:
        try:
            flop_analyzer = FlopCountAnalysis(model, input_tensor)
            flop_analyzer.unsupported_ops_warnings(False)
            flop_analyzer.uncalled_modules_warnings(False)
            flops_by_op = flop_analyzer.by_operator()
            result['conv1d_flops'] = flops_by_op.get('conv', 0)
        except Exception:
            pass

    return result


def aggregate_profiling_results(profiling_samples: list, conv1d_flops: int = 0,
                                 verbose: bool = True) -> Dict:
    """
    Aggregate profiling results across multiple samples.

    Args:
        profiling_samples: List of profiling results from profile_rnn_inference
        conv1d_flops: Conv1d FLOPs to add to each sample's total
        verbose: If True, print summary

    Returns:
        Dictionary with aggregated statistics:
        - peak/avg/std for cpu_memory, cpu_time, flops
        - per-sample arrays
    """
    cpu_mem = np.array([s['cpu_memory'] for s in profiling_samples])
    cpu_time = np.array([s['cpu_time'] for s in profiling_samples])
    flops = np.array([s['flops'] for s in profiling_samples])
    wall_clock_time = np.array([s.get('wall_clock_time', 0) for s in profiling_samples])

    # Add Conv1d FLOPs to total
    total_flops_per_sample = flops + conv1d_flops

    results = {
        # Per-sample arrays
        'cpu_memory_per_sample': cpu_mem.tolist(),
        'cpu_time_per_sample': cpu_time.tolist(),
        'flops_per_sample': flops.tolist(),
        'wall_clock_time_per_sample': wall_clock_time.tolist(),
        'conv1d_flops': conv1d_flops,

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

        # Aggregated FLOPs stats
        'peak_flops': int(np.max(total_flops_per_sample)),
        'avg_flops': float(np.mean(total_flops_per_sample)),
        'avg_flops_torch_profiler': float(np.mean(flops)),

        'num_samples_profiled': len(profiling_samples)
    }

    if verbose:
        print("\n" + "=" * 60)
        print("RNN Profiling Results")
        print("=" * 60)
        print(f"Samples profiled: {results['num_samples_profiled']}")
        print(f"\n[CPU Memory]")
        print(f"  Peak:    {results['peak_cpu_memory_bytes'] / 1024**2:.2f} MB")
        print(f"  Average: {results['avg_cpu_memory_bytes'] / 1024**2:.2f} MB")
        print(f"  Std:     {results['std_cpu_memory_bytes'] / 1024**2:.2f} MB")
        print(f"\n[Wall-Clock Time (actual inference)]")
        print(f"  Peak:    {results['peak_wall_clock_time_us'] / 1000:.2f} ms")
        print(f"  Average: {results['avg_wall_clock_time_us'] / 1000:.2f} ms")
        print(f"  Std:     {results['std_wall_clock_time_us'] / 1000:.2f} ms")
        print(f"\n[FLOPs]")
        print(f"  torch.profiler avg: {results['avg_flops_torch_profiler']:,.0f}")
        print(f"  Conv1d (fvcore):    {conv1d_flops:,}")
        print(f"  Total avg FLOPs:    {results['avg_flops']:,.0f}")
        print("=" * 60 + "\n")

    return results


def evaluate_rnn(model, model_config, eval_dataset, enable_profiling: bool = False,
                 num_profile_samples: int = 10) -> tuple:
    """
    Perform prediction using the trained RNN model.

    Args:
        model: Trained RNN model
        model_config: Model configuration
        eval_dataset: Evaluation dataset
        enable_profiling: If True, profile memory and CPU time across samples
        num_profile_samples: Number of samples to profile (default: 10)

    Returns:
        If enable_profiling=False:
            (average_loss, predicted_values_cache, true_values_cache, loss_per_batch)
        If enable_profiling=True:
            (average_loss, predicted_values_cache, true_values_cache, loss_per_batch, profiling_results)
    """
    batch_size = model_config.test_batch_size
    test_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    loss_function = torch.nn.MSELoss()
    model.eval()

    # Profiling storage
    profiling_samples = []
    conv1d_flops = 0

    with torch.no_grad():
        loss_per_batch = []
        predicted_values_cache = []
        true_values_cache = []

        for batch_idx, (inputs, label, noisy_data) in enumerate(test_loader):
            inputs = inputs.to(DEVICE) if model_config.percent_omission==0 else noisy_data.to(DEVICE)

            # Profile this batch if enabled and we haven't reached the limit
            if enable_profiling and len(profiling_samples) < num_profile_samples:
                # Get Conv1d FLOPs only on first sample
                get_conv1d = (len(profiling_samples) == 0)
                profile_result = profile_rnn_inference(model, inputs, get_conv1d_flops=get_conv1d)

                if get_conv1d:
                    conv1d_flops = profile_result['conv1d_flops']

                profiling_samples.append(profile_result)
                output = profile_result['output']
            else:
                output = model(inputs)

            if model_config.post_process_filt:
                output = model.post_process_predictions(output)

            loss_value = loss_function(output, label.to(DEVICE))
            loss_per_batch.append(loss_value.item())
            predicted_values_cache.append(output.cpu())
            true_values_cache.append(label.cpu())

    average_loss = np.sum(loss_per_batch) / len(loss_per_batch)

    # Aggregate and return profiling results
    if enable_profiling and profiling_samples:
        profiling_results = aggregate_profiling_results(
            profiling_samples, conv1d_flops, verbose=True
        )
        return average_loss, predicted_values_cache, true_values_cache, loss_per_batch, profiling_results

    return average_loss, predicted_values_cache, true_values_cache, loss_per_batch


