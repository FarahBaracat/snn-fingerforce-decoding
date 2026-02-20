"""
Profiling utilities for logging model performance metrics to TensorBoard.

This module provides functions to log profiling results (memory, CPU time, FLOPs)
from RNN and SNN models to TensorBoard for visualization and comparison.

===============================================================================
FLOP CALCULATION EXAMPLE: GRU with S1, Flexion
===============================================================================

Model Configuration:
    - Input size (I): 121 channels
    - Hidden size (H): 5
    - Output size: 5
    - Batch size (B): 1
    - Timesteps (T): 2793
    - Conv1d kernel size: 50
    - rnn_bias: False (no bias in RNN layers)
    - FC bias: False (no bias in output layer)

1. GRU FLOPs (without bias)
---------------------------
GRU has 3 gates (reset, update, new), each requiring matrix multiplications
for both input and hidden state. Without bias terms:

    FLOPs_GRU = 6 × B × H × (I + H) × T
              = 6 × 1 × 5 × (121 + 5) × 2793
              = 6 × 5 × 126 × 2793
              = 10,557,540 FLOPs

Note: With bias=True, GRU would add 6×H bias additions per timestep:
    Bias FLOPs = 6 × B × H × T = 6 × 1 × 5 × 2793 = 83,790 FLOPs
    But our model uses bias=False, so this is not counted.

2. Fully Connected Layer FLOPs (without bias)
---------------------------------------------
FC layer maps hidden_size -> output_size, applied at each timestep.
Our model uses nn.Linear(..., bias=False):

    FLOPs_FC_matmul = 2 × B × H × output × T
                    = 2 × 1 × 5 × 5 × 2793
                    = 139,650 FLOPs

Note: With bias=True, FC would add: B × output × T = 13,965 FLOPs
    But our model uses bias=False, so FC total = 139,650 FLOPs

3. Conv1d (Exponential Filter) FLOPs
------------------------------------
Depthwise Conv1d with kernel_size=50 (NOT counted by torch.profiler):

    FLOPs_Conv1d = 2 × B × T × I × kernel_size
                 = 2 × 1 × 2793 × 121 × 50
                 = 33,811,300 FLOPs

4. Total FLOPs Summary
----------------------
    | Component           | FLOPs       | % of Total |
    |---------------------|-------------|------------|
    | Conv1d (exp filter) | 33,811,300  | 75.9%      |
    | GRU                 | 10,557,540  | 23.7%      |
    | FC output layer     | 139,650     | 0.3%       |
    | **Total**           | 44,508,490  | 100%       |

5. What the Profiler Reports
----------------------------
    - torch.profiler: 10,697,190 FLOPs (GRU + FC, excludes Conv1d)
    - fvcore Conv1d:  33,811,300 FLOPs
    - Combined total: 44,508,490 FLOPs

Note: torch.profiler does NOT count Conv1d FLOPs. We use fvcore to get
Conv1d FLOPs and add them to the torch.profiler total.

Note: If your profiler reports 10,711,155 instead of 10,697,190, check
whether bias=True is being used. The difference of ~14,000 FLOPs corresponds
to bias additions (83,790 for GRU bias + 13,965 for FC bias ≈ 97,755).

Comparison across RNN types (same configuration, bias=False):
    | RNN Type    | Gates | FLOPs per timestep    | Total (T=2793) |
    |-------------|-------|-----------------------|----------------|
    | Vanilla RNN | 1     | 2 × H × (I + H) = 1260| 3,519,180      |
    | GRU         | 3     | 6 × H × (I + H) = 3780| 10,557,540     |
    | LSTM        | 4     | 8 × H × (I + H) = 5040| 14,076,720     |

===============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional


def log_profiling_to_tensorboard(profiling_results: Dict,
                                  model: nn.Module,
                                  log_dir: str,
                                  model_name: str = "model",
                                  step: int = 0,
                                  log_model_graph: bool = True,
                                  input_tensor: Optional[torch.Tensor] = None) -> None:
    """
    Log profiling results to TensorBoard.

    Args:
        profiling_results: Dictionary from evaluate_rnn/evaluate_snn with enable_profiling=True
            Expected keys:
            - peak_cpu_memory_bytes, avg_cpu_memory_bytes, std_cpu_memory_bytes
            - peak_cpu_time_us, avg_cpu_time_us, std_cpu_time_us
            - peak_flops, avg_flops, avg_flops_torch_profiler
            - conv1d_flops (optional)
            - num_samples_profiled
        model: The model (for parameter counting and graph)
        log_dir: Directory for TensorBoard logs
        model_name: Name prefix for logged metrics (e.g., "RNN", "SNN")
        step: Step index (e.g., fold number for cross-validation)
        log_model_graph: Whether to log model architecture graph
        input_tensor: Input tensor (required if log_model_graph=True)

    Usage:
        from force_regression.evaluation.profiling import log_profiling_to_tensorboard

        # After evaluate_rnn with profiling
        avg_loss, preds, targets, losses, profiling_results = evaluate_rnn(
            model, config, dataset, enable_profiling=True
        )

        # Log to TensorBoard
        log_profiling_to_tensorboard(
            profiling_results,
            model,
            log_dir="./logs/rnn_profile",
            model_name="RNN",
            step=fold_i
        )

        # View with: tensorboard --logdir=./logs/rnn_profile
    """
    writer = SummaryWriter(log_dir=log_dir)
    # Get model parameters - either from model directly or from profiling_results
    if model is not None:
        # Model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    else:
        total_params = profiling_results['total_params'] if 'total_params' in profiling_results else 0
        trainable_params = profiling_results['trainable_params'] if 'trainable_params' in profiling_results else 0
        frozen_params = total_params - trainable_params
        # Calculate param size in MiB from counted_params list
        param_size_bytes = sum(
            param_num * torch.tensor([], dtype=dtype).element_size()
            for name, size, requires_grad, param_num, dtype in profiling_results.get('counted_params', [])
        )
        param_size_mb = param_size_bytes / 1024**2
        
    writer.add_scalar(f'{model_name}/parameters/total', total_params, step)
    writer.add_scalar(f'{model_name}/parameters/trainable', trainable_params, step)
    writer.add_scalar(f'{model_name}/parameters/frozen', frozen_params, step)
    writer.add_scalar(f'{model_name}/parameters/size_mb', param_size_mb, step)

    # CPU Memory (in MiB for readability)
    writer.add_scalar(f'{model_name}/memory/peak_cpu_mib',
                     profiling_results['peak_cpu_memory_bytes'] / 1024**2, step)
    writer.add_scalar(f'{model_name}/memory/avg_cpu_mib',
                     profiling_results['avg_cpu_memory_bytes'] / 1024**2, step)
    writer.add_scalar(f'{model_name}/memory/std_cpu_mib',
                     profiling_results['std_cpu_memory_bytes'] / 1024**2, step)

    # CPU Memory (in bytes for precision)
    writer.add_scalar(f'{model_name}/memory/peak_cpu_bytes',
                     profiling_results['peak_cpu_memory_bytes'], step)
    writer.add_scalar(f'{model_name}/memory/avg_cpu_bytes',
                     profiling_results['avg_cpu_memory_bytes'], step)
    writer.add_scalar(f'{model_name}/memory/std_cpu_bytes',
                     profiling_results['std_cpu_memory_bytes'], step)

    # CPU Time (in ms for readability)
    writer.add_scalar(f'{model_name}/time/peak_cpu_ms',
                     profiling_results['peak_cpu_time_us'] / 1000, step)
    writer.add_scalar(f'{model_name}/time/avg_cpu_ms',
                     profiling_results['avg_cpu_time_us'] / 1000, step)
    writer.add_scalar(f'{model_name}/time/std_cpu_ms',
                     profiling_results['std_cpu_time_us'] / 1000, step)

    # CPU Time (in microseconds for precision)
    writer.add_scalar(f'{model_name}/time/peak_cpu_us',
                     profiling_results['peak_cpu_time_us'], step)
    writer.add_scalar(f'{model_name}/time/avg_cpu_us',
                     profiling_results['avg_cpu_time_us'], step)
    writer.add_scalar(f'{model_name}/time/std_cpu_us',
                     profiling_results['std_cpu_time_us'], step)

    # Wall-clock time (actual inference time without profiler overhead)
    if 'avg_wall_clock_time_us' in profiling_results:
        writer.add_scalar(f'{model_name}/time/peak_wall_clock_ms',
                         profiling_results.get('peak_wall_clock_time_us', 0) / 1000, step)
        writer.add_scalar(f'{model_name}/time/avg_wall_clock_ms',
                         profiling_results['avg_wall_clock_time_us'] / 1000, step)
        writer.add_scalar(f'{model_name}/time/std_wall_clock_ms',
                         profiling_results.get('std_wall_clock_time_us', 0) / 1000, step)
        writer.add_scalar(f'{model_name}/time/avg_wall_clock_us',
                         profiling_results['avg_wall_clock_time_us'], step)

    # FLOPs
    writer.add_scalar(f'{model_name}/computation/avg_flops',
                     profiling_results['avg_flops'], step)
    writer.add_scalar(f'{model_name}/computation/peak_flops',
                     profiling_results['peak_flops'], step)
    writer.add_scalar(f'{model_name}/computation/avg_flops_torch_profiler',
                     profiling_results['avg_flops_torch_profiler'], step)
    # writer.add_scalar(f'{model_name}/computation/avg_macs_torch_profiler',
    #                  profiling_results['avg_flops_torch_profiler'], step)
    # Conv1d FLOPs (optional, from fvcore)
    conv1d_flops = profiling_results.get('conv1d_flops', 0)
    if conv1d_flops > 0:
        writer.add_scalar(f'{model_name}/computation/conv1d_flops', conv1d_flops, step)

    # SNN Effective Operations (if available)
    if 'synaptic_ops' in profiling_results:
        # Core operation counts - SynOps (spike-dependent)
        writer.add_scalar(f'{model_name}/snn_ops/synaptic_ops', profiling_results['synaptic_ops'], step)
        writer.add_scalar(f'{model_name}/snn_ops/synaptic_ops_per_sample', profiling_results['synaptic_ops_per_sample'], step)
        writer.add_scalar(f'{model_name}/snn_ops/synops_mac', profiling_results.get('synops_mac', 0), step)
        writer.add_scalar(f'{model_name}/snn_ops/synops_ac', profiling_results.get('synops_ac', 0), step)
        writer.add_scalar(f'{model_name}/snn_ops/synops_ac_fc2', profiling_results.get('synops_ac_fc2', 0), step)
        writer.add_scalar(f'{model_name}/snn_ops/synops_flops', profiling_results.get('synops_flops', 0), step)
        # Dense operations (membrane/synapse dynamics)
        writer.add_scalar(f'{model_name}/snn_ops/mac_ops', profiling_results.get('mac_ops', 0), step)
        writer.add_scalar(f'{model_name}/snn_ops/add_ops', profiling_results.get('add_ops', 0), step)
        writer.add_scalar(f'{model_name}/snn_ops/membrane_synapse_flops', profiling_results.get('membrane_synapse_flops', 0), step)
        writer.add_scalar(f'{model_name}/snn_ops/comparison_ops', profiling_results.get('comparison_ops', 0), step)
        # Totals
        writer.add_scalar(f'{model_name}/snn_ops/total_effective_flops', profiling_results.get('total_effective_flops', 0), step)
        writer.add_scalar(f'{model_name}/snn_ops/total_effective_flops_per_sample', profiling_results.get('total_effective_flops_per_sample', 0), step)
        writer.add_scalar(f'{model_name}/snn_ops/synaptic_percentage', profiling_results.get('synaptic_percentage', 0), step)

        # Spike counts and rates
        writer.add_scalar(f'{model_name}/snn_spikes/total_input_spikes', profiling_results.get('total_input_spikes', 0), step)
        writer.add_scalar(f'{model_name}/snn_spikes/total_output_spikes', profiling_results.get('total_output_spikes', 0), step)
        writer.add_scalar(f'{model_name}/snn_spikes/total_encoding_spikes', profiling_results.get('total_encoding_spikes', 0), step)
        if 'encoding_spike_rate' in profiling_results:
            writer.add_scalar(f'{model_name}/snn_spikes/encoding_spike_rate', profiling_results['encoding_spike_rate'], step)
        if 'input_spike_rate' in profiling_results:
            writer.add_scalar(f'{model_name}/snn_spikes/input_spike_rate', profiling_results['input_spike_rate'], step)
        writer.add_scalar(f'{model_name}/snn_spikes/output_spike_rate', profiling_results.get('output_spike_rate', 0), step)

        # Dimensions
        writer.add_scalar(f'{model_name}/snn_dims/timesteps', profiling_results.get('timesteps', 0), step)
        writer.add_scalar(f'{model_name}/snn_dims/num_inputs', profiling_results.get('num_inputs', 0), step)
        writer.add_scalar(f'{model_name}/snn_dims/num_outputs', profiling_results.get('num_outputs', 0), step)

        # Comparison with dense ANN
        if 'dense_ann_flops' in profiling_results:
            writer.add_scalar(f'{model_name}/snn_comparison/dense_ann_flops', profiling_results['dense_ann_flops'], step)
        if 'reduction_vs_dense' in profiling_results:
            writer.add_scalar(f'{model_name}/snn_comparison/flop_reduction_vs_dense', profiling_results['reduction_vs_dense'], step)

    # Model graph
    if log_model_graph and input_tensor is not None:
        try:
            writer.add_graph(model, input_tensor)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")

    # Build counted parameters table if available
    counted_params_table = ""
    counted_params = profiling_results.get('counted_params', [])
    if counted_params:
        counted_params_table = "\n### Parameter Details\n"
        counted_params_table += "| Layer Name | Shape | Trainable | Params | Dtype |\n"
        counted_params_table += "|------------|-------|-----------|--------|-------|\n"
        for name, size, requires_grad, param_num, dtype in counted_params:
            trainable_str = "Yes" if requires_grad else "No"
            dtype_str = str(dtype).replace("torch.", "")
            counted_params_table += f"| {name} | {size} | {trainable_str} | {param_num:,} | {dtype_str} |\n"
        counted_params_table += "\n"

    # Build SNN effective ops table if available
    snn_ops_table = ""
    if 'synaptic_ops' in profiling_results:
        # Determine spike type (encoding or input)
        spike_rate_label = "Encoding spike rate" if 'encoding_spike_rate' in profiling_results else "Input spike rate"
        spike_rate_value = profiling_results.get('encoding_spike_rate', profiling_results.get('input_spike_rate', 0))
        total_spikes_label = "Total encoding spikes" if profiling_results.get('total_encoding_spikes', 0) > 0 else "Total input spikes"
        total_spikes_value = profiling_results.get('total_encoding_spikes', 0) or profiling_results.get('total_input_spikes', 0)

        # Determine SynOps type for display
        synops_mac = profiling_results.get('synops_mac', 0)
        synops_ac = profiling_results.get('synops_ac', 0)
        synops_ac_fc2 = profiling_results.get('synops_ac_fc2', 0)
        synops_flops = profiling_results.get('synops_flops', 0)
        membrane_synapse_flops = profiling_results.get('membrane_synapse_flops', 0)

        # Build fc1 SynOps row
        if synops_mac > 0:
            fc1_synops_row = f"| - fc1 SynOps (MAC, encoding) | {synops_mac:,} | {synops_mac*2:,} |"
        else:
            fc1_synops_row = f"| - fc1 SynOps (AC, non-enc) | {synops_ac:,} | {synops_ac:,} |"

        # Build fc2 SynOps row (if applicable)
        fc2_synops_row = ""
        if synops_ac_fc2 > 0:
            fc2_synops_row = f"\n| - fc2 SynOps (AC, out->filt) | {synops_ac_fc2:,} | {synops_ac_fc2:,} |"

        snn_ops_table = f"""
### SNN Spike Activity
| Metric | Value |
|--------|-------|
| Timesteps | {profiling_results.get('timesteps', 0):,} |
| {total_spikes_label} | {total_spikes_value:,} |
| Total output spikes | {profiling_results.get('total_output_spikes', 0):,} |
| {spike_rate_label} | {spike_rate_value:.4f} |
| Output spike rate | {profiling_results.get('output_spike_rate', 0):.4f} |

### SNN FLOP Breakdown
**Spike-Dependent (Sparse) Operations:**
| Operation Type | Count | FLOPs |
|----------------|-------|-------|
| Synaptic Ops (SynOps) total | {profiling_results.get('synaptic_ops', 0):,} | {synops_flops:,} |
{fc1_synops_row}{fc2_synops_row}

**Dense Operations (membrane/synapse dynamics):**
| Operation Type | Count | FLOPs |
|----------------|-------|-------|
| MAC operations | {profiling_results.get('mac_ops', 0):,} | {profiling_results.get('mac_ops', 0)*2:,} |
| Add operations | {profiling_results.get('add_ops', 0):,} | {profiling_results.get('add_ops', 0):,} |
| Dense subtotal | - | {membrane_synapse_flops:,} |

**Totals:**
| Metric | Value |
|--------|-------|
| Total effective FLOPs | {profiling_results.get('total_effective_flops', 0):,} |
| = SynOps FLOPs + Dense FLOPs | {synops_flops:,} + {membrane_synapse_flops:,} |
| Effective FLOPs per sample | {profiling_results.get('total_effective_flops_per_sample', 0):,.0f} |
| SynOps FLOPs as % of total | {profiling_results.get('synaptic_percentage', 0):.1f}% |
| Comparison operations | {profiling_results.get('comparison_ops', 0):,} |
"""
        # Add comparison with dense ANN if available
        if 'dense_ann_flops' in profiling_results:
            snn_ops_table += f"""
### FLOP Reduction vs Dense ANN
| Metric | Value |
|--------|-------|
| Dense ANN FLOPs | {profiling_results['dense_ann_flops']:,} |
| SNN effective FLOPs | {profiling_results.get('total_effective_flops', 0):,} |
| Reduction factor | {profiling_results.get('reduction_vs_dense', 0):.2f}x |
"""

    # Summary text
    summary_text = f"""
## {model_name} Profiling Summary (Step {step})

### Parameters
| Metric | Value |
|--------|-------|
| Total | {total_params:,} |
| Trainable | {trainable_params:,} |
| Frozen | {frozen_params:,} |
| Size | {param_size_mb:.2f} MiB |
{counted_params_table}
### CPU Memory
| Metric | MiB | Bytes |
|--------|-----|-------|
| Peak | {profiling_results['peak_cpu_memory_bytes'] / 1024**2:.2f} | {profiling_results['peak_cpu_memory_bytes']:,} |
| Average | {profiling_results['avg_cpu_memory_bytes'] / 1024**2:.2f} | {profiling_results['avg_cpu_memory_bytes']:,.0f} |
| Std | {profiling_results['std_cpu_memory_bytes'] / 1024**2:.2f} | {profiling_results['std_cpu_memory_bytes']:,.0f} |

### Profiler CPU Time (includes profiler overhead)
| Metric | ms | us |
|--------|-----|-----|
| Peak | {profiling_results['peak_cpu_time_us'] / 1000:.2f} | {profiling_results['peak_cpu_time_us']:,} |
| Average | {profiling_results['avg_cpu_time_us'] / 1000:.2f} | {profiling_results['avg_cpu_time_us']:,.0f} |
| Std | {profiling_results['std_cpu_time_us'] / 1000:.2f} | {profiling_results['std_cpu_time_us']:,.0f} |

### Wall-Clock Time (actual inference latency)
| Metric | ms | us |
|--------|-----|-----|
| Peak | {profiling_results.get('peak_wall_clock_time_us', 0) / 1000:.3f} | {profiling_results.get('peak_wall_clock_time_us', 0):,.0f} |
| Average | {profiling_results.get('avg_wall_clock_time_us', 0) / 1000:.3f} | {profiling_results.get('avg_wall_clock_time_us', 0):,.0f} |
| Std | {profiling_results.get('std_wall_clock_time_us', 0) / 1000:.3f} | {profiling_results.get('std_wall_clock_time_us', 0):,.0f} |

### FLOPs (torch.profiler)
| Metric | Value |
|--------|-------|
| torch.profiler avg | {profiling_results['avg_flops_torch_profiler']:,.0f} |
| Conv1d (fvcore) | {conv1d_flops:,} |
| Total avg | {profiling_results['avg_flops']:,.0f} |
{snn_ops_table}
### Profiling Info
- Samples profiled: {profiling_results['num_samples_profiled']}
"""

    writer.add_text(f'{model_name}/summary', summary_text, step)
    writer.close()

    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"View with: tensorboard --logdir={log_dir}")


def log_profiling_comparison_to_tensorboard(profiling_results_list: list,
                                             model_names: list,
                                             log_dir: str,
                                             step: int = 0) -> None:
    """
    Log profiling results for multiple models to TensorBoard for comparison.

    Args:
        profiling_results_list: List of profiling result dictionaries
        model_names: List of model names (same length as profiling_results_list)
        log_dir: Directory for TensorBoard logs
        step: Step index
    """
    writer = SummaryWriter(log_dir=log_dir)

    for profiling_results, model_name in zip(profiling_results_list, model_names):
        # CPU Memory
        writer.add_scalar(f'comparison/memory/peak_cpu_mb/{model_name}',
                         profiling_results['peak_cpu_memory_bytes'] / 1024**2, step)
        writer.add_scalar(f'comparison/memory/avg_cpu_mb/{model_name}',
                         profiling_results['avg_cpu_memory_bytes'] / 1024**2, step)

        # CPU Time
        writer.add_scalar(f'comparison/time/peak_cpu_ms/{model_name}',
                         profiling_results['peak_cpu_time_us'] / 1000, step)
        writer.add_scalar(f'comparison/time/avg_cpu_ms/{model_name}',
                         profiling_results['avg_cpu_time_us'] / 1000, step)

        # FLOPs
        writer.add_scalar(f'comparison/computation/avg_flops/{model_name}',
                         profiling_results['avg_flops'], step)

    writer.close()
    print(f"Comparison logs saved to: {log_dir}")


def create_profiling_log_dir(base_dir: str, subject: str, direction: str,
                              model_type: str, fold: int = None) -> str:
    """
    Create a standardized log directory path for profiling.

    Args:
        base_dir: Base directory for logs (e.g., "./logs")
        subject: Subject identifier (e.g., "S1", "S2")
        direction: Direction (e.g., "flex", "ext")
        model_type: Model type (e.g., "rnn_vanilla", "rnn_lstm", "snn")
        fold: Fold number (optional)

    Returns:
        Log directory path
    """
    if fold is not None:
        return f"{base_dir}/{subject}_{direction}_{model_type}_fold{fold}"
    return f"{base_dir}/{subject}_{direction}_{model_type}"


def aggregate_profiling_across_folds(profiling_results_list: list,
                                      log_dir: str,
                                      model_name: str = "model") -> Dict:
    """
    Aggregate profiling results across multiple folds and log a summary table.

    Args:
        profiling_results_list: List of profiling result dictionaries (one per fold)
        log_dir: Directory for TensorBoard logs
        model_name: Name prefix for logged metrics

    Returns:
        Dictionary with aggregated statistics across folds:
        - peak/avg for memory and CPU time (across all folds)
        - per-fold values

    Usage:
        # Collect profiling results during training loop
        all_profiling_results = []
        for fold_i in range(num_folds):
            ...
            _, _, _, _, profiling_results = evaluate_rnn(..., enable_profiling=True)
            all_profiling_results.append(profiling_results)

        # After all folds, create summary
        summary = aggregate_profiling_across_folds(
            all_profiling_results,
            log_dir="./logs/rnn_profiler/summary",
            model_name="RNN"
        )
    """
    num_folds = len(profiling_results_list)

    # Extract per-fold metrics
    peak_memory_per_fold = [r['peak_cpu_memory_bytes'] for r in profiling_results_list]
    avg_memory_per_fold = [r['avg_cpu_memory_bytes'] for r in profiling_results_list]
    peak_time_per_fold = [r['peak_cpu_time_us'] for r in profiling_results_list]
    avg_time_per_fold = [r['avg_cpu_time_us'] for r in profiling_results_list]
    avg_flops_per_fold = [r['avg_flops'] for r in profiling_results_list]

    # Wall-clock time (actual inference time without profiler overhead)
    avg_wall_clock_per_fold = [r.get('avg_wall_clock_time_us', 0) for r in profiling_results_list]
    peak_wall_clock_per_fold = [r.get('peak_wall_clock_time_us', 0) for r in profiling_results_list]

    # SNN-specific metrics
    synaptic_ops_per_fold = [r.get('synaptic_ops', 0) for r in profiling_results_list]
    synaptic_ops_per_sample_per_fold = [r.get('synaptic_ops_per_sample', 0) for r in profiling_results_list]
    mac_ops_per_fold = [r.get('mac_ops', 0) for r in profiling_results_list]
    add_ops_per_fold = [r.get('add_ops', 0) for r in profiling_results_list]
    comparison_ops_per_fold = [r.get('comparison_ops', 0) for r in profiling_results_list]
    total_effective_flops_per_fold = [r.get('total_effective_flops', 0) for r in profiling_results_list]
    synaptic_percentage_per_fold = [r.get('synaptic_percentage', 0) for r in profiling_results_list]
    reduction_vs_dense_per_fold = [r.get('reduction_vs_dense', 0) for r in profiling_results_list]

    # Spike rates
    encoding_spike_rate_per_fold = [r.get('encoding_spike_rate', r.get('input_spike_rate', 0)) for r in profiling_results_list]
    output_spike_rate_per_fold = [r.get('output_spike_rate', 0) for r in profiling_results_list]

    # Compute cross-fold statistics
    summary = {
        'num_folds': num_folds,
        # Memory across folds
        'peak_memory_across_folds_bytes': int(np.max(peak_memory_per_fold)),
        'avg_memory_across_folds_bytes': float(np.mean(avg_memory_per_fold)),
        'std_memory_across_folds_bytes': float(np.std(avg_memory_per_fold)),
        # Time across folds (from profiler)
        'peak_time_across_folds_us': int(np.max(peak_time_per_fold)),
        'avg_time_across_folds_us': float(np.mean(avg_time_per_fold)),
        'std_time_across_folds_us': float(np.std(avg_time_per_fold)),
        # Wall-clock time across folds (actual inference time)
        'peak_wall_clock_across_folds_us': float(np.max(peak_wall_clock_per_fold)) if any(peak_wall_clock_per_fold) else 0,
        'avg_wall_clock_across_folds_us': float(np.mean(avg_wall_clock_per_fold)) if any(avg_wall_clock_per_fold) else 0,
        'std_wall_clock_across_folds_us': float(np.std(avg_wall_clock_per_fold)) if any(avg_wall_clock_per_fold) else 0,
        # FLOPs across folds
        'avg_flops_across_folds': float(np.mean(avg_flops_per_fold)),
        # SNN-specific aggregated metrics
        'avg_synaptic_ops_across_folds': float(np.mean(synaptic_ops_per_fold)) if any(synaptic_ops_per_fold) else 0,
        'avg_synaptic_ops_per_sample': float(np.mean(synaptic_ops_per_sample_per_fold)) if any(synaptic_ops_per_sample_per_fold) else 0,
        'avg_mac_ops_across_folds': float(np.mean(mac_ops_per_fold)) if any(mac_ops_per_fold) else 0,
        'avg_add_ops_across_folds': float(np.mean(add_ops_per_fold)) if any(add_ops_per_fold) else 0,
        'avg_comparison_ops_across_folds': float(np.mean(comparison_ops_per_fold)) if any(comparison_ops_per_fold) else 0,
        'avg_total_effective_flops': float(np.mean(total_effective_flops_per_fold)) if any(total_effective_flops_per_fold) else 0,
        'avg_synaptic_percentage': float(np.mean(synaptic_percentage_per_fold)) if any(synaptic_percentage_per_fold) else 0,
        'avg_reduction_vs_dense': float(np.mean(reduction_vs_dense_per_fold)) if any(reduction_vs_dense_per_fold) else 0,
        'avg_encoding_spike_rate': float(np.mean(encoding_spike_rate_per_fold)) if any(encoding_spike_rate_per_fold) else 0,
        'avg_output_spike_rate': float(np.mean(output_spike_rate_per_fold)) if any(output_spike_rate_per_fold) else 0,
        # Per-fold arrays
        'peak_memory_per_fold': peak_memory_per_fold,
        'avg_memory_per_fold': avg_memory_per_fold,
        'peak_time_per_fold': peak_time_per_fold,
        'avg_time_per_fold': avg_time_per_fold,
        'avg_flops_per_fold': avg_flops_per_fold,
        'avg_wall_clock_per_fold': avg_wall_clock_per_fold,
        'synaptic_ops_per_fold': synaptic_ops_per_fold,
    }

    # Log to TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Build per-fold table rows (with wall-clock time)
    fold_rows = ""
    for i in range(num_folds):
        wall_clock_ms = avg_wall_clock_per_fold[i] / 1000 if avg_wall_clock_per_fold[i] > 0 else 0
        fold_rows += f"| Fold {i} | {peak_memory_per_fold[i] / 1024**2:.2f} | {avg_memory_per_fold[i] / 1024**2:.2f} | {peak_time_per_fold[i] / 1000:.2f} | {avg_time_per_fold[i] / 1000:.2f} | {wall_clock_ms:.3f} | {avg_flops_per_fold[i]:,.0f} |\n"

    # Build SynOps per-fold table if applicable
    synops_fold_rows = ""
    has_synops = any(synaptic_ops_per_fold)
    if has_synops:
        for i in range(num_folds):
            synops_fold_rows += f"| Fold {i} | {synaptic_ops_per_fold[i]:,} | {synaptic_ops_per_sample_per_fold[i]:,.0f} | {encoding_spike_rate_per_fold[i]:.4f} | {output_spike_rate_per_fold[i]:.4f} |\n"

    # Build SynOps summary section
    synops_summary = ""
    if has_synops:
        synops_summary = f"""
### Synaptic Operations Summary
| Metric | Average Across Folds |
|--------|---------------------|
| Synaptic Ops (SynOps) | {summary['avg_synaptic_ops_across_folds']:,.0f} |
| SynOps per Sample | {summary['avg_synaptic_ops_per_sample']:,.0f} |
| MAC Ops | {summary['avg_mac_ops_across_folds']:,.0f} |
| Add Ops | {summary['avg_add_ops_across_folds']:,.0f} |
| Comparison Ops | {summary['avg_comparison_ops_across_folds']:,.0f} |
| Total Effective FLOPs | {summary['avg_total_effective_flops']:,.0f} |
| SynOps as % of FLOPs | {summary['avg_synaptic_percentage']:.1f}% |
| FLOP Reduction vs Dense ANN | {summary['avg_reduction_vs_dense']:.2f}x |

### Spike Rates
| Metric | Average Across Folds |
|--------|---------------------|
| Encoding/Input Spike Rate | {summary['avg_encoding_spike_rate']:.4f} |
| Output Spike Rate | {summary['avg_output_spike_rate']:.4f} |

### Per-Fold SynOps
| Fold | SynOps | SynOps/Sample | Enc Spike Rate | Out Spike Rate |
|------|--------|---------------|----------------|----------------|
{synops_fold_rows}"""

    # Wall-clock time section
    wall_clock_section = ""
    if summary['avg_wall_clock_across_folds_us'] > 0:
        wall_clock_section = f"""
### Wall-Clock Time (Actual Inference Time)
| Metric | ms | us |
|--------|-----|-----|
| Peak | {summary['peak_wall_clock_across_folds_us'] / 1000:.3f} | {summary['peak_wall_clock_across_folds_us']:,.0f} |
| Average | {summary['avg_wall_clock_across_folds_us'] / 1000:.3f} | {summary['avg_wall_clock_across_folds_us']:,.0f} |
| Std | {summary['std_wall_clock_across_folds_us'] / 1000:.3f} | {summary['std_wall_clock_across_folds_us']:,.0f} |

*Note: Wall-clock time measures actual inference latency without profiler overhead.*
"""

    summary_text = f"""
## {model_name} Profiling Summary Across {num_folds} Folds

### Per-Fold Metrics
| Fold | Peak Mem (MiB) | Avg Mem (MiB) | Peak Time (ms) | Avg Time (ms) | Wall-Clock (ms) | Avg FLOPs |
|------|----------------|---------------|----------------|---------------|-----------------|-----------|
{fold_rows}

### Aggregated Memory & Profiler Time
| Metric | Memory (MiB) | Memory (bytes) | Profiler Time (ms) | Profiler Time (us) |
|--------|--------------|----------------|--------------------|--------------------|
| Peak | {summary['peak_memory_across_folds_bytes'] / 1024**2:.2f} | {summary['peak_memory_across_folds_bytes']:,} | {summary['peak_time_across_folds_us'] / 1000:.2f} | {summary['peak_time_across_folds_us']:,} |
| Average | {summary['avg_memory_across_folds_bytes'] / 1024**2:.2f} | {summary['avg_memory_across_folds_bytes']:,.0f} | {summary['avg_time_across_folds_us'] / 1000:.2f} | {summary['avg_time_across_folds_us']:,.0f} |
| Std | {summary['std_memory_across_folds_bytes'] / 1024**2:.2f} | {summary['std_memory_across_folds_bytes']:,.0f} | {summary['std_time_across_folds_us'] / 1000:.2f} | {summary['std_time_across_folds_us']:,.0f} |
{wall_clock_section}
### FLOPs
| Metric | Value |
|--------|-------|
| Average across folds | {summary['avg_flops_across_folds']:,.0f} |
{synops_summary}
"""

    writer.add_text(f'{model_name}/cross_fold_summary', summary_text, 0)

    # Log scalars for the aggregated values
    writer.add_scalar(f'{model_name}/cross_fold/peak_memory_mib',
                     summary['peak_memory_across_folds_bytes'] / 1024**2, 0)
    writer.add_scalar(f'{model_name}/cross_fold/avg_memory_mib',
                     summary['avg_memory_across_folds_bytes'] / 1024**2, 0)
    writer.add_scalar(f'{model_name}/cross_fold/peak_time_ms',
                     summary['peak_time_across_folds_us'] / 1000, 0)
    writer.add_scalar(f'{model_name}/cross_fold/avg_time_ms',
                     summary['avg_time_across_folds_us'] / 1000, 0)
    writer.add_scalar(f'{model_name}/cross_fold/avg_flops',
                     summary['avg_flops_across_folds'], 0)

    # Wall-clock time scalars
    if summary['avg_wall_clock_across_folds_us'] > 0:
        writer.add_scalar(f'{model_name}/cross_fold/peak_wall_clock_ms',
                         summary['peak_wall_clock_across_folds_us'] / 1000, 0)
        writer.add_scalar(f'{model_name}/cross_fold/avg_wall_clock_ms',
                         summary['avg_wall_clock_across_folds_us'] / 1000, 0)
        writer.add_scalar(f'{model_name}/cross_fold/avg_wall_clock_us',
                         summary['avg_wall_clock_across_folds_us'], 0)

    # SNN-specific scalars
    if has_synops:
        writer.add_scalar(f'{model_name}/cross_fold/avg_synaptic_ops',
                         summary['avg_synaptic_ops_across_folds'], 0)
        writer.add_scalar(f'{model_name}/cross_fold/avg_synaptic_ops_per_sample',
                         summary['avg_synaptic_ops_per_sample'], 0)
        writer.add_scalar(f'{model_name}/cross_fold/avg_mac_ops',
                         summary['avg_mac_ops_across_folds'], 0)
        writer.add_scalar(f'{model_name}/cross_fold/avg_add_ops',
                         summary['avg_add_ops_across_folds'], 0)
        writer.add_scalar(f'{model_name}/cross_fold/avg_comparison_ops',
                         summary['avg_comparison_ops_across_folds'], 0)
        writer.add_scalar(f'{model_name}/cross_fold/avg_total_effective_flops',
                         summary['avg_total_effective_flops'], 0)
        writer.add_scalar(f'{model_name}/cross_fold/avg_synaptic_percentage',
                         summary['avg_synaptic_percentage'], 0)
        writer.add_scalar(f'{model_name}/cross_fold/avg_reduction_vs_dense',
                         summary['avg_reduction_vs_dense'], 0)
        writer.add_scalar(f'{model_name}/cross_fold/avg_encoding_spike_rate',
                         summary['avg_encoding_spike_rate'], 0)
        writer.add_scalar(f'{model_name}/cross_fold/avg_output_spike_rate',
                         summary['avg_output_spike_rate'], 0)

    writer.close()

    # Print summary to console
    print("\n" + "=" * 90)
    print(f"{model_name} Profiling Summary Across {num_folds} Folds")
    print("=" * 90)
    print(f"\n{'Fold':<8} {'Peak Mem (MiB)':<16} {'Avg Mem (MiB)':<16} {'Peak Time (ms)':<16} {'Avg Time (ms)':<14} {'Wall-Clock (ms)':<16}")
    print("-" * 90)
    for i in range(num_folds):
        wall_clock_ms = avg_wall_clock_per_fold[i] / 1000 if avg_wall_clock_per_fold[i] > 0 else 0
        print(f"{i:<8} {peak_memory_per_fold[i] / 1024**2:<16.2f} {avg_memory_per_fold[i] / 1024**2:<16.2f} "
              f"{peak_time_per_fold[i] / 1000:<16.2f} {avg_time_per_fold[i] / 1000:<14.2f} {wall_clock_ms:<16.3f}")
    print("-" * 90)
    avg_wall_clock_ms = summary['avg_wall_clock_across_folds_us'] / 1000 if summary['avg_wall_clock_across_folds_us'] > 0 else 0
    print(f"{'Peak':<8} {summary['peak_memory_across_folds_bytes'] / 1024**2:<16.2f} {'-':<16} "
          f"{summary['peak_time_across_folds_us'] / 1000:<16.2f} {'-':<14} {'-':<16}")
    print(f"{'Average':<8} {summary['avg_memory_across_folds_bytes'] / 1024**2:<16.2f} {'-':<16} "
          f"{summary['avg_time_across_folds_us'] / 1000:<16.2f} {'-':<14} {avg_wall_clock_ms:<16.3f}")
    print(f"\nAverage FLOPs across folds: {summary['avg_flops_across_folds']:,.0f}")

    # Print SynOps summary if applicable
    if has_synops:
        print(f"\n--- SNN Synaptic Operations ---")
        print(f"Average SynOps: {summary['avg_synaptic_ops_across_folds']:,.0f}")
        print(f"Average SynOps/sample: {summary['avg_synaptic_ops_per_sample']:,.0f}")
        print(f"Average Encoding Spike Rate: {summary['avg_encoding_spike_rate']:.4f}")
        print(f"Average Output Spike Rate: {summary['avg_output_spike_rate']:.4f}")
        print(f"FLOP Reduction vs Dense: {summary['avg_reduction_vs_dense']:.2f}x")

    print("=" * 90 + "\n")

    print(f"Cross-fold summary saved to: {log_dir}")
    return summary


def log_neurobench_to_tensorboard(neurobench_results: Dict,
                                   log_dir: str,
                                   model_name: str = "SNN",
                                   step: int = 0) -> None:
    """
    Log NeuroBench benchmark results to TensorBoard.

    Args:
        neurobench_results: Dictionary from Benchmark.run() with keys like
            'Footprint', 'ConnectionSparsity', 'ActivationSparsity', 'SynapticOperations'
        log_dir: Directory for TensorBoard logs
        model_name: Name prefix for logged metrics
        step: Step index (e.g., fold number)
    """
    writer = SummaryWriter(log_dir=log_dir)

    for metric_name, value in neurobench_results.items():
        if value is None:
            continue
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (int, float)):
                    writer.add_scalar(f'{model_name}/neurobench/{metric_name}/{sub_key}', sub_val, step)
        elif isinstance(value, (int, float)):
            writer.add_scalar(f'{model_name}/neurobench/{metric_name}', value, step)

    # Summary text table (skip list-typed entries like parameters/buffers)
    rows = ""
    for metric_name, value in neurobench_results.items():
        if value is None or isinstance(value, list):
            continue
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                rows += f"| {metric_name}/{sub_key} | {sub_val} |\n"
        else:
            rows += f"| {metric_name} | {value} |\n"

    # Build parameter breakdown table
    params_table = ""
    parameters = neurobench_results.get('parameters', [])
    if parameters:
        params_table = "\n### Parameter Details\n"
        params_table += "| Layer Name | Shape | Trainable | Params | Dtype |\n"
        params_table += "|------------|-------|-----------|--------|-------|\n"
        for name, shape, requires_grad, numel, dtype in parameters:
            trainable_str = "Yes" if requires_grad else "No"
            dtype_str = str(dtype).replace("torch.", "")
            params_table += f"| {name} | {shape} | {trainable_str} | {numel:,} | {dtype_str} |\n"

    # Build buffer breakdown table
    buffers_table = ""
    buffers = neurobench_results.get('buffers', [])
    if buffers:
        buffers_table = "\n### Buffer Details\n"
        buffers_table += "| Buffer Name | Shape | Elements | Dtype |\n"
        buffers_table += "|-------------|-------|----------|-------|\n"
        for name, shape, numel, dtype in buffers:
            dtype_str = str(dtype).replace("torch.", "")
            buffers_table += f"| {name} | {shape} | {numel:,} | {dtype_str} |\n"

    summary_text = f"""
## {model_name} NeuroBench Results (Fold {step})

| Metric | Value |
|--------|-------|
{rows}{params_table}{buffers_table}"""

    writer.add_text(f'{model_name}/neurobench_summary', summary_text, step)
    writer.close()
    print(f"NeuroBench TensorBoard logs saved to: {log_dir}")


def aggregate_neurobench_across_folds(neurobench_results_list: list,
                                       log_dir: str,
                                       model_name: str = "SNN") -> Dict:
    """
    Aggregate NeuroBench results across multiple folds and log a summary.

    Args:
        neurobench_results_list: List of NeuroBench result dicts (one per fold)
        log_dir: Directory for TensorBoard logs
        model_name: Name prefix for logged metrics

    Returns:
        Dictionary with per-metric mean/std across folds
    """
    if not neurobench_results_list:
        return {}

    # Flatten results: expand dict-valued metrics into key/subkey pairs
    def flatten_results(results_dict):
        flat = {}
        for key, value in results_dict.items():
            if value is None:
                continue
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, (int, float)):
                        flat[f'{key}/{sub_key}'] = sub_val
            elif isinstance(value, (int, float)):
                flat[key] = value
        return flat

    flattened_list = [flatten_results(r) for r in neurobench_results_list]

    # Collect all flattened keys
    all_keys = set()
    for r in flattened_list:
        all_keys.update(r.keys())

    num_folds = len(flattened_list)
    summary = {'num_folds': num_folds}

    writer = SummaryWriter(log_dir=log_dir)

    # Per-fold table
    fold_rows = ""
    for fold_i, r in enumerate(flattened_list):
        row_vals = " | ".join(f"{r.get(k, 'N/A'):.4f}" if isinstance(r.get(k), float)
                              else str(r.get(k, 'N/A'))
                              for k in sorted(all_keys))
        fold_rows += f"| Fold {fold_i} | {row_vals} |\n"

    header_cols = " | ".join(sorted(all_keys))
    header_sep = " | ".join(["-------"] * len(all_keys))

    # Aggregate statistics
    agg_rows = ""
    for key in sorted(all_keys):
        values = [r[key] for r in flattened_list if key in r]
        if values:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            summary[f'{key}_mean'] = mean_val
            summary[f'{key}_std'] = std_val
            writer.add_scalar(f'{model_name}/neurobench_cross_fold/{key}_mean', mean_val, 0)
            writer.add_scalar(f'{model_name}/neurobench_cross_fold/{key}_std', std_val, 0)
            agg_rows += f"| {key} | {mean_val:.4f} | {std_val:.4f} |\n"

    summary_text = f"""
## {model_name} NeuroBench Summary Across {num_folds} Folds

### Per-Fold Results
| Fold | {header_cols} |
|------|{header_sep}|
{fold_rows}
### Aggregated Results
| Metric | Mean | Std |
|--------|------|-----|
{agg_rows}"""

    writer.add_text(f'{model_name}/neurobench_cross_fold_summary', summary_text, 0)
    writer.close()

    # Print to console
    print("\n" + "=" * 70)
    print(f"{model_name} NeuroBench Summary Across {num_folds} Folds")
    print("=" * 70)
    for key in sorted(all_keys):
        values = [r[key] for r in flattened_list if key in r]
        if values:
            print(f"  {key}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
    print("=" * 70 + "\n")

    print(f"NeuroBench cross-fold summary saved to: {log_dir}")
    return summary


# ---------------------------------------------------------------------------
# Per-fold profiling orchestration
# ---------------------------------------------------------------------------

def run_snn_fold_profiling(snn_config, data_config, unsegmented_dataset,
                            rep_on_rep, fold_i, subject_id, base_log_dir="./logs/snn_profiler"):
    """
    Profile the SNN on unsegmented data for one fold and log results to TensorBoard.

    Args:
        snn_config: Nested Hydra/wandb SNN config.
        data_config: DataConfig instance.
        unsegmented_dataset: EMGDataset built from unsegmented inference split.
        rep_on_rep: Fold string e.g. '1on2'.
        fold_i: Fold index (used as TensorBoard step).
        subject_id: Remapped subject identifier string.
        base_log_dir: Root directory for TensorBoard logs.

    Returns:
        (profiling_results, neurobench_results) — neurobench_results may be None.
    """
    from force_regression.models.snn import profile_snn_on_unsegmented_data

    topology = snn_config.decoder_type["topology"]
    profiling_results = profile_snn_on_unsegmented_data(
        snn_config, data_config, unsegmented_dataset, rep_on_rep, fold_i, num_profile_samples=5
    )
    neurobench_results = profiling_results.pop('neurobench_results', None)

    log_profiling_to_tensorboard(
        profiling_results=profiling_results,
        model=None,
        log_dir=create_profiling_log_dir(
            base_dir=base_log_dir,
            subject=subject_id,
            direction=data_config.direction,
            model_type=f"snn_{topology}",
        ),
        model_name="SNN",
        step=fold_i,
        log_model_graph=False,
        input_tensor=None,
    )

    if neurobench_results is not None:
        log_neurobench_to_tensorboard(
            neurobench_results=neurobench_results,
            log_dir=create_profiling_log_dir(
                base_dir=base_log_dir,
                subject=subject_id,
                direction=data_config.direction,
                model_type=f"snn_{topology}_neurobench",
            ),
            model_name="SNN",
            step=fold_i,
        )

    return profiling_results, neurobench_results


def run_rnn_neurobench(rnn_reg, test_dataset, data_config, rnn_config,
                       fold_i, subject_id, base_log_dir="./logs/rnn_profiler"):
    """
    Run NeuroBench static benchmark on the RNN model and log results to TensorBoard.

    Args:
        rnn_reg: Trained RNN model instance.
        test_dataset: PyTorch Dataset for the test fold.
        data_config: DataConfig instance.
        rnn_config: Nested Hydra/wandb RNN config.
        fold_i: Fold index (used as TensorBoard step).
        subject_id: Remapped subject identifier string.
        base_log_dir: Root directory for TensorBoard logs.

    Returns:
        neurobench_results dict (augmented with parameter/buffer metadata).
    """
    from torch.utils.data import DataLoader
    from neurobench.benchmarks import Benchmark
    from neurobench.metrics.static import ConnectionSparsity, Footprint
    from neurobench.models import TorchModel

    neuro_model = TorchModel(rnn_reg)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    nb_benchmark = Benchmark(neuro_model, test_loader, [], [], [[Footprint, ConnectionSparsity], []])
    neurobench_results = nb_benchmark.run(device='cpu')

    neurobench_results['parameters'] = [
        (name, list(p.shape), p.requires_grad, p.numel(), p.dtype)
        for name, p in rnn_reg.named_parameters()
    ]
    neurobench_results['buffers'] = [
        (name, list(b.shape), b.numel(), b.dtype)
        for name, b in rnn_reg.named_buffers()
    ]

    log_neurobench_to_tensorboard(
        neurobench_results=neurobench_results,
        log_dir=create_profiling_log_dir(
            base_dir=base_log_dir,
            subject=subject_id,
            direction=data_config.direction,
            model_type=f"rnn_{rnn_config.decoder_type['rnn_type']}_neurobench",
        ),
        model_name="RNN",
        step=fold_i,
    )
    return neurobench_results


def run_rnn_fold_profiling(rnn_reg, flat_rnn_config, test_dataset, data_config, rnn_config,
                            fold_i, subject_id, timesteps, num_inputs,
                            base_log_dir="./logs/rnn_profiler"):
    """
    Profile the RNN on the test split for one fold and log results to TensorBoard.

    Calls evaluate_rnn with profiling enabled, logs the profiling results, then
    runs the NeuroBench static benchmark.

    Args:
        rnn_reg: Trained RNN model instance.
        flat_rnn_config: Flat SimpleNamespace config (used by evaluate_rnn).
        test_dataset: PyTorch Dataset for the test fold.
        data_config: DataConfig instance.
        rnn_config: Nested Hydra/wandb RNN config.
        fold_i: Fold index (used as TensorBoard step).
        subject_id: Remapped subject identifier string.
        timesteps: Number of time steps in one sample (for sample input tensor).
        num_inputs: Number of input channels (for sample input tensor).
        base_log_dir: Root directory for TensorBoard logs.

    Returns:
        Tuple of (test_eloss, test_predicted_values_cache, test_true_values_cache,
                  test_loss_per_batch, profiling_results, neurobench_results).
    """
    import torch
    from force_regression.models.rnn import evaluate_rnn

    test_eloss, test_predicted_values_cache, test_true_values_cache, test_loss_per_batch, profiling_results = evaluate_rnn(
        rnn_reg, flat_rnn_config, test_dataset, enable_profiling=True
    )

    sample_input = torch.randn(1, timesteps, num_inputs)
    log_profiling_to_tensorboard(
        profiling_results=profiling_results,
        model=rnn_reg,
        log_dir=create_profiling_log_dir(
            base_dir=base_log_dir,
            subject=subject_id,
            direction=data_config.direction,
            model_type=f"rnn_{rnn_config.decoder_type['rnn_type']}",
        ),
        model_name="RNN",
        step=fold_i,
        log_model_graph=(fold_i == 0),
        input_tensor=sample_input if fold_i == 0 else None,
    )

    neurobench_results = run_rnn_neurobench(
        rnn_reg, test_dataset, data_config, rnn_config, fold_i, subject_id, base_log_dir
    )

    return (test_eloss, test_predicted_values_cache, test_true_values_cache,
            test_loss_per_batch, profiling_results, neurobench_results)
