'''
Neurocell v2: GPU-Optimal Von Neumann Networks (VNNs) in JAX.

Copyright (c) 2026 Shekhar S. Chandra. All rights reserved.
Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

@author Shakes
'''
__author__ = "Shekhar S. Chandra (Shakes)"
__copyright__ = "Copyright (c) 2026 Shekhar S. Chandra"
__license__ = "LGPL-3.0"
__version__ = "1.0.0"

from neurocell.topology import PortSpec, CoordinatePortSpec, ChipTopology2D
from neurocell.params import (
    STATE_ACTIVE,
    STATE_SIGNAL,
    STATE_ACTIVE_VALUE,
    STATE_SIGNAL_VALUE,
    SUPPORTED_CODD_MODES,
    VNNParams,
    PrecomputedStepParams,
    compute_codd_state_probs,
    init_vnn_params,
    precompute_step_params,
    export_legacy_haiku_params,
    import_legacy_haiku_params,
)
from neurocell import filters
from neurocell.kernels_greens import (
    rfft2_pruned,
    irfft2_pruned,
    conv2d_grouped_filter_grad,
    install_lean_rfft_transpose,
    conv2d_batched_nchw,
    apply_spatial_filter_2d,
    precompute_filter_rfft2,
    rfft2_filter_grad_to_oihw,
    hoist_filter_rfft2,
    conv2d_roi_rfft_step,
    conv2d_valid_rfft_precomputed,
    hoist_filter_rfft2_zero_fwd,
    conv2d_hybrid_roi_step,
)
from neurocell.kernels_pallas import (
    is_supported_pallas_activation,
    pallas_fused_roi_step_epilogue,
    pallas_standardize_output_port,
    pallas_fused_output_port_epilogue,
    rfft2_unshifted_filter_grad_to_oihw,
    hoist_filter_rfft2_nopad,
    hoist_filter_rfft2_zero_nopad,
    conv2d_hybrid_nopad_roi_step,
    conv2d_small_kernel_roi_step,
    conv2d_pruned_rfft_roi_step,
)
from neurocell.epilogue import (
    standardize_spatial_nchw,
    fused_roi_step_epilogue,
    fused_roi_epilogue,
    fused_output_port_epilogue,
)
from neurocell.propagator import (
    SUPPORTED_BACKENDS,
    select_optimal_backend,
    inject_inputs_2d,
    inject_inputs_roi_2d,
    propagate_vnn_2d,
)
from neurocell.adaptive_mask import (
    AdaptiveMaskSpec,
    estimate_effective_filter_radii,
    compute_adaptive_spatial_mask,
)
from neurocell.model import VonNeumannNetwork
from neurocell.datasets import (
    ALUDeviceDataset,
    generate_arithmetic,
    generate_arithmetic_logic,
    prepare_alu_device_dataset,
)
from neurocell.trainer import (
    VNNTrainer,
    compute_annealed_temperature,
    packbits_u8_on_device,
    default_mse_loss,
    default_exact_match_metric,
)
from neurocell.visualize import (
    unroll_vnn_propagation_2d,
    load_checkpoint_with_topology,
    compute_vnn_diagnostics,
    plot_vnn_parameter_suite,
    animate_vnn_propagation,
    export_interactive_explorer_html,
)

__all__ = [
    "PortSpec",
    "CoordinatePortSpec",
    "ChipTopology2D",
    "STATE_ACTIVE",
    "STATE_SIGNAL",
    "STATE_ACTIVE_VALUE",
    "STATE_SIGNAL_VALUE",
    "SUPPORTED_CODD_MODES",
    "VNNParams",
    "PrecomputedStepParams",
    "compute_codd_state_probs",
    "init_vnn_params",
    "precompute_step_params",
    "export_legacy_haiku_params",
    "import_legacy_haiku_params",
    "filters",
    "rfft2_pruned",
    "irfft2_pruned",
    "conv2d_grouped_filter_grad",
    "install_lean_rfft_transpose",
    "conv2d_batched_nchw",
    "apply_spatial_filter_2d",
    "precompute_filter_rfft2",
    "rfft2_filter_grad_to_oihw",
    "hoist_filter_rfft2",
    "conv2d_roi_rfft_step",
    "conv2d_valid_rfft_precomputed",
    "hoist_filter_rfft2_zero_fwd",
    "conv2d_hybrid_roi_step",
    "is_supported_pallas_activation",
    "pallas_fused_roi_step_epilogue",
    "pallas_standardize_output_port",
    "pallas_fused_output_port_epilogue",
    "rfft2_unshifted_filter_grad_to_oihw",
    "hoist_filter_rfft2_nopad",
    "hoist_filter_rfft2_zero_nopad",
    "conv2d_hybrid_nopad_roi_step",
    "conv2d_small_kernel_roi_step",
    "conv2d_pruned_rfft_roi_step",
    "standardize_spatial_nchw",
    "fused_roi_step_epilogue",
    "fused_roi_epilogue",
    "fused_output_port_epilogue",
    "SUPPORTED_BACKENDS",
    "select_optimal_backend",
    "inject_inputs_2d",
    "inject_inputs_roi_2d",
    "propagate_vnn_2d",
    "AdaptiveMaskSpec",
    "estimate_effective_filter_radii",
    "compute_adaptive_spatial_mask",
    "VonNeumannNetwork",
    "ALUDeviceDataset",
    "generate_arithmetic",
    "generate_arithmetic_logic",
    "prepare_alu_device_dataset",
    "VNNTrainer",
    "compute_annealed_temperature",
    "packbits_u8_on_device",
    "default_mse_loss",
    "default_exact_match_metric",
    "unroll_vnn_propagation_2d",
    "load_checkpoint_with_topology",
    "compute_vnn_diagnostics",
    "plot_vnn_parameter_suite",
    "animate_vnn_propagation",
    "export_interactive_explorer_html",
]
