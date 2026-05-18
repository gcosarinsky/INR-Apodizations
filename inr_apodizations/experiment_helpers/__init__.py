"""Helpers for INR DAS experiment workflows.

This package centralizes utilities used by training, tuning, and
baseline-evaluation scripts, so all consumers import from the package namespace.

Internal modules (prefixed with ``_``) are not part of the public API. All
public symbols are re-exported here so consumer scripts can continue using
``import inr_apodizations.experiment_helpers as helpers`` without changes.
"""

from __future__ import annotations

# Hardware / GPU diagnostics
from ._hardware import (
    BYTES_PER_GB,
    bytes_to_gb,
    get_tf_available_vram_info,
    gpu_mem,
)

# Run artifact persistence
from ._artifacts import (
    create_run_directories,
    history_length,
    load_previous_history,
    merge_training_histories,
    save_artifacts,
    save_debug_arrays,
    save_json_artifact,
    to_json_serializable,
)

# Keras Tuner utilities
from ._tuning import save_tuner_architecture_scores

# Dataset I/O and validation
from ._dataset_io import (
    build_coordinate_manager,
    load_delayed_samples_dataset,
    load_saved_beamforming_config,
    load_saved_scatterers,
    validate_dataset_shapes,
)

# Configuration and resume parsing
from ._config import (
    get_target_regeneration_override,
    load_experiment_config,
    parse_resume_config,
    parse_weight_regularization_config,
    setup_resume_state,
)

# Training pipeline setup
from ._training import (
    build_gaussian_loss_weights,
    build_training_pipelines,
    prepare_training_dataset,
    setup_output_directories_and_callbacks,
)

# Plot utilities
from ._plotting import (
    build_reflector_profile_context,
    plot_apodization_energy_comparison,
)

# ---------------------------------------------------------------------------
# Convenience re-exports from other inr_apodizations modules.
# These allow consumer scripts to access everything via ``helpers.XXX``
# without importing each sub-module directly.
# ---------------------------------------------------------------------------
from inr_apodizations.apodizations import compute_das_baseline_numpy
from inr_apodizations.dataset import (
    build_tf_dataset_by_examples,
    build_tf_dataset_by_indices,
    generate_unit_gaussian_mask,
    split_train_validation_examples,
    split_train_validation_indices,
)
from inr_apodizations.evaluation import (
    compute_scatterer_metrics,
    compute_validation_and_reference_metrics,
)
from inr_apodizations.evaluation.scatterers import (
    plot_scatterer_evaluation,
    plot_scatterer_snr_ratio,
)
from inr_apodizations.plots import (
    plot_apodization_before_after,
    plot_das_comparison_db,
    plot_training_curves,
    to_db,
)
