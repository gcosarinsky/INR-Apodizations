"""Evaluation utilities for INR apodization experiments.

This sub-package provides scatterer-centric evaluation tools:
- Disk/region masks for SNR computation (regions.py)
- Lateral/axial profile extraction and FWHM measurement (profiles.py)
- Unified SNR calculation (snr.py)
- Batch input normalization and validation (batch.py)
- Aggregation and persistence helpers (summary.py)
- High-level orchestrator combining all of the above (metrics.py)
- Baseline DAS reference computation (baseline.py)
- Scatterer selection utilities for profile plots (scatterers.py)
"""

from inr_apodizations.evaluation.batch import (
    normalize_scatterer_batch,
    validate_image_batch,
)
from inr_apodizations.evaluation.regions import (
    build_disk_context,
    build_roi_masks,
    compute_background_statistics,
)
from inr_apodizations.evaluation.profiles import (
    extract_reflector_profiles,
    compute_fwhm,
    compute_fwhm_batch,
)
from inr_apodizations.evaluation.snr import (
    compute_reflector_snr,
)
from inr_apodizations.evaluation.summary import (
    aggregate_reflector_metrics,
    build_snr_persistence_bundle,
    extract_snr_from_metrics,
)
from inr_apodizations.evaluation.metrics import (
    compute_scatterer_metrics,
    compute_validation_and_reference_metrics,
    compute_validation_mae_and_scatterer_metrics,
)
from inr_apodizations.evaluation.baseline import (
    build_validation_weights,
    resolve_baseline_f_number,
    compute_reference_apodizations,
    compute_validation_baseline_metrics,
    build_scatterer_reference_bundle,
    extract_scatterer_snr,
    find_latest_baseline_reference,
)
from inr_apodizations.evaluation.scatterers import (
    select_reflector_scatterer_index,
    select_reflector_scatterer,
)

__all__ = [
    # batch
    "normalize_scatterer_batch",
    "validate_image_batch",
    # regions
    "build_disk_context",
    "build_roi_masks",
    "compute_background_statistics",
    # profiles
    "extract_reflector_profiles",
    "compute_fwhm",
    "compute_fwhm_batch",
    # snr
    "compute_reflector_snr",
    # summary
    "aggregate_reflector_metrics",
    "build_snr_persistence_bundle",
    "extract_snr_from_metrics",
    # high-level
    "compute_scatterer_metrics",
    "compute_validation_and_reference_metrics",
    "compute_validation_mae_and_scatterer_metrics",
    # baseline
    "build_validation_weights",
    "resolve_baseline_f_number",
    "compute_reference_apodizations",
    "compute_validation_baseline_metrics",
    "build_scatterer_reference_bundle",
    "extract_scatterer_snr",
    "find_latest_baseline_reference",
    # scatterers
    "select_reflector_scatterer_index",
    "select_reflector_scatterer",
    "compute_validation_mae_and_scatterer_metrics",
]
