"""Backward-compatibility shim for baseline apodization evaluation helpers.

All substantive logic has been migrated to ``inr_apodizations.evaluation.baseline``.
This module re-exports the public API so that existing callers keep working without
changes.  Only ``load_validation_scatterers`` lives here because it depends on the
sandbox-only ``helpers`` module.
"""

from __future__ import annotations

import numpy as np

from inr_apodizations.evaluation.baseline import (  # noqa: F401
    build_scatterer_reference_bundle,
    build_validation_weights,
    compute_reference_apodizations,
    compute_validation_baseline_metrics,
    extract_scatterer_snr,
    find_latest_baseline_reference,
    resolve_baseline_f_number,
)

import helpers


def load_validation_scatterers(
    dataset_folder: str, validation_indices: np.ndarray
) -> list[np.ndarray]:
    """Load scatterers for the requested validation indices in millimeters.

    Args:
        dataset_folder: Path string to the delayed-samples dataset folder.
        validation_indices: Array of integer indices to load.

    Returns:
        List of ``(n_scatterers, 2)`` arrays with [x_mm, z_mm] per example.
    """
    scatterers_all = helpers.load_saved_scatterers(dataset_folder)
    scatterers_batch: list[np.ndarray] = []
    for idx in validation_indices:
        scatterers_example = np.asarray(scatterers_all[int(idx)], dtype=np.float32).copy()
        scatterers_example[:, :2] *= 1000.0
        scatterers_batch.append(scatterers_example[:, :2])
    return scatterers_batch
