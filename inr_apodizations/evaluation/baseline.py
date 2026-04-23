"""Baseline DAS evaluation helpers for INR apodization experiments.

Provides dataset-weighting, reference apodization computation, batch
baseline-image generation, and MAE reference calculation. These helpers
are used by both the standalone baseline evaluation script and the INR
training loop to establish pre-training reference metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from inr_apodizations.apodizations import (
    compute_das_baseline_numpy,
    compute_dynamic_apodizations_tf,
)
from inr_apodizations.evaluation.metrics import compute_validation_mae_and_scatterer_metrics
from inr_apodizations.evaluation.summary import (
    build_snr_persistence_bundle,
    extract_snr_from_metrics,
)
from inr_apodizations.modeling.metrics import PixelWeightedMAE
import inr_apodizations.sandbox_helpers as helpers


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


def build_validation_weights(gaussian_masks: np.ndarray, cfg_user: dict) -> np.ndarray:
    """Build per-pixel validation weights from Gaussian masks.

    Args:
        gaussian_masks: Gaussian mask tensor with shape ``(N, Z, X)``.
        cfg_user: Training/baseline configuration mapping.

    Returns:
        Per-pixel weights with the same shape as ``gaussian_masks``.

    Raises:
        ValueError: If the resolved lambda is negative.
    """
    mask_weighting_cfg = dict(cfg_user["training"].get("mask_weighting", {}))
    pixel_weight_lambda = float(
        mask_weighting_cfg.get(
            "pixel_weight_lambda",
            mask_weighting_cfg.get("lambda", 0.0),
        )
    )
    if pixel_weight_lambda < 0.0:
        raise ValueError(
            "training.mask_weighting.pixel_weight_lambda must be >= 0 "
            "(legacy key training.mask_weighting.lambda is also accepted)"
        )
    weights = 1.0 + pixel_weight_lambda * gaussian_masks.astype(np.float32, copy=False)
    return weights.astype(np.float32, copy=False)


def resolve_baseline_f_number(cfg_user: dict, kp: object) -> float:
    """Resolve the baseline f-number used for Hanning and boxcar references.

    Args:
        cfg_user: Training/baseline configuration mapping. Reads
            ``training.baseline_f_number`` when present.
        kp: KernelParameters2D instance used as fallback source.

    Returns:
        Resolved f-number as a float.
    """
    baseline_f_number_cfg = cfg_user["training"].get("baseline_f_number", None)
    if baseline_f_number_cfg is None:
        return float(getattr(kp, "f_number"))
    try:
        return float(baseline_f_number_cfg)
    except Exception:
        return float(getattr(kp, "f_number"))


def compute_reference_apodizations(
    cm,
    baseline_f_number: float,
    scaled_features: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Hanning and boxcar apodization tensors as NumPy arrays.

    Args:
        cm: CoordinateManager instance.
        baseline_f_number: F-number for the dynamic aperture.
        scaled_features: Whether to use scaled CoordinateManager features.

    Returns:
        Tuple ``(hanning_weights, boxcar_weights)`` each with shape
        ``(n_elements, nz, nx)`` and dtype ``float32``.

    Raises:
        ValueError: If the apodization tensors could not be computed.
    """
    apods_h = compute_dynamic_apodizations_tf(
        cm=cm,
        f_number=baseline_f_number,
        methods=("hanning",),
        scaled=scaled_features,
    )
    apods_b = compute_dynamic_apodizations_tf(
        cm=cm,
        f_number=baseline_f_number,
        methods=("boxcar",),
        scaled=scaled_features,
    )
    hanning_weights = apods_h.get("hanning")
    boxcar_weights = apods_b.get("boxcar")
    if hanning_weights is None or boxcar_weights is None:
        raise ValueError(
            f"Reference apodizations were not computed for f_number={baseline_f_number}."
        )
    return hanning_weights.numpy(), boxcar_weights.numpy()


def compute_validation_baseline_metrics(
    delayed: np.ndarray,
    noise: np.ndarray | None,
    validation_indices: np.ndarray,
    targets: np.ndarray,
    validation_sample_weights: np.ndarray,
    hanning_weights_np: np.ndarray,
    boxcar_weights_np: np.ndarray,
    eval_noise_enabled: bool,
    eval_noise_scale: float,
    eval_batch_size: int,
    cm,
    scatterers_batch: list[np.ndarray],
    radius_mm: float,
    hist_bins: int,
) -> tuple[dict[str, np.ndarray], dict, dict[str, float]]:
    """Compute baseline DAS images, validation metrics and reference MAE values.

    Processes the validation split in mini-batches to keep memory bounded.
    Produces uniform, Hanning and boxcar DAS images along with weighted-MAE
    metrics relative to the targets.

    Args:
        delayed: Full dataset of delayed samples ``(N, E, Z, X)``, complex.
        noise: Optional pre-computed noise array ``(N, E, Z, X)``, complex.
        validation_indices: 1-D array of integer indices into ``delayed``.
        targets: Full dataset of target images ``(N, Z, X)``, float32.
        validation_sample_weights: Per-pixel weights aligned with
            ``validation_indices``, shape ``(n_val, Z, X)``.
        hanning_weights_np: Hanning apodization ``(E, Z, X)``, float32.
        boxcar_weights_np: Boxcar apodization ``(E, Z, X)``, float32.
        eval_noise_enabled: Whether to add ``noise * eval_noise_scale``.
        eval_noise_scale: Scale factor applied to the noise array.
        eval_batch_size: Number of examples processed per iteration.
        cm: CoordinateManager instance.
        scatterers_batch: Per-example scatterer positions in mm (x, z).
        radius_mm: Disk radius for scatterer SNR evaluation.
        hist_bins: Number of bins for background histogram.

    Returns:
        Tuple ``(images_abs_eval, validation_bundle, pre_training_reference_mae)``:
        - ``images_abs_eval``: dict mapping method names to ``(n_val, Z, X)``
          absolute-value images.
        - ``validation_bundle``: output of
          :func:`~metrics.compute_validation_mae_and_scatterer_metrics`.
        - ``pre_training_reference_mae``: dict with keys ``"zero"``,
          ``"uniform"``, ``"hanning"``, ``"boxcar"``.

    Raises:
        ValueError: If ``eval_noise_enabled`` is True but ``noise`` is None.
    """
    validation_targets = targets[validation_indices].astype(np.float32, copy=False)

    m_zero = PixelWeightedMAE(name="ref_zero")
    m_uniform = PixelWeightedMAE(name="ref_uniform")
    m_hanning = PixelWeightedMAE(name="ref_hanning")
    m_boxcar = PixelWeightedMAE(name="ref_boxcar")

    uniform_val_abs_list: list[np.ndarray] = []
    hanning_val_abs_list: list[np.ndarray] = []
    boxcar_val_abs_list: list[np.ndarray] = []

    n_val = int(len(validation_indices))
    for start in range(0, n_val, eval_batch_size):
        end = min(start + eval_batch_size, n_val)
        idx_chunk = validation_indices[start:end]
        local_slice = slice(start, end)

        if eval_noise_enabled:
            if noise is None:
                raise ValueError(
                    "Eval noise requested but dataset does not contain precomputed noise."
                )
            sig_chunk = delayed[idx_chunk].astype(np.complex64, copy=False)
            noise_chunk = noise[idx_chunk].astype(np.complex64, copy=False)
            val_delayed_chunk_np = sig_chunk + eval_noise_scale * noise_chunk
        else:
            val_delayed_chunk_np = delayed[idx_chunk].astype(np.complex64, copy=False)

        y_true_chunk = tf.convert_to_tensor(validation_targets[local_slice])
        weights_chunk = tf.convert_to_tensor(validation_sample_weights[local_slice])

        m_zero.update_state(y_true_chunk, tf.zeros_like(y_true_chunk), sample_weight=weights_chunk)

        uniform_pred_np = compute_das_baseline_numpy(val_delayed_chunk_np)
        hanning_pred_np = compute_das_baseline_numpy(
            delayed_samples=val_delayed_chunk_np,
            apodization=hanning_weights_np,
        )
        boxcar_pred_np = compute_das_baseline_numpy(
            delayed_samples=val_delayed_chunk_np,
            apodization=boxcar_weights_np,
        )

        m_uniform.update_state(
            y_true_chunk,
            tf.convert_to_tensor(uniform_pred_np, dtype=tf.float32),
            sample_weight=weights_chunk,
        )
        m_hanning.update_state(
            y_true_chunk,
            tf.convert_to_tensor(hanning_pred_np, dtype=tf.float32),
            sample_weight=weights_chunk,
        )
        m_boxcar.update_state(
            y_true_chunk,
            tf.convert_to_tensor(boxcar_pred_np, dtype=tf.float32),
            sample_weight=weights_chunk,
        )

        uniform_val_abs_list.append(uniform_pred_np)
        hanning_val_abs_list.append(hanning_pred_np)
        boxcar_val_abs_list.append(boxcar_pred_np)

    images_abs_eval = {
        "uniform": np.concatenate(uniform_val_abs_list, axis=0),
        "hanning": np.concatenate(hanning_val_abs_list, axis=0),
        "boxcar": np.concatenate(boxcar_val_abs_list, axis=0),
    }

    validation_bundle = compute_validation_mae_and_scatterer_metrics(
        images_abs=images_abs_eval,
        targets=validation_targets,
        scatterers_xy=scatterers_batch,
        cm=cm,
        sample_weights=validation_sample_weights,
        radius_mm=radius_mm,
        hist_bins=hist_bins,
    )

    pre_training_reference_mae = {
        "zero": float(m_zero.result().numpy()),
        "uniform": float(m_uniform.result().numpy()),
        "hanning": float(m_hanning.result().numpy()),
        "boxcar": float(m_boxcar.result().numpy()),
    }

    return images_abs_eval, validation_bundle, pre_training_reference_mae


def build_scatterer_reference_bundle(
    scatterer_metrics: dict,
    scatterers_batch: list[np.ndarray],
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Summarize scatterer metrics and produce arrays for SNR persistence.

    Args:
        scatterer_metrics: Dict mapping method names to their metric bundles.
        scatterers_batch: Per-example scatterer coordinate arrays in mm.

    Returns:
        Tuple ``(summary_dict, arrays_dict)`` as produced by
        :func:`~summary.build_snr_persistence_bundle`.
    """
    return build_snr_persistence_bundle(
        metrics_dict=scatterer_metrics,
        scatterers_xy=scatterers_batch,
    )


def extract_scatterer_snr(metrics: dict) -> np.ndarray:
    """Return pointwise SNR values from a scatterer-metric bundle.

    Args:
        metrics: Output of :func:`~metrics.compute_scatterer_metrics`.

    Returns:
        1-D array of SNR values (peak / background_rms) per scatterer.
    """
    return extract_snr_from_metrics(metrics)


def find_latest_baseline_reference(
    dataset_folder: str,
    baseline_f_number: float,
    baseline_output_root: str | Path,
) -> tuple[dict | None, dict | None]:
    """Find the most recent matching baseline summary and its SNR arrays.

    Scans the baseline output root for ``baseline_summary.json`` files that
    match ``dataset_folder`` and ``baseline_f_number``, ordered by
    modification time (most recent first).

    Search is backward compatible with historical layouts:
    1) ``<baseline_output_root>/`` (current default)
    2) ``<baseline_output_root>/baseline_apodizations/`` (legacy)
    3) ``<baseline_output_root>/evaluation/baseline/`` (legacy)

    Args:
        dataset_folder: Absolute path to the delayed-samples dataset used
            during the baseline run.
        baseline_f_number: F-number that must match the candidate summary.
        baseline_output_root: Root folder where baseline outputs are stored.

    Returns:
        Tuple ``(summary_dict, arrays_dict)`` for the best matching run, or
        ``(None, None)`` if no match is found.
    """
    baseline_output_root = Path(baseline_output_root)
    candidate_roots = [
        baseline_output_root,
        baseline_output_root / "baseline_apodizations",
        baseline_output_root / "evaluation" / "baseline",
    ]

    existing_roots: list[Path] = []
    seen: set[Path] = set()
    for root in candidate_roots:
        if root.exists() and root not in seen:
            existing_roots.append(root)
            seen.add(root)

    if not existing_roots:
        return None, None

    summary_candidates: list[Path] = []
    for root in existing_roots:
        summary_candidates.extend(root.rglob("baseline_summary.json"))

    summary_candidates = sorted(summary_candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    for summary_path in summary_candidates:
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                candidate_summary = json.load(handle)
        except Exception:
            continue

        if str(candidate_summary.get("dataset_folder")) != str(dataset_folder):
            continue
        candidate_f_number = candidate_summary.get("baseline_f_number")
        if candidate_f_number is not None and abs(float(candidate_f_number) - float(baseline_f_number)) > 1e-9:
            continue

        npz_path = summary_path.with_name("scatterer_reference_metrics.npz")
        if not npz_path.exists():
            continue

        with np.load(npz_path, allow_pickle=True) as baseline_npz:
            baseline_reference_arrays = {key: np.array(baseline_npz[key]) for key in baseline_npz.files}
        return candidate_summary, baseline_reference_arrays

    return None, None
