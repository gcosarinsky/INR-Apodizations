"""Shared helpers for baseline apodization evaluation.

This module centralizes the dataset-weighting, baseline DAS computation,
scatterer loading, and SNR reference persistence logic used by both the
standalone baseline script and the INR training script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from inr_apodizations.apodizations import compute_dynamic_apodizations_tf
from inr_apodizations.modeling.metrics import PixelWeightedMAE

import helpers


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


def load_validation_scatterers(dataset_folder: str, validation_indices: np.ndarray) -> list[np.ndarray]:
    """Load scatterers for the requested validation indices in millimeters."""
    scatterers_all = helpers.load_saved_scatterers(dataset_folder)
    scatterers_batch: list[np.ndarray] = []
    for idx in validation_indices:
        scatterers_example = np.asarray(scatterers_all[int(idx)], dtype=np.float32).copy()
        scatterers_example[:, :2] *= 1000.0
        scatterers_batch.append(scatterers_example[:, :2])
    return scatterers_batch


def resolve_baseline_f_number(cfg_user: dict, kp: object) -> float:
    """Resolve the baseline f-number used for Hanning and boxcar references."""
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
    """Compute Hanning and boxcar apodization tensors as NumPy arrays."""
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
    """Compute baseline images, validation metrics, and reference MAE values."""
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
        # `validation_sample_weights` is expected to be aligned with
        # `validation_indices` (length == n_val). Use the local slice
        # here instead of indexing by absolute dataset indices `idx_chunk`.
        weights_chunk = tf.convert_to_tensor(validation_sample_weights[local_slice])

        m_zero.update_state(y_true_chunk, tf.zeros_like(y_true_chunk), sample_weight=weights_chunk)

        uniform_pred_np = helpers.compute_das_baseline_numpy(val_delayed_chunk_np)
        hanning_pred_np = helpers.compute_das_baseline_numpy(
            delayed_samples=val_delayed_chunk_np,
            apodization=hanning_weights_np,
        )
        boxcar_pred_np = helpers.compute_das_baseline_numpy(
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

    validation_bundle = helpers.compute_validation_mae_and_scatterer_metrics(
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
    """Summarize scatterer metrics and persist arrays needed for SNR ratios."""
    summary: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}

    first_positions = np.vstack([np.asarray(item, dtype=np.float32)[:, :2] for item in scatterers_batch])
    arrays["scatterer_x_mm"] = first_positions[:, 0].astype(np.float32, copy=False)
    arrays["scatterer_z_mm"] = first_positions[:, 1].astype(np.float32, copy=False)

    for method_name, metrics in scatterer_metrics.items():
        view = metrics.get("aggregated", metrics)
        peaks = np.asarray(view.get("peak_amplitudes", np.empty(0)), dtype=np.float64)
        bg_rms = np.asarray(view.get("point_background_rms", np.empty(0)), dtype=np.float64)
        if bg_rms.size == 0:
            bg_rms_per_example = np.asarray(
                view.get("background_rms_per_example", np.empty(0)), dtype=np.float64
            )
            example_indices = np.asarray(view.get("peak_example_indices", np.empty(0)), dtype=np.int32)
            if bg_rms_per_example.size > 0 and example_indices.size == peaks.size:
                bg_rms = bg_rms_per_example[example_indices]
            else:
                bg_rms = np.full(peaks.shape, max(float(view.get("background_rms", 0.0)), 1e-12))

        snr = peaks / np.maximum(bg_rms, 1e-12)
        arrays[f"peak_amplitudes_{method_name}"] = peaks.astype(np.float32, copy=False)
        arrays[f"background_rms_{method_name}"] = bg_rms.astype(np.float32, copy=False)
        arrays[f"snr_{method_name}"] = snr.astype(np.float32, copy=False)
        arrays[f"peak_example_indices_{method_name}"] = np.asarray(
            view.get("peak_example_indices", np.empty(0)), dtype=np.int32
        )
        arrays[f"peak_scatterer_indices_{method_name}"] = np.asarray(
            view.get("peak_scatterer_indices", np.empty(0)), dtype=np.int32
        )

        summary[method_name] = {
            "n_points": int(peaks.size),
            "background_rms": float(view.get("background_rms", 0.0)),
            "peak_mean": float(peaks.mean()) if peaks.size > 0 else 0.0,
            "peak_std": float(peaks.std()) if peaks.size > 0 else 0.0,
            "peak_max": float(peaks.max()) if peaks.size > 0 else 0.0,
            "snr_mean": float(snr.mean()) if snr.size > 0 else 0.0,
            "snr_std": float(snr.std()) if snr.size > 0 else 0.0,
            "snr_max": float(snr.max()) if snr.size > 0 else 0.0,
        }

    return summary, arrays


def extract_scatterer_snr(metrics: dict) -> np.ndarray:
    """Return pointwise SNR values from a scatterer-metric bundle."""
    view = metrics.get("aggregated", metrics)
    peaks = np.asarray(view.get("peak_amplitudes", np.empty(0)), dtype=np.float64)
    bg_rms = np.asarray(view.get("point_background_rms", np.empty(0)), dtype=np.float64)
    if bg_rms.size == 0:
        bg_rms_per_example = np.asarray(view.get("background_rms_per_example", np.empty(0)), dtype=np.float64)
        example_indices = np.asarray(view.get("peak_example_indices", np.empty(0)), dtype=np.int32)
        if bg_rms_per_example.size > 0 and example_indices.size == peaks.size:
            bg_rms = bg_rms_per_example[example_indices]
        else:
            bg_rms = np.full(peaks.shape, 1e-12, dtype=np.float64)
    return peaks / np.maximum(bg_rms, 1e-12)


def find_latest_baseline_reference(
    dataset_folder: str,
    baseline_f_number: float,
    sandbox_output_root: str | Path,
) -> tuple[dict | None, dict | None]:
    """Find the most recent matching baseline summary and its SNR arrays."""
    sandbox_output_root = Path(sandbox_output_root)
    baseline_search_root = sandbox_output_root / "baseline_apodizations"
    if not baseline_search_root.exists():
        return None, None

    summary_candidates = sorted(
        baseline_search_root.rglob("baseline_summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
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
