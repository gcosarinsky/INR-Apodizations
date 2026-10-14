"""Aggregation and persistence helpers for scatterer evaluation results.

Converts per-example metric dicts into batch-level summaries and produces
flat NumPy dictionaries suitable for npz persistence.
"""

from __future__ import annotations

import numpy as np

from inr_apodizations.evaluation.snr import compute_reflector_snr


def aggregate_reflector_metrics(
    per_example_metrics: list[dict],
    background_pixels_batch: list[np.ndarray],
    image_maxima: np.ndarray,
    return_background_hist: bool = False,
    hist_bins: int = 50,
) -> dict:
    """Aggregate a list of per-example metric dicts into a batch-level summary.

    Args:
        per_example_metrics: List of dicts, one per image example, each
            containing at least ``peak_amplitudes``, ``background_rms``,
            ``lateral_profiles``, ``axial_profiles``, and optional
            ``lateral_profile_offsets_mm`` / ``axial_profile_offsets_mm``.
        background_pixels_batch: List of 1D arrays of background pixel values
            (one per example), used for the global RMS and histogram.
        image_maxima: 1D array ``(N,)`` of per-image maximum values.
        return_background_hist: If True, compute a histogram over all
            background pixels pooled across examples.
        hist_bins: Number of histogram bins.

    Returns:
        Aggregated dict with keys (subset shown):
        - ``peak_amplitudes``, ``peak_example_indices``, ``peak_scatterer_indices``
        - ``background_rms``, ``background_rms_per_example``, ``point_background_rms``
        - ``lateral_profiles``, ``axial_profiles``, offsets
        - ``n_examples``, ``n_scatterers_total``
        - Optionally ``background_hist_counts``, ``background_hist_edges``.
    """
    agg_peaks: list[np.ndarray] = []
    agg_ex_idx: list[np.ndarray] = []
    agg_sc_idx: list[np.ndarray] = []
    agg_pt_bg: list[np.ndarray] = []
    agg_pt_max: list[np.ndarray] = []
    agg_lat: list[np.ndarray] = []
    agg_axl: list[np.ndarray] = []
    bg_rms_per_example = np.empty(len(per_example_metrics), dtype=np.float64)
    lateral_offsets_mm = np.empty(0, dtype=np.float64)
    axial_offsets_mm = np.empty(0, dtype=np.float64)

    for ex_idx, metrics in enumerate(per_example_metrics):
        peaks = np.asarray(metrics["peak_amplitudes"], dtype=np.float64)
        n_pts = peaks.shape[0]
        bg_rms_val = float(metrics["background_rms"])
        lat = np.asarray(metrics.get("lateral_profiles", np.empty((0, 0))), dtype=np.float64)
        axl = np.asarray(metrics.get("axial_profiles", np.empty((0, 0))), dtype=np.float64)
        bg_rms_per_example[ex_idx] = bg_rms_val

        agg_peaks.append(peaks)
        agg_ex_idx.append(np.full(n_pts, ex_idx, dtype=np.int32))
        agg_sc_idx.append(np.arange(n_pts, dtype=np.int32))
        agg_pt_bg.append(np.full(n_pts, bg_rms_val, dtype=np.float64))
        agg_pt_max.append(np.full(n_pts, float(image_maxima[ex_idx]), dtype=np.float64))
        agg_lat.append(lat)
        agg_axl.append(axl)

        if ex_idx == 0:
            lateral_offsets_mm = np.asarray(
                metrics.get("lateral_profile_offsets_mm", np.empty(0)), dtype=np.float64
            )
            axial_offsets_mm = np.asarray(
                metrics.get("axial_profile_offsets_mm", np.empty(0)), dtype=np.float64
            )

    if agg_peaks:
        peak_amplitudes = np.concatenate(agg_peaks)
        peak_example_indices = np.concatenate(agg_ex_idx)
        peak_scatterer_indices = np.concatenate(agg_sc_idx)
        point_bg_rms = np.concatenate(agg_pt_bg)
        point_img_max = np.concatenate(agg_pt_max)
        lateral_profiles = np.concatenate(agg_lat, axis=0)
        axial_profiles = np.concatenate(agg_axl, axis=0)
    else:
        peak_amplitudes = np.empty(0, dtype=np.float64)
        peak_example_indices = np.empty(0, dtype=np.int32)
        peak_scatterer_indices = np.empty(0, dtype=np.int32)
        point_bg_rms = np.empty(0, dtype=np.float64)
        point_img_max = np.empty(0, dtype=np.float64)
        lateral_profiles = np.empty((0, lateral_offsets_mm.size), dtype=np.float64)
        axial_profiles = np.empty((0, axial_offsets_mm.size), dtype=np.float64)

    non_empty = [p for p in background_pixels_batch if p.size > 0]
    global_bg_rms = (
        float(np.sqrt(np.mean(np.concatenate(non_empty) ** 2))) if non_empty else 0.0
    )

    aggregated: dict = {
        "peak_amplitudes": peak_amplitudes,
        "peak_example_indices": peak_example_indices,
        "peak_scatterer_indices": peak_scatterer_indices,
        "background_rms": global_bg_rms,
        "background_rms_per_example": bg_rms_per_example,
        "point_background_rms": point_bg_rms,
        "image_maxima": np.asarray(image_maxima, dtype=np.float64),
        "point_image_maxima": point_img_max,
        "n_examples": len(per_example_metrics),
        "n_scatterers_total": int(peak_amplitudes.size),
        "lateral_profiles": lateral_profiles,
        "axial_profiles": axial_profiles,
        "lateral_profile_offsets_mm": lateral_offsets_mm,
        "axial_profile_offsets_mm": axial_offsets_mm,
    }

    if return_background_hist:
        if non_empty:
            all_bg = np.concatenate(non_empty)
            counts, edges = np.histogram(all_bg, bins=hist_bins)
        else:
            counts, edges = np.histogram(np.asarray([], dtype=np.float64), bins=hist_bins)
        aggregated["background_hist_counts"] = counts
        aggregated["background_hist_edges"] = edges

    return aggregated


def _get_aggregated_view(metrics: dict) -> dict:
    """Return the aggregated sub-dict or the dict itself for single images."""
    return metrics.get("aggregated", metrics)


def build_snr_persistence_bundle(
    metrics_dict: dict[str, dict],
    scatterers_xy: np.ndarray | list[np.ndarray],
    eps: float = 1e-12,
) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    """Build per-method SNR summaries and flat arrays for npz persistence.

    For each method in ``metrics_dict`` the function computes SNR using the
    aggregated peak amplitudes and point-level background RMS (falling back to
    the global background RMS or a per-example value when pointwise data is
    not available).

    Args:
        metrics_dict: Mapping from method name to scatterer metric dict (as
            returned by :func:`~inr_apodizations.evaluation.metrics.compute_scatterer_metrics`).
        scatterers_xy: Scatterer positions, either ``(N, 2)`` or a list of
            ``(Ni, 2)`` arrays.  Used only to extract position arrays for
            persistence; coordinates are read from the aggregated index arrays
            when available.
        eps: Epsilon for SNR computation.

    Returns:
        Tuple ``(summary, arrays)`` where:
        - ``summary``: dict keyed by method name with scalar statistics.
        - ``arrays``: flat dict of NumPy arrays ready for ``np.savez``.
    """
    summary: dict[str, dict] = {}
    arrays: dict[str, np.ndarray] = {}

    # Flatten scatterer positions for persistence
    if isinstance(scatterers_xy, np.ndarray) and scatterers_xy.ndim == 2:
        flat_xy = scatterers_xy.astype(np.float32, copy=False)
    elif isinstance(scatterers_xy, (list, np.ndarray)):
        items = list(scatterers_xy) if not isinstance(scatterers_xy, list) else scatterers_xy
        if items:
            try:
                flat_xy = np.vstack([np.asarray(it, dtype=np.float32)[:, :2] for it in items])
            except Exception:
                flat_xy = np.empty((0, 2), dtype=np.float32)
        else:
            flat_xy = np.empty((0, 2), dtype=np.float32)
    else:
        flat_xy = np.empty((0, 2), dtype=np.float32)

    arrays["scatterer_x_mm"] = flat_xy[:, 0] if flat_xy.ndim == 2 and flat_xy.shape[0] > 0 else np.empty(0, dtype=np.float32)
    arrays["scatterer_z_mm"] = flat_xy[:, 1] if flat_xy.ndim == 2 and flat_xy.shape[0] > 0 else np.empty(0, dtype=np.float32)

    for method_name, metrics in metrics_dict.items():
        view = _get_aggregated_view(metrics)

        peaks = np.asarray(view.get("peak_amplitudes", np.empty(0)), dtype=np.float64)
        bg_rms_pt = np.asarray(view.get("point_background_rms", np.empty(0)), dtype=np.float64)

        # Resolve background RMS for SNR: prefer point-level, fall back to per-example, then global
        if bg_rms_pt.size == peaks.size and peaks.size > 0:
            bg_for_snr = bg_rms_pt
        else:
            bg_rms_per_ex = np.asarray(view.get("background_rms_per_example", np.empty(0)), dtype=np.float64)
            example_indices = np.asarray(view.get("peak_example_indices", np.empty(0)), dtype=np.int32)
            if bg_rms_per_ex.size > 0 and example_indices.size == peaks.size:
                bg_for_snr = bg_rms_per_ex[example_indices]
            else:
                bg_scalar = float(view.get("background_rms", 1e-12))
                bg_for_snr = np.full(peaks.shape, max(bg_scalar, 1e-12), dtype=np.float64)

        snr = compute_reflector_snr(peaks, bg_for_snr, eps=eps)

        lat = np.asarray(view.get("lateral_profiles", np.empty((0, 0))), dtype=np.float64)
        axl = np.asarray(view.get("axial_profiles", np.empty((0, 0))), dtype=np.float64)

        if "lateral_profile_offsets_mm" not in arrays:
            arrays["lateral_profile_offsets_mm"] = np.asarray(
                view.get("lateral_profile_offsets_mm", np.empty(0)), dtype=np.float32
            )
        if "axial_profile_offsets_mm" not in arrays:
            arrays["axial_profile_offsets_mm"] = np.asarray(
                view.get("axial_profile_offsets_mm", np.empty(0)), dtype=np.float32
            )

        arrays[f"peak_amplitudes_{method_name}"] = peaks.astype(np.float32, copy=False)
        arrays[f"background_rms_{method_name}"] = bg_for_snr.astype(np.float32, copy=False)
        arrays[f"snr_{method_name}"] = snr.astype(np.float32, copy=False)
        arrays[f"peak_example_indices_{method_name}"] = np.asarray(
            view.get("peak_example_indices", np.empty(0)), dtype=np.int32
        )
        arrays[f"peak_scatterer_indices_{method_name}"] = np.asarray(
            view.get("peak_scatterer_indices", np.empty(0)), dtype=np.int32
        )
        arrays[f"lateral_profiles_{method_name}"] = lat.astype(np.float32, copy=False)
        arrays[f"axial_profiles_{method_name}"] = axl.astype(np.float32, copy=False)

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


def extract_snr_from_metrics(metrics: dict, eps: float = 1e-12) -> np.ndarray:
    """Return pointwise SNR values from a scatterer-metric bundle.

    Args:
        metrics: Dict as returned by compute_scatterer_metrics (single or batch).
        eps: Epsilon passed to :func:`~inr_apodizations.evaluation.snr.compute_reflector_snr`.

    Returns:
        1D float64 array of SNR values, one per reflector.
    """
    view = _get_aggregated_view(metrics)
    peaks = np.asarray(view.get("peak_amplitudes", np.empty(0)), dtype=np.float64)
    bg_pt = np.asarray(view.get("point_background_rms", np.empty(0)), dtype=np.float64)

    if bg_pt.size == peaks.size and peaks.size > 0:
        bg = bg_pt
    else:
        bg_per_ex = np.asarray(view.get("background_rms_per_example", np.empty(0)), dtype=np.float64)
        ex_idx = np.asarray(view.get("peak_example_indices", np.empty(0)), dtype=np.int32)
        if bg_per_ex.size > 0 and ex_idx.size == peaks.size:
            bg = bg_per_ex[ex_idx]
        else:
            bg = np.full(peaks.shape, max(float(view.get("background_rms", 1e-12)), 1e-12))

    return compute_reflector_snr(peaks, bg, eps=eps)
