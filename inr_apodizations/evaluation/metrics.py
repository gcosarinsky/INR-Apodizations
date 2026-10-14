"""High-level orchestrator for per-scatterer evaluation.

This module combines region masks, profile extraction, background statistics,
and SNR computation into a unified public API.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.evaluation.batch import normalize_scatterer_batch
from inr_apodizations.evaluation.regions import (
    build_disk_context,
    build_roi_masks,
    compute_background_statistics,
    compute_peak_amplitudes,
)
from inr_apodizations.evaluation.profiles import extract_reflector_profiles
from inr_apodizations.evaluation.summary import aggregate_reflector_metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_single_image_metrics(
    image: np.ndarray,
    scatterers: np.ndarray,
    disk_ctx: dict,
    profile_half_x_mm: float | None = None,
    profile_half_z_mm: float | None = None,
    return_masks: bool = False,
    return_background_hist: bool = False,
    hist_bins: int = 50,
    cm: CoordinateManager | None = None,
) -> tuple[dict, np.ndarray]:
    """Compute scatterer metrics for a single 2D image.

    Args:
        image: 2D array ``(nz, nx)``.
        scatterers: ``(N, 2)`` array of positions in mm.
        disk_ctx: Disk geometry context from :func:`~regions.build_disk_context`.
        profile_half_x_mm: Lateral half-width for profile extraction (mm).
            Defaults to the disk radius when None.
        profile_half_z_mm: Axial half-width for profile extraction (mm).
            Defaults to the disk radius when None.
        return_masks: If True, include per-scatterer boolean masks in the result.
        return_background_hist: If True, include histogram of background pixels.
        hist_bins: Number of histogram bins.
        cm: CoordinateManager, required when ``profile_half_x_mm`` or
            ``profile_half_z_mm`` are provided (or always needed for profiles).

    Returns:
        Tuple ``(metrics_dict, background_pixels)``.
    """
    image = np.asarray(image, dtype=np.float64)
    individual_masks, union_mask = build_roi_masks(scatterers, disk_ctx)
    peak_amps = compute_peak_amplitudes(image, individual_masks)
    bg_stats = compute_background_statistics(
        image,
        union_mask,
        return_hist=return_background_hist,
        hist_bins=hist_bins,
    )

    # Profile extraction – use disk radius as default window size
    dx = float(disk_ctx["dx"])
    dz = float(disk_ctx["dz"])
    rx = int(disk_ctx["rx"])
    rz = int(disk_ctx["rz"])
    default_half_x_mm = rx * dx
    default_half_z_mm = rz * dz

    half_x_mm = profile_half_x_mm if profile_half_x_mm is not None else default_half_x_mm
    half_z_mm = profile_half_z_mm if profile_half_z_mm is not None else default_half_z_mm

    if cm is not None:
        prof = extract_reflector_profiles(
            image,
            scatterers,
            cm,
            half_width_lateral_mm=half_x_mm,
            half_width_axial_mm=half_z_mm,
        )
        lateral_profiles = prof["lateral_profiles"]
        axial_profiles = prof["axial_profiles"]
        lateral_offsets_mm = prof["lateral_offsets_mm"]
        axial_offsets_mm = prof["axial_offsets_mm"]
    else:
        lateral_profiles = np.empty((len(scatterers), 0), dtype=np.float64)
        axial_profiles = np.empty((len(scatterers), 0), dtype=np.float64)
        lateral_offsets_mm = np.empty(0, dtype=np.float64)
        axial_offsets_mm = np.empty(0, dtype=np.float64)

    result: dict = {
        "peak_amplitudes": peak_amps,
        "background_rms": bg_stats["background_rms"],
        "lateral_profiles": lateral_profiles,
        "axial_profiles": axial_profiles,
        "lateral_profile_offsets_mm": lateral_offsets_mm,
        "axial_profile_offsets_mm": axial_offsets_mm,
    }

    if return_background_hist:
        result["background_hist_counts"] = bg_stats.get("background_hist_counts")
        result["background_hist_edges"] = bg_stats.get("background_hist_edges")

    if return_masks:
        result["individual_masks"] = individual_masks
        result["background_mask"] = ~union_mask

    return result, bg_stats["background_pixels"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_scatterer_metrics(
    image: np.ndarray,
    scatterers: np.ndarray | Sequence[np.ndarray],
    cm: CoordinateManager,
    radius_mm: float = 1.0,
    profile_half_lateral_mm: float | None = None,
    profile_half_axial_mm: float | None = None,
    return_masks: bool = False,
    return_background_hist: bool = False,
    hist_bins: int = 50,
) -> dict:
    """Compute per-scatterer evaluation metrics for one image or a batch.

    For each scatterer the function extracts:
    - Peak amplitude within a disk ROI of ``radius_mm``.
    - Background RMS from pixels outside the union of all disk ROIs.
    - Lateral and axial max-projection profiles (window controlled by
      ``profile_half_*_mm``; defaults to the disk radius).

    Args:
        image: 2D ``(nz, nx)`` or 3D ``(B, nz, nx)`` amplitude image.
        scatterers: Scatterer positions in mm, accepted formats:
            ``(N, 2)``, ``(B, N, 2)``, or a sequence of ``(Ni, 2)`` arrays.
        cm: CoordinateManager providing physical coordinate arrays.
        radius_mm: Disk radius in mm for ROI/SNR computation.
        profile_half_lateral_mm: Lateral half-width for profile extraction in mm.
            Defaults to ``radius_mm``.
        profile_half_axial_mm: Axial half-width for profile extraction in mm.
            Defaults to ``radius_mm``.
        return_masks: If True, include ``individual_masks`` and
            ``background_mask`` in per-example results.
        return_background_hist: If True, include background histogram.
        hist_bins: Histogram bin count (used when ``return_background_hist``).

    Returns:
        For 2D input: flat metrics dict with keys ``peak_amplitudes``,
        ``background_rms``, ``lateral_profiles``, ``axial_profiles``, offsets,
        and optionally masks / histogram.

        For 3D input: dict with keys ``per_example`` (list of per-image
        dicts), ``aggregated`` (batch-level summary), and ``batch_size``.

    Raises:
        ValueError: If image dimensions are unsupported or inputs are invalid.
    """
    image_array = np.asarray(image)
    if image_array.ndim not in (2, 3):
        raise ValueError("image must be 2D (nz, nx) or 3D (B, nz, nx)")
    if radius_mm <= 0.0:
        raise ValueError("radius_mm must be > 0")

    if image_array.ndim == 2:
        disk_ctx = build_disk_context(image_array.shape, cm, radius_mm)
        scatterer_batch, _ = normalize_scatterer_batch(scatterers, batch_size=1)
        result, _ = _compute_single_image_metrics(
            image_array,
            scatterer_batch[0],
            disk_ctx,
            profile_half_x_mm=profile_half_lateral_mm,
            profile_half_z_mm=profile_half_axial_mm,
            return_masks=return_masks,
            return_background_hist=return_background_hist,
            hist_bins=hist_bins,
            cm=cm,
        )
        return result

    batch_size = int(image_array.shape[0])
    disk_ctx = build_disk_context(tuple(image_array.shape[1:]), cm, radius_mm)
    scatterer_batch, _ = normalize_scatterer_batch(scatterers, batch_size=batch_size)

    per_example_metrics: list[dict] = []
    background_pixels_batch: list[np.ndarray] = []
    image_maxima = np.max(image_array, axis=(1, 2)).astype(np.float64, copy=False)

    for img_ex, sc_ex in zip(image_array, scatterer_batch):
        metrics_ex, bg_pixels = _compute_single_image_metrics(
            img_ex,
            sc_ex,
            disk_ctx,
            profile_half_x_mm=profile_half_lateral_mm,
            profile_half_z_mm=profile_half_axial_mm,
            return_masks=return_masks,
            return_background_hist=return_background_hist,
            hist_bins=hist_bins,
            cm=cm,
        )
        per_example_metrics.append(metrics_ex)
        background_pixels_batch.append(bg_pixels)

    aggregated = aggregate_reflector_metrics(
        per_example_metrics,
        background_pixels_batch,
        image_maxima=image_maxima,
        return_background_hist=return_background_hist,
        hist_bins=hist_bins,
    )

    return {
        "per_example": per_example_metrics,
        "aggregated": aggregated,
        "batch_size": batch_size,
    }


def compute_validation_and_reference_metrics(
    images_abs: dict[str, np.ndarray],
    targets: np.ndarray,
    scatterers_xy: np.ndarray | Sequence[np.ndarray] | None,
    cm: CoordinateManager,
    sample_weights: np.ndarray | None = None,
    radius_mm: float = 1.5,
    profile_half_lateral_mm: float | None = None,
    profile_half_axial_mm: float | None = None,
    hist_bins: int = 50,
) -> dict:
    """Compute validation MAE, scatterer metrics, and reference baseline MAEs.

    Args:
        images_abs: Dict mapping method name to 2D ``(Z, X)`` or 3D
            ``(B, Z, X)`` absolute-amplitude images.
        targets: Ground-truth images with shape ``(Z, X)`` or ``(B, Z, X)``.
        scatterers_xy: Scatterer positions in mm; None to skip scatterer metrics.
        cm: CoordinateManager providing geometric context.
        sample_weights: Optional per-pixel weights for masked MAE computation,
            same shape as ``targets``.
        radius_mm: Disk radius in mm for SNR computation.
        profile_half_lateral_mm: Lateral half-width for profile extraction (mm).
        profile_half_axial_mm: Axial half-width for profile extraction (mm).
        hist_bins: Histogram bin count for background statistics.

    Returns:
        Dictionary with keys:
        - ``mae_by_method``: per-method unweighted MAE (float).
        - ``masked_mae_by_method``: per-method weighted MAE (empty when no weights).
        - ``scatterer_metrics``: per-method scatterer metric dicts.
        - ``reference_mae``: ``{"zero": float, "uniform"?: float, ...}``.

    Raises:
        ValueError: If target/image shapes are invalid or inconsistent.
    """
    targets_array = np.asarray(targets)
    if targets_array.ndim not in (2, 3):
        raise ValueError("targets must be 2D (Z, X) or 3D (B, Z, X)")

    targets_batch = targets_array[np.newaxis, ...] if targets_array.ndim == 2 else targets_array

    weights_batch = None
    if sample_weights is not None:
        wa = np.asarray(sample_weights)
        if wa.ndim == 2:
            weights_batch = wa[np.newaxis, ...]
        elif wa.ndim == 3:
            weights_batch = wa
        else:
            raise ValueError("sample_weights must be 2D (Z, X) or 3D (B, Z, X)")
        if weights_batch.shape != targets_batch.shape:
            raise ValueError(
                f"sample_weights shape {weights_batch.shape} does not match "
                f"targets shape {targets_batch.shape}"
            )

    mae_by_method: dict[str, float] = {}
    masked_mae_by_method: dict[str, float] = {}
    normalized_images: dict[str, np.ndarray] = {}

    for method_name, image in images_abs.items():
        img_a = np.asarray(image)
        if img_a.ndim == 2:
            img_batch = img_a[np.newaxis, ...]
        elif img_a.ndim == 3:
            img_batch = img_a
        else:
            raise ValueError(
                f"images_abs['{method_name}'] must be 2D (Z, X) or 3D (B, Z, X)"
            )
        if img_batch.shape != targets_batch.shape:
            raise ValueError(
                f"images_abs['{method_name}'] shape {img_batch.shape} does not match "
                f"targets shape {targets_batch.shape}"
            )

        abs_err = np.abs(img_batch.astype(np.float64) - targets_batch.astype(np.float64))
        mae_by_method[method_name] = float(np.mean(abs_err))

        if weights_batch is not None:
            w = weights_batch.astype(np.float64)
            wsum = float(np.sum(w))
            masked_mae_by_method[method_name] = float(np.sum(abs_err * w) / max(wsum, 1e-12))

        normalized_images[method_name] = img_batch

    scatterer_metrics: dict = {}
    if scatterers_xy is not None:
        for method_name, img_batch in normalized_images.items():
            scatterer_metrics[method_name] = compute_scatterer_metrics(
                img_batch,
                scatterers_xy,
                cm,
                radius_mm=radius_mm,
                profile_half_lateral_mm=profile_half_lateral_mm,
                profile_half_axial_mm=profile_half_axial_mm,
                return_masks=False,
                return_background_hist=True,
                hist_bins=hist_bins,
            )

    reference_mae: dict[str, float] = {}
    t64 = targets_batch.astype(np.float64)
    reference_mae["zero"] = float(np.mean(np.abs(t64)))
    for rname in ("uniform", "boxcar", "hanning"):
        if rname in normalized_images:
            reference_mae[rname] = float(
                np.mean(np.abs(normalized_images[rname].astype(np.float64) - t64))
            )

    return {
        "mae_by_method": mae_by_method,
        "masked_mae_by_method": masked_mae_by_method,
        "scatterer_metrics": scatterer_metrics,
        "reference_mae": reference_mae,
    }
