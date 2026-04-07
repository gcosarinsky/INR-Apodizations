"""Scatterer metrics and plotting utilities separated from helpers.

This module contains the logic to compute per-scatterer metrics and
produce evaluation figures. Kept as a separate module to keep
`helpers.py` focused and small.
"""
from __future__ import annotations

from typing import Sequence
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from inr_apodizations.coordinate_manager import CoordinateManager


def _normalize_scatterer_batch(
    scatterers: np.ndarray | Sequence[np.ndarray],
    batch_size: int,
) -> tuple[list[np.ndarray], bool]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    if isinstance(scatterers, np.ndarray):
        if scatterers.ndim == 2:
            scatterers_single = np.asarray(scatterers, dtype=np.float64)
            if scatterers_single.shape[1] != 2:
                raise ValueError("scatterers must have shape (N, 2) with columns [x, z] in mm")
            return [scatterers_single.copy() for _ in range(batch_size)], True

        if scatterers.ndim == 3:
            if scatterers.shape[0] != batch_size or scatterers.shape[2] != 2:
                raise ValueError(
                    "scatterers batch array must have shape (B, N, 2) matching image batch size"
                )
            return [np.asarray(item, dtype=np.float64) for item in scatterers], False

        if scatterers.ndim == 1 and scatterers.dtype == object:
            scatterer_items = list(scatterers)
        else:
            raise ValueError(
                "scatterers must be (N, 2), (B, N, 2), or a sequence/object-array of (Ni, 2) arrays"
            )
    else:
        scatterer_items = list(scatterers)

    if len(scatterer_items) != batch_size:
        raise ValueError("scatterers batch length must match image batch size")

    normalized_batch: list[np.ndarray] = []
    for scatterer_item in scatterer_items:
        scatterer_array = np.asarray(scatterer_item, dtype=np.float64)
        if scatterer_array.ndim != 2 or scatterer_array.shape[1] != 2:
            raise ValueError(
                "each scatterer batch item must have shape (Ni, 2) with columns [x, z] in mm"
            )
        normalized_batch.append(scatterer_array)

    return normalized_batch, False


def _build_scatterer_disk_context(
    image_shape: tuple[int, int],
    cm: CoordinateManager,
    radius_mm: float,
) -> dict:
    if len(image_shape) != 2:
        raise ValueError("image_shape must be a 2D shape (nz, nx)")
    if radius_mm <= 0.0:
        raise ValueError("radius_mm must be > 0")

    nz, nx = image_shape
    coords = cm.get_coordinates_1d(scaled=False)
    x_coords = np.asarray(coords["x"], dtype=np.float64)
    z_coords = np.asarray(coords["z"], dtype=np.float64)

    if x_coords.shape[0] != nx or z_coords.shape[0] != nz:
        raise ValueError(
            "image shape does not match CoordinateManager 1D coordinates dimensions"
        )

    dx = float(np.abs(x_coords[1] - x_coords[0])) if nx > 1 else 1.0
    dz = float(np.abs(z_coords[1] - z_coords[0])) if nz > 1 else 1.0

    rx = max(1, round(radius_mm / dx))
    rz = max(1, round(radius_mm / dz))
    r = max(rx, rz)
    gy, gx = np.ogrid[-r: r + 1, -r: r + 1]
    disk = (gx / rx) ** 2 + (gy / rz) ** 2 <= 1.0

    return {
        "nz": nz,
        "nx": nx,
        "x_coords": x_coords,
        "z_coords": z_coords,
        "disk": disk,
        "r": r,
    }


def _compute_scatterer_metrics_single(
    image: np.ndarray,
    scatterers: np.ndarray,
    disk_ctx: dict,
    return_masks: bool = False,
    return_background_hist: bool = False,
    hist_bins: int = 50,
) -> tuple[dict, np.ndarray]:
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be 2D (nz, nx)")

    scatterers = np.asarray(scatterers, dtype=np.float64)
    if scatterers.ndim != 2 or scatterers.shape[1] != 2:
        raise ValueError("scatterers must be (N, 2) with columns [x, z] in mm")

    nz = int(disk_ctx["nz"])
    nx = int(disk_ctx["nx"])
    if image.shape != (nz, nx):
        raise ValueError("image shape must match the CoordinateManager geometry")

    x_coords = disk_ctx["x_coords"]
    z_coords = disk_ctx["z_coords"]
    disk = disk_ctx["disk"]
    r = int(disk_ctx["r"])

    n_scatterers = scatterers.shape[0]
    union_mask = np.zeros((nz, nx), dtype=bool)
    individual_masks = np.zeros((n_scatterers, nz, nx), dtype=bool) if return_masks else None
    peak_amplitudes = np.empty(n_scatterers, dtype=np.float64)

    for i, (x0, z0) in enumerate(scatterers):
        ix = int(np.argmin(np.abs(x_coords - x0)))
        iz = int(np.argmin(np.abs(z_coords - z0)))

        iz0 = iz - r
        iz1 = iz + r + 1
        ix0 = ix - r
        ix1 = ix + r + 1

        disk_z0 = max(0, -iz0)
        disk_z1 = disk.shape[0] - max(0, iz1 - nz)
        disk_x0 = max(0, -ix0)
        disk_x1 = disk.shape[1] - max(0, ix1 - nx)

        img_z0 = max(0, iz0)
        img_z1 = min(nz, iz1)
        img_x0 = max(0, ix0)
        img_x1 = min(nx, ix1)

        disk_patch = disk[disk_z0:disk_z1, disk_x0:disk_x1]

        local_mask = np.zeros((nz, nx), dtype=bool)
        local_mask[img_z0:img_z1, img_x0:img_x1] = disk_patch
        union_mask |= local_mask

        masked_pixels = image[local_mask]
        peak_amplitudes[i] = float(masked_pixels.max()) if masked_pixels.size > 0 else 0.0

        if individual_masks is not None:
            individual_masks[i] = local_mask

    background_mask = ~union_mask
    background_pixels = image[background_mask]
    background_rms = (
        float(np.sqrt(np.mean(background_pixels ** 2))) if background_pixels.size > 0 else 0.0
    )

    result = {
        "peak_amplitudes": peak_amplitudes,
        "background_rms": background_rms,
    }

    if return_background_hist:
        counts, bin_edges = np.histogram(background_pixels, bins=hist_bins)
        result["background_hist_counts"] = counts
        result["background_hist_edges"] = bin_edges

    if individual_masks is not None:
        result["individual_masks"] = individual_masks
        result["background_mask"] = background_mask

    return result, np.asarray(background_pixels, dtype=np.float64)


def _aggregate_scatterer_metrics_batch(
    per_example_metrics: list[dict],
    background_pixels_batch: list[np.ndarray],
    image_maxima: np.ndarray,
    return_background_hist: bool,
    hist_bins: int,
) -> dict:
    aggregated_peaks: list[np.ndarray] = []
    aggregated_example_indices: list[np.ndarray] = []
    aggregated_scatterer_indices: list[np.ndarray] = []
    point_background_rms: list[np.ndarray] = []
    point_image_maxima: list[np.ndarray] = []
    background_rms_per_example = np.empty(len(per_example_metrics), dtype=np.float64)

    for example_idx, metrics in enumerate(per_example_metrics):
        peaks = np.asarray(metrics["peak_amplitudes"], dtype=np.float64)
        n_points = peaks.shape[0]
        background_rms_value = float(metrics["background_rms"])
        background_rms_per_example[example_idx] = background_rms_value

        aggregated_peaks.append(peaks)
        aggregated_example_indices.append(np.full(n_points, example_idx, dtype=np.int32))
        aggregated_scatterer_indices.append(np.arange(n_points, dtype=np.int32))
        point_background_rms.append(np.full(n_points, background_rms_value, dtype=np.float64))
        point_image_maxima.append(np.full(n_points, image_maxima[example_idx], dtype=np.float64))

    if aggregated_peaks:
        peak_amplitudes = np.concatenate(aggregated_peaks)
        peak_example_indices = np.concatenate(aggregated_example_indices)
        peak_scatterer_indices = np.concatenate(aggregated_scatterer_indices)
        point_bg_rms = np.concatenate(point_background_rms)
        point_img_max = np.concatenate(point_image_maxima)
    else:
        peak_amplitudes = np.empty(0, dtype=np.float64)
        peak_example_indices = np.empty(0, dtype=np.int32)
        peak_scatterer_indices = np.empty(0, dtype=np.int32)
        point_bg_rms = np.empty(0, dtype=np.float64)
        point_img_max = np.empty(0, dtype=np.float64)

    aggregated = {
        "peak_amplitudes": peak_amplitudes,
        "peak_example_indices": peak_example_indices,
        "peak_scatterer_indices": peak_scatterer_indices,
        "background_rms": (
            float(np.sqrt(np.mean(np.concatenate(background_pixels_batch) ** 2)))
            if any(pixels.size > 0 for pixels in background_pixels_batch)
            else 0.0
        ),
        "background_rms_per_example": background_rms_per_example,
        "point_background_rms": point_bg_rms,
        "image_maxima": np.asarray(image_maxima, dtype=np.float64),
        "point_image_maxima": point_img_max,
        "n_examples": len(per_example_metrics),
        "n_scatterers_total": int(peak_amplitudes.size),
    }

    if return_background_hist:
        background_pixels_non_empty = [pixels for pixels in background_pixels_batch if pixels.size > 0]
        if background_pixels_non_empty:
            background_pixels = np.concatenate(background_pixels_non_empty)
            counts, bin_edges = np.histogram(background_pixels, bins=hist_bins)
        else:
            counts, bin_edges = np.histogram(np.asarray([], dtype=np.float64), bins=hist_bins)
        aggregated["background_hist_counts"] = counts
        aggregated["background_hist_edges"] = bin_edges

    return aggregated


def _get_plot_metric_view(metrics: dict) -> dict:
    return metrics.get("aggregated", metrics)


def _validate_aligned_aggregated_points(ref_metrics: dict, cmp_metrics: dict) -> None:
    ref_view = _get_plot_metric_view(ref_metrics)
    cmp_view = _get_plot_metric_view(cmp_metrics)

    ref_examples = ref_view.get("peak_example_indices")
    cmp_examples = cmp_view.get("peak_example_indices")
    ref_scatterers = ref_view.get("peak_scatterer_indices")
    cmp_scatterers = cmp_view.get("peak_scatterer_indices")

    if ref_examples is None or cmp_examples is None:
        return

    if not np.array_equal(ref_examples, cmp_examples) or not np.array_equal(
        ref_scatterers, cmp_scatterers
    ):
        raise ValueError(
            "Batch scatterer metrics are not aligned across compared methods; "
            "ensure all methods use the same scatterer sets per example"
        )


def compute_scatterer_metrics(
    image: np.ndarray,
    scatterers: np.ndarray | Sequence[np.ndarray],
    cm: CoordinateManager,
    radius_mm: float = 1.0,
    return_masks: bool = False,
    return_background_hist: bool = False,
    hist_bins: int = 50,
) -> dict:
    image_array = np.asarray(image)
    if image_array.ndim not in (2, 3):
        raise ValueError("image must be 2D (nz, nx) or 3D (B, nz, nx)")
    if radius_mm <= 0.0:
        raise ValueError("radius_mm must be > 0")

    if image_array.ndim == 2:
        disk_ctx = _build_scatterer_disk_context(image_array.shape, cm, radius_mm)
        scatterer_batch, _ = _normalize_scatterer_batch(scatterers, batch_size=1)
        result, _ = _compute_scatterer_metrics_single(
            image_array,
            scatterer_batch[0],
            disk_ctx,
            return_masks=return_masks,
            return_background_hist=return_background_hist,
            hist_bins=hist_bins,
        )
        return result

    batch_size = int(image_array.shape[0])
    disk_ctx = _build_scatterer_disk_context(tuple(image_array.shape[1:]), cm, radius_mm)
    scatterer_batch, _ = _normalize_scatterer_batch(scatterers, batch_size=batch_size)

    per_example_metrics: list[dict] = []
    background_pixels_batch: list[np.ndarray] = []
    image_maxima = np.max(image_array, axis=(1, 2)).astype(np.float64, copy=False)

    for image_example, scatterers_example in zip(image_array, scatterer_batch):
        metrics_single, background_pixels = _compute_scatterer_metrics_single(
            image_example,
            scatterers_example,
            disk_ctx,
            return_masks=return_masks,
            return_background_hist=return_background_hist,
            hist_bins=hist_bins,
        )
        per_example_metrics.append(metrics_single)
        background_pixels_batch.append(background_pixels)

    aggregated = _aggregate_scatterer_metrics_batch(
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


def plot_scatterer_evaluation(
    images_abs: dict,
    scatterers_xy: np.ndarray | Sequence[np.ndarray],
    cm: CoordinateManager,
    output_dir: str,
    radius_mm: float = 1.5,
    hist_bins: int = 50,
    compare_pairs: list | None = None,
    extent: tuple | None = None,
    vmin_db: float = -60.0,
    vmax_db: float = 0.0,
    cmap: str = "gray",
    example_suffix: str = "",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    if compare_pairs is None:
        compare_pairs = []

    for ref_name, cmp_name in compare_pairs:
        if ref_name not in images_abs:
            raise ValueError(f"Reference method '{ref_name}' not in images_abs.")
        if cmp_name not in images_abs:
            raise ValueError(f"Compared method '{cmp_name}' not in images_abs.")

    image_ndims = {method_name: np.asarray(image).ndim for method_name, image in images_abs.items()}
    invalid_methods = [name for name, ndim in image_ndims.items() if ndim not in (2, 3)]
    if invalid_methods:
        raise ValueError(
            "All images_abs entries must be 2D or 3D arrays. Invalid methods: "
            + ", ".join(invalid_methods)
        )

    uses_batch = any(ndim == 3 for ndim in image_ndims.values())
    if uses_batch and any(ndim != 3 for ndim in image_ndims.values()):
        raise ValueError("All images_abs entries must be batch arrays when any method uses batch input")

    batch_size = 1
    if uses_batch:
        batch_sizes = {method_name: int(np.asarray(image).shape[0]) for method_name, image in images_abs.items()}
        unique_batch_sizes = set(batch_sizes.values())
        if len(unique_batch_sizes) != 1:
            raise ValueError("All batch inputs in images_abs must have the same batch size")
        batch_size = unique_batch_sizes.pop()

    first_ref = compare_pairs[0][0] if compare_pairs else None
    sfx = f"_{example_suffix}" if example_suffix else ""

    all_metrics: dict = {}
    for method_name, image in images_abs.items():
        need_masks = method_name == first_ref
        all_metrics[method_name] = compute_scatterer_metrics(
            image,
            scatterers_xy,
            cm,
            radius_mm=radius_mm,
            return_masks=need_masks,
            return_background_hist=True,
            hist_bins=hist_bins,
        )

    ref_metrics = all_metrics.get(first_ref) if first_ref is not None else None
    if extent is not None and first_ref is not None and ref_metrics is not None:
        ref_mask_metrics = (
            ref_metrics["per_example"][0] if "per_example" in ref_metrics else ref_metrics
        )
        if "background_mask" in ref_mask_metrics:
            ref_image_array = np.asarray(images_abs[first_ref])
            ref_img = ref_image_array[0] if ref_image_array.ndim == 3 else ref_image_array
            scatterer_union_mask = ~ref_mask_metrics["background_mask"]
            ref_max = float(np.max(ref_img))
            ref_db = 20.0 * np.log10((ref_img / (ref_max + 1e-8)) + 1e-8) if ref_max > 0 else ref_img

            fig_mask, axes_mask = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
            im0 = axes_mask[0].imshow(
                ref_db, cmap=cmap, vmin=vmin_db, vmax=vmax_db, extent=extent, aspect="auto",
            )
            axes_mask[0].set_title(f"DAS {first_ref} (dB)")
            axes_mask[0].set_xlabel("x (mm)")
            axes_mask[0].set_ylabel("z (mm)")
            axes_mask[1].imshow(
                scatterer_union_mask.astype(np.float32), cmap="gray",
                vmin=0.0, vmax=1.0, extent=extent, aspect="auto",
            )
            axes_mask[1].set_title("Total Scatterer Mask")
            axes_mask[1].set_xlabel("x (mm)")
            axes_mask[2].imshow(
                ref_db, cmap=cmap, vmin=vmin_db, vmax=vmax_db, extent=extent, aspect="auto",
            )
            overlay = np.ma.masked_where(~scatterer_union_mask, scatterer_union_mask.astype(np.float32))
            axes_mask[2].imshow(
                overlay, cmap="autumn", vmin=0.0, vmax=1.0,
                extent=extent, aspect="auto", alpha=0.35,
            )
            axes_mask[2].set_title(f"DAS {first_ref} + Scatterer Mask Overlay")
            axes_mask[2].set_xlabel("x (mm)")
            overlay_title = (
                f"DAS {first_ref}, scatterer mask and overlay (example 0){sfx}"
                if uses_batch
                else f"DAS {first_ref}, scatterer mask and overlay{sfx}"
            )
            fig_mask.suptitle(overlay_title)
            fig_mask.tight_layout()
            fig_mask.colorbar(im0, ax=axes_mask[0], fraction=0.046, pad=0.04).set_label("dB")
            fig_mask.savefig(
                os.path.join(output_dir, f"scatt_mask_overlay_{first_ref}{sfx}.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig_mask)

    tab_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    fig_hist, ax_hist = plt.subplots(1, 1, figsize=(10, 5))
    for idx, method_name in enumerate(images_abs.keys()):
        metric_view = _get_plot_metric_view(all_metrics[method_name])
        hist_counts = metric_view.get("background_hist_counts")
        hist_edges = metric_view.get("background_hist_edges")
        if hist_counts is not None and hist_edges is not None:
            bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2.0
            color = tab_colors[idx % len(tab_colors)]
            ax_hist.plot(
                bin_centers, hist_counts, marker="o", label=method_name,
                color=color, linewidth=2, markersize=4, alpha=0.7,
            )
            ax_hist.fill_between(bin_centers, hist_counts, alpha=0.2, color=color)
    ax_hist.set_xlabel("Amplitude (linear)")
    ax_hist.set_ylabel("Frequency")
    hist_title_suffix = f" ({batch_size} examples)" if uses_batch else ""
    ax_hist.set_title(f"Background noise histograms{hist_title_suffix}{sfx}")
    ax_hist.grid(True, alpha=0.3)
    ax_hist.legend()
    fig_hist.tight_layout()
    fig_hist.savefig(
        os.path.join(output_dir, f"scatt_background_hist{sfx}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig_hist)

    if compare_pairs:
        n_cols = len(compare_pairs)

        fig_snr, axes_snr = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
        for col_idx, (ref_name, cmp_name) in enumerate(compare_pairs):
            ax = axes_snr[0, col_idx]
            _validate_aligned_aggregated_points(all_metrics[ref_name], all_metrics[cmp_name])
            ref_view = _get_plot_metric_view(all_metrics[ref_name])
            cmp_view = _get_plot_metric_view(all_metrics[cmp_name])
            ref_peaks = np.asarray(ref_view["peak_amplitudes"], dtype=np.float64)
            cmp_peaks = np.asarray(cmp_view["peak_amplitudes"], dtype=np.float64)
            if ref_peaks.shape != cmp_peaks.shape:
                raise ValueError(
                    f"Compared methods '{ref_name}' and '{cmp_name}' do not have the same number of points"
                )
            if ref_peaks.size == 0:
                raise ValueError(
                    f"Compared methods '{ref_name}' and '{cmp_name}' do not contain scatterer points"
                )

            if "point_background_rms" in ref_view:
                ref_bg_rms = np.maximum(np.asarray(ref_view["point_background_rms"], dtype=np.float64), 1e-12)
                cmp_bg_rms = np.maximum(np.asarray(cmp_view["point_background_rms"], dtype=np.float64), 1e-12)
            else:
                ref_bg_rms = np.full(ref_peaks.shape, max(float(ref_view["background_rms"]), 1e-12))
                cmp_bg_rms = np.full(cmp_peaks.shape, max(float(cmp_view["background_rms"]), 1e-12))

            ref_snr = ref_peaks / ref_bg_rms
            cmp_snr = cmp_peaks / cmp_bg_rms
            ax_max = max(float(ref_snr.max()), float(cmp_snr.max()))
            ax.scatter(ref_snr, cmp_snr, s=15, alpha=0.7)
            ax.plot([0, ax_max], [0, ax_max], color="red", linewidth=1, linestyle="--", label="y = x")
            ax.set_xlabel(f"{ref_name} SNR (peak / bg_rms)")
            ax.set_ylabel(f"{cmp_name} SNR (peak / bg_rms)")
            title_suffix = (
                f"{len(ref_peaks)} points, {batch_size} examples"
                if uses_batch
                else f"{len(ref_peaks)} scatterers"
            )
            ax.set_title(f"{cmp_name} vs {ref_name} SNR ({title_suffix})")
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig_snr.suptitle(f"SNR comparison{sfx}")
        fig_snr.tight_layout()
        fig_snr.savefig(
            os.path.join(output_dir, f"scatt_snr_scatter{sfx}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig_snr)

        # Plot histogram of SNR ratio for each requested comparison pair
        for ref_name, cmp_name in compare_pairs:
            ref_view = _get_plot_metric_view(all_metrics[ref_name])
            cmp_view = _get_plot_metric_view(all_metrics[cmp_name])
            ref_peaks = np.asarray(ref_view["peak_amplitudes"], dtype=np.float64)
            cmp_peaks = np.asarray(cmp_view["peak_amplitudes"], dtype=np.float64)
            if ref_peaks.shape != cmp_peaks.shape:
                raise ValueError(
                    f"Compared methods '{ref_name}' and '{cmp_name}' do not have the same number of points"
                )
            if ref_peaks.size == 0:
                raise ValueError(
                    f"Compared methods '{ref_name}' and '{cmp_name}' do not contain scatterer points"
                )

            if "point_background_rms" in ref_view:
                ref_bg_rms = np.maximum(np.asarray(ref_view["point_background_rms"], dtype=np.float64), 1e-12)
                cmp_bg_rms = np.maximum(np.asarray(cmp_view["point_background_rms"], dtype=np.float64), 1e-12)
            else:
                ref_bg_rms = np.full(ref_peaks.shape, max(float(ref_view["background_rms"] or 1e-12), 1e-12))
                cmp_bg_rms = np.full(cmp_peaks.shape, max(float(cmp_view["background_rms"] or 1e-12), 1e-12))

            ref_snr = ref_peaks / ref_bg_rms
            cmp_snr = cmp_peaks / cmp_bg_rms
            ratio = cmp_snr / ref_snr

            fig_ratio, ax_ratio = plt.subplots(1, 1, figsize=(8, 5))
            ax_ratio.hist(ratio, bins=50, color="tab:blue", alpha=0.75, edgecolor="black")
            ax_ratio.set_xlabel(f"{cmp_name} / {ref_name} SNR ratio")
            ax_ratio.set_ylabel("Frequency")
            ax_ratio.set_title(f"SNR ratio histogram: {cmp_name} / {ref_name} ({len(ratio)} points)")
            ax_ratio.grid(True, alpha=0.3)
            fig_ratio.tight_layout()
            fig_ratio.savefig(
                os.path.join(output_dir, f"scatt_snr_ratio_hist_{ref_name}_vs_{cmp_name}{sfx}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig_ratio)

    return all_metrics


def plot_scatterer_snr_ratio(
    images_abs: dict,
    scatterers_xy: np.ndarray | Sequence[np.ndarray],
    cm: CoordinateManager,
    ref_method: str,
    cmp_method: str,
    radius_mm: float = 1.5,
    extent: tuple | None = None,
    cmap: str = "RdBu_r",
    scale: str = "linear",
    clip_percentiles: tuple[float, float] | None = (1.0, 99.0),
    point_size: int = 15,
    alpha: float = 0.7,
    eps: float = 1e-8,
    return_fig: bool = True,
) -> dict:
    """Plot per-reflector SNR ratio between two methods.

    The function computes for each reflector in the provided scatterer set the
    SNR = peak_amplitude / background_rms for both methods (reference and
    comparison), forms the ratio SNR_ref / SNR_cmp and produces a scatter
    plot in the (x, z) plane where each point color encodes that ratio.

    Parameters
    ----------
    images_abs:
        Mapping from method name to 2D or 3D (batch) absolute images.
    scatterers_xy:
        Per-example scatterer locations in mm. Accepted formats: (N,2),
        (B,N,2), or sequence/object-array of (Ni,2) arrays.
    cm:
        CoordinateManager used to validate geometry.
    ref_method, cmp_method:
        Keys in `images_abs` identifying the two methods to compare. Ratio is
        computed as SNR_ref / SNR_cmp.
    radius_mm:
        Radius used to compute local peak amplitude around each scatterer.
    extent:
        Optional imshow extent for axis labeling (passed through to plot).
    cmap:
        Matplotlib colormap name.
    scale:
        Either 'linear' or 'log' to control color mapping. If 'log', the
        plotted values are log10(ratio).
    clip_percentiles:
        Tuple (pmin, pmax) used to clip the color scale by percentiles. If
        None, no clipping is applied.
    point_size, alpha, eps:
        Scatter plotting parameters and numerical epsilon to avoid div-by-zero.
    return_fig:
        If True, returns the Matplotlib Figure object in the result dict.

    Returns
    -------
    dict
        Contains arrays used for plotting (`x`, `z`, `ratio`, `snr_ref`,
        `snr_cmp`) and optionally the `fig` object when `return_fig` is True.
    """
    # Validate methods
    if ref_method not in images_abs:
        raise ValueError(f"Reference method '{ref_method}' not found in images_abs")
    if cmp_method not in images_abs:
        raise ValueError(f"Compared method '{cmp_method}' not found in images_abs")

    # Determine batch usage and consistency (reuse logic from plot_scatterer_evaluation)
    image_ndims = {method_name: np.asarray(image).ndim for method_name, image in images_abs.items()}
    invalid_methods = [name for name, ndim in image_ndims.items() if ndim not in (2, 3)]
    if invalid_methods:
        raise ValueError(
            "All images_abs entries must be 2D or 3D arrays. Invalid methods: "
            + ", ".join(invalid_methods)
        )

    uses_batch = any(ndim == 3 for ndim in image_ndims.values())
    if uses_batch and any(ndim != 3 for ndim in image_ndims.values()):
        raise ValueError("All images_abs entries must be batch arrays when any method uses batch input")

    batch_size = 1
    if uses_batch:
        batch_sizes = {method_name: int(np.asarray(image).shape[0]) for method_name, image in images_abs.items()}
        unique_batch_sizes = set(batch_sizes.values())
        if len(unique_batch_sizes) != 1:
            raise ValueError("All batch inputs in images_abs must have the same batch size")
        batch_size = unique_batch_sizes.pop()

    # Normalize scatterers to per-example list
    scatterer_batch, _ = _normalize_scatterer_batch(scatterers_xy, batch_size=batch_size)
    # Compute metrics for both methods
    ref_metrics = compute_scatterer_metrics(
        images_abs[ref_method], scatterers_xy, cm, radius_mm=radius_mm, return_masks=False, return_background_hist=False
    )
    cmp_metrics = compute_scatterer_metrics(
        images_abs[cmp_method], scatterers_xy, cm, radius_mm=radius_mm, return_masks=False, return_background_hist=False
    )

    # Ensure alignment of aggregated points
    _validate_aligned_aggregated_points(ref_metrics, cmp_metrics)

    ref_view = _get_plot_metric_view(ref_metrics)
    cmp_view = _get_plot_metric_view(cmp_metrics)

    peaks_ref = np.asarray(ref_view.get("peak_amplitudes", np.empty(0)), dtype=np.float64)
    peaks_cmp = np.asarray(cmp_view.get("peak_amplitudes", np.empty(0)), dtype=np.float64)

    if peaks_ref.shape != peaks_cmp.shape:
        raise ValueError("Aggregated peak arrays have different shapes between reference and comparison methods")

    # Prefer point-level background rms if available
    bg_ref = np.asarray(ref_view.get("point_background_rms", None))
    bg_cmp = np.asarray(cmp_view.get("point_background_rms", None))
    if bg_ref is None or bg_ref.size == 0:
        # Fallback to per-example background_rms_per_example expanded per point
        bg_ref = np.empty_like(peaks_ref)
        bg_ref_per_example = ref_view.get("background_rms_per_example")
        if bg_ref_per_example is None:
            bg_ref.fill(eps)
        else:
            # expand per-example into per-point using peak_example_indices
            example_idx = np.asarray(ref_view.get("peak_example_indices", np.zeros(peaks_ref.shape, dtype=np.int32)))
            bg_ref = np.asarray(bg_ref_per_example, dtype=np.float64)[example_idx]

    if bg_cmp is None or bg_cmp.size == 0:
        bg_cmp = np.empty_like(peaks_cmp)
        bg_cmp_per_example = cmp_view.get("background_rms_per_example")
        if bg_cmp_per_example is None:
            bg_cmp.fill(eps)
        else:
            example_idx = np.asarray(cmp_view.get("peak_example_indices", np.zeros(peaks_cmp.shape, dtype=np.int32)))
            bg_cmp = np.asarray(bg_cmp_per_example, dtype=np.float64)[example_idx]

    # Compute SNRs and ratio (always comparison / reference)
    snr_ref = peaks_ref / (bg_ref + eps)
    snr_cmp = peaks_cmp / (bg_cmp + eps)
    ratio = snr_cmp / (snr_ref + eps)

    # Build positions aligned with aggregated ordering: concatenate scatterer_batch in example order
    if not scatterer_batch:
        pos = np.empty((0, 2), dtype=np.float64)
    else:
        pos = np.vstack([np.asarray(arr, dtype=np.float64) for arr in scatterer_batch])

    if ratio.size != pos.shape[0]:
        # It's possible that some examples have zero scatterers; build positions using peak indices
        peak_example_indices = np.asarray(ref_view.get("peak_example_indices", np.zeros(ratio.shape, dtype=np.int32)))
        peak_scatterer_indices = np.asarray(ref_view.get("peak_scatterer_indices", np.arange(ratio.size, dtype=np.int32)))
        coords_list: list[np.ndarray] = []
        for ex_idx, scat_idx in zip(peak_example_indices, peak_scatterer_indices):
            coords_list.append(scatterer_batch[int(ex_idx)][int(scat_idx)])
        pos = np.asarray(coords_list, dtype=np.float64)

    x = pos[:, 0]
    z = pos[:, 1]

    # Prepare plotting values and apply scale if requested
    plot_vals = ratio.copy()
    if scale == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            plot_vals = np.log10(plot_vals)

    # Clip by percentiles or compute vmin/vmax defaults
    if clip_percentiles is not None:
        pmin, pmax = float(clip_percentiles[0]), float(clip_percentiles[1])
        vmin = float(np.nanpercentile(plot_vals, pmin))
        vmax = float(np.nanpercentile(plot_vals, pmax))
    else:
        # fallback to data min/max
        vmin = float(np.nanmin(plot_vals)) if plot_vals.size > 0 else 0.0
        vmax = float(np.nanmax(plot_vals)) if plot_vals.size > 0 else 1.0

    # Use a diverging normalization centered at 1 (or 0 for log scale)
    center = 0.0 if scale == "log" else 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    # Improve visibility: no transparency, add black edge to markers, and set
    # a light gray background so central values (e.g. 1) don't blend with white.
    ax.set_facecolor("#f2f2f2")
    sc = ax.scatter(
        x,
        z,
        c=plot_vals,
        cmap=cmap,
        norm=norm,
        s=point_size,
        alpha=1.0,
        edgecolors="black",
        linewidths=0.4,
    )
    cbar = fig.colorbar(sc, ax=ax)
    if scale == "log":
        cbar.set_label("log10(SNR_cmp / SNR_ref)")
    else:
        cbar.set_label("SNR_cmp / SNR_ref")

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_title(f"Scatterer SNR ratio: {cmp_method} / {ref_method}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    result = {
        "x": x,
        "z": z,
        "ratio": ratio,
        "snr_ref": snr_ref,
        "snr_cmp": snr_cmp,
        "peaks_ref": peaks_ref,
        "peaks_cmp": peaks_cmp,
        "bg_ref": bg_ref,
        "bg_cmp": bg_cmp,
    }
    # Label indicating plotted ratio orientation (safe for filenames)
    result["ratio_label"] = f"{cmp_method}/{ref_method}"
    if return_fig:
        result["fig"] = fig

    return result
