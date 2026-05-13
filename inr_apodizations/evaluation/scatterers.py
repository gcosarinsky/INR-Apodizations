"""Scatterer utilities for profile-centered evaluation and plotting.

Provides helpers to resolve which scatterer to use as the center of a
reflector lateral/axial profile figure, plus plotting utilities for
scatterer-based metric comparisons.
"""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.evaluation.batch import normalize_scatterer_batch
from inr_apodizations.evaluation.metrics import compute_scatterer_metrics


def select_reflector_scatterer_index(
    scatterers_mm: np.ndarray,
    selection: str = "strongest",
    scatterer_idx: int | None = None,
) -> int:
    """Resolve the reflector index used for profile-centered plots.

    Args:
        scatterers_mm: Scatterer coordinates with columns ``[x_mm, z_mm]``
            or ``[x_mm, z_mm, reflectivity]``.
        selection: Selection mode. Supported values are ``"strongest"``
            (highest absolute reflectivity when column 2 is present) and
            ``"first"``.
        scatterer_idx: Optional explicit index. When provided it takes
            priority over ``selection``.

    Returns:
        Integer index of the selected scatterer.

    Raises:
        ValueError: If the array is empty, the index is out of range, or
            the selection mode is unsupported.
    """
    if scatterers_mm.shape[0] == 0:
        raise ValueError("No scatterers available for reflector profile selection.")

    if scatterer_idx is not None:
        if scatterer_idx < 0 or scatterer_idx >= scatterers_mm.shape[0]:
            raise ValueError(
                f"scatterer_idx={scatterer_idx} is out of range "
                f"[0, {scatterers_mm.shape[0] - 1}]."
            )
        return int(scatterer_idx)

    selection_normalized = selection.strip().lower()
    if selection_normalized == "strongest":
        if scatterers_mm.shape[1] >= 3:
            return int(np.argmax(np.abs(scatterers_mm[:, 2])))
        return 0
    if selection_normalized == "first":
        return 0
    raise ValueError("scatterer_selection must be 'strongest' or 'first'.")


def select_reflector_scatterer(
    scatterers_mm: np.ndarray,
    selection: str = "strongest",
    scatterer_idx: int | None = None,
) -> np.ndarray:
    """Select a single scatterer row to center reflector profile plots.

    Convenience wrapper around :func:`select_reflector_scatterer_index`
    that returns the full row instead of the index.

    Args:
        scatterers_mm: Scatterer coordinates with columns
            ``[x_mm, z_mm]`` or ``[x_mm, z_mm, reflectivity]``.
        selection: Selection mode — ``"strongest"`` or ``"first"``.
        scatterer_idx: Optional explicit index with priority over
            ``selection``.

    Returns:
        Selected scatterer row as a 1-D NumPy array.

    Raises:
        ValueError: If the array is empty, the index is out of range, or
            the selection mode is unsupported.
    """
    idx = select_reflector_scatterer_index(
        scatterers_mm,
        selection=selection,
        scatterer_idx=scatterer_idx,
    )
    return scatterers_mm[idx]


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
    all_metrics: dict | None = None,
) -> dict:
    """Plot scatterer-centric evaluation figures and return computed metrics.

    Args:
        images_abs: Mapping from method name to absolute image arrays.
            Each value can be 2D ``(z, x)`` or 3D ``(batch, z, x)``.
        scatterers_xy: Scatterer locations in mm, either single-example
            coordinates ``(N, 2)`` or batch-compatible structures.
        cm: Coordinate manager used by metric computation routines.
        output_dir: Directory where output figures are written.
        radius_mm: Radius in mm for local scatterer peak extraction.
        hist_bins: Number of bins used for background histograms.
        compare_pairs: Optional list of ``(ref_method, cmp_method)`` pairs
            for pairwise SNR scatter and ratio plots.
        extent: Optional plotting extent passed to ``imshow``.
        vmin_db: Lower dB limit for image visualizations.
        vmax_db: Upper dB limit for image visualizations.
        cmap: Colormap for image visualizations.
        example_suffix: Optional suffix appended to generated filenames.
        all_metrics: Optional precomputed metrics per method.

    Returns:
        Dictionary with metrics for all methods.

    Raises:
        ValueError: If method names are missing, array shapes are invalid,
            batch sizes do not match, or compared methods are not aligned.
    """
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
        raise ValueError(
            "All images_abs entries must be batch arrays when any method uses batch input"
        )

    batch_size = 1
    if uses_batch:
        batch_sizes = {
            method_name: int(np.asarray(image).shape[0])
            for method_name, image in images_abs.items()
        }
        unique_batch_sizes = set(batch_sizes.values())
        if len(unique_batch_sizes) != 1:
            raise ValueError("All batch inputs in images_abs must have the same batch size")
        batch_size = unique_batch_sizes.pop()

    first_ref = compare_pairs[0][0] if compare_pairs else None
    sfx = f"_{example_suffix}" if example_suffix else ""

    if all_metrics is None:
        all_metrics = {}
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
                ref_db,
                cmap=cmap,
                vmin=vmin_db,
                vmax=vmax_db,
                extent=extent,
                aspect="auto",
            )
            axes_mask[0].set_title(f"DAS {first_ref} (dB)")
            axes_mask[0].set_xlabel("x (mm)")
            axes_mask[0].set_ylabel("z (mm)")
            axes_mask[1].imshow(
                scatterer_union_mask.astype(np.float32),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                extent=extent,
                aspect="auto",
            )
            axes_mask[1].set_title("Total Scatterer Mask")
            axes_mask[1].set_xlabel("x (mm)")
            axes_mask[2].imshow(
                ref_db,
                cmap=cmap,
                vmin=vmin_db,
                vmax=vmax_db,
                extent=extent,
                aspect="auto",
            )
            overlay = np.ma.masked_where(
                ~scatterer_union_mask, scatterer_union_mask.astype(np.float32)
            )
            axes_mask[2].imshow(
                overlay,
                cmap="autumn",
                vmin=0.0,
                vmax=1.0,
                extent=extent,
                aspect="auto",
                alpha=0.35,
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
                dpi=150,
                bbox_inches="tight",
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
                bin_centers,
                hist_counts,
                marker="o",
                label=method_name,
                color=color,
                linewidth=2,
                markersize=4,
                alpha=0.7,
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
        dpi=150,
        bbox_inches="tight",
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
                ref_bg_rms = np.maximum(
                    np.asarray(ref_view["point_background_rms"], dtype=np.float64),
                    1e-12,
                )
                cmp_bg_rms = np.maximum(
                    np.asarray(cmp_view["point_background_rms"], dtype=np.float64),
                    1e-12,
                )
            else:
                ref_bg_rms = np.full(
                    ref_peaks.shape,
                    max(float(ref_view["background_rms"]), 1e-12),
                )
                cmp_bg_rms = np.full(
                    cmp_peaks.shape,
                    max(float(cmp_view["background_rms"]), 1e-12),
                )

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
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig_snr)

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
                ref_bg_rms = np.maximum(
                    np.asarray(ref_view["point_background_rms"], dtype=np.float64),
                    1e-12,
                )
                cmp_bg_rms = np.maximum(
                    np.asarray(cmp_view["point_background_rms"], dtype=np.float64),
                    1e-12,
                )
            else:
                ref_bg_rms = np.full(
                    ref_peaks.shape,
                    max(float(ref_view["background_rms"] or 1e-12), 1e-12),
                )
                cmp_bg_rms = np.full(
                    cmp_peaks.shape,
                    max(float(cmp_view["background_rms"] or 1e-12), 1e-12),
                )

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
    all_metrics: dict | None = None,
) -> dict:
    """Plot per-reflector SNR ratio between two methods.

    The function computes for each reflector in the provided scatterer set the
    SNR = peak_amplitude / background_rms for both methods (reference and
    comparison), forms the ratio SNR_cmp / SNR_ref and produces a scatter
    plot in the (x, z) plane where each point color encodes that ratio.

    Args:
        images_abs: Mapping from method name to 2D or 3D absolute images.
        scatterers_xy: Per-example scatterer locations in mm. Accepted formats:
            ``(N,2)``, ``(B,N,2)``, or sequence/object-array of ``(Ni,2)``.
        cm: CoordinateManager used to validate geometry.
        ref_method: Key in ``images_abs`` identifying the reference method.
        cmp_method: Key in ``images_abs`` identifying the compared method.
        radius_mm: Radius used to compute local peak amplitude around each
            scatterer.
        extent: Optional plotting extent for axis labeling.
        cmap: Matplotlib colormap name.
        scale: Either ``'linear'`` or ``'log'`` to control color mapping.
        clip_percentiles: Tuple ``(pmin, pmax)`` used to clip color scale by
            percentiles. If ``None``, no clipping is applied.
        point_size: Scatter marker size.
        alpha: Marker alpha parameter (kept for API compatibility).
        eps: Numerical epsilon to avoid division-by-zero.
        return_fig: If True, returns Matplotlib figure in the result dict.
        all_metrics: Optional precomputed metrics for both methods.

    Returns:
        Dictionary with plotting arrays and optionally ``fig``.

    Raises:
        ValueError: If methods are missing, array shapes are inconsistent,
            batch sizes do not match, or aggregated points are misaligned.
    """
    _ = alpha

    if ref_method not in images_abs:
        raise ValueError(f"Reference method '{ref_method}' not found in images_abs")
    if cmp_method not in images_abs:
        raise ValueError(f"Compared method '{cmp_method}' not found in images_abs")

    image_ndims = {method_name: np.asarray(image).ndim for method_name, image in images_abs.items()}
    invalid_methods = [name for name, ndim in image_ndims.items() if ndim not in (2, 3)]
    if invalid_methods:
        raise ValueError(
            "All images_abs entries must be 2D or 3D arrays. Invalid methods: "
            + ", ".join(invalid_methods)
        )

    uses_batch = any(ndim == 3 for ndim in image_ndims.values())
    if uses_batch and any(ndim != 3 for ndim in image_ndims.values()):
        raise ValueError(
            "All images_abs entries must be batch arrays when any method uses batch input"
        )

    batch_size = 1
    if uses_batch:
        batch_sizes = {
            method_name: int(np.asarray(image).shape[0])
            for method_name, image in images_abs.items()
        }
        unique_batch_sizes = set(batch_sizes.values())
        if len(unique_batch_sizes) != 1:
            raise ValueError("All batch inputs in images_abs must have the same batch size")
        batch_size = unique_batch_sizes.pop()

    scatterer_batch, _ = normalize_scatterer_batch(scatterers_xy, batch_size=batch_size)
    if all_metrics is not None and ref_method in all_metrics and cmp_method in all_metrics:
        ref_metrics = all_metrics[ref_method]
        cmp_metrics = all_metrics[cmp_method]
    else:
        ref_metrics = compute_scatterer_metrics(
            images_abs[ref_method],
            scatterers_xy,
            cm,
            radius_mm=radius_mm,
            return_masks=False,
            return_background_hist=False,
        )
        cmp_metrics = compute_scatterer_metrics(
            images_abs[cmp_method],
            scatterers_xy,
            cm,
            radius_mm=radius_mm,
            return_masks=False,
            return_background_hist=False,
        )

    _validate_aligned_aggregated_points(ref_metrics, cmp_metrics)

    ref_view = _get_plot_metric_view(ref_metrics)
    cmp_view = _get_plot_metric_view(cmp_metrics)

    peaks_ref = np.asarray(ref_view.get("peak_amplitudes", np.empty(0)), dtype=np.float64)
    peaks_cmp = np.asarray(cmp_view.get("peak_amplitudes", np.empty(0)), dtype=np.float64)

    if peaks_ref.shape != peaks_cmp.shape:
        raise ValueError(
            "Aggregated peak arrays have different shapes between reference and comparison methods"
        )

    bg_ref = np.asarray(ref_view.get("point_background_rms", None))
    bg_cmp = np.asarray(cmp_view.get("point_background_rms", None))
    if bg_ref is None or bg_ref.size == 0:
        bg_ref = np.empty_like(peaks_ref)
        bg_ref_per_example = ref_view.get("background_rms_per_example")
        if bg_ref_per_example is None:
            bg_ref.fill(eps)
        else:
            example_idx = np.asarray(
                ref_view.get("peak_example_indices", np.zeros(peaks_ref.shape, dtype=np.int32))
            )
            bg_ref = np.asarray(bg_ref_per_example, dtype=np.float64)[example_idx]

    if bg_cmp is None or bg_cmp.size == 0:
        bg_cmp = np.empty_like(peaks_cmp)
        bg_cmp_per_example = cmp_view.get("background_rms_per_example")
        if bg_cmp_per_example is None:
            bg_cmp.fill(eps)
        else:
            example_idx = np.asarray(
                cmp_view.get("peak_example_indices", np.zeros(peaks_cmp.shape, dtype=np.int32))
            )
            bg_cmp = np.asarray(bg_cmp_per_example, dtype=np.float64)[example_idx]

    snr_ref = peaks_ref / (bg_ref + eps)
    snr_cmp = peaks_cmp / (bg_cmp + eps)
    ratio = snr_cmp / (snr_ref + eps)

    if not scatterer_batch:
        pos = np.empty((0, 2), dtype=np.float64)
    else:
        pos = np.vstack([np.asarray(arr, dtype=np.float64) for arr in scatterer_batch])

    if ratio.size != pos.shape[0]:
        peak_example_indices = np.asarray(
            ref_view.get("peak_example_indices", np.zeros(ratio.shape, dtype=np.int32))
        )
        peak_scatterer_indices = np.asarray(
            ref_view.get("peak_scatterer_indices", np.arange(ratio.size, dtype=np.int32))
        )
        coords_list: list[np.ndarray] = []
        for ex_idx, scat_idx in zip(peak_example_indices, peak_scatterer_indices):
            coords_list.append(scatterer_batch[int(ex_idx)][int(scat_idx)])
        pos = np.asarray(coords_list, dtype=np.float64)

    x = pos[:, 0]
    z = pos[:, 1]

    plot_vals = ratio.copy()
    if scale == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            plot_vals = np.log10(plot_vals)

    if clip_percentiles is not None:
        pmin, pmax = float(clip_percentiles[0]), float(clip_percentiles[1])
        vmin = float(np.nanpercentile(plot_vals, pmin))
        vmax = float(np.nanpercentile(plot_vals, pmax))
    else:
        vmin = float(np.nanmin(plot_vals)) if plot_vals.size > 0 else 0.0
        vmax = float(np.nanmax(plot_vals)) if plot_vals.size > 0 else 1.0

    center = 0.0 if scale == "log" else 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
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
    result["ratio_label"] = f"{cmp_method}/{ref_method}"
    if return_fig:
        result["fig"] = fig

    return result
