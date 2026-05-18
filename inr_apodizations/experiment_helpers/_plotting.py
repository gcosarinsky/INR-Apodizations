"""Experiment-level plot utilities for apodization analysis and reflector profiles."""

from __future__ import annotations

import os
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from inr_apodizations.evaluation.scatterers import select_reflector_scatterer
from inr_apodizations.plots import to_db


def plot_apodization_energy_comparison(
    hanning_apod: np.ndarray,
    inr_apod_after: np.ndarray,
    output_path: str,
    extent: tuple[float, float, float, float],
    cmap: str = "viridis",
) -> None:
    """Plot pixel-wise apodization energy maps for Hanning and INR after training.

    The energy proxy is computed as ``sum(abs(apodization), axis=0)``, where
    axis 0 is the element dimension of the apodization map ``(E, Z, X)``.

    Args:
        hanning_apod: Hanning apodization with shape ``(E, Z, X)``.
        inr_apod_after: INR learned apodization with shape ``(E, Z, X)``.
        output_path: Output figure path.
        extent: Matplotlib imshow extent ``(xmin, xmax, zmax, zmin)`` in mm.
        cmap: Colormap for energy maps.

    Raises:
        ValueError: If input shapes are not 3D or not equal.
    """
    hanning = np.asarray(hanning_apod)
    inr_after = np.asarray(inr_apod_after)

    if hanning.ndim != 3 or inr_after.ndim != 3:
        raise ValueError("hanning_apod and inr_apod_after must be 3D arrays with shape (E, Z, X)")
    if hanning.shape != inr_after.shape:
        raise ValueError("hanning_apod and inr_apod_after must have identical shapes")

    hanning_energy = np.sum(np.abs(hanning), axis=0)
    inr_energy = np.sum(np.abs(inr_after), axis=0)
    delta_energy = inr_energy - hanning_energy

    vmax = float(max(np.max(hanning_energy), np.max(inr_energy)))
    if vmax <= 0.0:
        vmax = 1.0

    delta_abs = float(np.max(np.abs(delta_energy)))
    if delta_abs <= 0.0:
        delta_abs = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)

    axes[0].imshow(
        hanning_energy,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        extent=extent,
        aspect="auto",
    )
    axes[0].set_title("Hanning |sum_e |w_e||")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("z (mm)")

    im1 = axes[1].imshow(
        inr_energy,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        extent=extent,
        aspect="auto",
    )
    axes[1].set_title("INR after |sum_e |w_e||")
    axes[1].set_xlabel("x (mm)")

    im2 = axes[2].imshow(
        delta_energy,
        cmap="RdBu_r",
        vmin=-delta_abs,
        vmax=delta_abs,
        extent=extent,
        aspect="auto",
    )
    axes[2].set_title("INR - Hanning")
    axes[2].set_xlabel("x (mm)")

    fig.tight_layout(rect=[0, 0, 0.9, 1])
    cbar_energy_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im1, cax=cbar_energy_ax, label="sum_e |w_e|")

    cbar_delta_ax = fig.add_axes([0.955, 0.15, 0.015, 0.7])
    fig.colorbar(im2, cax=cbar_delta_ax, label="delta energy")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_reflector_profile_context(
    *,
    scatterers_mm: np.ndarray,
    profile_cfg: dict,
    images_linear: Mapping[str, np.ndarray],
    images_db: Mapping[str, np.ndarray],
    target_image: np.ndarray | None,
) -> dict:
    """Build a normalized context dict for reflector profile plotting.

    Args:
        scatterers_mm: Scatterers for one example in mm with at least [x, z].
        profile_cfg: User reflector profile config block.
        images_linear: Mapping of method name to linear image.
        images_db: Mapping of method name to dB image.
        target_image: Optional target image.

    Returns:
        Dictionary with selected scatterer, selected images and plotting flags.

    Raises:
        ValueError: If configuration is invalid or no images are selected.
    """
    if scatterers_mm is None or len(scatterers_mm) == 0:
        raise ValueError("Reflector profile plotting requires scatterers for the selected example.")

    line_length_mm = float(profile_cfg.get("line_length_mm", 5.0))
    overlay_profiles = bool(profile_cfg.get("overlay_profiles", True))
    use_db_profiles = bool(profile_cfg.get("use_db", True))
    include_target_profile = bool(profile_cfg.get("include_target", True))
    scatterer_selection = str(profile_cfg.get("scatterer_selection", "strongest"))
    scatterer_idx_raw = profile_cfg.get("scatterer_idx")
    scatterer_idx = None if scatterer_idx_raw is None else int(scatterer_idx_raw)

    selected_scatterer = select_reflector_scatterer(
        scatterers_mm,
        selection=scatterer_selection,
        scatterer_idx=scatterer_idx,
    )

    if use_db_profiles:
        profile_image_source = dict(images_db)
    else:
        profile_image_source = {
            name: np.abs(image) for name, image in images_linear.items()
        }

    if include_target_profile and target_image is not None:
        profile_image_source["target"] = (
            to_db(target_image, ref=float(np.max(np.abs(target_image))))
            if use_db_profiles
            else np.abs(target_image)
        )

    requested_profile_images = profile_cfg.get("images")
    if requested_profile_images is not None and not isinstance(requested_profile_images, list):
        raise ValueError("`reflector_lateral_profile.images` must be a YAML list when provided.")

    if requested_profile_images is None or len(requested_profile_images) == 0:
        profile_image_names = list(profile_image_source.keys())
    else:
        profile_image_names = [str(name).strip().lower() for name in requested_profile_images]

    missing_profile_images = [
        image_name for image_name in profile_image_names if image_name not in profile_image_source
    ]
    if missing_profile_images:
        raise ValueError(
            "Unknown images requested in `reflector_lateral_profile.images`: "
            f"{missing_profile_images}. Available images: {list(profile_image_source.keys())}"
        )

    selected_profile_images = {
        image_name: profile_image_source[image_name] for image_name in profile_image_names
    }
    if len(selected_profile_images) == 0:
        raise ValueError("No images available for `reflector_lateral_profile` plotting.")

    return {
        "selected_scatterer": selected_scatterer,
        "line_length_mm": line_length_mm,
        "overlay_profiles": overlay_profiles,
        "use_db_profiles": use_db_profiles,
        "selected_profile_images": selected_profile_images,
    }
