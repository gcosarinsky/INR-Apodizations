"""Helpers for interactive DAS visualization in das_standard_apodizations.py.

This module provides functions to generate DAS comparison figures for multiple
examples, suitable for use with the InteractiveImageNavigator.
"""
from __future__ import annotations

import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.kernels import KernelParameters2D


def to_db(image: np.ndarray, ref: float, eps: float = 1e-8) -> np.ndarray:
    """Convert linear magnitude image to dB.

    Args:
        image: Complex or real image in linear domain.
        ref: Positive reference magnitude.
        eps: Small epsilon to avoid numerical issues.

    Returns:
        Magnitude image in dB.
    """
    magnitude = np.abs(image)
    return 20.0 * np.log10((magnitude / (ref + eps)) + eps)


def generate_das_comparison_figure(
    example_idx: int,
    delayed_samples_all: np.ndarray,
    targets_all: Optional[np.ndarray],
    apods: dict,
    kp: KernelParameters2D,
    cm: CoordinateManager,
    cmap: str = "gray",
    vmin_db: float = -60.0,
    vmax_db: float = 0.0,
    normalize_per_image: bool = True,
) -> tuple[plt.Figure, dict]:
    """Generate a DAS comparison figure for a single example.

    Args:
        example_idx: Index of the example to process.
        delayed_samples_all: Full delayed samples array (n_examples, ...).
        targets_all: Full targets array or None.
        apods: Dictionary of apodization tensors (method_name -> tensor).
        kp: KernelParameters2D instance.
        cm: CoordinateManager instance.
        cmap: Matplotlib colormap name.
        vmin_db: Lower dB display bound.
        vmax_db: Upper dB display bound.
        normalize_per_image: If True, normalize each panel independently.

    Returns:
        Tuple of (matplotlib.figure.Figure, metadata_dict) where metadata includes
        DAS images, target image, etc. for potential further processing.
    """
    # Extract example data
    delayed_samples_np = np.asarray(delayed_samples_all[example_idx])
    delayed_samples_tf = tf.convert_to_tensor(delayed_samples_np, dtype=tf.complex64)

    # Compute DAS images
    das_images_linear = {"uniform": tf.reduce_sum(delayed_samples_tf, axis=0).numpy()}
    for method_name, apod_tensor in apods.items():
        weighted = delayed_samples_tf * tf.cast(apod_tensor, tf.complex64)
        das_images_linear[method_name] = tf.reduce_sum(weighted, axis=0).numpy()

    # Convert to dB
    if normalize_per_image:
        das_images_db = {
            name: to_db(image, ref=float(np.max(np.abs(image))))
            for name, image in das_images_linear.items()
        }
    else:
        shared_ref = max(float(np.max(np.abs(image))) for image in das_images_linear.values())
        das_images_db = {
            name: to_db(image, ref=shared_ref) for name, image in das_images_linear.items()
        }

    # Load target if available
    target_np = np.asarray(targets_all[example_idx]) if targets_all is not None else None
    target_db = None
    if target_np is not None:
        target_db = to_db(target_np, ref=float(np.max(np.abs(target_np))))
        das_images_db["target"] = target_db

    # Create figure
    extent = kp.get_imshow_extent()
    ordered_names = ["uniform"] + [name for name in das_images_db.keys()
                                    if name != "uniform" and name != "target"]
    if target_np is not None:
        ordered_names.append("target")

    n_panels = len(ordered_names)
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)
    width_per_panel = 5
    height_per_panel = 5

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width_per_panel * ncols, height_per_panel * nrows),
        sharex=True,
        sharey=True,
    )

    # Flatten axes to handle both single and multiple panels uniformly
    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    first_im = None
    for idx, image_name in enumerate(ordered_names):
        ax = axes[idx]
        im = ax.imshow(
            das_images_db[image_name],
            cmap=cmap,
            vmin=vmin_db,
            vmax=vmax_db,
            extent=extent,
            aspect="auto",
        )
        if first_im is None:
            first_im = im

        title_name = "Uniform" if image_name == "uniform" else image_name.capitalize()
        ax.set_title(f"DAS {title_name} (dB)")
        ax.set_xlabel("x (mm)")
        if idx % ncols == 0:
            ax.set_ylabel("z (mm)")

    # Hide unused subplots
    for idx in range(n_panels, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"DAS Comparison - Example {example_idx}", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    # Add colorbar
    if first_im is not None:
        cbar_ax = fig.add_axes([0.93, 0.1, 0.013, 0.8])
        fig.colorbar(first_im, cax=cbar_ax, label="dB")

    # Store metadata
    metadata = {
        "example_idx": example_idx,
        "das_images_db": das_images_db,
        "das_images_linear": das_images_linear,
        "target_db": target_db,
        "target_np": target_np,
    }

    return fig, metadata
