from pathlib import Path
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from loguru import logger
from tqdm import tqdm
import typer

from inr_apodizations.apodizations import extract_map_for_x
from inr_apodizations.config import FIGURES_DIR, PROCESSED_DATA_DIR
from inr_apodizations.coordinate_manager import CoordinateManager

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating plot from data...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Plot generation complete.")
    # -----------------------------------------


def to_db(image: np.ndarray, ref: float, eps: float = 1e-8) -> np.ndarray:
    """Convert an image magnitude from linear domain to decibels.

    Args:
        image: Real or complex image in linear domain.
        ref: Positive reference magnitude used as 0 dB.
        eps: Small value to avoid numerical instability.

    Returns:
        NumPy array with image values in dB.
    """
    magnitude = np.abs(image)
    return 20.0 * np.log10((magnitude / (ref + eps)) + eps)


def plot_training_curves(history: dict, output_path: str) -> None:
    """
    Save a training curve figure from a Keras history dictionary.

    Args:
        history: Mapping with metric lists, typically ``history.history``.
        output_path: Path to the output PNG file.

    Note:
        The docstring could be expanded to describe each supported metric
        in more detail. The current implementation uses a series of
        ``if`` statements to handle different metric names, which results
        in a somewhat clunky and non‑extensible way of managing metric
        visualization. Future enhancements should consider a more
        systematic mapping of metric names to plot configurations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    if "loss" in history:
        axes[0].plot(history["loss"], label="loss", color="black")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val_loss", color="tab:red")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    if axes[0].lines:
        axes[0].legend()

    # Plot MAE on the primary y-axis and SSIM on a secondary y-axis if present.
    has_metric = False
    mae_plotted = False
    ssim_plotted = False
    if "mae" in history or "val_mae" in history:
        if "mae" in history:
            axes[1].plot(history["mae"], label="mae", color="tab:blue")
            mae_plotted = True
        if "val_mae" in history:
            axes[1].plot(history["val_mae"], label="val_mae", color="tab:orange")
            mae_plotted = True
        has_metric = True

    ssim_ax = None
    if "ssim_metric" in history or "val_ssim_metric" in history:
        # Use a twin y-axis for SSIM (range ~[0,1]) to avoid mixing scales.
        ssim_ax = axes[1].twinx()
        if "ssim_metric" in history:
            ssim_ax.plot(history["ssim_metric"], label="ssim", color="tab:green")
            ssim_plotted = True
        if "val_ssim_metric" in history:
            ssim_ax.plot(history["val_ssim_metric"], label="val_ssim", color="tab:red")
            ssim_plotted = True
        has_metric = True

    if has_metric:
        axes[1].set_title("Metrics")
        axes[1].set_xlabel("Epoch")
        if mae_plotted:
            axes[1].set_ylabel("MAE")
            axes[1].grid(True, alpha=0.3)
        if ssim_plotted and ssim_ax is not None:
            ssim_ax.set_ylabel("SSIM")

        # Build combined legend from both axes if needed.
        lines, labels = axes[1].get_legend_handles_labels()
        if ssim_ax is not None:
            l2, lbl2 = ssim_ax.get_legend_handles_labels()
            lines += l2
            labels += lbl2
        if lines:
            axes[1].legend(lines, labels)
    else:
        axes[1].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_das_comparison_db(
    uniform_image: np.ndarray,
    inr_before_image: np.ndarray,
    inr_after_image: np.ndarray,
    target_image: np.ndarray,
    output_path: str,
    extent: tuple[float, float, float, float],
    cmap: str = "gray",
    vmin_db: float = -60.0,
    vmax_db: float = 0.0,
    normalize_each_image: bool = False,
) -> None:
    """Save a four-panel DAS comparison in decibels.

    Args:
        uniform_image: DAS image using uniform apodization.
        inr_before_image: INR reconstruction before training.
        inr_after_image: INR reconstruction after training.
        target_image: Reference target image.
        output_path: Path to the output PNG file.
        extent: Matplotlib imshow extent from beamforming geometry.
        cmap: Colormap used for all panels.
        vmin_db: Lower dB display bound.
        vmax_db: Upper dB display bound.
        normalize_each_image: If True, each panel uses its own maximum value as
            the 0 dB reference. If False, all panels share a common maximum
            reference across the full comparison.
    """
    images_linear = {
        "Uniform": np.asarray(uniform_image),
        "INR before": np.asarray(inr_before_image),
        "INR after": np.asarray(inr_after_image),
        "Target": np.asarray(target_image),
    }
    if normalize_each_image:
        images_db = {
            name: to_db(image, ref=float(np.max(np.abs(image))))
            for name, image in images_linear.items()
        }
    else:
        shared_ref = max(float(np.max(np.abs(image))) for image in images_linear.values())
        images_db = {
            name: to_db(image, ref=shared_ref) for name, image in images_linear.items()
        }

    # Dynamic layout: prefer a compact grid (up to 2 columns) to avoid excessively
    # wide horizontal figures. Each panel kept near square by default.
    n_panels = len(images_db)
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

    axes_flat = (
        np.array(axes).ravel()
        if isinstance(axes, (list, tuple, np.ndarray))
        else np.array([axes])
    )
    first_im = None
    for idx, (title, image_db) in enumerate(images_db.items()):
        ax = axes_flat[idx]
        current_im = ax.imshow(
            image_db,
            cmap=cmap,
            vmin=vmin_db,
            vmax=vmax_db,
            extent=extent,
            aspect="auto",
        )
        if first_im is None:
            first_im = current_im
        ax.set_title(f"DAS {title} (dB)")
        ax.set_xlabel("x (mm)")
        if idx % ncols == 0:
            ax.set_ylabel("z (mm)")

    # Hide any unused subplots.
    for j in range(n_panels, axes_flat.size):
        try:
            axes_flat[j].axis("off")
        except Exception:
            pass

    fig.suptitle("DAS comparison")
    # Leave room on the right for a single colorbar.
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    cbar_ax = fig.add_axes([0.92, 0.13, 0.02, 0.74])
    if first_im is not None:
        fig.colorbar(first_im, cax=cbar_ax, label="dB")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_apodization_before_after(
    cm: CoordinateManager,
    apod_before: np.ndarray,
    apod_after: np.ndarray,
    output_path: str,
    x_fixed: float = 0.0,
    z_fixed: float | None = None,
    z_profiles: list[float] | tuple[float, ...] | None = None,
    cmap: str = "viridis",
    hanning_apod: np.ndarray | None = None,
) -> None:
    """Save apodization maps and element-axis profiles at selected depths.

    Args:
        cm: Coordinate manager used to extract geometry coordinates.
        apod_before: INR weights before training with shape (E, Z, X).
        apod_after: INR weights after training with shape (E, Z, X).
        output_path: Path to the output PNG file.
        x_fixed: Lateral x value used for map extraction.
        z_fixed: Single depth used for profile extraction when ``z_profiles`` is not provided.
        z_profiles: Optional list/tuple of depths (mm) used for profile extraction.
        cmap: Colormap used for both maps.
    """
    # Plotting and profile requests are interpreted in physical units (mm)
    # to keep config values intuitive even when training uses scaled features.
    coords_phys = cm.get_coordinates_1d(scaled=False)
    x_elems = np.asarray(coords_phys["x_elem"])
    z_coords = np.asarray(coords_phys["z"])
    map_extent = (
        float(x_elems[0]),
        float(x_elems[-1]),
        float(z_coords[-1]),
        float(z_coords[0]),
    )

    map_before = extract_map_for_x(
        tf.convert_to_tensor(apod_before),
        cm,
        x_fixed=x_fixed,
        scaled=False,
    )
    map_after = extract_map_for_x(
        tf.convert_to_tensor(apod_after),
        cm,
        x_fixed=x_fixed,
        scaled=False,
    )

    # Convert maps to numpy arrays for plotting.
    mb = map_before.numpy()
    ma = map_after.numpy()
    vmin = float(min(float(mb.min()), float(ma.min())))
    vmax = float(max(float(mb.max()), float(ma.max())))

    # Determine z indices for profile extraction.
    if z_profiles is None or len(z_profiles) == 0:
        if z_fixed is None:
            z_indices = [len(z_coords) // 2]
        else:
            z_indices = [int(np.argmin(np.abs(z_coords - float(z_fixed))))]
    else:
        z_indices = [int(np.argmin(np.abs(z_coords - float(z)))) for z in z_profiles]
        # Keep order and avoid duplicated nearest-neighbor indices.
        z_indices = list(dict.fromkeys(z_indices))
    z_values = [float(z_coords[idx]) for idx in z_indices]

    # Lateral x index for profile extraction.
    x_coords = np.asarray(coords_phys["x"])
    x_idx = int(np.argmin(np.abs(x_coords - float(x_fixed))))

    apod_before_np = np.asarray(apod_before)
    apod_after_np = np.asarray(apod_after)
    hanning_np = np.asarray(hanning_apod) if hanning_apod is not None else None

    # Layout: 2x2 (maps on top row, profile on bottom-left, empty on bottom-right).
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )

    im0 = axes[0, 0].imshow(
        mb,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=map_extent,
        aspect="auto",
    )
    axes[0, 0].set_title("Apodization INR before")
    axes[0, 0].set_xlabel("Element lateral coordinate (mm)")
    axes[0, 0].set_ylabel("Depth z (mm)")

    im1 = axes[0, 1].imshow(
        ma,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=map_extent,
        aspect="auto",
    )
    axes[0, 1].set_title("Apodization INR after")
    axes[0, 1].set_xlabel("Element lateral coordinate (mm)")

    # Profile plot: one pair/triple of curves per selected depth on the same axes.
    depth_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(z_indices))))
    for color, z_idx, z_value in zip(depth_colors, z_indices, z_values):
        profile_before = apod_before_np[:, z_idx, x_idx]
        profile_after = apod_after_np[:, z_idx, x_idx]
        axes[1, 0].plot(
            x_elems,
            profile_before,
            label=f"INR before z={z_value:.2f} mm",
            linewidth=2,
            color=color,
            linestyle="-",
        )
        axes[1, 0].plot(
            x_elems,
            profile_after,
            label=f"INR after z={z_value:.2f} mm",
            linewidth=2,
            color=color,
            linestyle="--",
        )
        if hanning_np is not None:
            profile_hanning = hanning_np[:, z_idx, x_idx]
            axes[1, 0].plot(
                x_elems,
                profile_hanning,
                label=f"Hanning z={z_value:.2f} mm",
                linewidth=1.8,
                color=color,
                linestyle=":",
            )

    depth_list_text = ", ".join(f"{z:.2f}" for z in z_values)
    axes[1, 0].set_title(
        f"Profiles at x={x_fixed:.2f} mm, z=[{depth_list_text}] mm"
    )
    axes[1, 0].set_xlabel("Element lateral coordinate (mm)")
    axes[1, 0].set_ylabel("Apodization weight")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Bottom-right: show Hanning map if provided, else hide.
    if hanning_apod is not None:
        map_hanning = extract_map_for_x(
            tf.convert_to_tensor(hanning_apod),
            cm,
            x_fixed=x_fixed,
            scaled=False,
        )
        mh = map_hanning.numpy()
        im2 = axes[1, 1].imshow(
            mh,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=map_extent,
            aspect="auto",
        )
        axes[1, 1].set_title("Hanning apodization")
        axes[1, 1].set_xlabel("Element lateral coordinate (mm)")
        fig.colorbar(im2, ax=axes[1, 1], label="Weight")
    else:
        axes[1, 1].axis("off")

    # Colorbars for maps (original two maps).
    fig.colorbar(im0, ax=axes[0, 0], label="Weight")
    fig.colorbar(im1, ax=axes[0, 1], label="Weight")

    fig.suptitle(f"Apodization maps at x={x_fixed:.2f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    app()
