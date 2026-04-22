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
from inr_apodizations.utils import to_db

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


def _prepare_comparison_images_db(
    images_linear: dict[str, np.ndarray],
    normalize_each: bool = True,
) -> dict[str, np.ndarray]:
    """Convert a dictionary of linear-domain images to dB scale.

    Helper function to reduce duplication in comparison figure generators.

    Args:
        images_linear: Dictionary mapping image names to linear-domain magnitude arrays.
        normalize_each: If True, each image is normalized by its own maximum value.
            If False, all images share a common reference (the global maximum).

    Returns:
        Dictionary mapping image names to dB-scale arrays.
    """
    if normalize_each:
        return {
            name: to_db(image, ref=float(np.max(np.abs(image))))
            for name, image in images_linear.items()
        }
    else:
        shared_ref = max(float(np.max(np.abs(image))) for image in images_linear.values())
        return {name: to_db(image, ref=shared_ref) for name, image in images_linear.items()}


def _order_das_panel_names(images: dict[str, np.ndarray]) -> list[str]:
    """Return panel names with 'uniform' first and 'target' last when present."""
    ordered: list[str] = []
    if "uniform" in images:
        ordered.append("uniform")

    for name in images.keys():
        if name not in {"uniform", "target"}:
            ordered.append(name)

    if "target" in images:
        ordered.append("target")
    return ordered


def _render_das_comparison_panels(
    fig: plt.Figure,
    images_db: dict[str, np.ndarray],
    extent: tuple[float, float, float, float],
    cmap: str,
    vmin_db: float,
    vmax_db: float,
) -> plt.Axes | None:
    """Render DAS panels and return the first imshow artist for a shared colorbar."""
    ordered_names = _order_das_panel_names(images_db)
    n_panels = len(ordered_names)
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)

    axes: list[plt.Axes] = []
    for idx in range(nrows * ncols):
        axes.append(fig.add_subplot(nrows, ncols, idx + 1))

    first_im = None
    for idx, image_name in enumerate(ordered_names):
        ax = axes[idx]
        current_im = ax.imshow(
            images_db[image_name],
            cmap=cmap,
            vmin=vmin_db,
            vmax=vmax_db,
            extent=extent,
            aspect="auto",
        )
        if first_im is None:
            first_im = current_im

        title_name = "Uniform" if image_name == "uniform" else image_name.capitalize()
        ax.set_title(f"DAS {title_name} (dB)")
        ax.set_xlabel("x (mm)")
        if idx % ncols == 0:
            ax.set_ylabel("z (mm)")

    for idx in range(n_panels, len(axes)):
        axes[idx].axis("off")

    return first_im


def generate_das_comparison_figure(
    images_linear: dict[str, np.ndarray],
    extent: tuple[float, float, float, float],
    target_image: np.ndarray | None = None,
    title: str = "DAS comparison",
    existing_figure: plt.Figure | None = None,
    cmap: str = "gray",
    vmin_db: float = -60.0,
    vmax_db: float = 0.0,
    normalize_per_image: bool = True,
) -> tuple[plt.Figure, dict]:
    """Generate a DAS comparison figure from already reduced images.

    Args:
        images_linear: Dictionary mapping method names to reduced linear-domain
            images ``(nz, nx)``.
        extent: Matplotlib imshow extent as ``(xmin, xmax, zmax, zmin)`` in mm.
        target_image: Optional reduced target image in linear domain.
        title: Figure title.
        existing_figure: Optional pre-existing figure to render into.
        cmap: Matplotlib colormap name.
        vmin_db: Lower dB display bound.
        vmax_db: Upper dB display bound.
        normalize_per_image: If True, normalize each panel independently.

    Returns:
        Tuple of ``(figure, metadata_dict)`` with generated DAS images and target.
    """
    das_images_linear = {
        name: np.asarray(image)
        for name, image in images_linear.items()
    }

    target_np = np.asarray(target_image) if target_image is not None else None

    das_images_db = _prepare_comparison_images_db(
        das_images_linear,
        normalize_each=normalize_per_image,
    )

    target_db = None
    if target_np is not None:
        target_db = to_db(target_np, ref=float(np.max(np.abs(target_np))))
        das_images_db["target"] = target_db

    ordered_names = _order_das_panel_names(das_images_db)
    n_panels = max(1, len(ordered_names))
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)
    if existing_figure is None:
        fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
    else:
        fig = existing_figure
        fig.clear()
        fig.set_size_inches(5 * ncols, 5 * nrows)

    first_im = _render_das_comparison_panels(
        fig=fig,
        images_db=das_images_db,
        extent=extent,
        cmap=cmap,
        vmin_db=vmin_db,
        vmax_db=vmax_db,
    )

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])

    if first_im is not None:
        cbar_ax = fig.add_axes([0.93, 0.1, 0.013, 0.8])
        fig.colorbar(first_im, cax=cbar_ax, label="dB")

    metadata = {
        "das_images_db": das_images_db,
        "das_images_linear": das_images_linear,
        "target_db": target_db,
        "target_np": target_np,
    }

    return fig, metadata


def plot_training_curves(
    history: dict,
    output_path: str,
    reference_mae: dict | None = None,
) -> None:
    """Save a grouped training-curve figure from a Keras history dictionary.

    Args:
        history: Mapping with metric lists, typically ``history.history``.
        output_path: Path to the output PNG file.
        reference_mae: Optional baseline MAE values to overlay on the absolute
            reconstruction metrics panel.

    The figure groups metrics into dedicated panels for loss, absolute
    reconstruction errors, relative reconstruction errors, SSIM,
    regularization metrics, and any remaining tracked values. This keeps the
    plot compact while still exposing metrics such as ``reg_loss`` and
    ``reg_active_rate`` that were previously omitted.
    """

    if not history:
        raise ValueError("history must not be empty")

    def _is_regularization_metric(metric_name: str) -> bool:
        return (
            metric_name == "reg_loss"
            or metric_name.startswith("reg_")
            or "w_norm" in metric_name
        )

    def _is_loss_metric(metric_name: str) -> bool:
        return metric_name in {"loss", "val_loss"} or (
            metric_name.endswith("_loss") and not metric_name.startswith("reg_")
        )

    def _is_ssim_metric(metric_name: str) -> bool:
        return "ssim" in metric_name

    def _is_absolute_reconstruction_metric(metric_name: str) -> bool:
        if (
            _is_loss_metric(metric_name)
            or _is_regularization_metric(metric_name)
            or _is_ssim_metric(metric_name)
        ):
            return False
        if "relative" in metric_name:
            return False
        return any(token in metric_name for token in ("mae", "error"))

    def _is_relative_reconstruction_metric(metric_name: str) -> bool:
        if (
            _is_loss_metric(metric_name)
            or _is_regularization_metric(metric_name)
            or _is_ssim_metric(metric_name)
        ):
            return False
        return "relative" in metric_name

    ordered_keys = [name for name in history.keys() if len(history[name]) > 0]
    used_keys: set[str] = set()
    panels: list[tuple[str, list[str], str]] = []

    loss_keys = [name for name in ordered_keys if _is_loss_metric(name)]
    if loss_keys:
        panels.append(("Loss", loss_keys, "Loss"))
        used_keys.update(loss_keys)

    absolute_reconstruction_keys = [
        name
        for name in ordered_keys
        if name not in used_keys and _is_absolute_reconstruction_metric(name)
    ]
    if absolute_reconstruction_keys:
        panels.append(
            ("Absolute reconstruction metrics", absolute_reconstruction_keys, "Metric value")
        )
        used_keys.update(absolute_reconstruction_keys)

    relative_reconstruction_keys = [
        name
        for name in ordered_keys
        if name not in used_keys and _is_relative_reconstruction_metric(name)
    ]
    if relative_reconstruction_keys:
        panels.append(
            ("Relative reconstruction metrics", relative_reconstruction_keys, "Metric value")
        )
        used_keys.update(relative_reconstruction_keys)

    ssim_keys = [name for name in ordered_keys if name not in used_keys and _is_ssim_metric(name)]
    if ssim_keys:
        panels.append(("SSIM metrics", ssim_keys, "SSIM"))
        used_keys.update(ssim_keys)

    regularization_keys = [
        name for name in ordered_keys if name not in used_keys and _is_regularization_metric(name)
    ]
    if regularization_keys:
        panels.append(("Regularization metrics", regularization_keys, "Metric value"))
        used_keys.update(regularization_keys)

    other_keys = [name for name in ordered_keys if name not in used_keys]
    if other_keys:
        panels.append(("Other metrics", other_keys, "Metric value"))

    if not panels:
        raise ValueError("history does not contain plottable metrics")

    fig_height = max(3.0, 2.8 * len(panels))
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, fig_height), constrained_layout=True)
    if len(panels) == 1:
        axes = np.array([axes])

    palette = list(plt.rcParams["axes.prop_cycle"].by_key().get("color", []))
    if not palette:
        palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    def _plot_panel(axis, metric_names: list[str], title: str, ylabel: str, include_reference: bool = False) -> None:
        base_colors: dict[str, str] = {}
        color_index = 0

        for metric_name in metric_names:
            series = history[metric_name]
            base_name = metric_name[4:] if metric_name.startswith("val_") else metric_name
            if base_name not in base_colors:
                base_colors[base_name] = palette[color_index % len(palette)]
                color_index += 1

            line_style = "--" if metric_name.startswith("val_") else "-"
            axis.plot(series, label=metric_name, color=base_colors[base_name], linestyle=line_style)

        if include_reference and reference_mae:
            ref_colors = ["gray", "tab:purple", "tab:green", "tab:brown"]
            for idx, (ref_name, ref_value) in enumerate(reference_mae.items()):
                try:
                    ref_y = float(ref_value)
                except Exception:
                    continue
                ref_color = ref_colors[idx % len(ref_colors)]
                axis.axhline(
                    y=ref_y,
                    color=ref_color,
                    linestyle=":",
                    linewidth=1.2,
                    label="_nolegend_",
                )
                axis.annotate(
                    f"ref_{ref_name}",
                    xy=(0.975, ref_y),
                    xycoords=("axes fraction", "data"),
                    xytext=(-4, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=8,
                    color=ref_color,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
                )

        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        lines, labels = axis.get_legend_handles_labels()
        if lines:
            axis.legend(lines, labels, fontsize=9)

    for axis, (title, metric_names, ylabel) in zip(axes, panels, strict=False):
        include_reference = title == "Absolute reconstruction metrics"
        _plot_panel(axis, metric_names, title, ylabel, include_reference=include_reference)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
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
    baseline_name: str = "Uniform",
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
        str(baseline_name): np.asarray(uniform_image),
        "inr before": np.asarray(inr_before_image),
        "inr after": np.asarray(inr_after_image),
    }

    fig, _ = generate_das_comparison_figure(
        images_linear=images_linear,
        extent=extent,
        target_image=np.asarray(target_image),
        title="DAS comparison",
        cmap=cmap,
        vmin_db=vmin_db,
        vmax_db=vmax_db,
        normalize_per_image=normalize_each_image,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_lateral_reflector_profiles(
    images: dict[str, np.ndarray] | list[np.ndarray] | tuple[np.ndarray, ...],
    output_path: str,
    extent: tuple[float, float, float, float],
    x_center: float,
    z_center: float,
    line_length: float,
    thickness_mm: float = 0.0,
    overlay_profiles: bool = True,
    labels: list[str] | tuple[str, ...] | None = None,
    cm: CoordinateManager | None = None,
    vmin_db: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Plot horizontal reflector profiles centered at a physical coordinate.

    Args:
        images: Collection of 2D images to sample. It can be a dictionary with
            display names as keys or a list/tuple of arrays.
        output_path: Path to the output PNG file.
        extent: Matplotlib ``imshow`` extent as ``(xmin, xmax, zmax, zmin)`` in mm.
        x_center: Lateral center of the sampled line in mm.
        z_center: Depth center of the sampled line in mm.
        line_length: Total horizontal profile length in mm.
        thickness_mm: Vertical band thickness in mm used to aggregate profiles.
            If ``thickness_mm <= 0``, profiles are sampled on a single line.
            If ``thickness_mm > 0``, each profile point is the maximum value
            across the selected depth band.
        overlay_profiles: If True, all profiles are drawn on the same axes.
            If False, one subplot is created per image.
        labels: Optional labels used when ``images`` is a list or tuple.
        cm: Optional coordinate manager used to recover the physical image axes.
            When provided, its ``x`` and ``z`` coordinates are used to locate
            the nearest sampling indices. Otherwise, axes are reconstructed from
            ``extent``.
        vmin_db: Optional lower dB display bound for the profile plots. If None, 
        the minimum value is determined from the data.

    Returns:
        Tuple with the sampled lateral coordinates in mm and a dictionary mapping
        image names to extracted profile arrays.

    Raises:
        ValueError: If inputs are empty, inconsistent, non-2D, or if the sampled
            window falls outside the image domain.
    """
    if line_length <= 0.0:
        raise ValueError(f"`line_length` must be > 0, got {line_length}.")
    if thickness_mm < 0.0:
        raise ValueError(f"`thickness_mm` must be >= 0, got {thickness_mm}.")

    if isinstance(images, dict):
        image_items = [(str(name), np.asarray(image)) for name, image in images.items()]
    else:
        image_list = [np.asarray(image) for image in images]
        if len(image_list) == 0:
            raise ValueError("`images` must contain at least one image.")

        if labels is None:
            inferred_labels = [f"Image {idx + 1}" for idx in range(len(image_list))]
        else:
            if len(labels) != len(image_list):
                raise ValueError(
                    "`labels` length must match the number of images when `images` "
                    "is a list or tuple."
                )
            inferred_labels = [str(label) for label in labels]
        image_items = list(zip(inferred_labels, image_list))

    if len(image_items) == 0:
        raise ValueError("`images` must contain at least one image.")

    reference_shape = image_items[0][1].shape
    if len(reference_shape) != 2:
        raise ValueError(
            f"Expected 2D images, got shape {reference_shape} for '{image_items[0][0]}'."
        )

    for image_name, image in image_items:
        if image.ndim != 2:
            raise ValueError(f"Expected 2D image for '{image_name}', got shape {image.shape}.")
        if image.shape != reference_shape:
            raise ValueError(
                "All images must share the same shape. "
                f"Expected {reference_shape}, got {image.shape} for '{image_name}'."
            )

    nz, nx = reference_shape
    if cm is not None:
        coords_phys = cm.get_coordinates_1d(scaled=False)
        x_axis = np.asarray(coords_phys["x"])
        z_axis = np.asarray(coords_phys["z"])
        if x_axis.shape[0] != nx or z_axis.shape[0] != nz:
            raise ValueError(
                "CoordinateManager axes are inconsistent with image shape. "
                f"Expected (nz={nz}, nx={nx}), got (nz={z_axis.shape[0]}, nx={x_axis.shape[0]})."
            )
    else:
        x_min, x_max, z_max, z_min = [float(value) for value in extent]
        x_axis = np.linspace(x_min, x_max, nx)
        z_axis = np.linspace(z_min, z_max, nz)

    half_length = line_length / 2.0
    x_start = x_center - half_length
    x_end = x_center + half_length
    if x_start < x_axis.min() or x_end > x_axis.max():
        raise ValueError(
            "Requested horizontal line falls outside the image lateral extent. "
            f"Requested [{x_start:.3f}, {x_end:.3f}] mm, available "
            f"[{x_axis.min():.3f}, {x_axis.max():.3f}] mm."
        )
    if z_center < z_axis.min() or z_center > z_axis.max():
        raise ValueError(
            "Requested z_center falls outside the image depth extent. "
            f"Requested {z_center:.3f} mm, available "
            f"[{z_axis.min():.3f}, {z_axis.max():.3f}] mm."
        )

    z_idx = int(np.argmin(np.abs(z_axis - z_center)))
    x_start_idx = int(np.searchsorted(x_axis, x_start, side="left"))
    x_end_idx = int(np.searchsorted(x_axis, x_end, side="right"))
    if x_end_idx - x_start_idx < 2:
        raise ValueError(
            "The requested line_length is too short for the image sampling resolution."
        )

    if thickness_mm <= 0.0:
        z_start_idx = z_idx
        z_end_idx = z_idx + 1
    else:
        half_thickness = thickness_mm / 2.0
        z_start = z_center - half_thickness
        z_end = z_center + half_thickness
        z_start_idx = int(np.searchsorted(z_axis, z_start, side="left"))
        z_end_idx = int(np.searchsorted(z_axis, z_end, side="right"))
        z_start_idx = max(0, z_start_idx)
        z_end_idx = min(nz, z_end_idx)
        if z_end_idx - z_start_idx < 1:
            z_start_idx = z_idx
            z_end_idx = z_idx + 1

    sampled_x = x_axis[x_start_idx:x_end_idx]
    if thickness_mm <= 0.0:
        profiles = {
            image_name: np.asarray(image[z_idx, x_start_idx:x_end_idx])
            for image_name, image in image_items
        }
    else:
        profiles = {
            image_name: np.max(
                np.asarray(image[z_start_idx:z_end_idx, x_start_idx:x_end_idx]),
                axis=0,
            )
            for image_name, image in image_items
        }

    if overlay_profiles:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
        for image_name, profile in profiles.items():
            ax.plot(sampled_x, profile, linewidth=2, label=image_name)
            if vmin_db is not None:
                ax.set_ylim(vmin_db, 0.0)
        if thickness_mm <= 0.0:
            ax.set_title(
                "Lateral reflector profiles "
                f"at z={z_axis[z_idx]:.2f} mm centered on x={x_center:.2f} mm"
            )
        else:
            z0 = float(z_axis[z_start_idx])
            z1 = float(z_axis[z_end_idx - 1])
            ax.set_title(
                "Lateral reflector profiles (max over thickness) "
                f"z=[{z0:.2f}, {z1:.2f}] mm centered on x={x_center:.2f} mm"
            )
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.axvline(x_center, color="black", linestyle=":", linewidth=1.2)
        ax.legend()
    else:
        fig, axes = plt.subplots(
            len(profiles),
            1,
            figsize=(10, max(4, 3.2 * len(profiles))),
            sharex=True,
            constrained_layout=True,
        )
        axes_array = np.atleast_1d(axes)
        for ax, (image_name, profile) in zip(axes_array, profiles.items()):
            ax.plot(sampled_x, profile, linewidth=2)
            if vmin_db is not None:
                ax.set_ylim(vmin_db, 0.0)
            if thickness_mm <= 0.0:
                ax.set_title(
                    f"{image_name} at z={z_axis[z_idx]:.2f} mm centered on x={x_center:.2f} mm"
                )
            else:
                z0 = float(z_axis[z_start_idx])
                z1 = float(z_axis[z_end_idx - 1])
                ax.set_title(
                    f"{image_name} max over z=[{z0:.2f}, {z1:.2f}] mm centered on x={x_center:.2f} mm"
                )
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            ax.axvline(x_center, color="black", linestyle=":", linewidth=1.2)
        axes_array[-1].set_xlabel("x (mm)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)

    return sampled_x, profiles


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


if __name__ == "__main__":
    app()
