"""Quick sandbox validation for NSI reconstruction from delayed samples.

This module validates the NumPy NSI implementation using one sample from a
saved delayed-samples dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inr_apodizations.apodizations import (
    compute_das_baseline_numpy,
    compute_nsi_from_delayed_samples_numpy,
)
from inr_apodizations.config import DATA_DIR
from inr_apodizations.experiment_helpers import (
    build_coordinate_manager,
    load_delayed_samples_dataset,
)
from inr_apodizations.utils import find_latest_dataset_folder
from inr_apodizations.utils import to_db


def run_nsi_core_validation(
    dataset_folder: Path | None = None,
    dc: float = 0.05,
    f_number_override: float | None = None,
    return_payload: bool = False,
) -> dict:
    """Run a lightweight NSI consistency check on one delayed-samples example.

    Args:
        dataset_folder: Optional delayed-samples dataset folder. If None,
            the most recent dataset under ``data/delayed_samples_dataset`` is used.
        dc: NSI recombination offset.
        f_number_override: Optional F-number override. If None, the saved config
            value is used.
        return_payload: If True, also return intermediate arrays for quick
            interactive inspection in sandbox sessions.

    Returns:
        Dictionary with summary metrics and shape checks.

    Raises:
        ValueError: If loaded arrays have unexpected shape.
    """
    if dataset_folder is None:
        dataset_folder = find_latest_dataset_folder(DATA_DIR, "delayed_samples_dataset")

    delayed, _noise, _targets, _masks, _info = load_delayed_samples_dataset(str(dataset_folder))
    if delayed.ndim != 4:
        raise ValueError(f"Expected delayed shape (N, E, Z, X), got {delayed.shape}")

    cfg = np.load(Path(dataset_folder) / "cfg_delayed_samples.npy", allow_pickle=True).item()
    f_number = float(cfg["f_number"]) if f_number_override is None else float(f_number_override)

    _kp, cm = build_coordinate_manager(str(dataset_folder))
    coords = cm.get_coordinates_1d(scaled=False)
    x_coords = np.asarray(coords["x"], dtype=np.float32)
    z_coords = np.asarray(coords["z"], dtype=np.float32)
    x_elem_coords = np.asarray(coords["x_elem"], dtype=np.float32)

    sample = np.asarray(delayed[0], dtype=np.complex64)
    das_abs = compute_das_baseline_numpy(sample)

    img_sum, img_diff, nsi_img = compute_nsi_from_delayed_samples_numpy(
        delayed_samples=sample,
        x_coords=x_coords,
        z_coords=z_coords,
        x_elem_coords=x_elem_coords,
        f_number=f_number,
        dc=dc,
        normalize_by_n_subap=False,
    )

    _img_sum_norm, _img_diff_norm, nsi_img_norm = compute_nsi_from_delayed_samples_numpy(
        delayed_samples=sample,
        x_coords=x_coords,
        z_coords=z_coords,
        x_elem_coords=x_elem_coords,
        f_number=f_number,
        dc=dc,
        normalize_by_n_subap=True,
    )

    if img_sum.shape != sample.shape[1:] or img_diff.shape != sample.shape[1:] or nsi_img.shape != sample.shape[1:]:
        raise ValueError(
            "Unexpected NSI output shape. "
            f"Expected {(sample.shape[1], sample.shape[2])}, "
            f"got sum={img_sum.shape}, diff={img_diff.shape}, nsi={nsi_img.shape}"
        )

    summary = {
        "dataset_folder": str(dataset_folder),
        "sample_shape": tuple(int(v) for v in sample.shape),
        "sum_dtype": str(img_sum.dtype),
        "diff_dtype": str(img_diff.dtype),
        "nsi_dtype": str(nsi_img.dtype),
        "das_max": float(np.max(das_abs)),
        "nsi_max": float(np.max(nsi_img)),
        "nsi_norm_max": float(np.max(nsi_img_norm)),
        "nsi_mean": float(np.mean(nsi_img)),
        "nsi_norm_mean": float(np.mean(nsi_img_norm)),
        "nsi_has_nan": bool(np.isnan(nsi_img).any()),
        "nsi_has_inf": bool(np.isinf(nsi_img).any()),
    }

    print("NSI core validation summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if not return_payload:
        return summary

    return {
        "summary": summary,
        "sample": sample,
        "x_coords": x_coords,
        "z_coords": z_coords,
        "das_abs": das_abs,
        "img_sum": img_sum,
        "img_diff": img_diff,
        "nsi_img": nsi_img,
        "nsi_img_norm": nsi_img_norm,
    }


def plot_nsi_images_db(
    payload: dict,
    output_dir: Path | None = None,
    save_figure: bool = True,
    show_figure: bool = True,
    dpi: int = 150,
    cmap: str = "gray",
    vmin_db: float = -60.0,
    vmax_db: float = 0.0,
) -> tuple[plt.Figure, Path | None, dict[str, np.ndarray]]:
    """Create a dB panel plot for NSI-related images.

    Args:
        payload: Dictionary returned by ``run_nsi_core_validation(return_payload=True)``.
        output_dir: Optional output directory. If None, save under
            ``<dataset_folder>/nsi_sandbox``.
        save_figure: Whether to write the figure to disk.
        show_figure: Whether to display the figure window.
        dpi: Figure saving resolution.
        cmap: Matplotlib colormap.
        vmin_db: Lower display limit in dB.
        vmax_db: Upper display limit in dB.

    Returns:
        Tuple ``(figure, saved_path, db_images)``.
    """
    summary = payload["summary"]
    dataset_folder = Path(summary["dataset_folder"])
    if output_dir is None:
        output_dir = dataset_folder / "nsi_sandbox"
    output_dir.mkdir(parents=True, exist_ok=True)

    das_abs = np.asarray(payload["das_abs"], dtype=np.float32)
    img_sum = np.asarray(payload["img_sum"], dtype=np.complex64)
    img_diff = np.asarray(payload["img_diff"], dtype=np.complex64)
    nsi_img = np.asarray(payload["nsi_img"], dtype=np.float32)
    nsi_img_norm = np.asarray(payload["nsi_img_norm"], dtype=np.float32)

    x_coords = np.asarray(payload["x_coords"], dtype=np.float32)
    z_coords = np.asarray(payload["z_coords"], dtype=np.float32)
    extent = (float(x_coords[0]), float(x_coords[-1]), float(z_coords[-1]), float(z_coords[0]))

    images_linear = {
        "DAS Uniform": das_abs,
        "NSI Sum |S|": np.abs(img_sum),
        "NSI Diff |D|": np.abs(img_diff),
        "NSI": np.abs(nsi_img),
        "NSI (norm)": np.abs(nsi_img_norm),
    }

    images_db = {
        name: to_db(image, ref=float(np.max(np.abs(image))))
        for name, image in images_linear.items()
    }

    n_panels = len(images_db)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels + 1, 5), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    first_im = None
    for idx, (name, image_db) in enumerate(images_db.items()):
        current_im = axes[idx].imshow(
            image_db,
            cmap=cmap,
            vmin=vmin_db,
            vmax=vmax_db,
            extent=extent,
            aspect="auto",
        )
        if first_im is None:
            first_im = current_im
        axes[idx].set_title(f"{name} (dB)")
        axes[idx].set_xlabel("x (mm)")
        if idx == 0:
            axes[idx].set_ylabel("z (mm)")

    fig.suptitle("NSI reconstruction panels")
    fig.tight_layout(rect=[0, 0, 0.92, 1])
    cbar_ax = fig.add_axes([0.93, 0.1, 0.013, 0.78])
    fig.colorbar(first_im, cax=cbar_ax, label="dB")

    saved_path = None
    if save_figure:
        saved_path = output_dir / "nsi_db_panel_example0.png"
        fig.savefig(saved_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {saved_path}")

    if show_figure:
        plt.show()

    return fig, saved_path, images_db


# Auto-run for sandbox usage: execute everything and keep artifacts accessible.
NSI_RUN = run_nsi_core_validation(return_payload=True)
NSI_SUMMARY = NSI_RUN["summary"]
NSI_SAMPLE = NSI_RUN["sample"]
NSI_DAS = NSI_RUN["das_abs"]
NSI_SUM = NSI_RUN["img_sum"]
NSI_DIFF = NSI_RUN["img_diff"]
NSI_IMAGE = NSI_RUN["nsi_img"]
NSI_IMAGE_NORM = NSI_RUN["nsi_img_norm"]

NSI_DB_FIG, NSI_DB_FIG_PATH, NSI_DB_IMAGES = plot_nsi_images_db(NSI_RUN)

print("\nNSI sandbox run artifacts are ready:")
print("  NSI_SUMMARY, NSI_SAMPLE, NSI_DAS, NSI_SUM, NSI_DIFF, NSI_IMAGE, NSI_IMAGE_NORM")
print("  NSI_DB_FIG, NSI_DB_FIG_PATH, NSI_DB_IMAGES")
