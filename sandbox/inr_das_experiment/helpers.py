"""Helpers for the sandbox INR DAS experiment.

This module provides lightweight utilities to load the delayed-samples
dataset, create the coordinate/features context required by the INR,
split examples into train/validation sets, build tf.data pipelines, and
persist minimal artifacts.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
from datetime import datetime
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import yaml

from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.dataset import (
    build_tf_dataset_by_examples,
    build_tf_dataset_by_indices,
    generate_unit_gaussian_mask,
    split_train_validation_examples,
    split_train_validation_indices,
)
from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.plots import (
    plot_apodization_before_after,
    plot_das_comparison_db,
    plot_training_curves,
    to_db,
)


BYTES_PER_GB = float(1024**3)


def bytes_to_gb(n_bytes: int) -> float:
    """Convert bytes to gibibytes (GiB)."""
    return float(n_bytes) / BYTES_PER_GB


def gpu_mem() -> list[dict[str, int]] | str:
    """Query GPU memory info (total and free) via ``nvidia-smi``.

    Returns:
        List of dicts with ``total`` and ``free`` keys in MiB, one per GPU.
        Returns the string ``"nvidia-smi no disponible"`` if the query fails.
    """
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits".split()
        ).decode().strip()
        return [dict(zip(["total", "free"], map(int, line.split(",")))) for line in out.split("\n")]
    except Exception:
        return "nvidia-smi no disponible"


def get_tf_available_vram_info() -> tuple[int | None, str]:
    """Return currently free VRAM bytes on GPU:0 and source label.

    Uses ``gpu_mem()`` (nvidia-smi) to get the free memory at call time.

    Returns:
        Tuple ``(free_vram_bytes, source)``. If unavailable, returns
        ``(None, "unavailable")``.
    """
    result = gpu_mem()
    if isinstance(result, list) and result:
        free_mib = result[0]["free"]
        return int(free_mib * 1024 * 1024), "nvidia-smi"
    return None, "unavailable"


def load_saved_scatterers(folder: str) -> np.ndarray:
    """Load scatterer coordinates saved next to a delayed-samples dataset.

    Args:
        folder: Dataset folder containing ``scatterers.npy``.

    Returns:
        NumPy object array with one scatterer array per example.

    Raises:
        FileNotFoundError: If the scatterers file is missing.
    """
    scatterers_path = os.path.join(folder, "scatterers.npy")
    if not os.path.exists(scatterers_path):
        raise FileNotFoundError(
            "Scatterers file not found in %s. "
            "This dataset cannot regenerate targets with new sigma values." % folder
        )
    return np.load(scatterers_path, allow_pickle=True)


def _build_target_grids(folder: str) -> tuple[np.ndarray, np.ndarray]:
    """Build the target meshgrids used during dataset generation.

    Args:
        folder: Dataset folder containing ``cfg_delayed_samples.npy``.

    Returns:
        Tuple ``(x_grid, z_grid)`` with shape ``(nz, nx)``.
    """
    cfg = load_saved_beamforming_config(folder)
    kp = KernelParameters2D(cfg)
    x = np.linspace(kp.roi_effective[0], kp.roi_effective[1], kp.nx)
    z = np.linspace(kp.roi_effective[2], kp.roi_effective[3], kp.nz)
    return np.meshgrid(x, z)


def _regenerate_targets_and_masks(
    delayed: np.ndarray,
    scatterers: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    sigma_x: float,
    sigma_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute targets and gaussian masks from delayed samples and scatterers.

    Args:
        delayed: Delayed samples array with shape ``(N, E, Z, X)``.
        scatterers: Scatterer coordinates per example.
        x_grid: Target x meshgrid.
        z_grid: Target z meshgrid.
        sigma_x: Lateral gaussian sigma in mm.
        sigma_z: Axial gaussian sigma in mm.

    Returns:
        Tuple ``(targets, gaussian_masks)`` with shape ``(N, Z, X)``.

    Raises:
        ValueError: If the scatterer count does not match the dataset size.
    """
    if len(scatterers) != delayed.shape[0]:
        raise ValueError(
            "scatterers.npy example count does not match delayed_samples_dataset.npy"
        )

    targets = np.zeros((delayed.shape[0], delayed.shape[2], delayed.shape[3]), dtype=np.float32)
    gaussian_masks = np.zeros_like(targets)

    for idx in range(delayed.shape[0]):
        das_uniform = delayed[idx].sum(axis=0)
        scatterers_mm = 1000.0 * np.asarray(scatterers[idx], dtype=np.float32)
        gaussian_mask = generate_unit_gaussian_mask(
            scatterers_mm,
            x_grid,
            z_grid,
            sigma_x=sigma_x,
            sigma_z=sigma_z,
        )
        gaussian_masks[idx] = gaussian_mask.astype(np.float32, copy=False)
        targets[idx] = np.abs(das_uniform).astype(np.float32, copy=False) * gaussian_mask

    return targets, gaussian_masks


def load_delayed_samples_dataset(
    folder: str,
    sigma_x: float | None = None,
    sigma_z: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load delayed samples dataset, targets, gaussian masks and metadata from a dataset folder.

    Expects files inside `folder`: `delayed_samples_dataset.npy`,
    `targets_dataset.npy`, `gaussian_masks_dataset.npy` and
    `delayed_samples_info.yaml` (optional).

    Returns:
        delayed: np.ndarray, shape (N, E, Z, X), dtype complex64
        targets: np.ndarray, shape (N, Z, X), dtype float32
        gaussian_masks: np.ndarray, shape (N, Z, X), dtype float32, values in [0, 1]
        info: dict with parsed YAML metadata (empty dict if not present)

    Args:
        folder: Dataset folder containing delayed samples artifacts.
        sigma_x: Optional lateral sigma override in mm. Must be provided with ``sigma_z``.
        sigma_z: Optional axial sigma override in mm. Must be provided with ``sigma_x``.

    Raises:
        FileNotFoundError: If delayed samples, targets or gaussian masks files are missing.
        ValueError: If only one sigma override is provided or if sigma values are non-positive.
    """
    delayed_path = os.path.join(folder, "delayed_samples_dataset.npy")
    targets_path = os.path.join(folder, "targets_dataset.npy")
    masks_path = os.path.join(folder, "gaussian_masks_dataset.npy")
    info_path = os.path.join(folder, "delayed_samples_info.yaml")

    regenerate = sigma_x is not None or sigma_z is not None
    if regenerate and (sigma_x is None or sigma_z is None):
        raise ValueError("sigma_x and sigma_z must be provided together")
    if regenerate and (float(sigma_x) <= 0.0 or float(sigma_z) <= 0.0):
        raise ValueError("sigma_x and sigma_z must be > 0")

    if not os.path.exists(delayed_path) or not os.path.exists(targets_path):
        raise FileNotFoundError("Delayed samples or targets .npy not found in %s" % folder)
    if not os.path.exists(masks_path):
        raise FileNotFoundError(
            "gaussian_masks_dataset.npy not found in %s. Regenerate the dataset." % folder
        )

    delayed = np.load(delayed_path, allow_pickle=False)
    targets = np.load(targets_path, allow_pickle=False)
    gaussian_masks = np.load(masks_path, allow_pickle=False)

    info = {}
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = yaml.safe_load(f) or {}

    if regenerate:
        scatterers = load_saved_scatterers(folder)
        x_grid, z_grid = _build_target_grids(folder)
        targets, gaussian_masks = _regenerate_targets_and_masks(
            delayed,
            scatterers,
            x_grid,
            z_grid,
            sigma_x=float(sigma_x),
            sigma_z=float(sigma_z),
        )
        info = copy.deepcopy(info)
        info["runtime_target_override"] = {
            "enabled": True,
            "sigma_x": float(sigma_x),
            "sigma_z": float(sigma_z),
        }

    return delayed, targets, gaussian_masks, info


def get_target_sigma_override(config: dict) -> tuple[float | None, float | None]:
    """Extract optional target sigma overrides from sandbox configuration.

    Args:
        config: Sandbox experiment configuration mapping.

    Returns:
        Tuple ``(sigma_x, sigma_z)`` or ``(None, None)`` when disabled.

    Raises:
        ValueError: If the override section is enabled but incomplete.
    """
    override_cfg = dict(config.get("target_regeneration", {}))
    if not bool(override_cfg.get("enabled", False)):
        return None, None

    sigma_x = override_cfg.get("sigma_x")
    sigma_z = override_cfg.get("sigma_z")
    if sigma_x is None or sigma_z is None:
        raise ValueError(
            "target_regeneration.sigma_x and target_regeneration.sigma_z must be set "
            "when regeneration is enabled"
        )

    return float(sigma_x), float(sigma_z)


def load_saved_beamforming_config(folder: str) -> dict:
    """Load the delayed-samples configuration saved next to the dataset.

    Args:
        folder: Dataset folder containing ``cfg_delayed_samples.npy``.

    Returns:
        Configuration dictionary used to generate the dataset.

    Raises:
        FileNotFoundError: If the config file is missing.
    """
    cfg_path = os.path.join(folder, "cfg_delayed_samples.npy")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    return np.load(cfg_path, allow_pickle=True).item()


def build_coordinate_manager(
    dataset_folder: str,
    physical_feature_set: str = "distance_depth_edge",
) -> tuple[KernelParameters2D, CoordinateManager]:
    """Build ``KernelParameters2D`` and ``CoordinateManager`` from a dataset folder.

    Args:
        dataset_folder: Dataset folder containing the saved beamforming config.
        physical_feature_set: Physical feature variant used by ``CoordinateManager``.

    Returns:
        Tuple ``(kp, cm)`` with initialized kernel parameters and coordinate manager.
    """
    cfg = load_saved_beamforming_config(dataset_folder)
    kp = KernelParameters2D(cfg)
    cm = CoordinateManager(kp, physical_feature_set=physical_feature_set)
    return kp, cm


def validate_dataset_shapes(
    delayed: np.ndarray, targets: np.ndarray, gaussian_masks: np.ndarray
) -> None:
    """
    Basic assertions ensuring the dataset contract we rely on.

    Args:
        delayed: Delayed samples array, expected shape (N, E, Z, X), complex64.
        targets: Target images array, expected shape (N, Z, X), float32/64.
        gaussian_masks: Gaussian mask array, expected shape (N, Z, X), float32/64.

    Raises:
        AssertionError: If shapes, dtypes or batch sizes are inconsistent.
    """
    assert delayed.ndim == 4, "delayed must be (N, E, Z, X)"
    assert targets.ndim == 3, "targets must be (N, Z, X)"
    assert gaussian_masks.ndim == 3, "gaussian_masks must be (N, Z, X)"
    assert delayed.shape[0] == targets.shape[0], "N mismatch between delayed and targets"
    assert delayed.shape[0] == gaussian_masks.shape[0], "N mismatch between delayed and gaussian_masks"
    assert targets.shape == gaussian_masks.shape, "Shape mismatch between targets and gaussian_masks"
    assert np.iscomplexobj(delayed), "delayed_samples must be complex-valued"
    assert targets.dtype == np.float32 or targets.dtype == np.float64, "targets must be float"
    assert gaussian_masks.dtype == np.float32 or gaussian_masks.dtype == np.float64, "gaussian_masks must be float"


def build_mlp_inr(input_dim: int = 3,
                  hidden_units: int = 8,
                  n_hidden: int = 3,
                  activation: str = "relu",
                  output_activation: str = "sigmoid") -> tf.keras.Model:
    """
    Build a small MLP that maps `input_dim` features -> scalar weight [0,1].

    Returns a compiled but un-trained `tf.keras.Model` (caller compiles with chosen optimizer/loss).
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="coords")
    x = inputs
    for i in range(n_hidden):
        x = tf.keras.layers.Dense(hidden_units, activation=activation,
                                  name=f"dense_{i+1}")(x)
    outputs = tf.keras.layers.Dense(1, activation=output_activation, name="weight_out")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="inr_mlp")
    return model


def load_experiment_config(config_path: str) -> dict:
    """Load the sandbox experiment YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_run_directories(processed_root: str, sandbox_root: str) -> tuple[str, str, str]:
    """Create timestamped run directories for processed and sandbox outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_dir = os.path.join(processed_root, timestamp)
    sandbox_dir = os.path.join(sandbox_root, timestamp)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(sandbox_dir, exist_ok=True)
    return timestamp, processed_dir, sandbox_dir


def save_artifacts(output_dir: str, model: tf.keras.Model, history: dict, config: dict) -> None:
    """Save model and minimal artifacts into ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    # Save the Keras SavedModel folder (default) and an HDF5 copy (.h5)
    model_path = os.path.join(output_dir, "model.keras")
    model.save(model_path)
    # SavedModel folder is created above (model.keras). We intentionally
    # avoid exporting HDF5 (.h5) here to prevent failures with custom
    # objects; prefer SavedModel or the single-file .keras format.

    def _to_serializable(obj):
        """Recursively convert numpy/TF types into Python built-ins for JSON."""
        # Scalars
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        # Tensors
        if isinstance(obj, tf.Tensor):
            try:
                val = obj.numpy()
            except Exception:
                return str(obj)
            return _to_serializable(val)
        # NumPy arrays
        if isinstance(obj, np.ndarray):
            return _to_serializable(obj.tolist())
        # Dicts, lists, tuples
        if isinstance(obj, dict):
            return {str(k): _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        # Fallback for built-ins
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

    serializable_history = _to_serializable(history)

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as file:
        json.dump(serializable_history, file, indent=2)

    with open(os.path.join(output_dir, "train_config_info.yml"), "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def build_gaussian_loss_weights(
    gaussian_masks_array: np.ndarray, weighting_cfg: dict
) -> np.ndarray | None:
    """Build per-pixel loss weights directly from gaussian masks using ``1 + lambda * mask``.

    The gaussian mask is expected to be in [0, 1], so the resulting weights are
    in [1, 1 + lambda]. No normalization or clipping is applied.

    Args:
        gaussian_masks_array: Gaussian mask tensor with shape ``(N, Z, X)``,
            values in ``[0, 1]``.
        weighting_cfg: Configuration mapping under ``training.mask_weighting``.
            Expected key: ``lambda`` (float, default 3.0).

    Returns:
        Optional weight tensor with shape ``(N, Z, X)`` and dtype float32.
        Returns ``None`` when weighting is disabled.

    Raises:
        ValueError: If ``lambda`` is negative.
    """
    enabled = bool(weighting_cfg.get("enabled", False))
    if not enabled:
        return None

    weight_lambda = float(weighting_cfg.get("lambda", 3.0))
    if weight_lambda < 0.0:
        raise ValueError("training.mask_weighting.lambda must be >= 0")

    masks_float = gaussian_masks_array.astype(np.float32, copy=False)
    weights = 1.0 + weight_lambda * masks_float
    return weights.astype(np.float32, copy=False)


def save_debug_arrays(output_dir: str, arrays: dict[str, np.ndarray]) -> None:
    """Persist selected NumPy arrays for quick inspection."""
    os.makedirs(output_dir, exist_ok=True)
    for name, array in arrays.items():
        np.save(os.path.join(output_dir, f"{name}.npy"), array)


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

    im0 = axes[0].imshow(
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


def _normalize_scatterer_batch(
    scatterers: np.ndarray | Sequence[np.ndarray],
    batch_size: int,
) -> tuple[list[np.ndarray], bool]:
    """Normalize scatterer coordinates into a batch-aligned list.

    Args:
        scatterers: Either a single ``(N, 2)`` array, a stacked ``(B, N, 2)`` array,
            or a sequence/object-array with one ``(Ni, 2)`` array per example.
        batch_size: Number of images in the batch.

    Returns:
        Tuple ``(scatterer_batch, shared_across_batch)`` where ``scatterer_batch`` is a
        list of length ``batch_size`` and each item has shape ``(Ni, 2)`` in mm.

    Raises:
        ValueError: If scatterers cannot be matched to the requested batch size.
    """
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
    """Build reusable geometric context for scatterer masks.

    Args:
        image_shape: Image shape ``(nz, nx)``.
        cm: Coordinate manager used to retrieve x/z coordinates.
        radius_mm: Disk radius in mm.

    Returns:
        Dictionary with coordinates and precomputed elliptical disk kernel.

    Raises:
        ValueError: If image shape does not match coordinate-grid dimensions or radius is invalid.
    """
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
    """Compute scatterer metrics for a single image using precomputed geometry."""
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
    """Aggregate per-example scatterer metrics into a batch-level summary."""
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
    """Return the plot-ready metric section for single-example or batch outputs."""
    return metrics.get("aggregated", metrics)


def _validate_aligned_aggregated_points(ref_metrics: dict, cmp_metrics: dict) -> None:
    """Validate that aggregated point order matches across compared methods."""
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
    """Compute scatterer metrics for a single image or a batch of abs-DAS images.

    A quasi-circular disk (built once via ``np.ogrid``, morphological-style) is
    stamped around each scatterer's nearest pixel.  The union of all disks defines
    the scatterer region; its complement is used as background. Batch inputs reuse
    the same geometry and aggregate points across all examples.

    Args:
        image: Real-valued abs-DAS image with shape ``(nz, nx)`` or batch with shape
            ``(B, nz, nx)``.
        scatterers: Scatterer coordinates in mm. Supports a single ``(N, 2)`` array,
            a stacked batch ``(B, N, 2)``, or a sequence/object-array with one
            ``(Ni, 2)`` array per example.
        cm: CoordinateManager used to retrieve the image pixel grid in mm.
        radius_mm: Radius of the circular mask in mm.
        return_masks: If ``True``, include ``individual_masks`` (N, nz, nx) and
            ``background_mask`` (nz, nx) in the output dict for each evaluated example.
        return_background_hist: If ``True``, include amplitude histogram of
            background pixels in the output dict.
        hist_bins: Number of bins for the background histogram.

    Returns:
        For a single image, returns the historical flat dict with keys:

        - ``peak_amplitudes``: ``(N,)`` float array, max amplitude within each disk.
        - ``background_rms``: scalar float, RMS of image pixels outside all disks.
        - ``background_hist_counts`` / ``background_hist_edges``: only when
          ``return_background_hist=True``.
        - ``individual_masks``: only when ``return_masks=True``.
        - ``background_mask``: only when ``return_masks=True``.

        For a batch, returns:

        - ``per_example``: list of single-example metric dicts in batch order.
        - ``aggregated``: batch-level summary with concatenated peaks, global background
          RMS, optional aggregated histogram, point-to-example indices, and per-point
          background RMS / image maxima for batch plotting.
        - ``batch_size``: number of evaluated examples.

    Raises:
        ValueError: If image rank is not 2 or 3, if scatterers cannot be aligned to the
            batch, or if ``radius_mm`` is non-positive.
    """
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
    """Compute scatterer metrics for multiple methods and save single/batch figures.

    Produces:
    - Mask overlay figure for the first reference method and first example (requires ``extent``).
    - Background noise histogram with all methods overlaid.
    - Per-pair SNR scatter plot (peak / bg_rms) across one image or a whole batch.
    - Per-pair normalized scatter plot (peak / image_max) across one image or a whole batch.

    Args:
        images_abs: Dict mapping method names to absolute DAS images with shape ``(nz, nx)``
            or batches ``(B, nz, nx)``.
        scatterers_xy: Scatterer coordinates in mm. Supports ``(N, 2)``, ``(B, N, 2)``,
            or a sequence/object-array with one ``(Ni, 2)`` array per example.
        cm: CoordinateManager for pixel-grid lookups.
        output_dir: Directory where figures are saved.
        radius_mm: Disk radius around each scatterer for peak extraction.
        hist_bins: Number of background histogram bins.
        compare_pairs: List of ``(reference_name, compared_name)`` tuples.
            Each pair produces one column in the scatter figures.
        extent: Matplotlib imshow extent ``(xmin, xmax, zmax, zmin)`` in mm.
            Required for the mask overlay figure.
        vmin_db: Minimum dB value for DAS image display in overlay figure.
        vmax_db: Maximum dB value for DAS image display.
        cmap: Colormap for DAS images.
        example_suffix: Suffix appended to all output file names.

    Returns:
        Dict mapping method name to its metric dict (from ``compute_scatterer_metrics``).

    Raises:
        ValueError: If compare pairs reference unknown methods, if methods mix single and
            batch inputs, or if batch sizes are inconsistent.
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

    # Compute metrics for all methods
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

    # --- Mask overlay (first reference, requires extent) ---
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

    # --- Background noise histogram (all methods) ---
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

    # --- Scatter plots per pair ---
    if compare_pairs:
        n_cols = len(compare_pairs)

        # SNR scatter (peak / bg_rms)
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
            ax.scatter(ref_snr, cmp_snr, s=30, alpha=0.7)
            ax.plot([0, ax_max], [0, ax_max], color="red", linewidth=1, linestyle="--", label="y = x")
            ax.scatter(
                [1.0], [1.0], s=110, marker="D", facecolors="none",
                edgecolors="black", linewidths=1.5, label="bg_rms reference", zorder=5,
            )
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

        # Normalized scatter (peak / image_max)
        fig_norm, axes_norm = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
        for col_idx, (ref_name, cmp_name) in enumerate(compare_pairs):
            ax = axes_norm[0, col_idx]
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

            if "point_image_maxima" in ref_view:
                ref_max = np.maximum(np.asarray(ref_view["point_image_maxima"], dtype=np.float64), 1e-12)
                cmp_max = np.maximum(np.asarray(cmp_view["point_image_maxima"], dtype=np.float64), 1e-12)
                ref_bg_rms = np.asarray(ref_view["point_background_rms"], dtype=np.float64)
                cmp_bg_rms = np.asarray(cmp_view["point_background_rms"], dtype=np.float64)
            else:
                ref_max = np.full(ref_peaks.shape, max(float(np.max(np.asarray(images_abs[ref_name]))), 1e-12))
                cmp_max = np.full(cmp_peaks.shape, max(float(np.max(np.asarray(images_abs[cmp_name]))), 1e-12))
                ref_bg_rms = np.full(ref_peaks.shape, float(ref_view["background_rms"]))
                cmp_bg_rms = np.full(cmp_peaks.shape, float(cmp_view["background_rms"]))

            ax.scatter(ref_peaks / ref_max, cmp_peaks / cmp_max, s=30, alpha=0.7)
            ax.plot([0, 1], [0, 1], color="red", linewidth=1, linestyle="--", label="y = x")
            ax.scatter(
                [float(np.mean(ref_bg_rms / ref_max))], [float(np.mean(cmp_bg_rms / cmp_max))],
                s=110, marker="D", facecolors="none", edgecolors="black",
                linewidths=1.5, label="bg_rms", zorder=5,
            )
            ax.set_xlabel(f"{ref_name} peak / image max")
            ax.set_ylabel(f"{cmp_name} peak / image max")
            title_suffix = (
                f"{len(ref_peaks)} points, {batch_size} examples"
                if uses_batch
                else f"{len(ref_peaks)} scatterers"
            )
            ax.set_title(f"{cmp_name} vs {ref_name} normalized ({title_suffix})")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig_norm.suptitle(f"Normalized peak amplitude comparison{sfx}")
        fig_norm.tight_layout()
        fig_norm.savefig(
            os.path.join(output_dir, f"scatt_normalized_scatter{sfx}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig_norm)

    return all_metrics
