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
from typing import Tuple

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


def build_coordinate_manager(dataset_folder: str) -> tuple[KernelParameters2D, CoordinateManager]:
    """Build ``KernelParameters2D`` and ``CoordinateManager`` from a dataset folder."""
    cfg = load_saved_beamforming_config(dataset_folder)
    kp = KernelParameters2D(cfg)
    cm = CoordinateManager(kp)
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

    with open(os.path.join(output_dir, "config.yml"), "w", encoding="utf-8") as file:
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


def compute_scatterer_metrics(
    image: np.ndarray,
    scatterers: np.ndarray,
    cm: CoordinateManager,
    radius_mm: float = 1.0,
    return_masks: bool = False,
    return_background_hist: bool = False,
    hist_bins: int = 50,
) -> dict:
    """Compute peak amplitude per scatterer and background RMS from an abs-DAS image.

    A quasi-circular disk (built once via ``np.ogrid``, morphological-style) is
    stamped around each scatterer's nearest pixel.  The union of all disks defines
    the scatterer region; its complement is used as background.

    Args:
        image: Real-valued abs-DAS image with shape ``(nz, nx)``.
        scatterers: Array with shape ``(N, 2)`` where columns are ``[x, z]`` in mm.
        cm: CoordinateManager used to retrieve the image pixel grid in mm.
        radius_mm: Radius of the circular mask in mm.
        return_masks: If ``True``, include ``individual_masks`` (N, nz, nx) and
            ``background_mask`` (nz, nx) in the output dict.
        return_background_hist: If ``True``, include amplitude histogram of
            background pixels in the output dict.
        hist_bins: Number of bins for the background histogram.

    Returns:
        dict with keys:

        - ``peak_amplitudes``: ``(N,)`` float array, max amplitude within each disk.
        - ``background_rms``: scalar float, RMS of image pixels outside all disks.
        - ``background_hist_counts`` / ``background_hist_edges``: only when
          ``return_background_hist=True``.
        - ``individual_masks``: only when ``return_masks=True``.
        - ``background_mask``: only when ``return_masks=True``.

    Raises:
        ValueError: If ``image`` is not 2-D, ``scatterers`` is not ``(N, 2)``, or
            ``radius_mm`` is non-positive.
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D (nz, nx)")
    scatterers = np.asarray(scatterers, dtype=np.float64)
    if scatterers.ndim != 2 or scatterers.shape[1] != 2:
        raise ValueError("scatterers must be (N, 2) with columns [x, z] in mm")
    if radius_mm <= 0.0:
        raise ValueError("radius_mm must be > 0")

    nz, nx = image.shape
    coords = cm.get_coordinates_1d(scaled=False)
    x_coords = np.asarray(coords["x"])   # (nx,)
    z_coords = np.asarray(coords["z"])   # (nz,)

    dx = float(np.abs(x_coords[1] - x_coords[0])) if nx > 1 else 1.0
    dz = float(np.abs(z_coords[1] - z_coords[0])) if nz > 1 else 1.0

    # Build the disk kernel once (morphological disk using ogrid).
    # Pixel spacing may differ along x and z; use an elliptical footprint so
    # the physical radius is honoured on both axes.
    rx = max(1, round(radius_mm / dx))
    rz = max(1, round(radius_mm / dz))
    r = max(rx, rz)
    gy, gx = np.ogrid[-r: r + 1, -r: r + 1]
    # Ellipse equation: (gx/rx)^2 + (gy/rz)^2 <= 1
    disk = (gx / rx) ** 2 + (gy / rz) ** 2 <= 1.0  # shape (2r+1, 2r+1)

    n_scatterers = scatterers.shape[0]
    union_mask = np.zeros((nz, nx), dtype=bool)

    if return_masks:
        individual_masks = np.zeros((n_scatterers, nz, nx), dtype=bool)

    peak_amplitudes = np.empty(n_scatterers, dtype=np.float64)

    for i, (x0, z0) in enumerate(scatterers):
        # Nearest pixel indices
        ix = int(np.argmin(np.abs(x_coords - x0)))
        iz = int(np.argmin(np.abs(z_coords - z0)))

        # Clipped image region covered by the disk
        iz0 = iz - r
        iz1 = iz + r + 1
        ix0 = ix - r
        ix1 = ix + r + 1

        # Corresponding slice into the disk kernel (handles border cases)
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

        if return_masks:
            individual_masks[i] = local_mask

    background_mask = ~union_mask
    background_pixels = image[background_mask]
    background_rms = float(np.sqrt(np.mean(background_pixels ** 2))) if background_pixels.size > 0 else 0.0

    result = {
        "peak_amplitudes": peak_amplitudes,
        "background_rms": background_rms,
    }

    if return_background_hist:
        counts, bin_edges = np.histogram(background_pixels, bins=hist_bins)
        result["background_hist_counts"] = counts
        result["background_hist_edges"] = bin_edges

    if return_masks:
        result["individual_masks"] = individual_masks
        result["background_mask"] = background_mask

    return result
