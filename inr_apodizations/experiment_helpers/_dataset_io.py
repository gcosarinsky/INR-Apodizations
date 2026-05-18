"""Dataset I/O, validation, target regeneration, and coordinate-manager construction."""

from __future__ import annotations

import copy
import os
from typing import Tuple

import numpy as np
import yaml

from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.dataset import generate_unit_gaussian_mask
from inr_apodizations.kernels import KernelParameters2D


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
    alpha: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute targets and gaussian masks from delayed samples and scatterers.

    Args:
        delayed: Delayed samples array with shape ``(N, E, Z, X)``.
        scatterers: Scatterer coordinates per example.
        x_grid: Target x meshgrid.
        z_grid: Target z meshgrid.
        sigma_x: Lateral gaussian sigma in mm.
        sigma_z: Axial gaussian sigma in mm.
        alpha: Optional blending parameter for relaxed target formula.

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
        abs_das = np.abs(das_uniform).astype(np.float32, copy=False)
        if alpha is None:
            targets[idx] = abs_das * gaussian_mask
        else:
            a = np.float32(alpha)
            relaxed_mask = a + (np.float32(1.0) - a) * gaussian_mask
            targets[idx] = abs_das * relaxed_mask

    return targets, gaussian_masks


def load_delayed_samples_dataset(
    folder: str,
    sigma_x: float | None = None,
    sigma_z: float | None = None,
    alpha_override: float | None = None,
    load_noise: bool = True,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """
    Load delayed samples dataset, targets, gaussian masks and metadata from a dataset folder.

    Expects files inside `folder`: `delayed_samples_dataset.npy` (or
    `delayed_samples_signal.npy`), `targets_dataset.npy`,
    `targets_dataset.npy`, `gaussian_masks_dataset.npy` and
    `delayed_samples_info.yaml` (optional).

    Returns:
        delayed: np.ndarray, shape (N, E, Z, X), dtype complex64
        noise: np.ndarray or None, shape (N, E, Z, X), dtype complex64 when precomputed noise is available
        targets: np.ndarray, shape (N, Z, X), dtype float32
        gaussian_masks: np.ndarray, shape (N, Z, X), dtype float32, values in [0, 1]
        info: dict with parsed YAML metadata (empty dict if not present)

    Args:
        folder: Dataset folder containing delayed samples artifacts.
        sigma_x: Optional lateral sigma override in mm. Must be provided with ``sigma_z``.
        sigma_z: Optional axial sigma override in mm. Must be provided with ``sigma_x``.
        alpha_override: Optional blending parameter for regenerated targets. Must be in
            the interval [0.0, 1.0). When provided, targets will be regenerated using
            the same sigma overrides and the relaxed target formula
            ``alpha + (1-alpha) * gauss``. ``alpha_override=0`` reproduces the
            default behavior.
        load_noise: If ``True``, load or derive precomputed noise when available.
            If ``False``, skip noise loading and return ``None`` for ``noise``.

    Raises:
        FileNotFoundError: If delayed samples, targets or gaussian masks files are missing.
        ValueError: If only one sigma override is provided or if sigma values are non-positive.
    """
    targets_path = os.path.join(folder, "targets_dataset.npy")
    masks_path = os.path.join(folder, "gaussian_masks_dataset.npy")
    info_path = os.path.join(folder, "delayed_samples_info.yaml")

    regenerate = sigma_x is not None or sigma_z is not None or alpha_override is not None
    if regenerate and (sigma_x is None or sigma_z is None):
        raise ValueError("sigma_x and sigma_z must be provided together")
    if regenerate and (float(sigma_x) <= 0.0 or float(sigma_z) <= 0.0):
        raise ValueError("sigma_x and sigma_z must be > 0")

    if alpha_override is not None:
        alpha_val = float(alpha_override)
        if not (0.0 <= alpha_val < 1.0):
            raise ValueError("alpha_override must be in the interval [0.0, 1.0)")
        if not regenerate:
            raise ValueError("alpha_override requires sigma_x and sigma_z to be provided for regeneration")

    if not os.path.exists(targets_path):
        raise FileNotFoundError("targets_dataset.npy not found in %s" % folder)
    if not os.path.exists(masks_path):
        raise FileNotFoundError(
            "gaussian_masks_dataset.npy not found in %s. Regenerate the dataset." % folder
        )

    print("loading targets.npy and gaussian_masks.npy")
    targets = np.load(targets_path, allow_pickle=False)
    gaussian_masks = np.load(masks_path, allow_pickle=False)

    info = {}
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = yaml.safe_load(f) or {}

    signal_candidates = (
        os.path.join(folder, "delayed_samples_signal.npy"),
        os.path.join(folder, "delayed_samples_dataset.npy"),
        os.path.join(folder, "delayed_samples.npy"),
    )
    signal_path = None
    for p in signal_candidates:
        if os.path.exists(p):
            signal_path = p
            break

    combined_path = os.path.join(folder, "delayed_samples_combined.npy")
    noise_path = os.path.join(folder, "delayed_samples_noise.npy")

    noise = None
    if load_noise and os.path.exists(noise_path):
        print("Loading precomputed noise")
        noise = np.load(noise_path, allow_pickle=False)
        info = copy.deepcopy(info)
        info.setdefault("precomputed_noise_source", {})
        info["precomputed_noise_source"].update({"source": "noise_file"})

    print("Loading signal and/or combined delayed samples")
    if os.path.exists(combined_path):
        if signal_path is not None and load_noise:
            combined = np.load(combined_path, allow_pickle=False)
            signal = np.load(signal_path, allow_pickle=False)
            noise = combined.astype(np.complex64, copy=False) - signal.astype(np.complex64, copy=False)
            delayed = signal.astype(np.complex64, copy=False)
            info = copy.deepcopy(info)
            info.setdefault("precomputed_noise_source", {})
            info["precomputed_noise_source"].update({"source": "combined_minus_signal"})
        elif signal_path is not None:
            delayed = np.load(signal_path, allow_pickle=False).astype(np.complex64, copy=False)
        else:
            delayed = np.load(combined_path, allow_pickle=False).astype(np.complex64, copy=False)
            if load_noise:
                noise = None
                info = copy.deepcopy(info)
                info.setdefault("precomputed_noise_source", {})
                info["precomputed_noise_source"].update({"source": "combined_only"})
    else:
        if signal_path is not None:
            delayed = np.load(signal_path, allow_pickle=False).astype(np.complex64, copy=False)
            if noise is not None:
                info = copy.deepcopy(info)
                info.setdefault("precomputed_noise_source", {})
                info["precomputed_noise_source"].update({"source": "signal_and_noise_file"})
        else:
            fallback = os.path.join(folder, "delayed_samples_dataset.npy")
            if os.path.exists(fallback):
                delayed = np.load(fallback, allow_pickle=False).astype(np.complex64, copy=False)
            else:
                raise FileNotFoundError(
                    "No delayed-samples file found in %s. Searched for signal/combined/noise variants." % folder
                )

    if regenerate:
        print(
            "Regenerating targets and gaussian masks with sigma_x=%.3f mm, sigma_z=%.3f mm, alpha=%s"
            % (float(sigma_x), float(sigma_z), str(alpha_override))
        )
        scatterers = load_saved_scatterers(folder)
        x_grid, z_grid = _build_target_grids(folder)
        targets, gaussian_masks = _regenerate_targets_and_masks(
            delayed,
            scatterers,
            x_grid,
            z_grid,
            sigma_x=float(sigma_x),
            sigma_z=float(sigma_z),
            alpha=(float(alpha_override) if alpha_override is not None else None),
        )
        info = copy.deepcopy(info)
        info["runtime_target_override"] = {
            "enabled": True,
            "sigma_x": float(sigma_x),
            "sigma_z": float(sigma_z),
        }
        if alpha_override is not None:
            info["runtime_target_override"]["alpha"] = float(alpha_override)

    return delayed, noise, targets, gaussian_masks, info


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
    """Basic assertions ensuring the dataset contract we rely on.

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
