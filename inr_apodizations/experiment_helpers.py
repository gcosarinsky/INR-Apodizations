"""Helpers for INR DAS experiment workflows.

This module centralizes utilities used by training, tuning, and
baseline-evaluation scripts, so all consumers import from the package namespace.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import statistics
import subprocess
from datetime import datetime
from typing import Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.dataset import (
    build_tf_dataset_by_examples,
    build_tf_dataset_by_indices,
    generate_unit_gaussian_mask,
    split_train_validation_examples,
    split_train_validation_indices,
)
from inr_apodizations.evaluation import (
    compute_scatterer_metrics,
    compute_validation_and_reference_metrics,
)
from inr_apodizations.evaluation.scatterers import (
    plot_scatterer_evaluation,
    plot_scatterer_snr_ratio,
    select_reflector_scatterer,
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


def get_target_regeneration_override(config: dict) -> tuple[float | None, float | None, float | None]:
    """Extract optional target regeneration overrides from experiment configuration.

    Args:
        config: Experiment configuration mapping.

    Returns:
        Tuple ``(sigma_x, sigma_z, alpha)`` or ``(None, None, None)`` when disabled.

    Raises:
        ValueError: If the override section is enabled but incomplete.
    """
    override_cfg = dict(config.get("target_regeneration", {}))
    if not bool(override_cfg.get("enabled", False)):
        return None, None, None

    sigma_x = override_cfg.get("sigma_x")
    sigma_z = override_cfg.get("sigma_z")
    if sigma_x is None or sigma_z is None:
        raise ValueError(
            "target_regeneration.sigma_x and target_regeneration.sigma_z must be set "
            "when regeneration is enabled"
        )

    alpha = override_cfg.get("alpha")
    if alpha is not None:
        alpha_val = float(alpha)
        if not (0.0 <= alpha_val < 1.0):
            raise ValueError("target_regeneration.alpha must be in [0.0, 1.0)")
        alpha = alpha_val

    return float(sigma_x), float(sigma_z), (float(alpha) if alpha is not None else None)


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


from inr_apodizations.apodizations import compute_das_baseline_numpy  # noqa: E402


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


def load_experiment_config(config_path: str) -> dict:
    """Load the experiment YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_run_directories(processed_root: str, output_root: str) -> tuple[str, str, str]:
    """Create timestamped run directories for processed and experiment outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_dir = os.path.join(processed_root, timestamp)
    output_dir = os.path.join(output_root, timestamp)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return timestamp, processed_dir, output_dir


def save_artifacts(output_dir: str, model: tf.keras.Model, history: dict, config: dict) -> None:
    """Save model and minimal artifacts into ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "model.keras")
    model.save(model_path)

    def _to_serializable(obj):
        """Recursively convert numpy/TF types into Python built-ins for JSON."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, tf.Tensor):
            try:
                val = obj.numpy()
            except Exception:
                return str(obj)
            return _to_serializable(val)
        if isinstance(obj, np.ndarray):
            return _to_serializable(obj.tolist())
        if isinstance(obj, dict):
            return {str(k): _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
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
    """Build per-pixel loss weights directly from gaussian masks.

    The gaussian mask is expected to be in [0, 1], so the resulting weights are
    in ``[1, 1 + pixel_weight_lambda]``. No normalization or clipping is applied.

    Args:
        gaussian_masks_array: Gaussian mask tensor with shape ``(N, Z, X)``,
            values in ``[0, 1]``.
        weighting_cfg: Configuration mapping under ``training.mask_weighting``.
            Expected key: ``pixel_weight_lambda`` (float, default 3.0).
            Legacy key ``lambda`` is also accepted as fallback.

    Returns:
        Optional weight tensor with shape ``(N, Z, X)`` and dtype float32.
        Returns ``None`` when weighting is disabled.

    Raises:
        ValueError: If the resolved pixel-weight lambda is negative.
    """
    enabled = bool(weighting_cfg.get("enabled", False))
    if not enabled:
        return None

    weight_lambda = float(
        weighting_cfg.get(
            "pixel_weight_lambda",
            weighting_cfg.get("lambda", 3.0),
        )
    )
    if weight_lambda < 0.0:
        raise ValueError(
            "training.mask_weighting.pixel_weight_lambda must be >= 0 "
            "(legacy key training.mask_weighting.lambda is also accepted)"
        )

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


def save_tuner_architecture_scores(
    tuner,
    candidate_architectures,
    output_dir: str,
    objective_name: str = "val_mae",
    sort_by_best: bool = True,
) -> None:
    """Collect Keras Tuner trials and save per-architecture score summaries.

    Args:
        tuner: Keras Tuner instance after `.search()` has completed.
        candidate_architectures: List of architectures (indexed by `architecture_index`).
        output_dir: Folder where the artifacts will be saved (created if needed).
        objective_name: Metric name used as the tuner objective (default: "val_mae").
        sort_by_best: Whether to order the saved per-architecture summaries by
            `best_score` ascending (best/lowest first). Defaults to ``True``.

    Behavior:
        - Writes `tuner_trials.json` with one entry per trial (hyperparameters + score).
        - Writes `per_architecture_scores.json` mapping architecture index -> summary.
        - Writes `per_architecture_scores.csv` for quick inspection.
    """
    os.makedirs(output_dir, exist_ok=True)

    trials = getattr(getattr(tuner, "oracle", {}), "trials", {})
    trials_info = []

    arch_lookup = {}
    try:
        for i, arch in enumerate(candidate_architectures):
            arch_lookup[str(i)] = arch
    except Exception:
        arch_lookup = {}

    for trial_id, trial in (trials.items() if isinstance(trials, dict) else []):
        hp_obj = getattr(trial, "hyperparameters", None)
        hp_dict = {}
        if hp_obj is not None:
            if hasattr(hp_obj, "values"):
                try:
                    hp_dict = dict(hp_obj.values)
                except Exception:
                    hp_dict = {}
            else:
                try:
                    hp_dict = dict(getattr(hp_obj, "get_config", lambda: {})() or {})
                except Exception:
                    hp_dict = {}

        score = None
        try:
            if hasattr(trial, "score") and trial.score is not None:
                score = float(trial.score)
        except Exception:
            score = None

        if score is None:
            metrics_obj = getattr(trial, "metrics", None)
            if metrics_obj is not None:
                try:
                    if hasattr(metrics_obj, "get_last_value"):
                        val = metrics_obj.get_last_value(objective_name)
                        if val is not None:
                            score = float(val)
                except Exception:
                    try:
                        if hasattr(metrics_obj, "get_best_value"):
                            val = metrics_obj.get_best_value(objective_name)
                            if val is not None:
                                score = float(val)
                    except Exception:
                        score = None

        arch_index = None
        if isinstance(hp_dict, dict) and "architecture_index" in hp_dict:
            try:
                arch_index = int(hp_dict.get("architecture_index"))
            except Exception:
                arch_index = None

        hidden_units = None
        try:
            if arch_index is not None:
                hidden_units = arch_lookup.get(str(arch_index))
        except Exception:
            hidden_units = None

        trials_info.append({
            "trial_id": str(trial_id),
            "architecture_index": arch_index,
            "hidden_units": hidden_units,
            "hyperparameters": {
                k: (
                    int(v)
                    if isinstance(v, (np.integer,))
                    else (float(v) if isinstance(v, (np.floating,)) else v)
                )
                for k, v in hp_dict.items()
            },
            "score": (float(score) if score is not None else None),
        })

    per_arch = {}
    for trial_info in trials_info:
        ai = trial_info["architecture_index"]
        ai_key = str(ai) if ai is not None else "None"
        per_arch.setdefault(ai_key, {"n_trials": 0, "all_scores": [], "hidden_units": None})
        per_arch[ai_key]["n_trials"] += 1
        per_arch[ai_key]["all_scores"].append(trial_info["score"])
        if per_arch[ai_key]["hidden_units"] is None and trial_info.get("hidden_units") is not None:
            per_arch[ai_key]["hidden_units"] = trial_info.get("hidden_units")

    for ai_key, info in per_arch.items():
        scores = [s for s in info["all_scores"] if s is not None]
        if scores:
            info["best_score"] = float(min(scores))
            try:
                info["median_score"] = float(statistics.median(scores))
            except Exception:
                info["median_score"] = None
        else:
            info["best_score"] = None
            info["median_score"] = None

    if sort_by_best:

        def _best_score_key(item):
            info = item[1]
            score = info.get("best_score")
            return float(score) if score is not None else float("inf")

        ordered_items = sorted(per_arch.items(), key=_best_score_key)
    else:
        ordered_items = sorted(per_arch.items(), key=lambda x: (x[0] if x[0] != "None" else "zz"))

    ordered_per_arch = {k: v for k, v in ordered_items}

    trials_path = os.path.join(output_dir, "tuner_trials.json")
    with open(trials_path, "w", encoding="utf-8") as f:
        json.dump(trials_info, f, indent=2, default=str)

    arch_path = os.path.join(output_dir, "per_architecture_scores.json")
    with open(arch_path, "w", encoding="utf-8") as f:
        json.dump(ordered_per_arch, f, indent=2, default=str)

    csv_path = os.path.join(output_dir, "per_architecture_scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "architecture_index",
                "n_trials",
                "best_score",
                "median_score",
                "hidden_units",
                "all_scores",
            ]
        )
        for ai_key, info in ordered_items:
            writer.writerow(
                [
                    ai_key,
                    info.get("n_trials", 0),
                    info.get("best_score"),
                    info.get("median_score"),
                    json.dumps(info.get("hidden_units", None)),
                    json.dumps(info.get("all_scores", [])),
                ]
            )


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
