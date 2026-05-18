"""Training pipeline setup: dataset preparation, tf.data pipelines, output dirs, callbacks."""

from __future__ import annotations

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import tensorflow as tf

from inr_apodizations.dataset import build_tf_dataset_by_indices, split_train_validation_indices

from ._config import get_target_regeneration_override
from ._dataset_io import build_coordinate_manager, load_delayed_samples_dataset, validate_dataset_shapes
from ._hardware import gpu_mem


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


def prepare_training_dataset(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Load and prepare the delayed-samples dataset for train/validation workflows.

    This helper centralizes dataset path resolution, optional target regeneration,
    optional precomputed-noise loading for evaluation, split indices, pixel-loss
    weighting, and memory diagnostics.

    Args:
        cfg: Full experiment configuration mapping.

    Returns:
        Dictionary with arrays, metadata, coordinate objects, split indices,
        loss weights, and memory diagnostics used by training scripts.

    Raises:
        ValueError: If configured mask weighting is invalid.
    """
    from inr_apodizations import config as global_config

    dataset_folder = Path(str(cfg["io"]["dataset_folder"]))
    if not dataset_folder.is_absolute():
        dataset_folder = global_config.PROJ_ROOT / dataset_folder
    dataset_folder_str = str(dataset_folder)

    sigma_x_override, sigma_z_override, alpha_override = get_target_regeneration_override(dict(cfg))
    eval_noise_cfg = dict(cfg.get("eval_noise", {}))
    eval_noise_enabled = bool(eval_noise_cfg.get("enabled", False))

    delayed, noise, targets, gaussian_masks, info = load_delayed_samples_dataset(
        dataset_folder_str,
        sigma_x=sigma_x_override,
        sigma_z=sigma_z_override,
        alpha_override=alpha_override,
        load_noise=eval_noise_enabled,
    )
    validate_dataset_shapes(delayed, targets, gaussian_masks)

    physical_feature_set = str(
        cfg.get("model", {}).get("physical_feature_set", "distance_depth_edge")
    )
    kp, cm = build_coordinate_manager(
        dataset_folder_str,
        physical_feature_set=physical_feature_set,
    )

    delayed_dataset_bytes = int(delayed.nbytes)
    delayed_example_bytes = int(np.prod(delayed.shape[1:], dtype=np.int64) * delayed.dtype.itemsize)

    max_examples = cfg.get("training", {}).get("max_examples")
    if max_examples is not None:
        max_examples = int(max_examples)
        delayed = delayed[:max_examples]
        targets = targets[:max_examples]
        gaussian_masks = gaussian_masks[:max_examples]
        if noise is not None:
            noise = noise[:max_examples]

    mask_weighting_cfg = dict(cfg.get("training", {}).get("mask_weighting", {}))
    pixel_weight_lambda = float(
        mask_weighting_cfg.get(
            "pixel_weight_lambda",
            mask_weighting_cfg.get("lambda", 0.0),
        )
    )
    if pixel_weight_lambda < 0.0:
        raise ValueError(
            "training.mask_weighting.pixel_weight_lambda must be >= 0 "
            "(legacy key training.mask_weighting.lambda is also accepted)"
        )
    train_loss_weights = 1.0 + pixel_weight_lambda * gaussian_masks.astype(np.float32, copy=False)
    train_loss_weights = train_loss_weights.astype(np.float32, copy=False)

    train_idx, val_idx = split_train_validation_indices(
        n_examples=delayed.shape[0],
        train_fraction=float(cfg["training"]["train_fraction"]),
        seed=int(cfg["training"]["seed"]),
    )

    configured_batch_size = int(cfg["training"]["batch_size"])
    effective_batch_examples = min(configured_batch_size, int(train_idx.shape[0]))
    configured_batch_bytes = delayed_example_bytes * configured_batch_size
    effective_batch_bytes = delayed_example_bytes * effective_batch_examples

    gpu_mem_info = gpu_mem()
    if isinstance(gpu_mem_info, list) and gpu_mem_info:
        gpu_vram_source = "nvidia-smi"
        gpu_total_vram_bytes = int(gpu_mem_info[0]["total"] * 1024 * 1024)
        gpu_free_vram_bytes = int(gpu_mem_info[0]["free"] * 1024 * 1024)
        gpu_used_vram_bytes = gpu_total_vram_bytes - gpu_free_vram_bytes
        half_free_vram_bytes = int(0.5 * gpu_free_vram_bytes)
        batch_exceeds_half_free_vram = configured_batch_bytes > half_free_vram_bytes
    else:
        gpu_vram_source = "unavailable"
        gpu_total_vram_bytes = None
        gpu_free_vram_bytes = None
        gpu_used_vram_bytes = None
        half_free_vram_bytes = None
        batch_exceeds_half_free_vram = None

    return {
        "dataset_folder": dataset_folder_str,
        "sigma_x_override": sigma_x_override,
        "sigma_z_override": sigma_z_override,
        "alpha_override": alpha_override,
        "eval_noise_cfg": eval_noise_cfg,
        "eval_noise_enabled": eval_noise_enabled,
        "delayed": delayed,
        "noise": noise,
        "targets": targets,
        "gaussian_masks": gaussian_masks,
        "info": info,
        "kp": kp,
        "cm": cm,
        "train_loss_weights": train_loss_weights,
        "pixel_weight_lambda": pixel_weight_lambda,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "memory": {
            "delayed_dataset_bytes": delayed_dataset_bytes,
            "delayed_example_bytes": delayed_example_bytes,
            "configured_batch_size": configured_batch_size,
            "effective_batch_examples": effective_batch_examples,
            "configured_batch_bytes": configured_batch_bytes,
            "effective_batch_bytes": effective_batch_bytes,
            "gpu_vram_source": gpu_vram_source,
            "gpu_total_vram_bytes": gpu_total_vram_bytes,
            "gpu_free_vram_bytes": gpu_free_vram_bytes,
            "gpu_used_vram_bytes": gpu_used_vram_bytes,
            "half_free_vram_bytes": half_free_vram_bytes,
            "batch_exceeds_half_free_vram": batch_exceeds_half_free_vram,
        },
    }


def build_training_pipelines(
    delayed: np.ndarray,
    targets: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_loss_weights: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Build train/validation ``tf.data`` pipelines using shared defaults.

    Args:
        delayed: Delayed samples array with shape ``(N, E, Z, X)``.
        targets: Targets array with shape ``(N, Z, X)``.
        train_idx: Training indices.
        val_idx: Validation indices.
        train_loss_weights: Per-pixel training weights array ``(N, Z, X)``.
        batch_size: Batch size in examples.
        seed: Random seed used for deterministic shuffle.

    Returns:
        Tuple ``(train_ds, val_ds)``.
    """
    train_ds = build_tf_dataset_by_indices(
        delayed,
        targets,
        indices=train_idx,
        sample_weights=train_loss_weights,
        batch_size=int(batch_size),
        shuffle=True,
        seed=int(seed),
    )
    val_ds = build_tf_dataset_by_indices(
        delayed,
        targets,
        indices=val_idx,
        sample_weights=train_loss_weights,
        batch_size=int(batch_size),
        shuffle=False,
        seed=int(seed),
    )
    return train_ds, val_ds


def setup_output_directories_and_callbacks(
    cfg: Mapping[str, Any],
    default_output_root: str,
) -> tuple[str, Path, Path, list[tf.keras.callbacks.Callback], str]:
    """Create timestamped output directories and standard training callbacks.

    Args:
        cfg: Full experiment configuration mapping.
        default_output_root: Fallback output root under project root when no IO override exists.

    Returns:
        Tuple ``(output_dir, apodization_dir, snr_dir, callbacks, timestamp)``.
    """
    from inr_apodizations import config as global_config

    io_cfg = dict(cfg.get("io", {}))
    scripts_output_root = io_cfg.get("scripts_output_root")
    if scripts_output_root is None:
        legacy_sandbox_output_root = io_cfg.get("sandbox_output_root")
        if legacy_sandbox_output_root is not None:
            warnings.warn(
                "io.sandbox_output_root is deprecated; use io.scripts_output_root instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            scripts_output_root = legacy_sandbox_output_root
        else:
            scripts_output_root = default_output_root

    scripts_output_root_cfg = Path(str(scripts_output_root))
    if not scripts_output_root_cfg.is_absolute():
        output_root = global_config.PROJ_ROOT / scripts_output_root_cfg
    else:
        output_root = scripts_output_root_cfg

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = str(Path(output_root) / timestamp)
    os.makedirs(output_dir, exist_ok=True)
    apodization_dir = Path(output_dir) / "apodization"
    snr_dir = Path(output_dir) / "snr"
    apodization_dir.mkdir(parents=True, exist_ok=True)
    snr_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(cfg["training"]["early_stopping_patience"]),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=int(cfg["training"]["reduce_lr_patience"]),
            factor=0.5,
        ),
    ]
    return output_dir, apodization_dir, snr_dir, callbacks, timestamp
