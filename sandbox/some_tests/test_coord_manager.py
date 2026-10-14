"""Basic test for CoordinateManager.

Loads the latest delayed samples dataset, builds KernelParameters2D, instantiates
CoordinateManager, constructs features on GPU (if available) and reports GPU
memory usage before and after feature construction.

Code and docstrings in English; test prompts and prints in Spanish.
"""
import os
import time
import subprocess

import numpy as np
import tensorflow as tf

from inr_apodizations.config import DATA_DIR
from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.utils import find_latest_dataset_folder


def _get_gpu_memory_tf():
    """Return GPU memory used in bytes, or None if unavailable."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return None
    try:
        info = tf.config.experimental.get_memory_info('GPU:0')
        return int(info.get('current', 0))
    except Exception:
        # Fallback to nvidia-smi
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                stderr=subprocess.DEVNULL,
            )
            used_mb = int(out.decode('utf-8').strip().splitlines()[0])
            return used_mb * 1024 * 1024
        except Exception:
            return None


def test_coordinate_manager_gpu_memory():
    """Basic test: build CoordinateManager features and report VRAM usage."""
    latest = find_latest_dataset_folder(DATA_DIR, "delayed_samples_dataset")
    print(f"Usando dataset: {latest}")

    # Try to load config saved by the dataset creation script
    cfg_path = latest / 'cfg_delayed_samples.npy'
    if not cfg_path.exists():
        # Try alternative names
        cfg_path = latest / 'cfg_delayed_samples.npy'
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found in {latest}")

    cfg = np.load(cfg_path, allow_pickle=True).item()

    # Create kernel parameters
    kp = KernelParameters2D(cfg)
    print("KernelParameters2D created:", kp)

    cm = CoordinateManager(kp)
    print("CoordinateManager instantiated")

    # Measure GPU memory before building features
    used_before = _get_gpu_memory_tf()
    if used_before is None:
        print("No GPU detected or unable to query GPU memory. Exiting test.")
        return
    print(f"VRAM usada antes: {used_before / (1024**2):.2f} MiB")

    # Build features on GPU (if available) by using device context
    try:
        with tf.device('/GPU:0'):
            features_mm = cm.get_features_grid(scaled=False)
            # Force tensor materialization if possible
            _ = features_mm.shape
            # small sleep to allow allocation to settle
            time.sleep(0.5)
            used_after_mm = _get_gpu_memory_tf()
            print(f"VRAM usada después features mm: {used_after_mm / (1024**2):.2f} MiB")

            # Build scaled features
            features_scaled = cm.get_features_grid(scaled=True)
            _ = features_scaled.shape
            time.sleep(0.5)
            used_after_scaled = _get_gpu_memory_tf()
            print(f"VRAM usada después features scaled: {used_after_scaled / (1024**2):.2f} MiB")
    except RuntimeError as e:
        print("Error al forzar dispositivo GPU:", e)
        raise

    # Basic shape asserts
    n_elem = kp.n_elements
    nz, nx = kp.nz, kp.nx
    assert tuple(features_mm.shape) == (n_elem, nz, nx, 3)
    assert tuple(features_scaled.shape) == (n_elem, nz, nx, 3)

    # Report deltas
    delta_mm = (used_after_mm - used_before) / (1024**2) if used_after_mm is not None else None
    delta_scaled = (used_after_scaled - used_after_mm) / (1024**2) if (used_after_scaled is not None and used_after_mm is not None) else None
    print(f"Incremento VRAM (features mm): {delta_mm:.2f} MiB")
    print(f"Incremento VRAM (features scaled): {delta_scaled:.2f} MiB")


if __name__ == '__main__':
    test_coordinate_manager_gpu_memory()
