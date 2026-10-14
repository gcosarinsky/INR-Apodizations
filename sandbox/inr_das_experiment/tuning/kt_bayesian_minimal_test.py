"""Minimal BayesianOptimization test for Keras Tuner with XLA/libdevice handling.

This script is a tiny, self-contained example to reproduce the XLA/libdevice
environment setup without running any heavy beamforming pipelines.

How it helps:
- Searches for CUDA `libdevice.*.bc` and sets `XLA_FLAGS` before importing
  TensorFlow. If not found, it disables XLA auto-jit via `TF_XLA_FLAGS`.
- Runs a tiny BayesianOptimization search on a toy regression problem.

Use for quick local debugging of the libdevice/XLA environment.
"""
from __future__ import annotations

import os
import glob
import numpy as np
from pathlib import Path


def _find_libdevice_dir() -> str | None:
    """Search common CUDA install locations for a libdevice directory.

    Returns the directory containing `libdevice.*.bc` or None if not found.
    """
    env_vars = ("CUDA_PATH", "CUDA_HOME", "CUDA_DIR", "CUDA_ROOT")
    for ev in env_vars:
        root = os.environ.get(ev)
        if not root:
            continue
        candidate = os.path.join(root, "nvvm", "libdevice")
        if os.path.isdir(candidate):
            for f in os.listdir(candidate):
                if f.startswith("libdevice") and f.endswith(".bc"):
                    return candidate

    # Typical Windows CUDA install locations
    cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(cuda_root):
        for sub in os.listdir(cuda_root):
            candidate = os.path.join(cuda_root, sub, "nvvm", "libdevice")
            if os.path.isdir(candidate):
                for f in os.listdir(candidate):
                    if f.startswith("libdevice") and f.endswith(".bc"):
                        return candidate

    # Fallback: recursive glob search in common roots
    search_roots = [os.environ.get("CUDA_PATH", ""), "/usr/local/cuda", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA"]
    for root in filter(None, search_roots):
        try:
            for match in glob.glob(os.path.join(root, "**", "libdevice.*.bc"), recursive=True):
                return os.path.dirname(match)
        except Exception:
            continue

    return None


# Set XLA flags early, before importing tensorflow
_libdevice_dir = _find_libdevice_dir()
_disable_xla_fallback = False
if _libdevice_dir:
    os.environ.setdefault("XLA_FLAGS", f"--xla_gpu_cuda_data_dir={_libdevice_dir}")
    print(f"XLA_FLAGS set to: {_libdevice_dir}")
else:
    # Disable XLA auto-jit to avoid crashes when libdevice is missing
    _disable_xla_fallback = True
    os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
    print("libdevice not found; setting TF_XLA_FLAGS to disable XLA auto-jit.")


def build_and_run_tuner(max_trials: int = 4, executions_per_trial: int = 1) -> None:
    """Builds a minimal model and runs Keras Tuner BayesianOptimization.

    Args:
        max_trials: maximum number of tuner trials.
        executions_per_trial: executions per trial.
    """
    # Delayed import to ensure XLA flags are applied
    import tensorflow as tf
    import keras_tuner as kt

    if _disable_xla_fallback:
        try:
            tf.config.optimizer.set_jit(False)
            print("TensorFlow JIT/XLA disabled for this run.")
        except Exception:
            pass

    tf.keras.utils.set_random_seed(42)

    # Toy dataset (regression)
    x = np.random.RandomState(0).randn(300, 16).astype(np.float32)
    y = (np.sum(x, axis=1) + 0.1 * np.random.randn(x.shape[0])).astype(np.float32)

    def model_builder(hp):
        units = hp.Int("units", min_value=4, max_value=64, step=4)
        lr = hp.Float("lr", 1e-4, 1e-1, sampling="log")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(x.shape[1],)),
            tf.keras.layers.Dense(units, activation="relu"),
            tf.keras.layers.Dense(units // 2, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
        return model

    tuner = kt.BayesianOptimization(
        model_builder,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory=str(Path.cwd() / "kt_minimal_tuner_logs"),
        project_name="minimal_bayes_test",
    )

    print("Starting a tiny BayesianOptimization search (this is lightweight)...")
    tuner.search(x, y, epochs=3, validation_split=0.2, verbose=2)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best hyperparameters found:")
    print(f"  units: {best_hps.get('units')}")
    print(f"  lr: {best_hps.get('lr')}")


if __name__ == "__main__":
    # Do not run heavy tasks — this remains a small, quick test.
    build_and_run_tuner(max_trials=4, executions_per_trial=1)
