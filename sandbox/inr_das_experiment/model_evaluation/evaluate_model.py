"""
Evaluate INR model apodization vs classical windows.

This script loads a saved Keras model (INR) and a delayed-samples dataset,
computes apodizations (INR, Hanning, Boxcar, Uniform), applies them to
the delayed samples to obtain DAS reconstructions, and plots the results.

Design decisions:
- The script is simple (no argparse/main). It reads `evaluation_config.yml`.
- Docstrings and comments in English; user-facing messages in Spanish.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
import tensorflow as tf

from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.modeling.trainer import DasInrTrainer
from inr_apodizations.apodizations import compute_dynamic_apodizations_tf
from inr_apodizations.config import PROJ_ROOT

# Temporary path fix: ensure `sandbox/inr_das_experiment` is on sys.path
# so the local `helpers.py` module can be imported when running this script
import sys as _sys
from pathlib import Path as _Path_for_helpers
_helpers_dir = _Path_for_helpers(__file__).resolve().parent.parent
if str(_helpers_dir) not in _sys.path:
    _sys.path.insert(0, str(_helpers_dir))

import helpers


def _load_config(cfg_path: Path) -> dict[str, Any]:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_delayed_samples(path: Path) -> np.ndarray:
    arr = np.load(path)
    # expected shape: (n_examples, n_elements, nz, nx)
    return arr


# Note: apodization helpers removed — use compute_dynamic_apodizations_tf and
# the INR trainer outputs. Fallback 1D windows are intentionally disabled.


# ===== Load config =====
script_dir = PROJ_ROOT / "sandbox" / "inr_das_experiment" / "model_evaluation"
cfg = _load_config(script_dir / "evaluation_config.yml")

io_cfg = cfg.get("io", {})
sim_cfg = cfg.get("simulation", {})
bf_cfg = cfg.get("beamforming", {})

model_path = Path(PROJ_ROOT / io_cfg.get("model_path"))
delayed_samples_path = Path(PROJ_ROOT / io_cfg.get("delayed_samples_path"))

if not delayed_samples_path.exists():
    raise FileNotFoundError(f"Delayed samples not found: {delayed_samples_path}")

# ===== Load delayed samples =====
delayed = _load_delayed_samples(delayed_samples_path)
# Use first example
delayed0 = delayed[0]  # shape (n_elements, nz, nx)

n_elements = delayed0.shape[0]

# ===== Build coordinate manager (try dataset folder pattern like training) =====
dataset_folder = delayed_samples_path.parent
try:
    kp, cm = helpers.build_coordinate_manager(str(dataset_folder))
except FileNotFoundError:
    # Fallback: build KernelParameters2D from config blocks
    kp_cfg = {
        "fs": sim_cfg.get("fs", 20.832),
        "c1": sim_cfg.get("c1", 1.54),
        "pitch": sim_cfg.get("pitch", 0.3),
        "f1": bf_cfg.get("f1", 1.0),
        "f2": bf_cfg.get("f2", 10.0),
        "f_number": bf_cfg.get("f_number", 1.0),
        "x_step": bf_cfg.get("x_step", 0.1),
        "z_step": bf_cfg.get("z_step", 0.1),
        "t_start": sim_cfg.get("t_start", 0.0),
        "taps": bf_cfg.get("taps", 62),
        "n_batch": bf_cfg.get("n_batch", 0),
        "n_elements": int(sim_cfg.get("n_elements", n_elements)),
        "n_ch": bf_cfg.get("n_ch", int(sim_cfg.get("n_elements", n_elements))),
        "n_angles": int(len(np.arange(*sim_cfg.get("angles", [-5, 6, 1])))),
        "n_samples": int(sim_cfg.get("n_samples") or 1000),
        "roi_user": bf_cfg.get("roi_user"),
        "blocksize_img": tuple(bf_cfg.get("blocksize_img", [32, 8])),
    }
    kp = KernelParameters2D(kp_cfg)
    cm = CoordinateManager(kp)

# ===== Load model (optional) =====
model = None
if model_path.exists():
    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception:
        # try common artifact names
        if model_path.is_dir():
            candidate = model_path / "model.keras"
            if candidate.exists():
                model = tf.keras.models.load_model(str(candidate))
            else:
                h5s = list(model_path.glob("*.h5"))
                if h5s:
                    model = tf.keras.models.load_model(str(h5s[0]))

# If model still not found, exit with error (script requires a model)
if model is None:
    _sys.exit(f"Error: no INR model found at {model_path}. The script requires an INR model to continue.")

# ===== Compute apodizations =====
# Classical dynamic apodizations via CoordinateManager
f_number = float(bf_cfg.get("f_number", 1.0))
dyn_apods = compute_dynamic_apodizations_tf(cm, f_number, methods=("boxcar", "hanning"), scaled=False)

# Prepare DAS images dictionary
images = {}

# Uniform DAS
uniform_img = delayed0.sum(axis=0)
images["uniform"] = uniform_img

# Hanning and Boxcar DAS
if "hanning" in dyn_apods:
    hanning_w = dyn_apods["hanning"].numpy()  # shape (E, Z, X)
    hanning_img = (delayed0 * hanning_w).sum(axis=0)
    images["hanning"] = hanning_img
else:
    raise RuntimeError("compute_dynamic_apodizations_tf did not return 'hanning' apodization. Ensure CoordinateManager features are available.")

if "boxcar" in dyn_apods:
    box_w = dyn_apods["boxcar"].numpy()
    box_img = (delayed0 * box_w).sum(axis=0)
    images["boxcar"] = box_img
else:
    raise RuntimeError("compute_dynamic_apodizations_tf did not return 'boxcar' apodization. Ensure CoordinateManager features are available.")

# INR via DasInrTrainer if model available
if model is not None:
    # `scaled_features` MUST be specified in a train_config_info.yml adjacent to the model.
    try:
        if model_path.exists():
            if model_path.is_dir():
                train_info = model_path / "train_config_info.yml"
            else:
                train_info = model_path.parent / "train_config_info.yml"
        else:
            train_info = None

        if train_info is None or not train_info.exists():
            _sys.exit(
                f"Error: `scaled_features` must be specified in train_config_info.yml next to the model. Not found: {train_info or model_path}"
            )

        with open(train_info, encoding="utf-8") as _f:
            train_info_dict = yaml.safe_load(_f) or {}

        if not isinstance(train_info_dict, dict):
            _sys.exit(f"Error: invalid format in {train_info}; expected a YAML mapping.")

        scaled_features = bool(train_info_dict["experiment"]["model"].get("scaled_features"))
        
        if not scaled_features:
            _sys.exit(
                f"Error: `scaled_features` not found in {train_info}; it must be defined at top-level or under 'model'."
            )
    except Exception as _e:
        _sys.exit(f"Error reading {train_info}: {_e}")

    features_grid = cm.get_features_grid(scaled=scaled_features)
    feature_chunk_size = int(65536)
    trainer = DasInrTrainer(
        apodization_model=model,
        features_grid=features_grid,
        feature_chunk_size=feature_chunk_size,
        weight_regularization_enabled=False,
    )
    weights_grid = trainer.predict_weights_grid(training=False).numpy()  # (E, Z, X)
    inr_img = (delayed0 * weights_grid).sum(axis=0)
    images["inr"] = inr_img

# ===== Produce DAS images and plot =====

# Convert to dB for plotting (use per-image maximum rather than shared reference)
images_db = {}
for k, v in images.items():
    ref = np.max(np.abs(v))
    images_db[k] = 20.0 * np.log10(np.abs(v) / (ref + 1e-12) + 1e-12)

# Plot grid (force a 2x2 layout)
labels = list(images_db.keys())
rows, cols = 2, 2
num = len(labels)
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
axes = np.atleast_2d(axes)

for idx, label in enumerate(labels):
    r = idx // cols
    c = idx % cols
    ax = axes[r, c]
    im = ax.imshow(images_db[label], aspect="auto", cmap="gray", vmin=-60, vmax=0)
    ax.set_title(label)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Hide unused axes (if fewer than 4 images)
for j in range(len(labels), rows * cols):
    r = j // cols
    c = j % cols
    axes[r, c].axis("off")

plt.tight_layout()
out_root = Path(io_cfg.get("output_root", script_dir / "outputs"))
out_root = (Path.cwd() / out_root).resolve()
out_root.mkdir(parents=True, exist_ok=True)
plot_path = out_root / "evaluate_apodizations_quicklook.png"
fig.savefig(plot_path, dpi=150)
plt.show()

print("Evaluate complete. Plot saved to:", plot_path)
