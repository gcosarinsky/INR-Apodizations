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
from inr_apodizations.plots import plot_lateral_reflector_profiles

# Temporary path fix: ensure `sandbox/inr_das_experiment` is on sys.path
# so the local `helpers.py` module can be imported when running this script
import sys as _sys
from pathlib import Path as _Path_for_helpers
_helpers_dir = _Path_for_helpers(__file__).resolve().parent.parent
if str(_helpers_dir) not in _sys.path:
    _sys.path.insert(0, str(_helpers_dir))

import helpers
plt.ion()  # interactive mode for plotting

def _load_config(cfg_path: Path) -> dict[str, Any]:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_delayed_samples(path: Path) -> np.ndarray:
    arr = np.load(path)
    # expected shape: (n_examples, n_elements, nz, nx)
    return arr


def _build_grid_row_definitions(cfg: dict[str, Any], line_length_margin_mm: float) -> list[dict[str, float]]:
    """Derive horizontal reflector-row definitions from phantom grid config.

    Args:
        cfg: Full evaluation config dictionary.
        line_length_margin_mm: Extra lateral margin added to the row span.

    Returns:
        List of dictionaries with keys ``z_center``, ``x_center`` and ``line_length``.

    Raises:
        ValueError: If phantom mode/config is invalid or inconsistent.
    """
    phantom_cfg = cfg.get("phantom", {})
    mode = str(phantom_cfg.get("mode", "")).strip().lower()
    if mode != "grid":
        raise ValueError(
            "Este flujo de perfiles laterales requiere `phantom.mode: grid` en evaluation_config.yml."
        )

    grid_cfg = phantom_cfg.get("grid")
    if not isinstance(grid_cfg, dict):
        raise ValueError("Falta el bloque `phantom.grid` en evaluation_config.yml.")

    required = (
        "x_count",
        "z_count",
        "x_center_mm",
        "z_start_mm",
        "x_spacing_mm",
        "z_spacing_mm",
    )
    missing = [name for name in required if name not in grid_cfg]
    if missing:
        raise ValueError(f"Faltan campos requeridos en `phantom.grid`: {missing}")

    x_count = int(grid_cfg["x_count"])
    z_count = int(grid_cfg["z_count"])
    x_center_mm = float(grid_cfg["x_center_mm"])
    z_start_mm = float(grid_cfg["z_start_mm"])
    x_spacing_mm = float(grid_cfg["x_spacing_mm"])
    z_spacing_mm = float(grid_cfg["z_spacing_mm"])

    if x_count <= 0 or z_count <= 0:
        raise ValueError("`x_count` y `z_count` deben ser enteros positivos en `phantom.grid`.")
    if x_spacing_mm <= 0.0 or z_spacing_mm <= 0.0:
        raise ValueError("`x_spacing_mm` y `z_spacing_mm` deben ser > 0 en `phantom.grid`.")
    if line_length_margin_mm < 0.0:
        raise ValueError("`line_length_margin_mm` debe ser >= 0.")

    x_indices = np.arange(x_count, dtype=np.float32)
    x_offsets = (x_indices - (x_count - 1) / 2.0) * x_spacing_mm
    x_positions = x_center_mm + x_offsets

    x_row_center = float(np.mean(x_positions))
    x_row_span = float(np.max(x_positions) - np.min(x_positions)) if x_count > 1 else float(x_spacing_mm)
    line_length = x_row_span + float(line_length_margin_mm)

    rows = []
    for z_idx in range(z_count):
        z_center = z_start_mm + z_idx * z_spacing_mm
        rows.append(
            {
                "z_center": float(z_center),
                "x_center": x_row_center,
                "line_length": float(line_length),
            }
        )
    return rows


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

# ===== Reflector lateral profiles by row (from phantom.grid) =====
profile_cfg = cfg.get("reflector_lateral_profiles", {})
profiles_enabled = bool(profile_cfg.get("enabled", True))
if profiles_enabled:
    methods_cfg = profile_cfg.get("methods", ["uniform", "hanning", "boxcar", "inr"])
    if not isinstance(methods_cfg, list) or len(methods_cfg) == 0:
        raise ValueError(
            "`reflector_lateral_profiles.methods` debe ser una lista no vacia en evaluation_config.yml."
        )

    selected_method_names = [str(name).strip().lower() for name in methods_cfg]
    # Preserve order while removing duplicates.
    selected_method_names = list(dict.fromkeys(selected_method_names))

    missing_methods = [name for name in selected_method_names if name not in images_db]
    if missing_methods:
        raise RuntimeError(
            "No se pueden generar perfiles laterales: faltan metodos en images_db: "
            f"{missing_methods}"
        )

    selected_images_db = {name: np.asarray(images_db[name]) for name in selected_method_names}
    line_length_margin_mm = float(profile_cfg.get("line_length_margin_mm", 1.0))
    thickness_mm = float(profile_cfg.get("thickness_mm", 0.0))
    vmin_db = float(profile_cfg.get("vmin_db", -60.0))
    save_individual_rows = bool(profile_cfg.get("save_individual_rows", False))
    row_defs = _build_grid_row_definitions(cfg, line_length_margin_mm=line_length_margin_mm)

    nrows = len(row_defs)
    fig_rows, axes_rows = plt.subplots(
        nrows,
        1,
        figsize=(10, max(4.0, 2.9 * nrows)),
        sharex=False,
        constrained_layout=True,
    )
    axes_rows = np.atleast_1d(axes_rows)

    tmp_rows_dir = out_root / "_tmp_reflector_rows"
    tmp_rows_dir.mkdir(parents=True, exist_ok=True)

    for row_idx, row_def in enumerate(row_defs):
        row_png = tmp_rows_dir / f"row_{row_idx:02d}.png"
        sampled_x, profiles = plot_lateral_reflector_profiles(
            images=selected_images_db,
            output_path=str(row_png),
            extent=kp.get_imshow_extent(),
            x_center=float(row_def["x_center"]),
            z_center=float(row_def["z_center"]),
            line_length=float(row_def["line_length"]),
            thickness_mm=thickness_mm,
            overlay_profiles=True,
            cm=cm,
            vmin_db=vmin_db,
        )

        ax = axes_rows[row_idx]
        for method_name in selected_method_names:
            ax.plot(sampled_x, profiles[method_name], linewidth=2, label=method_name)
        ax.set_ylim(vmin_db, 0.0)
        ax.grid(True, alpha=0.3)
        ax.axvline(float(row_def["x_center"]), color="black", linestyle=":", linewidth=1.2)
        ax.set_ylabel("Amplitude (dB)")
        ax.set_title(
            "Reflector row profile "
            f"z={float(row_def['z_center']):.2f} mm, len={float(row_def['line_length']):.2f} mm"
        )
        if row_idx == 0:
            legend_cols = min(4, max(1, len(selected_method_names)))
            ax.legend(ncol=legend_cols, fontsize=9)

        if not save_individual_rows and row_png.exists():
            row_png.unlink()

    axes_rows[-1].set_xlabel("x (mm)")
    profiles_plot_path = out_root / "evaluate_reflector_row_profiles_db.png"
    fig_rows.savefig(profiles_plot_path, dpi=150)

print("Evaluate complete. Plot saved to:", plot_path)
if profiles_enabled:
    print("Row profiles plot saved to:", profiles_plot_path)
