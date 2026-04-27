"""
Evaluate INR model apodization vs classical windows.

This script loads a saved Keras model (INR) and a delayed-samples dataset,
computes apodizations (INR, Hanning, Boxcar, Uniform), applies them to
the delayed samples to obtain DAS reconstructions, and plots the results.

Design decisions:
- The script is simple (no argparse/main). It reads `configs/numeric_phantom_evaluation_config.yml`.
- Docstrings and comments in English; user-facing messages in Spanish.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
from datetime import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
import tensorflow as tf

from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.modeling.trainer import DasInrTrainer
from inr_apodizations.apodizations import compute_dynamic_apodizations_tf
from inr_apodizations.config import CONFIGS_DIR, PROJ_ROOT
from inr_apodizations.dataset import generate_das_modulated_target
from inr_apodizations.evaluation.profiles import compute_fwhm_batch, extract_reflector_profiles
from inr_apodizations.evaluation import compute_reflector_snr, compute_scatterer_metrics
import inr_apodizations.sandbox_helpers as helpers

plt.ion()  # interactive mode for plotting

def _load_config(cfg_path: Path) -> dict[str, Any]:
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_delayed_samples(path: Path) -> np.ndarray:
    arr = np.load(path)
    # expected shape: (n_examples, n_elements, nz, nx)
    return arr


def _resolve_latest_delayed_samples_path(io_cfg: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve delayed-samples artifact from latest simulation metadata.

    Returns:
        Tuple ``(delayed_samples_file, latest_run_folder)``.

    Raises:
        FileNotFoundError: If expected folders/files are missing.
        ValueError: If metadata format/content is invalid.
    """
    simulation_root_cfg = io_cfg.get(
        "simulation_output_root", "sandbox/inr_das_experiment/evaluation/delayed_samples"
    )
    simulation_root = Path(PROJ_ROOT / simulation_root_cfg)
    if not simulation_root.exists() or not simulation_root.is_dir():
        raise FileNotFoundError(
            "No existe `io.simulation_output_root` o no es carpeta: "
            f"{simulation_root}"
        )

    run_folders = [path for path in simulation_root.iterdir() if path.is_dir()]
    if len(run_folders) == 0:
        raise FileNotFoundError(
            "No se encontraron corridas en `io.simulation_output_root`: "
            f"{simulation_root}"
        )

    latest_run = max(run_folders, key=lambda path: path.stat().st_mtime)
    simulation_info_path = latest_run / "simulation_info.yml"
    if not simulation_info_path.exists():
        raise FileNotFoundError(
            "No se encontro simulation_info.yml en la corrida mas reciente: "
            f"{simulation_info_path}"
        )

    simulation_info = _load_config(simulation_info_path)
    delayed_samples_cfg = simulation_info.get("delayed_samples_path")
    if not isinstance(delayed_samples_cfg, str) or len(delayed_samples_cfg.strip()) == 0:
        raise ValueError(
            "Falta `delayed_samples_path` valido en simulation_info.yml: "
            f"{simulation_info_path}"
        )

    delayed_samples_path = Path(delayed_samples_cfg)
    if not delayed_samples_path.is_absolute():
        delayed_samples_path = (PROJ_ROOT / delayed_samples_path).resolve()

    if not delayed_samples_path.exists() or delayed_samples_path.is_dir():
        raise FileNotFoundError(
            "`delayed_samples_path` en simulation_info.yml no apunta a un archivo valido: "
            f"{delayed_samples_path}"
        )

    return delayed_samples_path, latest_run


def _build_grid_reflector_points(cfg: dict[str, Any]) -> np.ndarray:
    """Build reflector positions ``(N, 2)`` in mm from ``phantom.grid`` config."""
    phantom_cfg = cfg.get("phantom", {})
    mode = str(phantom_cfg.get("mode", "")).strip().lower()
    if mode != "grid":
        raise ValueError(
            "Este flujo de perfiles por reflector requiere `phantom.mode: grid` en numeric_phantom_evaluation_config.yml."
        )

    grid_cfg = phantom_cfg.get("grid")
    if not isinstance(grid_cfg, dict):
        raise ValueError("Falta el bloque `phantom.grid` en numeric_phantom_evaluation_config.yml.")

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

    x_indices = np.arange(x_count, dtype=np.float64)
    x_offsets = (x_indices - (x_count - 1) / 2.0) * x_spacing_mm
    x_positions = x_center_mm + x_offsets

    points: list[list[float]] = []
    for z_idx in range(z_count):
        z_pos = z_start_mm + z_idx * z_spacing_mm
        for x_pos in x_positions:
            points.append([float(x_pos), float(z_pos)])

    return np.asarray(points, dtype=np.float64)


def _resolve_reflector_indices(indices_cfg: Any, n_reflectors: int) -> np.ndarray:
    """Resolve user-selected reflector indices from YAML.

    Accepted values are ``"all"`` (default), or a list of integer indices.
    """
    if isinstance(indices_cfg, str) and indices_cfg.strip().lower() == "all":
        return np.arange(n_reflectors, dtype=np.int32)
    if indices_cfg is None:
        return np.arange(n_reflectors, dtype=np.int32)
    if not isinstance(indices_cfg, list) or len(indices_cfg) == 0:
        raise ValueError(
            "`reflector_lateral_profiles.reflector_indices` debe ser 'all' o una lista no vacia de enteros."
        )

    resolved = np.asarray(indices_cfg, dtype=np.int32)
    if np.any(resolved < 0) or np.any(resolved >= n_reflectors):
        raise ValueError(
            "`reflector_lateral_profiles.reflector_indices` contiene indices fuera de rango. "
            f"Rango valido: [0, {n_reflectors - 1}]"
        )
    return np.unique(resolved)


# Note: apodization helpers removed — use compute_dynamic_apodizations_tf and
# the INR trainer outputs. Fallback 1D windows are intentionally disabled.


# ===== Load config =====
script_dir = Path(__file__).resolve().parent
cfg = _load_config(CONFIGS_DIR / "numeric_phantom_evaluation_config.yml")

io_cfg = cfg.get("io", {})
sim_cfg = cfg.get("simulation", {})
bf_cfg = cfg.get("beamforming", {})

plot_cfg = cfg.get("plots", {})
fontsize_legend = int(plot_cfg.get("legend_fontsize", 8))
fontsize_title = int(plot_cfg.get("title_fontsize", 10))
fontsize_axis = int(plot_cfg.get("axis_fontsize", 10))

target_train_cfg = cfg.get("training_target", {})
target_train_enabled = bool(target_train_cfg.get("enabled", False))
target_train_sigma_x = float(target_train_cfg.get("sigma_x_mm", 0.5))
target_train_sigma_z = float(target_train_cfg.get("sigma_z_mm", 0.5))
target_train_alpha = float(target_train_cfg.get("alpha", 0.0))
target_train_include_in_profiles = bool(target_train_cfg.get("include_in_profiles", True))
target_train_label = str(target_train_cfg.get("display_label", "target"))
if target_train_enabled:
    if not (0.0 <= target_train_alpha < 1.0):
        raise ValueError("`training_target.alpha` must be in [0, 1).")
    if target_train_sigma_x <= 0.0 or target_train_sigma_z <= 0.0:
        raise ValueError("`training_target.sigma_x_mm` and `sigma_z_mm` must be > 0.")

model_path = Path(PROJ_ROOT / io_cfg.get("model_path"))
delayed_samples_path, latest_simulation_run = _resolve_latest_delayed_samples_path(io_cfg)
print("Corrida de simulacion seleccionada:", latest_simulation_run)
print("Usando delayed samples:", delayed_samples_path)

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

# Build spatial grids in mm for target image computation
_x_coords = np.linspace(kp.roi_effective[0], kp.roi_effective[1], kp.nx)
_z_coords = np.linspace(kp.roi_effective[2], kp.roi_effective[3], kp.nz)
x_grid, z_grid = np.meshgrid(_x_coords, _z_coords)

# ===== Load model (conditional on reflector_lateral_profiles.inr_enabled) =====
inr_enabled = bool(cfg.get("reflector_lateral_profiles", {}).get("inr_enabled", True))
model = None
if inr_enabled:
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

    # If model still not found and INR is required, exit
    if model is None:
        sys.exit(f"Error: no INR model found at {model_path}. Set inr_enabled: false in reflector_lateral_profiles to skip INR evaluation.")

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
            sys.exit(
                f"Error: `scaled_features` must be specified in train_config_info.yml next to the model. Not found: {train_info or model_path}"
            )

        with open(train_info, encoding="utf-8") as _f:
            train_info_dict = yaml.safe_load(_f) or {}

        if not isinstance(train_info_dict, dict):
            sys.exit(f"Error: invalid format in {train_info}; expected a YAML mapping.")

        scaled_features = bool(train_info_dict["experiment"]["model"].get("scaled_features"))
        
        if not scaled_features:
            sys.exit(
                f"Error: `scaled_features` not found in {train_info}; it must be defined at top-level or under 'model'."
            )
    except Exception as _e:
        sys.exit(f"Error reading {train_info}: {_e}")

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

# Training target: Gaussian-modulated uniform DAS (mirrors the dataset creation pipeline)
if target_train_enabled:
    _reflector_points_for_target = _build_grid_reflector_points(cfg)
    target_img = generate_das_modulated_target(
        images["uniform"],
        _reflector_points_for_target,
        x_grid,
        z_grid,
        sigma_x=target_train_sigma_x,
        sigma_z=target_train_sigma_z,
        alpha=target_train_alpha,
    )
    images[target_train_label] = target_img

# ===== Produce DAS images and plot =====

# Convert to dB for plotting (use per-image maximum rather than shared reference)
images_db = {}
for k, v in images.items():
    ref = np.max(np.abs(v))
    images_db[k] = 20.0 * np.log10(np.abs(v) / (ref + 1e-12) + 1e-12)

# Build display list: target always last; if 5 images total, drop boxcar to keep 2x2
display_labels = list(images_db.keys())
if target_train_enabled and target_train_label in display_labels:
    display_labels.remove(target_train_label)
    display_labels.append(target_train_label)
    if len(display_labels) == 5 and "boxcar" in display_labels:
        display_labels.remove("boxcar")
labels = display_labels
rows, cols = 2, 2
extent = kp.get_imshow_extent()
vmin_db = -60.0
vmax_db = 0.0

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(5 * cols, 4 * rows),
    sharex=True,
    sharey=True,
)
axes_flat = np.asarray(axes).ravel()
first_im = None

for idx, label in enumerate(labels):
    ax = axes_flat[idx]
    im = ax.imshow(
        images_db[label],
        aspect="auto",
        cmap="gray",
        vmin=vmin_db,
        vmax=vmax_db,
        extent=extent,
    )
    if first_im is None:
        first_im = im
    ax.set_title(label, fontsize=fontsize_title)
    ax.set_xlabel("x (mm)", fontsize=fontsize_axis)
    if idx % cols == 0:
        ax.set_ylabel("z (mm)", fontsize=fontsize_axis)

# Hide unused axes (if fewer than 4 images)
for j in range(len(labels), rows * cols):
    axes_flat[j].axis("off")

fig.suptitle("DAS comparison", fontsize=fontsize_title)
fig.tight_layout(rect=[0, 0, 0.92, 1])
cbar_ax = fig.add_axes([0.93, 0.1, 0.013, 0.78])
if first_im is not None:
    fig.colorbar(first_im, cax=cbar_ax, label="dB")

out_root = Path(io_cfg.get("evaluation_output_root", script_dir / "outputs"))
out_root = (Path.cwd() / out_root).resolve()
out_root.mkdir(parents=True, exist_ok=True)

# Save each evaluation run under a timestamped subfolder.
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_out_dir = out_root / run_timestamp
run_out_dir.mkdir(parents=True, exist_ok=False)

with (run_out_dir / "numeric_phantom_evaluation_config_info.yml").open("w", encoding="utf-8") as _f:
    yaml.safe_dump(cfg, _f, sort_keys=False)

plot_path = run_out_dir / "evaluate_apodizations_quicklook.png"
fig.savefig(plot_path, dpi=150)
plt.show()

# ===== Reflector profiles and FWHM (per reflector) =====
profile_cfg = cfg.get("reflector_lateral_profiles", {})
profiles_enabled = bool(profile_cfg.get("enabled", True))
if profiles_enabled:
    methods_cfg = profile_cfg.get("methods", ["uniform", "hanning", "boxcar", "inr"])
    if not isinstance(methods_cfg, list) or len(methods_cfg) == 0:
        raise ValueError(
            "`reflector_lateral_profiles.methods` debe ser una lista no vacia en numeric_phantom_evaluation_config.yml."
        )

    selected_method_names = [str(name).strip().lower() for name in methods_cfg]
    # Preserve order while removing duplicates.
    selected_method_names = list(dict.fromkeys(selected_method_names))

    # Append training target to profiles if enabled and requested
    if target_train_enabled and target_train_include_in_profiles and target_train_label not in selected_method_names:
        selected_method_names.append(target_train_label)

    missing_methods = [name for name in selected_method_names if name not in images_db]
    if missing_methods:
        raise RuntimeError(
            "No se pueden generar perfiles por reflector: faltan metodos en images_db: "
            f"{missing_methods}"
        )

    selected_images_abs = {name: np.asarray(images[name], dtype=np.float64) for name in selected_method_names}

    half_width_lateral_mm = float(profile_cfg.get("half_width_lateral_mm", 1.5))
    half_width_axial_mm = float(profile_cfg.get("half_width_axial_mm", 1.0))
    vmin_db = float(profile_cfg.get("vmin_db", -60.0))
    snr_radius_mm = float(profile_cfg.get("snr_radius_mm", 1.5))

    reflector_points = _build_grid_reflector_points(cfg)
    selected_indices = _resolve_reflector_indices(
        profile_cfg.get("reflector_indices", "all"),
        n_reflectors=int(reflector_points.shape[0]),
    )
    selected_points = reflector_points[selected_indices]

    method_profiles: dict[str, dict[str, np.ndarray | int]] = {}
    fwhm_rows: list[dict[str, float | int | str]] = []
    snr_by_method: dict[str, dict[str, np.ndarray | float]] = {}

    for method_name in selected_method_names:
        extracted = extract_reflector_profiles(
            image=selected_images_abs[method_name],
            scatterers=selected_points,
            cm=cm,
            half_width_lateral_mm=half_width_lateral_mm,
            half_width_axial_mm=half_width_axial_mm,
        )
        lateral_profiles = np.asarray(extracted["lateral_profiles"], dtype=np.float64)
        axial_profiles = np.asarray(extracted["axial_profiles"], dtype=np.float64)
        lateral_offsets_mm = np.asarray(extracted["lateral_offsets_mm"], dtype=np.float64)
        axial_offsets_mm = np.asarray(extracted["axial_offsets_mm"], dtype=np.float64)

        fwhm_lat = compute_fwhm_batch(lateral_profiles, lateral_offsets_mm)
        fwhm_ax = compute_fwhm_batch(axial_profiles, axial_offsets_mm)

        for local_idx, refl_idx in enumerate(selected_indices.tolist()):
            fwhm_rows.append(
                {
                    "reflector_index": int(refl_idx),
                    "method": method_name,
                    "fwhm_lateral_mm": float(fwhm_lat["fwhm_mm"][local_idx]),
                    "fwhm_axial_mm": float(fwhm_ax["fwhm_mm"][local_idx]),
                }
            )

        scatterer_metrics = compute_scatterer_metrics(
            image=selected_images_abs[method_name],
            scatterers=selected_points,
            cm=cm,
            radius_mm=snr_radius_mm,
            profile_half_lateral_mm=None,
            profile_half_axial_mm=None,
            return_masks=False,
            return_background_hist=False,
        )
        peak_amplitudes = np.asarray(scatterer_metrics["peak_amplitudes"], dtype=np.float64)
        background_rms = float(scatterer_metrics["background_rms"])
        snr_values = compute_reflector_snr(peak_amplitudes, background_rms)
        snr_values_db = compute_reflector_snr(peak_amplitudes, background_rms, return_db=True)
        snr_by_method[method_name] = {
            "snr": snr_values,
            "snr_db": snr_values_db,
            "peak_amplitudes": peak_amplitudes,
            "background_rms": background_rms,
        }

        method_profiles[method_name] = {
            "lateral_profiles": lateral_profiles,
            "axial_profiles": axial_profiles,
            "lateral_offsets_mm": lateral_offsets_mm,
            "axial_offsets_mm": axial_offsets_mm,
        }

    # Determine which profile dimensions to plot (lateral / axial / both)
    profile_type = str(profile_cfg.get("profile_type", "both")).strip().lower()
    if profile_type not in ("lateral", "axial", "both"):
        raise ValueError(
            f"`reflector_lateral_profiles.profile_type` must be 'lateral', 'axial', or 'both'. "
            f"Got: '{profile_type}'"
        )
    show_lateral = profile_type in ("lateral", "both")
    show_axial = profile_type in ("axial", "both")

    # Save per-reflector figures (one figure per selected reflector)
    profiles_dir = run_out_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, refl_idx in enumerate(selected_indices.tolist()):
        x_mm = float(reflector_points[refl_idx, 0])
        z_mm = float(reflector_points[refl_idx, 1])

        n_subplots = int(show_lateral) + int(show_axial)
        fig_ref, axes_ref_raw = plt.subplots(
            1, n_subplots, figsize=(6 * n_subplots, 4), constrained_layout=True
        )
        axes_ref_arr = np.atleast_1d(axes_ref_raw)

        # Map role → axes object
        ax_lat = axes_ref_arr[0] if show_lateral else None
        ax_ax = axes_ref_arr[1] if (show_lateral and show_axial) else (axes_ref_arr[0] if show_axial else None)

        for method_name in selected_method_names:
            lat_offsets = np.asarray(method_profiles[method_name]["lateral_offsets_mm"], dtype=np.float64)
            ax_offsets = np.asarray(method_profiles[method_name]["axial_offsets_mm"], dtype=np.float64)
            lat_profile = np.asarray(method_profiles[method_name]["lateral_profiles"], dtype=np.float64)[local_idx]
            ax_profile = np.asarray(method_profiles[method_name]["axial_profiles"], dtype=np.float64)[local_idx]

            lat_peak = float(np.nanmax(np.abs(lat_profile))) if lat_profile.size > 0 else 1.0
            ax_peak = float(np.nanmax(np.abs(ax_profile))) if ax_profile.size > 0 else 1.0
            lat_db_arr = 20.0 * np.log10(np.maximum(np.abs(lat_profile), 1e-12) / max(lat_peak, 1e-12))
            ax_db_arr = 20.0 * np.log10(np.maximum(np.abs(ax_profile), 1e-12) / max(ax_peak, 1e-12))

            if show_lateral:
                ax_lat.plot(lat_offsets, lat_db_arr, linewidth=2, label=method_name)
            if show_axial:
                ax_ax.plot(ax_offsets, ax_db_arr, linewidth=2, label=method_name)

        if show_lateral:
            ax_lat.set_ylim(vmin_db, 0.0)
            ax_lat.grid(True, alpha=0.3)
            ax_lat.set_xlabel("Lateral offset (mm)")
            ax_lat.set_ylabel("Amplitude (dB)")
            ax_lat.set_title("Lateral profile")
            ax_lat.legend(ncol=min(4, len(selected_method_names)), fontsize=fontsize_legend)
        if show_axial:
            ax_ax.set_ylim(vmin_db, 0.0)
            ax_ax.grid(True, alpha=0.3)
            ax_ax.set_xlabel("Axial offset (mm)")
            ax_ax.set_ylabel("Amplitude (dB)")
            ax_ax.set_title("Axial profile")
            if not show_lateral:
                ax_ax.legend(ncol=min(4, len(selected_method_names)), fontsize=fontsize_legend)

        fig_ref.suptitle(
            f"Reflector {refl_idx} at x={x_mm:.2f} mm, z={z_mm:.2f} mm"
        )
        fig_ref.savefig(
            profiles_dir / f"reflector_{refl_idx:03d}_profiles_db.png",
            dpi=150,
        )
        plt.close(fig_ref)
    # Persist FWHM summary as CSV in wide format: one row per reflector,
    # with two columns per method (lateral and axial).
    fwhm_csv_path = run_out_dir / "reflector_fwhm_summary.csv"
    # Pivot rows into a mapping reflector_index -> {colname: value}
    per_reflector: dict[int, dict[str, float]] = {}
    for row in fwhm_rows:
        rid = int(row["reflector_index"])
        method = str(row["method"])
        lat_key = f"{method}_fwhm_lateral_mm"
        ax_key = f"{method}_fwhm_axial_mm"
        if rid not in per_reflector:
            per_reflector[rid] = {}
        per_reflector[rid][lat_key] = float(row.get("fwhm_lateral_mm", float("nan")))
        per_reflector[rid][ax_key] = float(row.get("fwhm_axial_mm", float("nan")))

    for method_name, snr_data in snr_by_method.items():
        snr_vals = np.asarray(snr_data["snr"], dtype=np.float64)
        snr_vals_db = np.asarray(snr_data["snr_db"], dtype=np.float64)
        peak_vals = np.asarray(snr_data["peak_amplitudes"], dtype=np.float64)
        bg_rms = float(snr_data["background_rms"])
        for local_idx, rid in enumerate(selected_indices.tolist()):
            if int(rid) not in per_reflector:
                per_reflector[int(rid)] = {}
            per_reflector[int(rid)][f"{method_name}_snr"] = float(snr_vals[local_idx])
            per_reflector[int(rid)][f"{method_name}_snr_db"] = float(snr_vals_db[local_idx])
            per_reflector[int(rid)][f"{method_name}_peak_amplitude"] = float(peak_vals[local_idx])
            per_reflector[int(rid)][f"{method_name}_background_rms"] = bg_rms

    # Build header: reflector_index, then for each method FWHM and SNR values.
    header = ["reflector_index"]
    for m in selected_method_names:
        header.append(f"{m}_fwhm_lateral_mm")
        header.append(f"{m}_fwhm_axial_mm")
        header.append(f"{m}_snr")
        header.append(f"{m}_snr_db")
        header.append(f"{m}_peak_amplitude")
        header.append(f"{m}_background_rms")

    with open(fwhm_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # Write rows in the same order as selected_indices
        for rid in selected_indices.tolist():
            row_vals = [int(rid)]
            vals = per_reflector.get(int(rid), {})

            def _format_or_empty(value: object) -> str:
                if value is None:
                    return ""
                if isinstance(value, (float, np.floating)):
                    return "" if np.isnan(float(value)) else f"{float(value):.6f}"
                return ""

            for m in selected_method_names:
                lat_key = f"{m}_fwhm_lateral_mm"
                ax_key = f"{m}_fwhm_axial_mm"
                snr_key = f"{m}_snr"
                snr_db_key = f"{m}_snr_db"
                peak_key = f"{m}_peak_amplitude"
                bg_key = f"{m}_background_rms"
                row_vals.append(_format_or_empty(vals.get(lat_key, None)))
                row_vals.append(_format_or_empty(vals.get(ax_key, None)))
                row_vals.append(_format_or_empty(vals.get(snr_key, None)))
                row_vals.append(_format_or_empty(vals.get(snr_db_key, None)))
                row_vals.append(_format_or_empty(vals.get(peak_key, None)))
                row_vals.append(_format_or_empty(vals.get(bg_key, None)))
            writer.writerow(row_vals)

    # ===== Summary plots: FWHM lateral and SNR vs reflector index =====
    summary_fig, (ax_fwhm, ax_snr) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )
    refl_ids_ordered = selected_indices.tolist()

    for method_name in selected_method_names:
        fwhm_lat_vals = [
            float(per_reflector.get(int(rid), {}).get(f"{method_name}_fwhm_lateral_mm", float("nan")))
            for rid in refl_ids_ordered
        ]
        snr_db_vals = [
            float(per_reflector.get(int(rid), {}).get(f"{method_name}_snr_db", float("nan")))
            for rid in refl_ids_ordered
        ]
        ax_fwhm.plot(refl_ids_ordered, fwhm_lat_vals, marker="o", linewidth=2, label=method_name)
        ax_snr.plot(refl_ids_ordered, snr_db_vals, marker="o", linewidth=2, label=method_name)

    ax_fwhm.set_xlabel("Reflector index")
    ax_fwhm.set_ylabel("FWHM lateral (mm)")
    ax_fwhm.set_title("Lateral resolution vs reflector index")
    ax_fwhm.legend(fontsize=fontsize_legend)
    ax_fwhm.grid(True, alpha=0.3)

    ax_snr.set_xlabel("Reflector index")
    ax_snr.set_ylabel("SNR (dB)")
    ax_snr.set_title("SNR vs reflector index")
    ax_snr.legend(fontsize=fontsize_legend)
    ax_snr.grid(True, alpha=0.3)

    summary_fig_path = run_out_dir / "resolution_and_snr_summary.png"
    summary_fig.savefig(summary_fig_path, dpi=150)
    plt.close(summary_fig)

print("Evaluate complete. Plot saved to:", plot_path)
if profiles_enabled:
    print("Per-reflector profiles saved in:", profiles_dir)
    print("FWHM + SNR summary saved to:", fwhm_csv_path)
    print("Resolution and SNR summary plot saved to:", summary_fig_path)
