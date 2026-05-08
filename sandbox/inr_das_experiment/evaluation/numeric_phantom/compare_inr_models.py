"""
Compare two INR models on numeric phantom delayed samples using FWHM and SNR.

This script reads configs/numeric_phantom_evaluation_config.yml, loads the
latest delayed-samples artifact from io.simulation_output_root, reconstructs
two INR DAS images, computes per-reflector FWHM and SNR, and saves only the
summary comparison figure.

Design decisions:
- Simple sandbox script (no argparse/main).
- Docstrings and comments in English; user-facing messages in Spanish.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from inr_apodizations.config import CONFIGS_DIR, PROJ_ROOT
from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.evaluation import compute_reflector_snr, compute_scatterer_metrics
from inr_apodizations.evaluation.profiles import compute_fwhm_batch, extract_reflector_profiles
from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.modeling.trainer import DasInrTrainer
import inr_apodizations.sandbox_helpers as helpers

plt.ion()


def _load_config(cfg_path: Path) -> dict[str, Any]:
    """Load a YAML configuration file as a Python mapping."""
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_delayed_samples(path: Path) -> np.ndarray:
    """Load delayed samples array.

    Expected shape is (n_examples, n_elements, nz, nx).
    """
    return np.load(path)


def _resolve_path(path_cfg: str) -> Path:
    """Resolve configured path against project root when needed."""
    path = Path(path_cfg)
    if path.is_absolute():
        return path
    return (PROJ_ROOT / path).resolve()


def _resolve_latest_delayed_samples_path(io_cfg: dict[str, Any]) -> tuple[Path, Path]:
    """Resolve delayed-samples artifact from latest simulation metadata.

    Returns:
        Tuple (delayed_samples_file, latest_run_folder).

    Raises:
        FileNotFoundError: If expected folders/files are missing.
        ValueError: If metadata format/content is invalid.
    """
    simulation_root_cfg = io_cfg.get(
        "simulation_output_root",
        "sandbox/inr_das_experiment/outputs/evaluation/numeric_phantom/delayed_samples",
    )
    simulation_root = _resolve_path(str(simulation_root_cfg))

    if not simulation_root.exists() or not simulation_root.is_dir():
        raise FileNotFoundError(
            "No existe io.simulation_output_root o no es carpeta: "
            f"{simulation_root}"
        )

    run_folders = [path for path in simulation_root.iterdir() if path.is_dir()]
    if len(run_folders) == 0:
        raise FileNotFoundError(
            "No se encontraron corridas en io.simulation_output_root: "
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
            "Falta delayed_samples_path valido en simulation_info.yml: "
            f"{simulation_info_path}"
        )

    delayed_samples_path = _resolve_path(delayed_samples_cfg)
    if not delayed_samples_path.exists() or delayed_samples_path.is_dir():
        raise FileNotFoundError(
            "delayed_samples_path en simulation_info.yml no apunta a un archivo valido: "
            f"{delayed_samples_path}"
        )

    return delayed_samples_path, latest_run


def _build_grid_reflector_points(cfg: dict[str, Any]) -> np.ndarray:
    """Build reflector positions (N, 2) in mm from phantom.grid config."""
    phantom_cfg = cfg.get("phantom", {})
    mode = str(phantom_cfg.get("mode", "")).strip().lower()
    if mode != "grid":
        raise ValueError(
            "Este flujo de perfiles por reflector requiere phantom.mode: grid en numeric_phantom_evaluation_config.yml."
        )

    grid_cfg = phantom_cfg.get("grid")
    if not isinstance(grid_cfg, dict):
        raise ValueError("Falta el bloque phantom.grid en numeric_phantom_evaluation_config.yml.")

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
        raise ValueError(f"Faltan campos requeridos en phantom.grid: {missing}")

    x_count = int(grid_cfg["x_count"])
    z_count = int(grid_cfg["z_count"])
    x_center_mm = float(grid_cfg["x_center_mm"])
    z_start_mm = float(grid_cfg["z_start_mm"])
    x_spacing_mm = float(grid_cfg["x_spacing_mm"])
    z_spacing_mm = float(grid_cfg["z_spacing_mm"])

    if x_count <= 0 or z_count <= 0:
        raise ValueError("x_count y z_count deben ser enteros positivos en phantom.grid.")
    if x_spacing_mm <= 0.0 or z_spacing_mm <= 0.0:
        raise ValueError("x_spacing_mm y z_spacing_mm deben ser > 0 en phantom.grid.")

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
    """Resolve user-selected reflector indices from YAML."""
    if isinstance(indices_cfg, str) and indices_cfg.strip().lower() == "all":
        return np.arange(n_reflectors, dtype=np.int32)
    if indices_cfg is None:
        return np.arange(n_reflectors, dtype=np.int32)
    if not isinstance(indices_cfg, list) or len(indices_cfg) == 0:
        raise ValueError(
            "reflector_lateral_profiles.reflector_indices debe ser 'all' o una lista no vacia de enteros."
        )

    resolved = np.asarray(indices_cfg, dtype=np.int32)
    if np.any(resolved < 0) or np.any(resolved >= n_reflectors):
        raise ValueError(
            "reflector_lateral_profiles.reflector_indices contiene indices fuera de rango. "
            f"Rango valido: [0, {n_reflectors - 1}]"
        )
    return np.unique(resolved)


def _resolve_model_artifact_path(model_path_cfg: str) -> Path:
    """Resolve a model artifact path from config.

    Accepted inputs:
    - Direct path to a Keras model artifact.
    - Directory containing model.keras.
    - Directory containing one or more .h5 artifacts.
    """
    model_path = _resolve_path(model_path_cfg)
    candidates: list[Path] = []

    if model_path.is_file():
        candidates.append(model_path)
    elif model_path.is_dir():
        candidates.append(model_path / "model.keras")
        candidates.extend(sorted(model_path.glob("*.h5")))
    else:
        # If configured path does not exist, still try common artifact variants.
        candidates.append(model_path)
        candidates.append(model_path / "model.keras")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "No se encontro un artefacto de modelo valido para la ruta configurada: "
        f"{model_path_cfg}. Candidatos revisados: {candidates}"
    )


def _read_scaled_features_flag(model_artifact_path: Path) -> bool:
    """Read scaled_features from train_config_info.yml associated with a model."""
    train_info_path = model_artifact_path.parent / "train_config_info.yml"
    if not train_info_path.exists():
        raise FileNotFoundError(
            "No se encontro train_config_info.yml junto al modelo: "
            f"{train_info_path}"
        )

    with open(train_info_path, encoding="utf-8") as f:
        train_info = yaml.safe_load(f) or {}

    if not isinstance(train_info, dict):
        raise ValueError(f"Formato invalido en {train_info_path}; se esperaba un mapping YAML.")

    if "scaled_features" in train_info:
        return bool(train_info["scaled_features"])

    experiment_cfg = train_info.get("experiment", {})
    model_cfg = experiment_cfg.get("model", {}) if isinstance(experiment_cfg, dict) else {}
    if isinstance(model_cfg, dict) and "scaled_features" in model_cfg:
        return bool(model_cfg["scaled_features"])

    raise ValueError(
        "No se encontro scaled_features en train_config_info.yml. "
        f"Archivo revisado: {train_info_path}"
    )


def _default_model_label(model_path_cfg: str) -> str:
    """Build a display label from configured model path."""
    path = Path(model_path_cfg)
    if path.name:
        return path.name
    return str(path)


def _predict_model_image(
    *,
    model_path_cfg: str,
    cm: CoordinateManager,
    delayed0: np.ndarray,
    feature_chunk_size: int,
) -> tuple[np.ndarray, bool, Path, str]:
    """Load a model and reconstruct its DAS image on delayed samples."""
    model_artifact_path = _resolve_model_artifact_path(model_path_cfg)
    model = tf.keras.models.load_model(str(model_artifact_path))

    # Derive neuron counts for Dense-like layers to build a descriptive label
    neuron_counts: list[int] = []
    for layer in model.layers:
        units = getattr(layer, "units", None)
        if isinstance(units, int):
            neuron_counts.append(int(units))

    # Exclude final output layer if it is a single unit (1)
    if len(neuron_counts) > 0 and neuron_counts[-1] == 1:
        neuron_counts = neuron_counts[:-1]

    if len(neuron_counts) > 0:
        generated_label = "-".join(str(u) for u in neuron_counts)
    else:
        generated_label = model.name

    scaled_features = _read_scaled_features_flag(model_artifact_path)
    features_grid = cm.get_features_grid(scaled=scaled_features)

    trainer = DasInrTrainer(
        apodization_model=model,
        features_grid=features_grid,
        feature_chunk_size=feature_chunk_size,
        weight_regularization_enabled=False,
    )
    weights_grid = trainer.predict_weights_grid(training=False).numpy()

    if weights_grid.shape != delayed0.shape:
        raise ValueError(
            "Shape mismatch entre delayed samples y pesos INR. "
            f"delayed0: {delayed0.shape}, weights_grid: {weights_grid.shape}, "
            f"modelo: {model_artifact_path}"
        )

    image = (delayed0 * weights_grid).sum(axis=0)
    return image, scaled_features, model_artifact_path, generated_label


# ===== Load config =====
script_dir = Path(__file__).resolve().parent
cfg = _load_config(CONFIGS_DIR / "numeric_phantom_evaluation_config.yml")

io_cfg = cfg.get("io", {})
sim_cfg = cfg.get("simulation", {})
bf_cfg = cfg.get("beamforming", {})
plot_cfg = cfg.get("plots", {})
profile_cfg = cfg.get("reflector_lateral_profiles", {})
comparison_cfg = cfg.get("model_comparison", {})

fontsize_legend = int(plot_cfg.get("legend_fontsize", 8))

model_1_path_cfg = comparison_cfg.get("model_1_path")
model_2_path_cfg = comparison_cfg.get("model_2_path")
if not isinstance(model_1_path_cfg, str) or len(model_1_path_cfg.strip()) == 0:
    raise ValueError("Falta model_comparison.model_1_path en numeric_phantom_evaluation_config.yml.")
if not isinstance(model_2_path_cfg, str) or len(model_2_path_cfg.strip()) == 0:
    raise ValueError("Falta model_comparison.model_2_path en numeric_phantom_evaluation_config.yml.")

model_1_label = str(
    comparison_cfg.get("model_1_label", _default_model_label(model_1_path_cfg))
).strip()
model_2_label = str(
    comparison_cfg.get("model_2_label", _default_model_label(model_2_path_cfg))
).strip()
if len(model_1_label) == 0 or len(model_2_label) == 0:
    raise ValueError("model_comparison.model_1_label y model_2_label no pueden ser vacios.")
if model_1_label == model_2_label:
    raise ValueError("model_comparison.model_1_label y model_2_label deben ser distintos.")

# ===== Load delayed samples =====
delayed_samples_path, latest_simulation_run = _resolve_latest_delayed_samples_path(io_cfg)
print("Corrida de simulacion seleccionada:", latest_simulation_run)
print("Usando delayed samples:", delayed_samples_path)

delayed = _load_delayed_samples(delayed_samples_path)
delayed0 = delayed[0]
n_elements = delayed0.shape[0]

# ===== Build coordinate manager =====
dataset_folder = delayed_samples_path.parent
try:
    kp, cm = helpers.build_coordinate_manager(str(dataset_folder))
except FileNotFoundError:
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

# ===== Reconstruct both models =====
feature_chunk_size = int(comparison_cfg.get("feature_chunk_size", 65536))
if feature_chunk_size <= 0:
    raise ValueError("model_comparison.feature_chunk_size debe ser un entero positivo.")

model_1_image, model_1_scaled, model_1_artifact, model_1_generated_label = _predict_model_image(
    model_path_cfg=model_1_path_cfg,
    cm=cm,
    delayed0=delayed0,
    feature_chunk_size=feature_chunk_size,
)
model_2_image, model_2_scaled, model_2_artifact, model_2_generated_label = _predict_model_image(
    model_path_cfg=model_2_path_cfg,
    cm=cm,
    delayed0=delayed0,
    feature_chunk_size=feature_chunk_size,
)

# Overwrite user labels with generated neuron-count labels when available
if model_1_generated_label:
    model_1_label = model_1_generated_label
if model_2_generated_label:
    model_2_label = model_2_generated_label

if model_1_scaled != model_2_scaled:
    raise ValueError(
        "Los modelos no usan el mismo scaled_features; comparacion no valida. "
        f"{model_1_label}: {model_1_scaled}, {model_2_label}: {model_2_scaled}"
    )

images = {
    model_1_label: np.abs(np.asarray(model_1_image)).astype(np.float64),
    model_2_label: np.abs(np.asarray(model_2_image)).astype(np.float64),
}

# ===== Reflector metrics =====
half_width_lateral_mm = float(profile_cfg.get("half_width_lateral_mm", 1.5))
half_width_axial_mm = float(profile_cfg.get("half_width_axial_mm", 1.0))
snr_radius_mm = float(profile_cfg.get("snr_radius_mm", 1.5))

snr_y_lim_cfg = profile_cfg.get("snr_y_lim_db", None)
snr_y_lim_db: tuple[float, float] | None = None
if snr_y_lim_cfg is not None:
    if not isinstance(snr_y_lim_cfg, (list, tuple)) or len(snr_y_lim_cfg) != 2:
        raise ValueError(
            "reflector_lateral_profiles.snr_y_lim_db must be null or a [ymin, ymax] list."
        )
    snr_y_min = float(snr_y_lim_cfg[0])
    snr_y_max = float(snr_y_lim_cfg[1])
    if snr_y_max <= snr_y_min:
        raise ValueError("reflector_lateral_profiles.snr_y_lim_db must satisfy ymax > ymin.")
    snr_y_lim_db = (snr_y_min, snr_y_max)

reflector_points = _build_grid_reflector_points(cfg)
selected_indices = _resolve_reflector_indices(
    profile_cfg.get("reflector_indices", "all"),
    n_reflectors=int(reflector_points.shape[0]),
)
selected_points = reflector_points[selected_indices]

fwhm_lateral_by_model: dict[str, np.ndarray] = {}
snr_db_by_model: dict[str, np.ndarray] = {}

for model_label, image_abs in images.items():
    extracted = extract_reflector_profiles(
        image=image_abs,
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
    # Keep axial FWHM computation for parity with existing workflow, even if only
    # lateral FWHM is shown in the summary panel.
    _ = compute_fwhm_batch(axial_profiles, axial_offsets_mm)
    fwhm_lateral_by_model[model_label] = np.asarray(fwhm_lat["fwhm_mm"], dtype=np.float64)

    scatterer_metrics = compute_scatterer_metrics(
        image=image_abs,
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
    snr_db_by_model[model_label] = np.asarray(
        compute_reflector_snr(peak_amplitudes, background_rms, return_db=True),
        dtype=np.float64,
    )

# ===== Plot summary figure only =====
summary_fig, (ax_fwhm, ax_snr) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
refl_ids_ordered = selected_indices.tolist()

for model_label in (model_1_label, model_2_label):
    ax_fwhm.plot(
        refl_ids_ordered,
        fwhm_lateral_by_model[model_label].tolist(),
        marker="o",
        linewidth=2,
        label=model_label,
    )
    ax_snr.plot(
        refl_ids_ordered,
        snr_db_by_model[model_label].tolist(),
        marker="o",
        linewidth=2,
        label=model_label,
    )

ax_fwhm.set_xlabel("Reflector index")
ax_fwhm.set_ylabel("FWHM lateral (mm)")
ax_fwhm.set_title("Lateral resolution vs reflector index")
ax_fwhm.legend(fontsize=fontsize_legend)
ax_fwhm.grid(True, alpha=0.3)

ax_snr.set_xlabel("Reflector index")
ax_snr.set_ylabel("SNR (dB)")
ax_snr.set_title("SNR vs reflector index")
if snr_y_lim_db is not None:
    ax_snr.set_ylim(*snr_y_lim_db)
ax_snr.legend(fontsize=fontsize_legend)
ax_snr.grid(True, alpha=0.3)

# ===== Persist outputs =====
out_root = Path(io_cfg.get("evaluation_output_root", script_dir / "outputs"))
out_root = _resolve_path(str(out_root))
out_root.mkdir(parents=True, exist_ok=True)

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_out_dir = out_root / run_timestamp
run_out_dir.mkdir(parents=True, exist_ok=False)

comparison_info = {
    "model_1_label": model_1_label,
    "model_1_artifact": str(model_1_artifact),
    "model_1_scaled_features": bool(model_1_scaled),
    "model_2_label": model_2_label,
    "model_2_artifact": str(model_2_artifact),
    "model_2_scaled_features": bool(model_2_scaled),
    "delayed_samples_path": str(delayed_samples_path),
    "latest_simulation_run": str(latest_simulation_run),
}

with (run_out_dir / "numeric_phantom_evaluation_config_info.yml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
with (run_out_dir / "model_comparison_info.yml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(comparison_info, f, sort_keys=False)

summary_fig_path = run_out_dir / "model_comparison_fwhm_snr_summary.png"
summary_fig.savefig(summary_fig_path, dpi=150)
plt.close(summary_fig)

print("Comparacion completa.")
print("Modelo 1:", model_1_artifact)
print("Modelo 2:", model_2_artifact)
print("Figura de resumen guardada en:", summary_fig_path)
