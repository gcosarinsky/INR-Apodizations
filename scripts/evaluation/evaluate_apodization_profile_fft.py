"""Evaluate FFT magnitude of apodization profiles (INR vs baselines).

This script compares per-element apodization profiles at explicit
``(x_mm, z_mm)`` points and evaluates their spectral magnitude using FFT.

Workflow:
1. Load runtime configuration from YAML.
2. Build ``CoordinateManager`` from a delayed-samples dataset folder.
3. Load a trained INR checkpoint and infer the INR apodization map ``(E, Z, X)``.
4. Compute baseline apodizations on the same grid (Hanning, and optional Uniform).
5. Extract per-element profiles for all requested points, compute FFT magnitudes,
    and compare INR against selected baselines.
6. Save per-point figures, a summary figure, and an ``.npz`` bundle with raw data.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml

from inr_apodizations.apodizations import compute_dynamic_apodizations_tf, extract_profile_for_z
from inr_apodizations.config import CONFIGS_DIR, PROJ_ROOT
from inr_apodizations.experiment_helpers import build_coordinate_manager
from inr_apodizations.utils import to_db


CONFIG_PATH = CONFIGS_DIR / "apodization_fft_profiles.yml"


def load_yaml_config(config_path: Path) -> dict:
    """Load and validate YAML config for apodization-profile FFT evaluation.

    Args:
        config_path: Absolute path to YAML config.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required keys are missing or malformed.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    if not isinstance(cfg, dict):
        raise ValueError("YAML config must be a dictionary.")

    required_top = ["io", "model", "analysis", "fft", "plots"]
    missing_top = [key for key in required_top if key not in cfg]
    if missing_top:
        raise ValueError(f"Missing top-level YAML keys: {missing_top}")

    io_required = ["dataset_folder", "model_path", "output_root"]
    io_missing = [key for key in io_required if key not in cfg["io"]]
    if io_missing:
        raise ValueError(f"Missing io keys: {io_missing}")

    analysis_cfg = cfg.get("analysis", {})
    points = analysis_cfg.get("points_xz_mm")
    if not isinstance(points, list) or len(points) == 0:
        raise ValueError("analysis.points_xz_mm must be a non-empty list of [x_mm, z_mm].")

    for idx, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"analysis.points_xz_mm[{idx}] must be [x_mm, z_mm].")

    n_fft = int(cfg.get("fft", {}).get("n_fft", 0))
    if n_fft <= 0:
        raise ValueError("fft.n_fft must be a positive integer.")

    return cfg


def resolve_project_path(path_like: str | Path) -> Path:
    """Resolve a path relative to project root when needed."""
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJ_ROOT / path
    return path


def resolve_model_file(model_path_cfg: str | Path) -> Path:
    """Resolve an INR model artifact file from a path or directory.

    Args:
        model_path_cfg: File path (``.keras``/``.h5``) or directory that contains one.

    Returns:
        Absolute path to the model file.

    Raises:
        FileNotFoundError: If no model file can be resolved.
    """
    model_path = resolve_project_path(model_path_cfg)
    if model_path.is_file():
        return model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    candidate_keras = model_path / "model.keras"
    if candidate_keras.exists():
        return candidate_keras

    h5_candidates = sorted(model_path.glob("*.h5"))
    if h5_candidates:
        return h5_candidates[0]

    raise FileNotFoundError(
        f"No model artifact found in {model_path}. Expected model.keras or at least one .h5 file."
    )


def load_train_info(model_file: Path) -> dict:
    """Load ``train_config_info.yml`` next to the INR model when available."""
    train_info_path = model_file.parent / "train_config_info.yml"
    if not train_info_path.exists():
        return {}

    with train_info_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        return {}
    return payload


def resolve_model_feature_settings(cfg: dict, train_info: dict) -> tuple[bool, str, list[str] | None, int]:
    """Resolve model-related feature settings with config override precedence.

    Returns:
        Tuple ``(scaled_features, physical_feature_set, physical_feature_components, feature_chunk_size)``.
    """
    model_cfg = cfg.get("model", {})

    scaled_override = model_cfg.get("scaled_features")
    if scaled_override is not None:
        scaled_features = bool(scaled_override)
    else:
        scaled_features = bool(
            train_info.get("experiment", {}).get("model", {}).get("scaled_features", False)
        )

    physical_feature_set = str(
        model_cfg.get(
            "physical_feature_set",
            train_info.get("experiment", {}).get("model", {}).get(
                "physical_feature_set", "distance_depth_edge"
            ),
        )
    )

    components_cfg = model_cfg.get("physical_feature_components")
    if isinstance(components_cfg, list):
        physical_feature_components = [str(token) for token in components_cfg]
    else:
        train_components = train_info.get("experiment", {}).get("model", {}).get(
            "physical_feature_components"
        )
        physical_feature_components = (
            [str(token) for token in train_components]
            if isinstance(train_components, list)
            else None
        )

    feature_chunk_size = int(model_cfg.get("feature_chunk_size", 65536))
    if feature_chunk_size <= 0:
        raise ValueError("model.feature_chunk_size must be > 0")

    return scaled_features, physical_feature_set, physical_feature_components, feature_chunk_size


def infer_inr_apodization_map(
    model: tf.keras.Model,
    features_grid: tf.Tensor,
    feature_chunk_size: int,
) -> tf.Tensor:
    """Infer INR apodization map with chunked forward passes.

    Args:
        model: Loaded INR model mapping features -> scalar weights.
        features_grid: Tensor with shape ``(E, Z, X, F)``.
        feature_chunk_size: Number of feature rows processed per chunk.

    Returns:
        Tensor with shape ``(E, Z, X)`` and dtype ``tf.float32``.
    """
    if features_grid.shape.rank != 4:
        raise ValueError(f"Expected features_grid rank 4, got shape {features_grid.shape}")

    n_elem, nz, nx, n_feat = [int(dim) for dim in features_grid.shape]
    features_flat = tf.reshape(tf.cast(features_grid, tf.float32), (-1, n_feat))

    n_rows = int(features_flat.shape[0])
    predictions: list[tf.Tensor] = []
    for start_idx in range(0, n_rows, feature_chunk_size):
        end_idx = min(start_idx + feature_chunk_size, n_rows)
        chunk_features = features_flat[start_idx:end_idx]
        chunk_pred = model(chunk_features, training=False)
        predictions.append(tf.reshape(tf.cast(chunk_pred, tf.float32), (-1,)))

    weights_flat = tf.concat(predictions, axis=0)
    expected_rows = n_elem * nz * nx
    if int(weights_flat.shape[0]) != expected_rows:
        raise ValueError(
            "INR model output shape mismatch. "
            f"Expected {expected_rows} scalar weights, got {int(weights_flat.shape[0])}."
        )

    return tf.reshape(weights_flat, (n_elem, nz, nx))


def select_window(profile_size: int, window_name: str) -> np.ndarray:
    """Create a 1D window for FFT preprocessing."""
    name = str(window_name).strip().lower()
    if name in ("none", "rect", "rectangular"):
        return np.ones(profile_size, dtype=np.float32)
    if name in ("hann", "hanning"):
        return np.hanning(profile_size).astype(np.float32)
    if name == "hamming":
        return np.hamming(profile_size).astype(np.float32)
    raise ValueError(f"Unsupported fft.window value: {window_name}")


def compute_fft_abs(profile: np.ndarray, n_fft: int) -> np.ndarray:
    """Compute one-sided FFT magnitude for a real profile."""
    return np.abs(np.fft.rfft(profile, n=n_fft)).astype(np.float32)


def find_nearest_index_and_value(values: np.ndarray, query: float) -> tuple[int, float]:
    """Find nearest index and coordinate value for a query scalar."""
    idx = int(np.argmin(np.abs(values - float(query))))
    return idx, float(values[idx])


def build_fft_db_many(
    fft_abs_curves: list[np.ndarray],
    mode: str,
    eps: float,
) -> list[np.ndarray]:
    """Convert multiple FFT magnitude curves to dB using selected reference policy."""
    mode_norm = str(mode).strip().lower()
    if len(fft_abs_curves) == 0:
        raise ValueError("At least one FFT curve is required for dB conversion.")

    if mode_norm == "pair_peak":
        reference = max(*[float(np.max(curve)) for curve in fft_abs_curves], eps)
        return [to_db(curve, ref=reference, eps=eps) for curve in fft_abs_curves]

    if mode_norm != "self_peak":
        raise ValueError("fft.db_reference must be 'self_peak' or 'pair_peak'.")

    output_curves: list[np.ndarray] = []
    for curve in fft_abs_curves:
        curve_ref = max(float(np.max(curve)), eps)
        output_curves.append(to_db(curve, ref=curve_ref, eps=eps))
    return output_curves


def save_point_figure(
    output_path: Path,
    element_x_mm: np.ndarray,
    freq_cpe: np.ndarray,
    inr_profile: np.ndarray,
    hanning_profile: np.ndarray,
    uniform_profile: np.ndarray | None,
    inr_fft_plot: np.ndarray,
    hanning_fft_plot: np.ndarray,
    uniform_fft_plot: np.ndarray | None,
    use_db: bool,
    db_min: float,
    freq_axis_max: float,
    point_label: str,
    fft_ylabel: str,
    dpi: int,
) -> None:
    """Save profile + FFT comparison for one selected point."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(element_x_mm, hanning_profile, label="Hanning", linewidth=2)
    if uniform_profile is not None:
        axes[0].plot(element_x_mm, uniform_profile, label="Uniform", linewidth=2)
    axes[0].plot(element_x_mm, inr_profile, label="INR", linewidth=2)
    axes[0].set_xlabel("Element lateral coordinate (mm)")
    axes[0].set_ylabel("Apodization weight")
    axes[0].set_title(f"Apodization profile | {point_label}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(freq_cpe, hanning_fft_plot, label="Hanning", linewidth=2)
    if uniform_fft_plot is not None:
        axes[1].plot(freq_cpe, uniform_fft_plot, label="Uniform", linewidth=2)
    axes[1].plot(freq_cpe, inr_fft_plot, label="INR", linewidth=2)
    axes[1].set_xlabel("Spatial frequency (cycles/element)")
    axes[1].set_ylabel(fft_ylabel)
    axes[1].set_title(f"FFT magnitude | {point_label}")
    axes[1].set_xlim(0.0, freq_axis_max)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    if use_db:
        axes[1].set_ylim(db_min, 5.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_summary_figure(
    output_path: Path,
    point_labels: list[str],
    freq_cpe: np.ndarray,
    inr_fft_matrix_plot: np.ndarray,
    hanning_fft_matrix_plot: np.ndarray,
    uniform_fft_matrix_plot: np.ndarray | None,
    use_db: bool,
    freq_axis_max: float,
    db_min: float,
    fft_ylabel: str,
    dpi: int,
) -> None:
    """Save a multi-row summary figure with one FFT panel per requested point."""
    n_points = len(point_labels)
    fig, axes = plt.subplots(n_points, 1, figsize=(10, max(3.2 * n_points, 4.0)), squeeze=False)

    for point_idx, label in enumerate(point_labels):
        ax = axes[point_idx, 0]
        ax.plot(freq_cpe, hanning_fft_matrix_plot[point_idx], label="Hanning", linewidth=2)
        if uniform_fft_matrix_plot is not None:
            ax.plot(freq_cpe, uniform_fft_matrix_plot[point_idx], label="Uniform", linewidth=2)
        ax.plot(freq_cpe, inr_fft_matrix_plot[point_idx], label="INR", linewidth=2)
        ax.set_title(f"FFT magnitude comparison | {label}")
        ax.set_xlabel("Spatial frequency (cycles/element)")
        ax.set_ylabel(fft_ylabel)
        ax.set_xlim(0.0, freq_axis_max)
        ax.set_ylim(db_min, 0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        if use_db:
            ax.set_ylim(db_min, 5.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run FFT comparison for INR and selected baseline apodization profiles."""
    cfg = load_yaml_config(CONFIG_PATH)

    dataset_folder = resolve_project_path(cfg["io"]["dataset_folder"])
    if not dataset_folder.exists():
        raise FileNotFoundError(f"Configured dataset folder does not exist: {dataset_folder}")

    model_file = resolve_model_file(cfg["io"]["model_path"])
    train_info = load_train_info(model_file)
    scaled_features, physical_feature_set, physical_feature_components, feature_chunk_size = resolve_model_feature_settings(
        cfg, train_info
    )

    kp, cm = build_coordinate_manager(
        dataset_folder=str(dataset_folder),
        physical_feature_set=physical_feature_set,
        physical_feature_components=physical_feature_components,
    )

    print(f"Using YAML config: {CONFIG_PATH}")
    print(f"Dataset folder: {dataset_folder}")
    print(f"Model file: {model_file}")
    print(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}")
    print(
        "Feature settings: "
        f"scaled_features={scaled_features}, "
        f"physical_feature_set={physical_feature_set}, "
        f"physical_feature_components={physical_feature_components}, "
        f"feature_chunk_size={feature_chunk_size}"
    )

    inr_model = tf.keras.models.load_model(str(model_file))
    features_grid = cm.get_features_grid(scaled=scaled_features)
    inr_weights = infer_inr_apodization_map(
        model=inr_model,
        features_grid=features_grid,
        feature_chunk_size=feature_chunk_size,
    )

    baseline_f_number_cfg = cfg.get("analysis", {}).get("baseline_f_number", None)
    if baseline_f_number_cfg is None:
        baseline_f_number = float(kp.f_number)
    else:
        baseline_f_number = float(baseline_f_number_cfg)

    include_uniform = bool(cfg.get("analysis", {}).get("include_uniform", False))

    hanning_tensor = compute_dynamic_apodizations_tf(
        cm=cm,
        f_number=baseline_f_number,
        methods=("hanning",),
        scaled=scaled_features,
    ).get("hanning")
    if hanning_tensor is None:
        raise ValueError("Could not compute Hanning apodization tensor.")

    uniform_tensor = tf.ones_like(hanning_tensor) if include_uniform else None

    points_requested = np.asarray(cfg["analysis"]["points_xz_mm"], dtype=np.float32)
    n_points = int(points_requested.shape[0])

    coords = cm.get_coordinates_1d(scaled=False)
    z_coords_mm = np.asarray(coords["z"], dtype=np.float32)
    x_coords_mm = np.asarray(coords["x"], dtype=np.float32)
    element_x_mm = np.asarray(coords["x_elem"], dtype=np.float32)

    n_fft = int(cfg["fft"]["n_fft"])
    fft_window_name = str(cfg["fft"].get("window", "none"))
    use_fft_db = bool(cfg["fft"].get("use_db", True))
    fft_db_reference = str(cfg["fft"].get("db_reference", "pair_peak"))
    fft_db_eps = float(cfg["fft"].get("db_eps", 1e-8))
    db_min = float(cfg["fft"].get("db_min", -60.0))

    if n_fft < int(element_x_mm.shape[0]):
        raise ValueError(
            f"fft.n_fft ({n_fft}) must be >= number of elements ({int(element_x_mm.shape[0])})."
        )

    window_vector = select_window(profile_size=int(element_x_mm.shape[0]), window_name=fft_window_name)

    fft_freq_cpe = np.fft.rfftfreq(n_fft, d=1.0).astype(np.float32)
    fft_freq_cpm = np.fft.rfftfreq(n_fft, d=float(kp.pitch)).astype(np.float32)
    freq_axis_max_cfg = cfg["fft"].get("freq_axis_max", None)
    if freq_axis_max_cfg is None:
        freq_axis_max = float(fft_freq_cpe[-1])
    else:
        freq_axis_max = min(float(freq_axis_max_cfg), float(fft_freq_cpe[-1]))
        if freq_axis_max <= 0.0:
            raise ValueError("fft.freq_axis_max must be > 0 when provided.")

    profiles_inr = np.zeros((n_points, element_x_mm.size), dtype=np.float32)
    profiles_hanning = np.zeros_like(profiles_inr)
    profiles_uniform = np.zeros_like(profiles_inr) if include_uniform else None
    selected_points_mm = np.zeros((n_points, 2), dtype=np.float32)
    selected_indices = np.zeros((n_points, 2), dtype=np.int32)

    fft_abs_inr = np.zeros((n_points, fft_freq_cpe.size), dtype=np.float32)
    fft_abs_hanning = np.zeros_like(fft_abs_inr)
    fft_abs_uniform = np.zeros_like(fft_abs_inr) if include_uniform else None

    point_labels: list[str] = []

    for point_idx, (x_req_mm, z_req_mm) in enumerate(points_requested):
        x_idx, x_sel_mm = find_nearest_index_and_value(x_coords_mm, float(x_req_mm))
        z_idx, z_sel_mm = find_nearest_index_and_value(z_coords_mm, float(z_req_mm))

        selected_points_mm[point_idx, :] = [x_sel_mm, z_sel_mm]
        selected_indices[point_idx, :] = [x_idx, z_idx]

        inr_profile = extract_profile_for_z(
            inr_weights,
            cm,
            z_fixed=float(z_req_mm),
            x_fixed=float(x_req_mm),
            scaled=False,
        ).numpy()
        hanning_profile = extract_profile_for_z(
            hanning_tensor,
            cm,
            z_fixed=float(z_req_mm),
            x_fixed=float(x_req_mm),
            scaled=False,
        ).numpy()
        uniform_profile = None
        if uniform_tensor is not None:
            uniform_profile = extract_profile_for_z(
                uniform_tensor,
                cm,
                z_fixed=float(z_req_mm),
                x_fixed=float(x_req_mm),
                scaled=False,
            ).numpy()

        inr_profile = np.asarray(inr_profile, dtype=np.float32)
        hanning_profile = np.asarray(hanning_profile, dtype=np.float32)
        if uniform_profile is not None:
            uniform_profile = np.asarray(uniform_profile, dtype=np.float32)

        profiles_inr[point_idx, :] = inr_profile
        profiles_hanning[point_idx, :] = hanning_profile
        if profiles_uniform is not None and uniform_profile is not None:
            profiles_uniform[point_idx, :] = uniform_profile

        inr_profile_windowed = inr_profile * window_vector
        hanning_profile_windowed = hanning_profile * window_vector
        uniform_profile_windowed = uniform_profile * window_vector if uniform_profile is not None else None

        fft_inr_abs = compute_fft_abs(inr_profile_windowed, n_fft=n_fft)
        fft_hanning_abs = compute_fft_abs(hanning_profile_windowed, n_fft=n_fft)
        fft_uniform_abs = (
            compute_fft_abs(uniform_profile_windowed, n_fft=n_fft)
            if uniform_profile_windowed is not None
            else None
        )

        fft_abs_inr[point_idx, :] = fft_inr_abs
        fft_abs_hanning[point_idx, :] = fft_hanning_abs
        if fft_abs_uniform is not None and fft_uniform_abs is not None:
            fft_abs_uniform[point_idx, :] = fft_uniform_abs

        point_labels.append(f"x = {x_sel_mm:.2f} mm,  z = {z_sel_mm:.2f} mm")

    if use_fft_db:
        fft_plot_inr = np.zeros_like(fft_abs_inr)
        fft_plot_hanning = np.zeros_like(fft_abs_hanning)
        fft_plot_uniform = np.zeros_like(fft_abs_uniform) if fft_abs_uniform is not None else None
        for point_idx in range(n_points):
            fft_curves = [fft_abs_inr[point_idx], fft_abs_hanning[point_idx]]
            if fft_abs_uniform is not None:
                fft_curves.append(fft_abs_uniform[point_idx])

            db_curves = build_fft_db_many(fft_curves, mode=fft_db_reference, eps=fft_db_eps)
            fft_plot_inr[point_idx, :] = db_curves[0].astype(np.float32)
            fft_plot_hanning[point_idx, :] = db_curves[1].astype(np.float32)
            if fft_plot_uniform is not None and len(db_curves) > 2:
                fft_plot_uniform[point_idx, :] = db_curves[2].astype(np.float32)
        fft_ylabel = "|FFT| (dB)"
    else:
        fft_plot_inr = fft_abs_inr.copy()
        fft_plot_hanning = fft_abs_hanning.copy()
        fft_plot_uniform = fft_abs_uniform.copy() if fft_abs_uniform is not None else None
        fft_ylabel = "|FFT| (linear)"

    output_root = resolve_project_path(cfg["io"]["output_root"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    dpi = int(cfg.get("plots", {}).get("dpi", 150))

    for point_idx in range(n_points):
        point_file = output_dir / f"fft_profile_point_{point_idx:02d}.png"
        save_point_figure(
            output_path=point_file,
            element_x_mm=element_x_mm,
            freq_cpe=fft_freq_cpe,
            inr_profile=profiles_inr[point_idx],
            hanning_profile=profiles_hanning[point_idx],
            uniform_profile=(profiles_uniform[point_idx] if profiles_uniform is not None else None),
            inr_fft_plot=fft_plot_inr[point_idx],
            hanning_fft_plot=fft_plot_hanning[point_idx],
            uniform_fft_plot=(fft_plot_uniform[point_idx] if fft_plot_uniform is not None else None),
            use_db=use_fft_db,
            db_min=db_min,
            freq_axis_max=freq_axis_max,
            point_label=point_labels[point_idx],
            fft_ylabel=fft_ylabel,
            dpi=dpi,
        )

    save_summary_figure(
        output_path=output_dir / "fft_profile_summary.png",
        point_labels=point_labels,
        freq_cpe=fft_freq_cpe,
        inr_fft_matrix_plot=fft_plot_inr,
        hanning_fft_matrix_plot=fft_plot_hanning,
        uniform_fft_matrix_plot=fft_plot_uniform,
        use_db=use_fft_db,
        freq_axis_max=freq_axis_max,
        db_min=db_min,
        fft_ylabel=fft_ylabel,
        dpi=dpi,
    )

    metadata = {
        "config_path": str(CONFIG_PATH),
        "dataset_folder": str(dataset_folder),
        "model_file": str(model_file),
        "scaled_features": scaled_features,
        "physical_feature_set": physical_feature_set,
        "feature_chunk_size": feature_chunk_size,
        "baseline_f_number": baseline_f_number,
        "include_uniform": include_uniform,
        "fft": {
            "n_fft": n_fft,
            "window": fft_window_name,
            "use_db": use_fft_db,
            "db_reference": fft_db_reference,
            "db_eps": fft_db_eps,
            "db_min": db_min,
            "freq_axis_max": freq_axis_max,
        },
        "points_requested_mm": points_requested.tolist(),
        "points_selected_mm": selected_points_mm.tolist(),
        "selected_indices_xz": selected_indices.tolist(),
        "timestamp": timestamp,
    }

    np.savez(
        output_dir / "fft_profile_data.npz",
        **{
            "element_x_mm": element_x_mm,
            "freq_cycles_per_element": fft_freq_cpe,
            "freq_cycles_per_mm": fft_freq_cpm,
            "points_requested_mm": points_requested,
            "points_selected_mm": selected_points_mm,
            "selected_indices_xz": selected_indices,
            "profile_inr": profiles_inr,
            "profile_hanning": profiles_hanning,
            "fft_abs_inr": fft_abs_inr,
            "fft_abs_hanning": fft_abs_hanning,
            "fft_plot_inr": fft_plot_inr,
            "fft_plot_hanning": fft_plot_hanning,
            "metadata_json": json.dumps(metadata, indent=2),
            **(
                {
                    "profile_uniform": profiles_uniform,
                    "fft_abs_uniform": fft_abs_uniform,
                    "fft_plot_uniform": fft_plot_uniform,
                }
                if include_uniform and profiles_uniform is not None and fft_abs_uniform is not None
                else {}
            ),
        },
    )

    with (output_dir / "run_info.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved FFT profile evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
