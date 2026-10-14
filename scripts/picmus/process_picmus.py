from datetime import datetime
from pathlib import Path

from inr_apodizations.config import CONFIGS_DIR, PROJ_ROOT
from inr_apodizations.apodizations import compute_nsi_from_delayed_samples_numpy
from inr_apodizations.coordinate_manager import CoordinateManager
from inr_apodizations.evaluation.io_utils import (
    extract_mixer_train_metadata,
    load_config_yaml,
    resolve_mixer_artifacts,
)
from inr_apodizations.evaluation.profiles import extract_reflector_profiles
from inr_apodizations.modeling.das_models import DasInrApodMixer
from inr_apodizations.picmus import (
    build_coordinate_manager,
    build_kernel_parameters,
    build_picmus_kernel_config,
    compute_hanning_das_image_chunked,
    compute_inr_das_image_chunked,
    compute_delayed_samples,
    compute_uniform_das_image,
    load_inr_model,
    load_picmus_hdf5,
    load_picmus_pipeline_config,
    select_angle_subset,
)
from inr_apodizations.utils import to_db

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cupy as cp
import numpy as np
import tensorflow as tf
plt.ion()


def _subplot_grid(n_items: int) -> tuple[int, int]:
    """Return a compact subplot grid for ``n_items`` images.

    Examples:
        4 -> (2, 2)
        5 -> (2, 3)
    """
    cols = int(np.ceil(np.sqrt(max(1, n_items))))
    rows = int(np.ceil(n_items / cols))
    return rows, cols


def _resolve_display_path(path_value: str | Path) -> Path:
    """Resolve a path for display in logs and figure metadata."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJ_ROOT / path).resolve()


def _load_manual_scatterers_mm(io_cfg: dict) -> np.ndarray:
    """Load optional manual scatterers from YAML io config and convert to mm."""

    def _parse_pairs(values, field_name: str, to_mm_scale: float) -> np.ndarray:
        if values is None:
            return np.empty((0, 2), dtype=np.float32)

        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == 2:
            arr = arr.reshape(1, 2)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"io.{field_name} must be [x, z] or a list of [x, z] pairs, got shape {arr.shape}"
            )

        return (arr * np.float32(to_mm_scale)).astype(np.float32, copy=False)

    manual_mm = _parse_pairs(io_cfg.get("manual_scatterers_mm"), "manual_scatterers_mm", 1.0)
    manual_m = _parse_pairs(io_cfg.get("manual_scatterers_m"), "manual_scatterers_m", 1000.0)

    if manual_mm.size == 0 and manual_m.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if manual_mm.size == 0:
        return manual_m
    if manual_m.size == 0:
        return manual_mm

    return np.concatenate([manual_mm, manual_m], axis=0)


def _profile_to_db(profile: np.ndarray, floor_db: float = -40.0) -> np.ndarray:
    """Convert one 1D profile to dB with robust local normalization."""
    prof = np.asarray(profile, dtype=np.float32)
    valid = np.isfinite(prof)
    if not np.any(valid):
        return np.full_like(prof, floor_db, dtype=np.float32)

    ref = float(np.nanmax(np.abs(prof[valid])))
    if ref <= 0.0:
        return np.full_like(prof, floor_db, dtype=np.float32)

    prof_db = to_db(np.abs(prof), ref=ref)
    prof_db = np.where(np.isfinite(prof_db), prof_db, floor_db)
    return np.maximum(prof_db, floor_db).astype(np.float32, copy=False)


def _plot_profile_pages(
    profile_metrics: dict[str, dict],
    scatterers_mm: np.ndarray,
    axis_kind: str,
    output_dir: Path,
    ts: str,
    max_scatterers_per_page: int = 12,
    separate_figures: bool = False,
) -> list[Path]:
    """Plot reflector profiles for lateral or axial direction.

    Args:
        profile_metrics: Extracted profiles keyed by method name.
        scatterers_mm: Reflector coordinates in millimeters.
        axis_kind: Profile direction, either ``"lateral"`` or ``"axial"``.
        output_dir: Directory where figures are saved.
        ts: Timestamp used in output filenames.
        max_scatterers_per_page: Maximum profiles in a combined figure.
        separate_figures: Whether to save one figure per scatterer.
    """
    if axis_kind not in {"lateral", "axial"}:
        raise ValueError(f"axis_kind must be 'lateral' or 'axial', got {axis_kind}")
    if max_scatterers_per_page <= 0:
        raise ValueError("max_scatterers_per_page must be a positive integer.")

    method_names = list(profile_metrics.keys())
    if not method_names:
        return []

    key_profiles = f"{axis_kind}_profiles"
    key_offsets = f"{axis_kind}_offsets_mm"
    offsets_mm = np.asarray(profile_metrics[method_names[0]][key_offsets], dtype=np.float32)
    n_scatterers = int(scatterers_mm.shape[0])

    if n_scatterers == 0:
        return []

    scatterers_per_figure = 1 if separate_figures else max_scatterers_per_page
    n_pages = int(np.ceil(n_scatterers / scatterers_per_figure))
    saved_paths: list[Path] = []

    for page_idx in range(n_pages):
        start = page_idx * scatterers_per_figure
        end = min(start + scatterers_per_figure, n_scatterers)
        n_this_page = end - start
        rows, cols = _subplot_grid(n_this_page)

        fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.6 * rows), squeeze=False)
        axes_flat = axes.ravel()

        for local_idx, scatterer_idx in enumerate(range(start, end)):
            ax = axes_flat[local_idx]
            x_mm, z_mm = scatterers_mm[scatterer_idx]
            for method_name in method_names:
                profiles_arr = np.asarray(profile_metrics[method_name][key_profiles])
                profile_db = _profile_to_db(profiles_arr[scatterer_idx])
                display_name = "Proposed method" if method_name == "Mixer" else method_name
                ax.plot(offsets_mm, profile_db, label=display_name, linewidth=1.3)

            ax.set_title(f"Pt {scatterer_idx}: x={x_mm:.2f} mm, z={z_mm:.2f} mm")
            ax.set_xlabel(f"{axis_kind.capitalize()} offset [mm]")
            ax.set_ylabel("Amplitude [dB]")
            ax.set_ylim(-40.0, 1.0)
            ax.grid(True, alpha=0.25)
            if local_idx == 0:
                ax.legend(loc="lower left", fontsize=8)

        for local_idx in range(n_this_page, len(axes_flat)):
            axes_flat[local_idx].axis("off")

        fig.suptitle(
            f"PICMUS {axis_kind.capitalize()} profiles (scatterers {start}-{end - 1})",
            fontsize=12,
        )
        fig.tight_layout()

        output_path = output_dir / f"picmus_{axis_kind}_profiles_{ts}_p{page_idx + 1:02d}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def _overlay_profile_windows_on_uniform(
    ax,
    scatterers_mm: np.ndarray,
    cm: CoordinateManager,
    profile_metrics_uniform: dict,
) -> None:
    """Overlay extraction windows used for reflector profiles on Uniform DAS."""
    coords = cm.get_coordinates_1d(scaled=False)
    x_coords = np.asarray(coords["x"], dtype=np.float32)
    z_coords = np.asarray(coords["z"], dtype=np.float32)

    if x_coords.size < 2 or z_coords.size < 2:
        return

    dx = float(np.abs(x_coords[1] - x_coords[0]))
    dz = float(np.abs(z_coords[1] - z_coords[0]))
    half_x = int(profile_metrics_uniform["half_x_px"])
    half_z = int(profile_metrics_uniform["half_z_px"])

    width_mm = 2.0 * half_x * dx
    height_mm = 2.0 * half_z * dz

    for x_mm, z_mm in np.asarray(scatterers_mm, dtype=np.float32):
        ix = int(np.argmin(np.abs(x_coords - x_mm)))
        iz = int(np.argmin(np.abs(z_coords - z_mm)))
        x_center = float(x_coords[ix])
        z_center = float(z_coords[iz])

        rect = Rectangle(
            (x_center - width_mm / 2.0, z_center - height_mm / 2.0),
            width_mm,
            height_mm,
            fill=False,
            edgecolor="tab:cyan",
            linewidth=0.8,
            alpha=0.5,
        )
        ax.add_patch(rect)

"""Run PICMUS delayed-sample pipeline and compute uniform DAS image.

Args:
    config_path: Path to a PICMUS YAML config.

Returns:
    Tuple ``(das_uniform, delayed_samples, cfg, kp)``.

Raises:
    FileNotFoundError: If config or PICMUS files are missing.
    ValueError: If any configuration or shape validation fails.
"""

config_path = CONFIGS_DIR / "picmus_beamforming.yml"
pipeline_cfg = load_picmus_pipeline_config(config_path)

io_cfg = pipeline_cfg["io"]
picmus_data_path = io_cfg["picmus_data_path"]
print(f"Loading PICMUS data from: {picmus_data_path}")
angles, rf_real, x_axis_mm, z_axis_mm, scatterers_mm = load_picmus_hdf5(
    picmus_data_path=picmus_data_path,
    rf_file=io_cfg["rf_file"],
    scan_file=io_cfg["scan_file"],
    phantom_file=io_cfg["phantom_file"]
)

manual_scatterers_mm = _load_manual_scatterers_mm(io_cfg)
n_phantom_scatterers = int(scatterers_mm.shape[0])
if manual_scatterers_mm.size > 0:
    scatterers_mm = np.concatenate(
        [np.asarray(scatterers_mm, dtype=np.float32), manual_scatterers_mm],
        axis=0,
    )

angles_sel, rf_sel = select_angle_subset(angles, rf_real, pipeline_cfg["angle_subset"])
cfg = build_picmus_kernel_config(pipeline_cfg, angles_sel, rf_sel, x_axis_mm, z_axis_mm)
kp = build_kernel_parameters(cfg)

print("Computing delayed samples...")
delayed_samples = compute_delayed_samples(rf_sel, angles_sel, kp, cfg)
print("Delayed samples computed.")
das_uniform = compute_uniform_das_image(delayed_samples)

# Release cached CuPy GPU memory before running apodization on TensorFlow.
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()

print("PICMUS pipeline completed.")
print(f"Angles used: {cfg['n_angles']}")
print(f"RF shape used: {rf_sel.shape}")
print(f"Delayed samples shape: {delayed_samples.shape}")
print(f"Uniform DAS image shape: {das_uniform.shape}")
print(f"Effective ROI [xmin, xmax, zmin, zmax] (mm): {kp.roi_effective}")
print(f"Loaded phantom scatterers: {n_phantom_scatterers}")
print(f"Loaded manual scatterers from YAML: {manual_scatterers_mm.shape[0]}")
print(f"Total scatterers used: {scatterers_mm.shape[0]}")

output_dir = PROJ_ROOT / Path("scripts/outputs/picmus")
output_dir.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# Compute apodizations if enabled
print("\nProcessing apodizations...")
cm = build_coordinate_manager(kp)
apod_dict = {}  # Store computed apodizations
figure_info_lines = ["Models used:"]

# Hanning apodization
if pipeline_cfg["apodizations"]["hanning"]["enabled"]:
    f_number = float(pipeline_cfg["apodizations"]["hanning"]["f_number"])
    z_chunk_size = int(pipeline_cfg["apodizations"]["hanning"].get("z_chunk_size", 16))
    angle_chunk_size = int(pipeline_cfg["apodizations"]["hanning"].get("angle_chunk_size", 1))
    print(
        "Computing Hanning DAS in chunks "
        f"(f_number={f_number}, z_chunk_size={z_chunk_size}, angle_chunk_size={angle_chunk_size})..."
    )
    das_hanning = compute_hanning_das_image_chunked(
        delayed_samples=delayed_samples,
        cm=cm,
        f_number=f_number,
        z_chunk_size=z_chunk_size,
        angle_chunk_size=angle_chunk_size,
    )
    apod_dict["Hanning"] = das_hanning
    print(f"  Hanning DAS computed: shape={das_hanning.shape}")

# INR apodization
if pipeline_cfg["apodizations"]["inr"]["enabled"]:
    model_path = pipeline_cfg["apodizations"]["inr"]["model_path"]
    inr_model_display_path = _resolve_display_path(model_path)
    feature_batch_size = int(pipeline_cfg["apodizations"]["inr"].get("feature_batch_size", 32768))
    z_chunk_size = int(pipeline_cfg["apodizations"]["inr"].get("z_chunk_size", 16))
    angle_chunk_size = int(pipeline_cfg["apodizations"]["inr"].get("angle_chunk_size", 1))
    print(f"Loading INR model from: {model_path}")
    try:
        inr_model = load_inr_model(model_path)
        figure_info_lines.append(f"INR path: {inr_model_display_path}")
        print(
            "Computing INR DAS in chunks "
            f"(feature_batch_size={feature_batch_size}, z_chunk_size={z_chunk_size}, "
            f"angle_chunk_size={angle_chunk_size})..."
        )
        das_inr = compute_inr_das_image_chunked(
            delayed_samples=delayed_samples,
            model=inr_model,
            cm=cm,
            feature_batch_size=feature_batch_size,
            z_chunk_size=z_chunk_size,
            angle_chunk_size=angle_chunk_size,
            scaled_features=True,
        )
        apod_dict["INR"] = das_inr
        print(f"  INR DAS computed: shape={das_inr.shape}")
    except FileNotFoundError as e:
        figure_info_lines.append(f"INR path: unavailable ({inr_model_display_path})")
        print(f"  Warning: {e}")

# Mixer apodization
mixer_cfg = pipeline_cfg["apodizations"].get("mixer", {})
if bool(mixer_cfg.get("enabled", False)):
    mixer_path = mixer_cfg.get("model_path", "")
    mixer_model_display_path = _resolve_display_path(mixer_path)
    feature_chunk_size = int(mixer_cfg.get("feature_chunk_size", 65536))
    print(f"Loading mixer model from: {mixer_path}")
    try:
        model_file, combiner_weights_file, train_info_file, model_run_dir = resolve_mixer_artifacts(
            mixer_path
        )
        scaled_features, n_apodizations, physical_feature_set, physical_feature_components = (
            extract_mixer_train_metadata(train_info_file)
        )
        train_info = load_config_yaml(train_info_file)
        resolved_mixer_cfg = train_info.get("resolved_mixer", {}) if isinstance(train_info, dict) else {}
        mixer_head_cfg = (
            resolved_mixer_cfg.get("mixer_head", {})
            if isinstance(resolved_mixer_cfg, dict)
            else {}
        )

        apodization_model = tf.keras.models.load_model(str(model_file), compile=False)
        out_shape = apodization_model.output_shape
        if isinstance(out_shape, tuple):
            output_last_dim = int(out_shape[-1])
        else:
            output_last_dim = int(out_shape[0][-1])
        if output_last_dim != n_apodizations:
            raise ValueError(
                "Mixer MLP output shape does not match n_apodizations from train metadata: "
                f"model output last dim={output_last_dim}, n_apodizations={n_apodizations}"
            )

        # Build a coordinate manager matching mixer training feature conventions.
        mixer_cm = CoordinateManager(
            kp,
            physical_feature_set=physical_feature_set,
            physical_feature_components=physical_feature_components,
        )
        features_grid = mixer_cm.get_features_grid(scaled=scaled_features)

        mixer_model = DasInrApodMixer(
            apodization_model=apodization_model,
            features_grid=features_grid,
            feature_chunk_size=feature_chunk_size,
            n_apodizations=n_apodizations,
            weight_regularization_enabled=False,
            mixer_head_config=mixer_head_cfg,
        )

        if delayed_samples.ndim != 4:
            raise ValueError(
                "Expected delayed_samples with shape (n_angles, n_elements, nz, nx), "
                f"got {delayed_samples.shape}"
            )

        # Mixer expects (batch, n_elements, nz, nx), so reduce angle dimension first.
        delayed_for_mixer = np.asarray(np.sum(delayed_samples, axis=0), dtype=np.complex64)
        delayed_batch = tf.convert_to_tensor(np.expand_dims(delayed_for_mixer, axis=0))

        # Warmup pass to build pixel_combiner before loading learned combiner weights.
        _warmup_image, _warmup_weights_grid = mixer_model.reconstruct_image(
            delayed_batch, training=False
        )

        combiner_weights_npz = np.load(combiner_weights_file)
        mixer_model.load_mixer_head_from_npz(combiner_weights_npz)

        mixer_combined_batch, _weights_grid = mixer_model.reconstruct_image(
            delayed_batch, training=False
        )
        mixer_combined_img = np.asarray(mixer_combined_batch.numpy()[0], dtype=np.float32)
        apod_dict["Mixer"] = mixer_combined_img
        figure_info_lines.append(f"Proposed method INR path: {model_file}")
        print(
            "  Mixer DAS computed: "
            f"shape={mixer_combined_img.shape}, run_dir={model_run_dir}"
        )
    except (FileNotFoundError, ValueError, OSError, KeyError) as e:
        figure_info_lines.append(f"Mixer path: unavailable ({mixer_model_display_path})")
        print(f"  Warning: {e}")

# NSI apodization
if pipeline_cfg["apodizations"]["nsi"]["enabled"]:
    nsi_cfg = pipeline_cfg["apodizations"]["nsi"]
    f_number = float(nsi_cfg["f_number"])
    dc = float(nsi_cfg.get("dc", 0.05))
    normalize_by_n_subap = bool(nsi_cfg.get("normalize_by_n_subap", False))

    coords = cm.get_coordinates_1d(scaled=False)
    x_coords = np.asarray(coords["x"], dtype=np.float32)
    z_coords = np.asarray(coords["z"], dtype=np.float32)
    x_elem_coords = np.asarray(coords["x_elem"], dtype=np.float32)

    if delayed_samples.ndim != 4:
        raise ValueError(
            "Expected delayed_samples with shape (n_angles, n_elements, nz, nx), "
            f"got {delayed_samples.shape}"
        )

    delayed_for_nsi = np.asarray(np.sum(delayed_samples, axis=0), dtype=np.complex64)

    print(
        "Computing NSI image "
        f"(f_number={f_number}, dc={dc}, normalize_by_n_subap={normalize_by_n_subap})..."
    )
    _img_sum, _img_diff, nsi_img = compute_nsi_from_delayed_samples_numpy(
        delayed_samples=delayed_for_nsi,
        x_coords=x_coords,
        z_coords=z_coords,
        x_elem_coords=x_elem_coords,
        f_number=f_number,
        dc=dc,
        normalize_by_n_subap=normalize_by_n_subap,
    )
    if np.isnan(nsi_img).any() or np.isinf(nsi_img).any():
        print("  Warning: NSI image contains NaN/Inf values.")
    apod_dict["NSI"] = nsi_img.astype(np.float32, copy=False)
    print(f"  NSI image computed: shape={nsi_img.shape}")

# Compute lateral/axial reflector profiles for all plotted reconstructed images.
images_for_profiles = apod_dict
profile_metrics: dict[str, dict] = {}
for method_name, method_img in images_for_profiles.items():
    profile_metrics[method_name] = extract_reflector_profiles(
        image=np.asarray(method_img, dtype=np.float32),
        scatterers=np.asarray(scatterers_mm, dtype=np.float32),
        cm=cm,
        half_width_lateral_mm=1.5,
        half_width_axial_mm=1.0,
    )

profile_plot_cfg = pipeline_cfg.get("profile_plots", {})
separate_profile_figures = bool(profile_plot_cfg.get("separate_figures", False))

lateral_profile_figs = _plot_profile_pages(
    profile_metrics=profile_metrics,
    scatterers_mm=np.asarray(scatterers_mm, dtype=np.float32),
    axis_kind="lateral",
    output_dir=output_dir,
    ts=ts,
    separate_figures=separate_profile_figures,
)
axial_profile_figs = _plot_profile_pages(
    profile_metrics=profile_metrics,
    scatterers_mm=np.asarray(scatterers_mm, dtype=np.float32),
    axis_kind="axial",
    output_dir=output_dir,
    ts=ts,
    separate_figures=separate_profile_figures,
)

print("\nReflector profiles computed.")
print(f"Lateral profile figures: {len(lateral_profile_figs)}")
print(f"Axial profile figures: {len(axial_profile_figs)}")

# Plot the requested DAS, NSI, and proposed-method images in one row.
image_methods = [
    ("Hanning", "DAS"),
    ("NSI", "NSI"),
    ("Mixer", "Proposed method"),
]
image_methods = [item for item in image_methods if item[0] in apod_dict]
if image_methods:
    rows, cols = 1, len(image_methods)
    fig = plt.figure(figsize=(5 * cols, 5 + 1.8), constrained_layout=True)
    grid = fig.add_gridspec(rows + 1, cols, height_ratios=[1, 0.24], wspace=0.05)
    axes_flat = np.asarray(
        [fig.add_subplot(grid[0, idx]) for idx in range(cols)],
        dtype=object,
    )
    info_ax = fig.add_subplot(grid[rows, :])
    active_axes = list(axes_flat)
    extent = kp.get_imshow_extent()

    for idx, (method_name, display_name) in enumerate(image_methods):
        das_img = apod_dict[method_name]
        das_apod_db = to_db(das_img, ref=float(np.max(np.abs(das_img))))
        im = axes_flat[idx].imshow(
            das_apod_db,
            aspect="auto",
            cmap="gray",
            extent=extent,
            vmin=-60,
            vmax=0,
        )
        if idx == 0:
            im0 = im
        axes_flat[idx].set_title(f"{display_name} (dB)")
        axes_flat[idx].set_xlabel("Lateral [mm]")
        if idx == 0:
            axes_flat[idx].set_ylabel("Axial [mm]")
        else:
            axes_flat[idx].set_ylabel("")
            axes_flat[idx].tick_params(axis="y", labelleft=False)

    fig.colorbar(im0, ax=active_axes, label="dB", fraction=0.028, pad=0.03)

    info_ax.axis("off")
    info_ax.text(
        0.01,
        0.95,
        "\n".join(figure_info_lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )

    figure_path = output_dir / f"picmus_apodizations_{ts}.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")

    plt.show()

    print("\nApodization images plotted.")
    print(f"Figure saved to: {figure_path}")

if lateral_profile_figs:
    print("Saved lateral profile figures:")
    for p in lateral_profile_figs:
        print(f"  {p}")

if axial_profile_figs:
    print("Saved axial profile figures:")
    for p in axial_profile_figs:
        print(f"  {p}")


