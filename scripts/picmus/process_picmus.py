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
angles, rf_real, x_axis_mm, z_axis_mm = load_picmus_hdf5(
    picmus_data_path=picmus_data_path,
    rf_file=io_cfg["rf_file"],
    scan_file=io_cfg["scan_file"],
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
        figure_info_lines.append(f"Mixer INR path: {model_file}")
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

# Plot all apodizations side-by-side if any were computed
if apod_dict:
    n_images = len(apod_dict) + 1  # +1 for uniform
    rows, cols = _subplot_grid(n_images)
    fig = plt.figure(figsize=(5 * cols, 5 * rows + 1.8), constrained_layout=True)
    grid = fig.add_gridspec(rows + 1, cols, height_ratios=[1] * rows + [0.24])
    axes_flat = np.asarray(
        [fig.add_subplot(grid[idx // cols, idx % cols]) for idx in range(rows * cols)],
        dtype=object,
    )
    info_ax = fig.add_subplot(grid[rows, :])
    active_axes = list(axes_flat[:n_images])
    extent = kp.get_imshow_extent()

    # Uniform DAS
    das_uniform_db = to_db(das_uniform, ref=float(np.max(np.abs(das_uniform))))
    im0 = axes_flat[0].imshow(
        das_uniform_db,
        aspect="auto",
        cmap="gray",
        extent=extent,
        vmin=-60,
        vmax=0,
    )
    axes_flat[0].set_title("Uniform DAS (dB)")
    axes_flat[0].set_xlabel("Lateral [mm]")
    axes_flat[0].set_ylabel("Axial [mm]")

    # Apodized images
    for idx, (apod_name, das_img) in enumerate(apod_dict.items(), start=1):
        das_apod_db = to_db(das_img, ref=float(np.max(np.abs(das_img))))
        im = axes_flat[idx].imshow(
            das_apod_db,
            aspect="auto",
            cmap="gray",
            extent=extent,
            vmin=-60,
            vmax=0,
        )
        axes_flat[idx].set_title(f"{apod_name} DAS (dB)")
        axes_flat[idx].set_xlabel("Lateral [mm]")
        axes_flat[idx].set_ylabel("Axial [mm]")

    # Hide unused axes when the grid has spare cells (e.g., 5 images -> 2x3).
    for idx in range(n_images, rows * cols):
        axes_flat[idx].axis("off")

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

    output_dir = PROJ_ROOT / Path("scripts/outputs/picmus")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_path = output_dir / f"picmus_apodizations_{ts}.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")

    plt.show()

    print("\nApodization images plotted.")
    print(f"Figure saved to: {figure_path}")


