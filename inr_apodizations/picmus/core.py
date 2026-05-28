from __future__ import annotations

from pathlib import Path

import cupy as cp
import h5py
import numpy as np
import yaml
from scipy import signal

import inr_apodizations.hilbert_coef as hilb
from inr_apodizations.apodizations import compute_das_baseline_numpy
from inr_apodizations.config import CUDA_DIR
from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.coordinate_manager import CoordinateManager


def load_picmus_pipeline_config(config_path: str | Path) -> dict:
    """Load PICMUS processing configuration from a YAML file.

    Args:
        config_path: Path to YAML config.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required sections are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"PICMUS config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_sections = ["io", "probe", "acquisition", "beamforming", "angle_subset"]
    missing = [name for name in required_sections if name not in cfg]
    if missing:
        raise ValueError(f"Missing sections in PICMUS config: {missing}")

    return cfg


def load_picmus_hdf5(
    picmus_data_path: str | Path,
    rf_file: str,
    scan_file: str,
    phantom_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load RF, angle, and scan axes arrays from PICMUS HDF5 files.

    Args:
        picmus_data_path: Base folder that contains PICMUS HDF5 files.
        rf_file: RF HDF5 file path relative to ``picmus_data_path``.
        scan_file: Scan HDF5 file path relative to ``picmus_data_path``.
        phantom_file: Phantom HDF5 file path relative to ``picmus_data_path``.

    Returns:
        Tuple ``(angles, rf_real, x_axis_mm, z_axis_mm, scatterers_mm)`` where:
            angles: 1D transmit angles array.
            rf_real: RF real-part data with shape (n_angles, n_elements, n_samples).
            x_axis_mm: Scan lateral axis in millimeters.
            z_axis_mm: Scan axial axis in millimeters.
            scatterers_mm: Scatterer positions with shape (n_scatterers, 2) in millimeters,
                ordered as [x_mm, z_mm].

    Raises:
        FileNotFoundError: If RF, scan, or phantom file does not exist.
        KeyError: If expected HDF5 keys do not exist.
        ValueError: If RF shape, scan axes, or scatterer positions are invalid.
    """
    base_path = Path(picmus_data_path)
    rf_path = base_path / rf_file
    scan_path = base_path / scan_file
    phantom_path = base_path / phantom_file

    if not rf_path.exists():
        raise FileNotFoundError(f"RF dataset file not found: {rf_path}")
    if not scan_path.exists():
        raise FileNotFoundError(f"Scan file not found: {scan_path}")
    if not phantom_path.exists():
        raise FileNotFoundError(f"Phantom file not found: {phantom_path}")

    with h5py.File(rf_path, "r") as rf_h5:
        rf_dset = rf_h5["US"]["US_DATASET0000"]
        angles = np.asarray(rf_dset["angles"][:], dtype=np.float32)
        rf_real = np.asarray(rf_dset["data"]["real"][:])

    with h5py.File(scan_path, "r") as scan_h5:
        scan_dset = scan_h5["US"]["US_DATASET0000"]
        x_axis_mm = 1000.0 * np.ravel(scan_dset["x_axis"][:])
        z_axis_mm = 1000.0 * np.ravel(scan_dset["z_axis"][:])

    with h5py.File(phantom_path, "r") as phantom_h5:
        phantom_dset = phantom_h5["US"]["US_DATASET0000"]
        scatterers_positions = None

        if "scatterers_positions" in phantom_dset:
            scatterers_positions = np.asarray(phantom_dset["scatterers_positions"][:])
        else:
            def _find_scatterers(_name: str, obj):
                nonlocal scatterers_positions
                if (
                    scatterers_positions is None
                    and isinstance(obj, h5py.Dataset)
                    and obj.name.endswith("scatterers_positions")
                ):
                    scatterers_positions = np.asarray(obj[:])

            phantom_dset.visititems(_find_scatterers)

        if scatterers_positions is None:
            raise KeyError(
                "Expected dataset 'scatterers_positions' not found under 'US/US_DATASET0000' "
                f"in phantom HDF5 file {phantom_path}"
            )

    scatterers_positions = np.asarray(scatterers_positions, dtype=np.float32)
    if scatterers_positions.ndim != 2 or scatterers_positions.shape[0] != 3:
        raise ValueError(
            "Expected scatterers_positions with shape (3, N), got "
            f"{scatterers_positions.shape}"
        )
    if scatterers_positions.shape[1] == 0:
        raise ValueError("Expected at least one scatterer position in phantom file.")

    scatterers_mm = 1000.0 * scatterers_positions[[0, 2], :].T

    if rf_real.ndim != 3:
        raise ValueError(f"Expected rf_real with 3 dimensions, got shape {rf_real.shape}")
    if x_axis_mm.size < 2 or z_axis_mm.size < 2:
        raise ValueError(
            f"Expected x_axis/z_axis with at least 2 values, got {x_axis_mm.size} and {z_axis_mm.size}"
        )

    return (
        angles,
        rf_real,
        x_axis_mm.astype(np.float32),
        z_axis_mm.astype(np.float32),
        scatterers_mm.astype(np.float32),
    )


def select_angle_subset(
    angles: np.ndarray,
    rf_real: np.ndarray,
    subset_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Select transmit-angle subset and slice RF data consistently.

    Args:
        angles: 1D angles array with shape (n_angles,).
        rf_real: RF data with shape (n_angles, n_elements, n_samples).
        subset_cfg: Subset configuration dictionary with:
            - mode: "all" or "indices"
            - indices: list[int], required for mode="indices"

    Returns:
        Tuple ``(angles_subset, rf_subset)``.

    Raises:
        ValueError: If subset mode or indices are invalid.
    """
    mode = str(subset_cfg.get("mode", "all")).strip().lower()

    if mode == "all":
        return angles, rf_real

    if mode != "indices":
        raise ValueError(f"Unsupported angle_subset mode: {mode}. Use 'all' or 'indices'.")

    indices = subset_cfg.get("indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError("angle_subset.indices must be a non-empty list when mode='indices'.")

    idx = np.asarray(indices, dtype=np.int32)
    if np.any(idx < 0) or np.any(idx >= angles.shape[0]):
        raise ValueError(
            f"angle_subset.indices out of bounds for n_angles={angles.shape[0]}: {indices}"
        )

    return angles[idx], rf_real[idx, :, :]


def build_picmus_kernel_config(
    pipeline_cfg: dict,
    angles: np.ndarray,
    rf_real: np.ndarray,
    x_axis_mm: np.ndarray,
    z_axis_mm: np.ndarray,
) -> dict:
    """Build a KernelParameters2D-compatible configuration from PICMUS inputs.

    Args:
        pipeline_cfg: PICMUS pipeline config loaded from YAML.
        angles: Selected transmit angles.
        rf_real: Selected RF data with shape (n_angles, n_elements, n_samples).
        x_axis_mm: Scan lateral axis in millimeters.
        z_axis_mm: Scan axial axis in millimeters.

    Returns:
        Dictionary compatible with ``KernelParameters2D``.

    Raises:
        ValueError: If RF dimensions mismatch configured transducer settings.
    """
    n_angles, n_elements_data, n_samples = rf_real.shape

    probe_cfg = pipeline_cfg["probe"]
    acq_cfg = pipeline_cfg["acquisition"]
    bf_cfg = pipeline_cfg["beamforming"]

    n_elements_cfg = int(probe_cfg["n_elements"])
    if n_elements_cfg != n_elements_data:
        raise ValueError(
            "n_elements in YAML does not match RF data shape: "
            f"yaml={n_elements_cfg}, data={n_elements_data}"
        )

    roi_user = [
        float(x_axis_mm[0]),
        float(x_axis_mm[-1]),
        float(z_axis_mm[0]),
        float(z_axis_mm[-1]),
    ]

    auto_grid_step = bool(bf_cfg.get("auto_grid_step", True))
    if auto_grid_step:
        x_step = float(np.mean(np.diff(x_axis_mm)))
        z_step = float(np.mean(np.diff(z_axis_mm)))
    else:
        x_step = float(bf_cfg["x_step"])
        z_step = float(bf_cfg["z_step"])

    bandwidth_fraction = float(probe_cfg["bandwidth"]) / 100.0
    fc = float(probe_cfg["fc"])
    f1 = float(bf_cfg.get("f1", fc * (1.0 - bandwidth_fraction / 2.0)))
    f2 = float(bf_cfg.get("f2", fc * (1.0 + bandwidth_fraction / 2.0)))

    cfg = {
        "pitch": float(probe_cfg["pitch"]),
        "element_width": float(probe_cfg["element_width"]),
        "element_height": float(probe_cfg["element_height"]),
        "n_elements": n_elements_cfg,
        "fc": fc,
        "fs": float(acq_cfg["fs"]),
        "bandwidth": float(probe_cfg["bandwidth"]),
        "n_pulses": float(acq_cfg["n_pulses"]),
        "c1": float(probe_cfg["c1"]),
        "f1": f1,
        "f2": f2,
        "f_number": float(bf_cfg["f_number"]),
        "x_step": x_step,
        "z_step": z_step,
        "t_start": float(acq_cfg.get("t_start", 0.0)),
        "taps": int(bf_cfg["taps"]),
        "n_batch": int(bf_cfg.get("n_batch", 0)),
        "n_ch": int(bf_cfg.get("n_ch", n_elements_cfg)),
        "n_angles": int(n_angles),
        "n_samples": int(n_samples),
        "roi_user": roi_user,
        "blocksize_img": tuple(bf_cfg["blocksize_img"]),
        "wave_source_mode": int(bf_cfg.get("wave_source_mode", 0)),
        "angles": angles.astype(np.float32),
    }

    return cfg


def build_kernel_parameters(cfg: dict) -> KernelParameters2D:
    """Construct and validate ``KernelParameters2D`` from PICMUS config.

    Args:
        cfg: Kernel configuration dictionary.

    Returns:
        Initialized ``KernelParameters2D`` instance.

    Raises:
        ValueError: If kernel-required parameters are inconsistent.
    """
    kp = KernelParameters2D(cfg)

    if kp.n_angles != int(cfg["n_angles"]):
        raise ValueError("Kernel n_angles is inconsistent with configuration.")
    if kp.n_elements != int(cfg["n_elements"]):
        raise ValueError("Kernel n_elements is inconsistent with configuration.")

    return kp


def normalize_to_int16(rf_data: np.ndarray, reduce_factor: float = 1.0) -> np.ndarray:
    """Convert RF data to ``int16`` with symmetric max-abs normalization.

    Args:
        rf_data: Input RF data in float or integer dtype.
        reduce_factor: Optional multiplicative scale after normalization.

    Returns:
        RF data as int16.
    """
    max_val = float(np.max(np.abs(rf_data)))
    if max_val == 0.0:
        return np.zeros_like(rf_data, dtype=np.int16)

    scaled = (rf_data / max_val) * (32767.0 * float(reduce_factor))
    return scaled.astype(np.int16)


def _load_cuda_module_and_kernels(kp: KernelParameters2D) -> tuple[cp.RawKernel, cp.RawKernel]:
    """Load CUDA kernels used for FIR filtering and delayed-sample gathering.

    Args:
        kp: Kernel parameters object used for enum consistency check.

    Returns:
        Tuple ``(fir_kernel, gather_kernel)``.

    Raises:
        FileNotFoundError: If any kernel source file is missing.
    """
    code_files = [
        "constants.h",
        "enum_parameters.c",
        "fir_filter.cu",
        "pwi_1pix_per_thread.cu",
    ]
    code = ""
    for file_name in code_files:
        path = CUDA_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(f"CUDA code file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            code += f.read() + "\n"

    kp.check_enum_consistency(code)
    module = cp.RawModule(code=code, options=("--use_fast_math",))
    return module.get_function("fir_filter"), module.get_function("pwi_gather_delayed_samples")


def compute_delayed_samples(
    rf_real: np.ndarray,
    angles: np.ndarray,
    kp: KernelParameters2D,
    cfg: dict,
) -> np.ndarray:
    """Compute delayed samples on GPU from PICMUS RF data.

    Args:
        rf_real: RF real-part array with shape (n_angles, n_elements, n_samples).
        angles: Selected transmit angles with shape (n_angles,).
        kp: Kernel parameters.
        cfg: Kernel configuration containing FIR settings.

    Returns:
        Delayed samples array with shape (n_angles, n_elements, nz, nx), dtype complex64.

    Raises:
        ValueError: If RF/angle dimensions are incompatible.
    """
    if rf_real.shape[0] != angles.shape[0]:
        raise ValueError(
            "RF and angles mismatch: "
            f"rf_real.shape[0]={rf_real.shape[0]}, angles.shape[0]={angles.shape[0]}"
        )

    rf_int16 = normalize_to_int16(rf_real)

    fir_kernel, gather_kernel = _load_cuda_module_and_kernels(kp)

    int_params = cp.asarray(kp.get_int_array(), dtype=cp.int32)
    float_params = cp.asarray(kp.get_float_array(), dtype=cp.float32)
    angles_gpu = cp.asarray(angles, dtype=cp.float32)

    bandpass_coef = signal.firwin(
        int(cfg["taps"]) + 1,
        [2.0 * float(cfg["f1"]) / float(cfg["fs"]), 2.0 * float(cfg["f2"]) / float(cfg["fs"])],
        pass_zero=False,
    )
    bandpass_coef_gpu = cp.asarray(bandpass_coef, dtype=cp.float32)
    hilb_coef_gpu = cp.asarray(hilb.coef, dtype=cp.float32)

    rf_gpu = cp.asarray(rf_int16)
    rf_filt_gpu = cp.zeros_like(rf_gpu, dtype=cp.int16)
    rf_imag_gpu = cp.zeros_like(rf_gpu, dtype=cp.int16)

    nblock = 128
    n_ascans = rf_gpu.shape[0] * rf_gpu.shape[1]
    grid_size = ((n_ascans + nblock - 1) // nblock,)
    block_size = (nblock,)

    fir_kernel(grid_size, block_size, (int_params, rf_gpu, bandpass_coef_gpu, rf_filt_gpu))
    fir_kernel(grid_size, block_size, (int_params, rf_filt_gpu, hilb_coef_gpu, rf_imag_gpu))

    delayed_samples_gpu = cp.zeros((kp.n_angles, kp.n_elements, kp.nz, kp.nx), dtype=cp.complex64)
    gather_kernel(
        kp.gridsize_img,
        kp.blocksize_img,
        (int_params, float_params, angles_gpu, rf_filt_gpu, rf_imag_gpu, delayed_samples_gpu),
    )

    cp.cuda.Device().synchronize()
    return cp.asnumpy(delayed_samples_gpu)


def compute_uniform_das_image(delayed_samples: np.ndarray) -> np.ndarray:
    """Compute uniform DAS image by summing angles and receive elements.

    Args:
        delayed_samples: Complex delayed samples with shape
            (n_angles, n_elements, nz, nx).

    Returns:
        Uniform DAS image magnitude with shape (nz, nx), dtype float32.

    Raises:
        ValueError: If input shape is invalid.
    """
    if delayed_samples.ndim != 4:
        raise ValueError(
            "Expected delayed_samples with 4 dimensions "
            f"(n_angles, n_elements, nz, nx), got {delayed_samples.shape}"
        )

    summed = delayed_samples.sum(axis=0).sum(axis=0)
    return np.abs(summed).astype(np.float32)


def compute_hanning_apodization_chunk(
    cm: CoordinateManager,
    f_number: float,
    z_start: int,
    z_end: int,
    scaled: bool = False,
) -> np.ndarray:
    """Compute a Hanning apodization chunk for a depth slice.

    Args:
        cm: CoordinateManager instance.
        f_number: F-number used for dynamic aperture.
        z_start: Inclusive start index for the z slice.
        z_end: Exclusive end index for the z slice.
        scaled: Whether to use scaled coordinates.

    Returns:
        Apodization chunk with shape (n_elements, z_end - z_start, nx), dtype float32.

    Raises:
        ValueError: If the z slice indices are invalid.
    """
    coords = cm.get_coordinates_1d(scaled=scaled)
    z_coords = np.asarray(coords["z"], dtype=np.float32)
    x_coords = np.asarray(coords["x"], dtype=np.float32)
    x_elem_coords = np.asarray(coords["x_elem"], dtype=np.float32)

    if z_start < 0 or z_end > z_coords.shape[0] or z_start >= z_end:
        raise ValueError(
            f"Invalid z slice [{z_start}, {z_end}) for nz={z_coords.shape[0]}"
        )

    z_chunk = z_coords[z_start:z_end]
    dist = np.abs(x_coords.reshape(1, 1, -1) - x_elem_coords.reshape(-1, 1, 1))
    ap_radius = z_chunk.reshape(1, -1, 1) / (2.0 * float(f_number))

    safe_ap = ap_radius + np.float32(1e-8)
    ratio = dist / safe_ap
    mask = ratio <= 1.0
    weights = 0.5 * (1.0 + np.cos(np.float32(np.pi) * ratio))
    weights = np.where(mask, weights, 0.0)
    return weights.astype(np.float32, copy=False)


def compute_hanning_das_image_chunked(
    delayed_samples: np.ndarray,
    cm: CoordinateManager,
    f_number: float,
    z_chunk_size: int = 32,
    angle_chunk_size: int | None = None,
    scaled: bool = False,
) -> np.ndarray:
    """Compute a Hanning-weighted DAS image using chunked accumulation.

    The reconstruction is performed by processing the depth axis in chunks and,
    optionally, the angle axis in smaller chunks. This avoids materializing the
    full apodization tensor and the full weighted delayed-sample tensor at once.

    Args:
        delayed_samples: Complex delayed samples with shape
            ``(n_angles, n_elements, nz, nx)``.
        cm: CoordinateManager instance.
        f_number: F-number used for dynamic aperture.
        z_chunk_size: Number of z samples processed per chunk.
        angle_chunk_size: Optional number of angles processed per chunk.
            If ``None`` or non-positive, all angles are processed at once.
        scaled: Whether to use scaled coordinates.

    Returns:
        Hanning-weighted DAS magnitude image with shape ``(nz, nx)``, dtype float32.

    Raises:
        ValueError: If the input shape or chunk sizes are invalid.
    """
    if delayed_samples.ndim != 4:
        raise ValueError(
            "Expected delayed_samples with 4 dimensions "
            f"(n_angles, n_elements, nz, nx), got {delayed_samples.shape}"
        )
    if z_chunk_size <= 0:
        raise ValueError(f"z_chunk_size must be a positive integer, got {z_chunk_size}")

    n_angles, n_elements, nz, nx = delayed_samples.shape
    if angle_chunk_size is None or int(angle_chunk_size) <= 0:
        angle_chunk_size = n_angles
    else:
        angle_chunk_size = int(angle_chunk_size)

    out = np.zeros((nz, nx), dtype=np.float32)

    for z_start in range(0, nz, z_chunk_size):
        z_end = min(z_start + z_chunk_size, nz)
        apod_chunk = compute_hanning_apodization_chunk(
            cm=cm,
            f_number=f_number,
            z_start=z_start,
            z_end=z_end,
            scaled=scaled,
        )

        chunk_complex = np.zeros((z_end - z_start, nx), dtype=np.complex64)
        for angle_start in range(0, n_angles, angle_chunk_size):
            angle_end = min(angle_start + angle_chunk_size, n_angles)
            delayed_chunk = delayed_samples[angle_start:angle_end, :, z_start:z_end, :]
            weighted_chunk = delayed_chunk * apod_chunk[np.newaxis, :, :, :]
            chunk_complex += weighted_chunk.sum(axis=(0, 1)).astype(np.complex64, copy=False)

        out[z_start:z_end, :] = np.abs(chunk_complex).astype(np.float32, copy=False)

    return out


def build_coordinate_manager(kp: KernelParameters2D) -> CoordinateManager:
    """Build coordinate manager from kernel parameters.

    Args:
        kp: Kernel parameters object.

    Returns:
        CoordinateManager instance.
    """
    return CoordinateManager(kp, physical_feature_set="distance_depth_edge")


def load_inr_model(model_path: str | Path):
    """Load a pre-trained INR model for apodization.

    Args:
        model_path: Path to saved model (e.g., model.keras).

    Returns:
        Loaded TensorFlow/Keras model.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    import tensorflow as tf

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"INR model not found: {path}")

    return tf.keras.models.load_model(path)


def compute_apodization_inr(
    model,
    cm: CoordinateManager,
    batch_size: int = 8192,
) -> np.ndarray:
    """Compute apodization from pre-trained INR model.

    Args:
        model: Loaded INR model (TensorFlow/Keras).
        cm: CoordinateManager instance.
        batch_size: Batch size for model inference.

    Returns:
        Apodization array with shape (n_elements, nz, nx), dtype float32.
    """
    import tensorflow as tf

    features_grid = cm.get_features_grid(scaled=True)  # (n_elem, nz, nx, 3)
    n_elem, nz, nx, _ = features_grid.shape

    feats_flat = tf.reshape(features_grid, [-1, features_grid.shape[-1]])
    apod_flat = []

    for i in range(0, feats_flat.shape[0], batch_size):
        batch = feats_flat[i : i + batch_size]
        batch_out = model(batch, training=False).numpy()
        apod_flat.append(batch_out)

    apod_flat = np.concatenate(apod_flat, axis=0)
    apod = apod_flat.reshape(n_elem, nz, nx).astype(np.float32)

    return apod


def _build_feature_chunk_from_coords(
    cm: CoordinateManager,
    z_start: int,
    z_end: int,
    scaled_features: bool,
) -> np.ndarray:
    """Build INR input features for a z-slice directly from coordinates.

    Args:
        cm: CoordinateManager instance.
        z_start: Inclusive start index for z chunk.
        z_end: Exclusive end index for z chunk.
        scaled_features: Whether to generate scaled features.

    Returns:
        Feature chunk with shape (n_elements, z_chunk, nx, n_features), float32.

    Raises:
        ValueError: If z indices are invalid or feature set is unsupported.
    """
    coords = cm.get_coordinates_1d(scaled=scaled_features)
    x_coords = np.asarray(coords["x"], dtype=np.float32)
    z_coords = np.asarray(coords["z"], dtype=np.float32)
    x_elem_coords = np.asarray(coords["x_elem"], dtype=np.float32)

    if z_start < 0 or z_end > z_coords.shape[0] or z_start >= z_end:
        raise ValueError(f"Invalid z slice [{z_start}, {z_end}) for nz={z_coords.shape[0]}")

    z_chunk = z_coords[z_start:z_end]
    target_shape = (x_elem_coords.shape[0], z_chunk.shape[0], x_coords.shape[0])
    dist_to_elem = np.abs(x_coords.reshape(1, 1, -1) - x_elem_coords.reshape(-1, 1, 1))
    dist_to_elem = np.broadcast_to(dist_to_elem, target_shape)
    depth = np.broadcast_to(z_chunk.reshape(1, -1, 1), target_shape)
    x_from_center = np.broadcast_to(np.abs(x_coords).reshape(1, 1, -1), target_shape)

    feature_set = cm.physical_feature_set
    if feature_set == "distance_depth":
        features = np.stack([dist_to_elem, depth], axis=-1)
    elif feature_set == "distance_depth_center":
        features = np.stack([dist_to_elem, depth, x_from_center], axis=-1)
    elif feature_set == "distance_depth_edge":
        if scaled_features:
            dist_to_edge = 0.5 - x_from_center
        else:
            dist_to_edge = cm.D_half - x_from_center
        features = np.stack([dist_to_elem, depth, dist_to_edge], axis=-1)
    else:
        raise ValueError(f"Unsupported physical_feature_set: {feature_set}")

    return features.astype(np.float32, copy=False)


def compute_inr_das_image_chunked(
    delayed_samples: np.ndarray,
    model,
    cm: CoordinateManager,
    feature_batch_size: int = 32768,
    z_chunk_size: int = 16,
    angle_chunk_size: int | None = 1,
    scaled_features: bool = True,
) -> np.ndarray:
    """Compute INR-weighted DAS image using chunked feature inference and accumulation.

    Args:
        delayed_samples: Complex delayed samples with shape
            ``(n_angles, n_elements, nz, nx)``.
        model: Loaded INR model (TensorFlow/Keras).
        cm: CoordinateManager instance.
        feature_batch_size: Number of flattened feature rows per INR inference batch.
        z_chunk_size: Number of z samples processed per chunk.
        angle_chunk_size: Optional number of angles processed per chunk.
            If ``None`` or non-positive, all angles are processed at once.
        scaled_features: Whether to feed scaled features to INR.

    Returns:
        INR-weighted DAS magnitude image with shape ``(nz, nx)``, dtype float32.

    Raises:
        ValueError: If delayed_samples shape or chunk sizes are invalid.
    """
    if delayed_samples.ndim != 4:
        raise ValueError(
            "Expected delayed_samples with 4 dimensions "
            f"(n_angles, n_elements, nz, nx), got {delayed_samples.shape}"
        )
    if z_chunk_size <= 0:
        raise ValueError(f"z_chunk_size must be a positive integer, got {z_chunk_size}")
    if feature_batch_size <= 0:
        raise ValueError(
            f"feature_batch_size must be a positive integer, got {feature_batch_size}"
        )

    n_angles, n_elements, nz, nx = delayed_samples.shape
    if angle_chunk_size is None or int(angle_chunk_size) <= 0:
        angle_chunk_size = n_angles
    else:
        angle_chunk_size = int(angle_chunk_size)

    out = np.zeros((nz, nx), dtype=np.float32)

    for z_start in range(0, nz, z_chunk_size):
        z_end = min(z_start + z_chunk_size, nz)
        features_chunk = _build_feature_chunk_from_coords(
            cm=cm,
            z_start=z_start,
            z_end=z_end,
            scaled_features=scaled_features,
        )

        z_len = z_end - z_start
        feats_flat = features_chunk.reshape(-1, features_chunk.shape[-1])
        pred_flat = []
        for i in range(0, feats_flat.shape[0], feature_batch_size):
            feats_batch = feats_flat[i : i + feature_batch_size]
            pred_batch = model(feats_batch, training=False).numpy()
            pred_flat.append(pred_batch)

        weights_chunk = np.concatenate(pred_flat, axis=0).reshape(n_elements, z_len, nx)
        weights_chunk = weights_chunk.astype(np.float32, copy=False)

        chunk_complex = np.zeros((z_len, nx), dtype=np.complex64)
        for angle_start in range(0, n_angles, angle_chunk_size):
            angle_end = min(angle_start + angle_chunk_size, n_angles)
            delayed_chunk = delayed_samples[angle_start:angle_end, :, z_start:z_end, :]
            weighted_chunk = delayed_chunk * weights_chunk[np.newaxis, :, :, :]
            chunk_complex += weighted_chunk.sum(axis=(0, 1)).astype(np.complex64, copy=False)

        out[z_start:z_end, :] = np.abs(chunk_complex).astype(np.float32, copy=False)

    return out


def apply_apodization_to_delayed_samples(
    delayed_samples: np.ndarray,
    apodization: np.ndarray,
) -> np.ndarray:
    """Apply apodization weights to delayed samples and compute DAS image.

    Wrapper around compute_das_baseline_numpy for PICMUS delayed samples
    with shape (n_angles, n_elements, nz, nx).

    Args:
        delayed_samples: Complex delayed samples with shape
            (n_angles, n_elements, nz, nx).
        apodization: Apodization weights with shape (n_elements, nz, nx).

    Returns:
        Apodized DAS image magnitude with shape (nz, nx), dtype float32.
    """
    n_angles, n_elements, nz, nx = delayed_samples.shape
    
    # Reshape for compute_das_baseline_numpy: (n_angles*n_elements, nz, nx)
    delayed_flat = delayed_samples.reshape(n_angles * n_elements, nz, nx)
    
    # Tile apodization to match shape
    apod_tiled = np.tile(apodization, (n_angles, 1, 1))
    
    return compute_das_baseline_numpy(delayed_flat, apod_tiled, return_complex=False)
