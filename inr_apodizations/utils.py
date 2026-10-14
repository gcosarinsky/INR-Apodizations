from pathlib import Path

import pymust
import numpy as np
import tensorflow as tf
import yaml


def to_db(image: np.ndarray, ref: float, eps: float = 1e-8) -> np.ndarray:
    """Convert an image magnitude from linear domain to decibels.

    Args:
        image: Real or complex image in linear domain.
        ref: Positive reference magnitude used as 0 dB.
        eps: Small value to avoid numerical instability.

    Returns:
        NumPy array with image values in dB.
    """
    magnitude = np.abs(image)
    return 20.0 * np.log10((magnitude / (ref + eps)) + eps)


def to_db_tensor(x, ref, eps: float = 1e-8):
    """Convert a real/complex tensor magnitude to decibels.

    Args:
        x: Input tensor with linear amplitudes (real or complex).
        ref: Reference amplitude for normalization (0 dB level).
        eps: Small positive constant to avoid log/division issues.

    Returns:
        Tensor with values in dB.

    Raises:
        tf.errors.InvalidArgumentError: If ``eps`` is not strictly positive.
    """
    eps_tensor = tf.cast(eps, tf.float32)
    tf.debugging.assert_positive(eps_tensor, message="eps must be > 0")

    magnitude = tf.cast(tf.abs(x), tf.float32)
    ref_tensor = tf.cast(ref, tf.float32)
    ref_safe = tf.maximum(ref_tensor, eps_tensor)

    normalized = magnitude / ref_safe
    normalized_safe = tf.maximum(normalized, eps_tensor)
    log10 = tf.math.log(normalized_safe) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
    return 20.0 * log10


def relative_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
    normalize_by: str = "y_pred",
    sample_weights: np.ndarray | None = None,
) -> float:
    """Compute RelativeMAE with configurable normalization reference.

    Formula:
        RelativeMAE(y_true, y_pred) = sum(abs_error * w) / (sum(abs(reference) * w) + eps)

    where ``reference`` is selected by ``normalize_by``:
        - ``"y_pred"`` -> reference is ``y_pred``
        - ``"y_true"`` -> reference is ``y_true``

    When ``sample_weights`` is not provided, unit weights are used, which
    preserves the previous unweighted behavior.

    Args:
        y_true: Ground-truth image array.
        y_pred: Predicted image array with the same shape as ``y_true``.
        eps: Small positive value to stabilize the denominator.
        normalize_by: Denominator reference selector. Allowed values are
            ``"y_pred"`` and ``"y_true"``.
        sample_weights: Optional per-pixel weights with the same shape as
            ``y_true``/``y_pred``.

    Returns:
        Relative MAE as a Python float.

    Raises:
        ValueError: If ``eps`` is not positive, shapes do not match, or
            ``normalize_by`` is not supported.
    """
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    y_true_arr = np.asarray(y_true, dtype=np.float32)
    y_pred_arr = np.asarray(y_pred, dtype=np.float32)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "relative_mae requires matching shapes: "
            f"y_true.shape={y_true_arr.shape}, y_pred.shape={y_pred_arr.shape}"
        )

    normalize_key = str(normalize_by).strip().lower()
    if normalize_key not in ("y_pred", "y_true"):
        raise ValueError("normalize_by must be either 'y_pred' or 'y_true'")

    if sample_weights is None:
        weights_arr = np.ones_like(y_true_arr, dtype=np.float32)
    else:
        weights_arr = np.asarray(sample_weights, dtype=np.float32)
        if weights_arr.shape != y_true_arr.shape:
            raise ValueError(
                "relative_mae requires matching sample_weights shape: "
                f"sample_weights.shape={weights_arr.shape}, expected {y_true_arr.shape}"
            )

    abs_error = np.abs(y_true_arr - y_pred_arr)
    reference = y_pred_arr if normalize_key == "y_pred" else y_true_arr
    weighted_abs_error = np.sum(abs_error * weights_arr, dtype=np.float32)
    weighted_reference_abs = np.sum(np.abs(reference) * weights_arr, dtype=np.float32)
    return float(weighted_abs_error / (weighted_reference_abs + np.float32(eps)))


def cfg_to_must_param(cfg):
    """
    Convert a config dictionary into a pymust.utils.Param object.

    Expected config structure:
        cfg["probe"]:
            - pitch (mm)
            - element_width (mm)
            - element_height (mm)
            - n_elements
            - fc (MHz)
            - bandwidth (%)
        cfg["plane_wave_acquisition"]:
            - fs (MHz)
        cfg["c1"]:
            - speed of sound (mm/us)

    Returns:
        pymust.utils.Param: Parameter object with SI units required by pymust.
    """

    param = pymust.utils.Param()
    param.fs = cfg["fs"] * 1e6                    # MHz -> Hz
    param.fc = cfg["fc"] * 1e6                  # MHz -> Hz
    param.pitch = cfg["pitch"] * 1e-3           # mm -> m
    param.Nelements = cfg["n_elements"]
    param.c = cfg["c1"] * 1e3                     # mm/us -> m/s
    param.bandwidth = cfg["bandwidth"]          # Percent bandwidth
    param.width = cfg["element_width"] * 1e-3   # mm -> m
    param.height = cfg["element_height"] * 1e-3 # mm -> m
    param.radius = np.inf

    # Optional linear array element positions:
    # x0 = param.pitch * (param.Nelements - 1) / 2
    # param.elements = np.arange(param.Nelements) * param.pitch - x0

    return param

def save_config_yaml(config_path, cfg, extra_params):
    """
    Save configuration and extra parameters to a YAML file.
    All lists and tuples are saved in flat (flow) style.

    Args:
        config_path (Path): Output YAML file path.
        cfg (dict): Main configuration dictionary.
        extra_params (dict): Additional parameters to save.
    """
    def flat_seq_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    yaml.add_representer(list, flat_seq_representer, Dumper=yaml.SafeDumper)
    yaml.add_representer(tuple, flat_seq_representer, Dumper=yaml.SafeDumper)

    def _convert_tuples(obj):
        """
        Recursively convert tuples to lists so YAML dumper does not emit !!python/tuple tags.
        """
        if isinstance(obj, tuple):
            return [_convert_tuples(v) for v in obj]
        if isinstance(obj, list):
            return [_convert_tuples(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _convert_tuples(v) for k, v in obj.items()}
        return obj

    config_to_save = _convert_tuples(dict(cfg))
    extra_clean = _convert_tuples(dict(extra_params))
    config_to_save.update(extra_clean)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_to_save, f, allow_unicode=True, Dumper=yaml.SafeDumper)


def find_latest_dataset_folder(data_dir: Path, dataset_subdir: str) -> Path:
    """Return the most recently modified dataset run folder.

    Args:
        data_dir: Base data directory.
        dataset_subdir: Dataset parent directory name inside data_dir.

    Returns:
        Path to the latest dataset run folder.

    Raises:
        FileNotFoundError: If the dataset directory does not exist or has no subfolders.
    """
    dataset_root = data_dir / dataset_subdir
    if not dataset_root.exists():
        raise FileNotFoundError(f"No dataset folder found at {dataset_root}")

    subfolders = [folder for folder in dataset_root.iterdir() if folder.is_dir()]
    if not subfolders:
        raise FileNotFoundError(f"No dataset subfolders found in {dataset_root}")

    return max(subfolders, key=lambda folder: folder.stat().st_mtime)
    