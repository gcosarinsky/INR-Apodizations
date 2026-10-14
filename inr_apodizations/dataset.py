from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import numpy as np
import tensorflow as tf


def generate_unit_gaussian_mask(scatterers, x_grid, z_grid, sigma_x=0.001, sigma_z=0.001, background_points=None):
    """Generate a unit-amplitude Gaussian mask centered at scatterer positions.

    Args:
        scatterers: Array-like with shape ``(n_scatterers, >=2)`` where the first
            two columns contain ``x`` and ``z`` coordinates.
        x_grid: Meshgrid x coordinates with shape ``(nz, nx)``. Ignored when
            ``background_points`` is provided.
        z_grid: Meshgrid z coordinates with shape ``(nz, nx)``. Ignored when
            ``background_points`` is provided.
        sigma_x: Lateral Gaussian standard deviation.
        sigma_z: Axial Gaussian standard deviation.
        background_points: Optional array with shape ``(n_points, 2)`` containing
            ``(x, z)`` coordinates where the mask should be evaluated.

    Returns:
        A ``float32`` mask with shape ``(nz, nx)`` when evaluated on a grid, or
        shape ``(n_points,)`` when evaluated on ``background_points``.
    """
    scatterers_array = np.asarray(scatterers, dtype=np.float32)
    sigma_x = np.float32(sigma_x)
    sigma_z = np.float32(sigma_z)
    inv_two_sigma_x2 = np.float32(1.0) / (np.float32(2.0) * sigma_x * sigma_x)
    inv_two_sigma_z2 = np.float32(1.0) / (np.float32(2.0) * sigma_z * sigma_z)

    if scatterers_array.size == 0:
        if background_points is not None:
            return np.zeros(len(background_points), dtype=np.float32)
        return np.zeros_like(x_grid, dtype=np.float32)

    scatterers_x = scatterers_array[:, 0][:, None]
    scatterers_z = scatterers_array[:, 1][:, None]

    if background_points is not None:
        points = np.asarray(background_points, dtype=np.float32)
        dx = points[:, 0][None, :] - scatterers_x
        dz = points[:, 1][None, :] - scatterers_z
        exponent = -(dx * dx * inv_two_sigma_x2 + dz * dz * inv_two_sigma_z2)
        return np.exp(exponent, dtype=np.float32).sum(axis=0, dtype=np.float32)

    x_grid_array = np.asarray(x_grid, dtype=np.float32)
    z_grid_array = np.asarray(z_grid, dtype=np.float32)
    dx = x_grid_array[None, :, :] - scatterers_x[:, None, :]
    dz = z_grid_array[None, :, :] - scatterers_z[:, None, :]
    exponent = -(dx * dx * inv_two_sigma_x2 + dz * dz * inv_two_sigma_z2)
    return np.exp(exponent, dtype=np.float32).sum(axis=0, dtype=np.float32)


def generate_das_modulated_target(das_image, scatterers, x_grid, z_grid, sigma_x=0.001, sigma_z=0.001, alpha=0.0):
    """
    Generate target by modulating a DAS image with a unit-amplitude Gaussian mask.

    das_image: ndarray (nz, nx), real or complex
    scatterers: (n_scatterers, >=2) - columns include x, z as first two values
    x_grid, z_grid: meshgrid arrays of shape (nz, nx)
    alpha: Blending factor for relaxed targets. When alpha=0, uses the modulated target. When alpha>0, adds a constant background to avoid zeros.

    TODO: target could be RF instead of abs
    """
    assert alpha >= 0.0 and alpha < 1.0, "alpha must be in the range [0, 1)"
    gaussian_mask = generate_unit_gaussian_mask(
        scatterers,
        x_grid,
        z_grid,
        sigma_x=sigma_x,
        sigma_z=sigma_z,
    )
    relaxed_mask = alpha + (1 - alpha) * gaussian_mask
    target = np.abs(das_image).astype(np.float32) * relaxed_mask
    return target.astype(np.float32)


def split_train_validation_examples(
    delayed: np.ndarray,
    targets: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the dataset along the example axis.

    If the dataset contains a single example, the same example is used for
    both training and validation so sandbox flows remain runnable.
    """
    if delayed.shape[0] != targets.shape[0]:
        raise ValueError("delayed and targets must have the same number of examples")

    n_examples = delayed.shape[0]
    if n_examples == 1:
        return delayed, targets, delayed.copy(), targets.copy()

    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError("train_fraction must be in the open interval (0, 1)")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_examples)
    rng.shuffle(indices)

    n_train = max(1, int(np.floor(n_examples * train_fraction)))
    n_train = min(n_train, n_examples - 1)
    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:])

    return delayed[train_idx], targets[train_idx], delayed[val_idx], targets[val_idx]


def split_train_validation_indices(
    n_examples: int,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split example indices into train/validation subsets.

    If there is a single example, the same index is returned for both
    training and validation so sandbox flows remain runnable.
    """
    if n_examples <= 0:
        raise ValueError("n_examples must be > 0")

    if n_examples == 1:
        idx = np.array([0], dtype=np.int64)
        return idx, idx.copy()

    if train_fraction <= 0.0 or train_fraction >= 1.0:
        raise ValueError("train_fraction must be in the open interval (0, 1)")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_examples, dtype=np.int64)
    rng.shuffle(indices)

    n_train = max(1, int(np.floor(n_examples * train_fraction)))
    n_train = min(n_train, n_examples - 1)
    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:])
    return train_idx, val_idx


def build_tf_dataset_by_examples(
    delayed: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray | None = None,
    batch_size: int = 1,
    shuffle: bool = True,
    seed: int | None = 42,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from full arrays.

    Args:
        delayed: Complex delayed samples with shape ``(N, E, Z, X)``.
        targets: Target images with shape ``(N, Z, X)``.
        sample_weights: Optional per-pixel loss weights with shape ``(N, Z, X)``.
            When provided, dataset elements follow the Keras tuple contract
            ``(inputs, targets, sample_weights)``.
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle the dataset.
        seed: Random seed for shuffling.

    Returns:
        A ``tf.data.Dataset`` yielding batches of ``(delayed, target)`` or
        ``(delayed, target, weight)`` tuples.
    """
    n_examples = delayed.shape[0]
    if sample_weights is not None:
        ds = tf.data.Dataset.from_tensor_slices((delayed, targets, sample_weights))
    else:
        ds = tf.data.Dataset.from_tensor_slices((delayed, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=n_examples, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_tf_dataset_by_indices(
    delayed: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    sample_weights: np.ndarray | None = None,
    batch_size: int = 1,
    shuffle: bool = True,
    seed: int | None = 42,
) -> tf.data.Dataset:
    """Build a dataset that streams examples selected by index.

    This avoids materializing full delayed/target arrays as TensorFlow constants,
    which can trigger large GPU allocations during pipeline creation.

    Args:
        delayed: Delayed samples array with shape ``(N, E, Z, X)``.
        targets: Target images array with shape ``(N, Z, X)``.
        indices: Example indices selected for the split.
        sample_weights: Optional per-pixel loss weights with shape ``(N, Z, X)``.
            When provided, dataset elements follow the Keras tuple contract
            ``(inputs, targets, sample_weights)``.
        batch_size: Number of examples per batch.
        shuffle: Whether to shuffle the selected indices.
        seed: Random seed used for shuffling.

    Returns:
        A ``tf.data.Dataset`` yielding either ``(delayed, target)`` or
        ``(delayed, target, sample_weight)`` batches.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("indices must be a 1D array")

    if sample_weights is not None:
        weights = np.asarray(sample_weights)
        if weights.ndim != 3:
            raise ValueError("sample_weights must be (N, Z, X)")
        if weights.shape[0] != delayed.shape[0]:
            raise ValueError("sample_weights N dimension must match delayed/targets")
        if weights.shape[1:] != targets.shape[1:]:
            raise ValueError("sample_weights (Z, X) must match targets")

    ds = tf.data.Dataset.from_tensor_slices(idx)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(idx), seed=seed, reshuffle_each_iteration=True)

    delayed_shape = tuple(delayed.shape[1:])
    targets_shape = tuple(targets.shape[1:])
    weights_shape = tuple(targets.shape[1:])

    def _load_np(example_idx):
        i = int(example_idx)
        delayed_example = delayed[i].astype(np.complex64, copy=False)
        target_example = targets[i].astype(np.float32, copy=False)
        if sample_weights is None:
            return delayed_example, target_example
        weight_example = sample_weights[i].astype(np.float32, copy=False)
        return delayed_example, target_example, weight_example

    def _load_tf(example_idx):
        if sample_weights is None:
            delayed_example, target_example = tf.numpy_function(
                _load_np,
                [example_idx],
                [tf.complex64, tf.float32],
            )
            delayed_example.set_shape(delayed_shape)
            target_example.set_shape(targets_shape)
            return delayed_example, target_example

        delayed_example, target_example, weight_example = tf.numpy_function(
            _load_np,
            [example_idx],
            [tf.complex64, tf.float32, tf.float32],
        )
        delayed_example.set_shape(delayed_shape)
        target_example.set_shape(targets_shape)
        weight_example.set_shape(weights_shape)
        return delayed_example, target_example, weight_example

    ds = ds.map(_load_tf, num_parallel_calls=1)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(1)
    return ds


# # --------- From cookiecutter template, not used ---------
# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = RAW_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     # ----------------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Processing dataset...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Processing dataset complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()
