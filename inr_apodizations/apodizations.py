"""Dynamic apodization utilities based on TensorFlow tensors.

This module provides standard dynamic aperture apodizations and helper
functions to extract slices from apodization tensors.
"""

import math

import numpy as np
import tensorflow as tf


def compute_dynamic_apodizations_tf(
    cm,
    f_number,
    methods=("boxcar", "hanning"),
    scaled=False,
    chunk_size: int | None = None,
    eps=1e-8,
):
    """Compute dynamic apodizations from CoordinateManager feature tensors.

    Args:
        cm: CoordinateManager instance.
        f_number: F-number used for dynamic aperture, where ap_radius = z / (2 * f_number).
        methods: Iterable containing any of "boxcar" and "hanning".
        scaled: Whether to use scaled coordinates/features from CoordinateManager.
        chunk_size: Optional number of z samples processed per chunk.
            If ``None`` or non-positive, compute the full grid at once.
        eps: Small epsilon used to avoid division by zero.

    Returns:
        Dictionary mapping method names to tensors with shape (n_elem, nz, nx)
        and dtype tf.float32.

    Raises:
        ZeroDivisionError: If f_number is zero.
        ValueError: If chunk_size is invalid.
    """
    out = {}
    coords = cm.get_coordinates_1d(scaled=scaled)
    x_coords = tf.constant(np.asarray(coords["x"], dtype=np.float32))
    z_coords = tf.constant(np.asarray(coords["z"], dtype=np.float32))
    x_elem_coords = tf.constant(np.asarray(coords["x_elem"], dtype=np.float32))

    x_elem_grid = tf.reshape(x_elem_coords, [-1, 1, 1])
    dist = tf.abs(tf.reshape(x_coords, [1, 1, -1]) - x_elem_grid)

    pi = tf.constant(math.pi, dtype=tf.float32)

    if chunk_size is None or int(chunk_size) <= 0:
        z_chunks = [z_coords]
    else:
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer or None")
        z_chunks = [z_coords[i : i + chunk_size] for i in range(0, int(z_coords.shape[0]), chunk_size)]

    boxcar_chunks = []
    hanning_chunks = []
    need_boxcar = "boxcar" in methods
    need_hanning = "hanning" in methods

    for z_chunk in z_chunks:
        z_grid = tf.reshape(z_chunk, [1, -1, 1])
        ap_radius = z_grid / (2.0 * float(f_number))

        if need_boxcar:
            boxcar_chunks.append(tf.cast(dist <= ap_radius, tf.float32))

        if need_hanning:
            safe_ap = ap_radius + tf.cast(eps, tf.float32)
            ratio = dist / safe_ap
            mask = ratio <= 1.0
            w_h = 0.5 * (1.0 + tf.cos(pi * ratio))
            w_h = tf.where(mask, w_h, tf.zeros_like(w_h))
            hanning_chunks.append(tf.cast(w_h, tf.float32))

    if "boxcar" in methods:
        out["boxcar"] = tf.concat(boxcar_chunks, axis=1) if boxcar_chunks else tf.zeros((x_elem_coords.shape[0], 0, x_coords.shape[0]), dtype=tf.float32)

    if "hanning" in methods:
        out["hanning"] = tf.concat(hanning_chunks, axis=1) if hanning_chunks else tf.zeros((x_elem_coords.shape[0], 0, x_coords.shape[0]), dtype=tf.float32)

    return out


def extract_map_for_x(apod_tensor, cm, x_fixed, scaled=False):
    """Extract a depth-element map for a fixed lateral x coordinate.

    Args:
        apod_tensor: Tensor with shape (n_elem, nz, nx).
        cm: CoordinateManager instance.
        x_fixed: Lateral x in the same units as CoordinateManager coordinates.
        scaled: Whether to use scaled coordinates.

    Returns:
        Tensor with shape (nz, n_elem).
    """
    coords = cm.get_coordinates_1d(scaled=scaled)
    x_coords = np.asarray(coords["x"])
    idx = int(np.argmin(np.abs(x_coords - x_fixed)))
    slice_nelem_nz = apod_tensor[:, :, idx]
    return tf.transpose(slice_nelem_nz, perm=[1, 0])


def extract_profile_for_z(apod_tensor, cm, z_fixed, x_fixed=0.0, scaled=False):
    """Extract an apodization profile at a fixed depth and lateral coordinate.

    Args:
        apod_tensor: Tensor with shape (n_elem, nz, nx).
        cm: CoordinateManager instance.
        z_fixed: Depth value.
        x_fixed: Lateral x value. No averaging across x is performed.
        scaled: Whether to use scaled coordinates.

    Returns:
        Tensor with shape (n_elem,) and dtype tf.float32.
    """
    coords = cm.get_coordinates_1d(scaled=scaled)
    z_coords = np.asarray(coords["z"])
    z_idx = int(np.argmin(np.abs(z_coords - z_fixed)))

    x_coords = np.asarray(coords["x"])
    x_idx = int(np.argmin(np.abs(x_coords - x_fixed)))
    slice_nelem = apod_tensor[:, z_idx, x_idx]
    return tf.cast(slice_nelem, tf.float32)


def compute_das_baseline_numpy(
    delayed_samples: np.ndarray,
    apodization: np.ndarray | None = None,
    return_complex: bool = False,
) -> np.ndarray:
    """Compute baseline DAS reconstruction in NumPy.

    Args:
        delayed_samples: Delayed samples with shape ``(E, Z, X)`` or
            ``(B, E, Z, X)`` and complex dtype.
        apodization: Optional apodization map with shape ``(E, Z, X)``.
            If ``None``, the baseline is uniform (all-ones weights).
        return_complex: If ``True``, return the complex DAS image.
            Otherwise return ``abs(image)`` as ``float32``.

    Returns:
        Reconstructed DAS image with shape ``(Z, X)`` for 3D input or
        ``(B, Z, X)`` for 4D input.

    Raises:
        ValueError: If input shapes are invalid or incompatible.
    """
    delayed_np = np.asarray(delayed_samples)
    if delayed_np.ndim not in (3, 4):
        raise ValueError(
            "delayed_samples must have shape (E, Z, X) or (B, E, Z, X); "
            f"got shape {delayed_np.shape}"
        )

    delayed_np = delayed_np.astype(np.complex64, copy=False)
    elem_axis = 0 if delayed_np.ndim == 3 else 1

    if apodization is None:
        weighted = delayed_np
    else:
        apod_np = np.asarray(apodization)
        if apod_np.ndim != 3:
            raise ValueError(
                "apodization must have shape (E, Z, X); "
                f"got shape {apod_np.shape}"
            )
        expected_shape = delayed_np.shape if delayed_np.ndim == 3 else delayed_np.shape[1:]
        if apod_np.shape != expected_shape:
            raise ValueError(
                "apodization shape mismatch; expected "
                f"{expected_shape}, got {apod_np.shape}"
            )
        weighted = delayed_np * apod_np.astype(np.complex64, copy=False)

    image_complex = np.sum(weighted, axis=elem_axis)
    if return_complex:
        return image_complex.astype(np.complex64, copy=False)
    return np.abs(image_complex).astype(np.float32, copy=False)


def _compute_nsi_geometry(
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    x_elem_coords: np.ndarray,
    f_number: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute index geometry used by NSI half-subapertures.

    Args:
        x_coords: Lateral pixel coordinates with shape ``(X,)``.
        z_coords: Axial pixel coordinates with shape ``(Z,)``.
        x_elem_coords: Element lateral coordinates with shape ``(E,)``.
        f_number: Dynamic aperture F-number.

    Returns:
        Tuple ``(center_idx, border_limit, mx_per_z)`` where:
            - ``center_idx`` has shape ``(X,)`` and stores the closest element
              index for each lateral pixel.
            - ``border_limit`` has shape ``(X,)`` and stores the maximum valid
              half-subaperture size that preserves symmetry at each x.
            - ``mx_per_z`` has shape ``(Z,)`` and stores the requested half size
              from the dynamic F-number rule.

    Raises:
        ValueError: If ``f_number`` or element spacing are invalid.
    """
    if f_number <= 0:
        raise ValueError(f"f_number must be > 0, got {f_number}")

    elem_step = np.diff(np.asarray(x_elem_coords, dtype=np.float32))
    elem_pitch = float(np.median(np.abs(elem_step))) if elem_step.size > 0 else 0.0
    if elem_pitch <= 0.0:
        raise ValueError("x_elem_coords must contain at least 2 monotonic elements")

    x_coords_np = np.asarray(x_coords, dtype=np.float32)
    z_coords_np = np.asarray(z_coords, dtype=np.float32)
    x_elem_np = np.asarray(x_elem_coords, dtype=np.float32)

    center_idx = np.argmin(
        np.abs(x_elem_np[:, None] - x_coords_np[None, :]),
        axis=0,
    ).astype(np.int32)

    n_elem = int(x_elem_np.shape[0])
    max_left = center_idx
    max_right = n_elem - center_idx
    border_limit = np.minimum(max_left, max_right).astype(np.int32)

    z_positive = np.maximum(z_coords_np, np.float32(0.0))
    mx_per_z = np.rint(z_positive / (np.float32(elem_pitch) * np.float32(f_number))).astype(np.int32)
    mx_per_z = np.maximum(mx_per_z, 1)

    return center_idx, border_limit, mx_per_z


def _compute_nsi_sum_diff_single(
    delayed_samples_single: np.ndarray,
    center_idx: np.ndarray,
    border_limit: np.ndarray,
    mx_per_z: np.ndarray,
    normalize_by_n_subap: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute NSI sum and difference images for one delayed-samples frame.

    Args:
        delayed_samples_single: Complex delayed samples with shape ``(E, Z, X)``.
        center_idx: Closest-element indices per x with shape ``(X,)``.
        border_limit: Max symmetric half-size per x with shape ``(X,)``.
        mx_per_z: Requested half-size per z with shape ``(Z,)``.
        normalize_by_n_subap: If ``True``, divide outputs by ``n_subap``.

    Returns:
        Tuple ``(img_sum, img_diff)`` as ``complex64`` arrays with shape ``(Z, X)``.
    """
    delayed_np = np.asarray(delayed_samples_single, dtype=np.complex64)
    n_elem, nz, nx = delayed_np.shape

    img_sum = np.zeros((nz, nx), dtype=np.complex64)
    img_diff = np.zeros((nz, nx), dtype=np.complex64)

    for iz in range(nz):
        mx_requested = int(mx_per_z[iz])
        for ix in range(nx):
            half_size = min(mx_requested, int(border_limit[ix]))
            if half_size < 1:
                continue

            c_idx = int(center_idx[ix])
            left_start = c_idx - half_size
            left_end = c_idx
            right_start = c_idx
            right_end = c_idx + half_size

            if left_start < 0 or right_end > n_elem:
                # Keep safe behavior for extreme edge pixels.
                continue

            q_left = np.sum(delayed_np[left_start:left_end, iz, ix], dtype=np.complex64)
            q_right = np.sum(delayed_np[right_start:right_end, iz, ix], dtype=np.complex64)

            q_sum = q_left + q_right
            q_diff = q_right - q_left

            if normalize_by_n_subap:
                n_subap = np.float32(2 * half_size)
                q_sum = q_sum / n_subap
                q_diff = q_diff / n_subap

            img_sum[iz, ix] = np.complex64(q_sum)
            img_diff[iz, ix] = np.complex64(q_diff)

    return img_sum, img_diff


def compute_nsi_from_delayed_samples_numpy(
    delayed_samples: np.ndarray,
    x_coords: np.ndarray,
    z_coords: np.ndarray,
    x_elem_coords: np.ndarray,
    f_number: float,
    dc: float = 0.05,
    normalize_by_n_subap: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute NSI from delayed samples using symmetric half-subapertures.

    This implementation follows the NSI v2 concept with dynamic boxcar
    subapertures and per-pixel symmetric half splits. By default, normalization
    by ``n_subap`` is disabled to preserve the original amplitude behavior.

    Args:
        delayed_samples: Complex delayed samples with shape ``(E, Z, X)`` or
            ``(B, E, Z, X)``.
        x_coords: Lateral coordinates for image columns with shape ``(X,)``.
        z_coords: Axial coordinates for image rows with shape ``(Z,)``.
        x_elem_coords: Lateral coordinates of probe elements with shape ``(E,)``.
        f_number: Dynamic aperture F-number.
        dc: NSI recombination offset.
        normalize_by_n_subap: If ``True``, divide ``S`` and ``D`` by
            subaperture size. Defaults to ``False``.

    Returns:
        Tuple ``(img_sum, img_diff, nsi_img)`` where:
            - ``img_sum`` is complex64 with shape ``(Z, X)`` or ``(B, Z, X)``.
            - ``img_diff`` is complex64 with shape ``(Z, X)`` or ``(B, Z, X)``.
            - ``nsi_img`` is float32 with shape ``(Z, X)`` or ``(B, Z, X)``.

    Raises:
        ValueError: If input shapes or dtypes are invalid.
    """
    delayed_np = np.asarray(delayed_samples)
    if delayed_np.ndim not in (3, 4):
        raise ValueError(
            "delayed_samples must have shape (E, Z, X) or (B, E, Z, X); "
            f"got shape {delayed_np.shape}"
        )
    if not np.iscomplexobj(delayed_np):
        raise ValueError("delayed_samples must be complex-valued")

    delayed_np = delayed_np.astype(np.complex64, copy=False)

    x_coords_np = np.asarray(x_coords, dtype=np.float32)
    z_coords_np = np.asarray(z_coords, dtype=np.float32)
    x_elem_np = np.asarray(x_elem_coords, dtype=np.float32)

    if x_coords_np.ndim != 1 or z_coords_np.ndim != 1 or x_elem_np.ndim != 1:
        raise ValueError("x_coords, z_coords and x_elem_coords must be 1D arrays")

    expected_3d = (x_elem_np.shape[0], z_coords_np.shape[0], x_coords_np.shape[0])
    if delayed_np.ndim == 3 and delayed_np.shape != expected_3d:
        raise ValueError(
            "Shape mismatch for delayed_samples (E, Z, X). Expected "
            f"{expected_3d}, got {delayed_np.shape}"
        )
    if delayed_np.ndim == 4 and delayed_np.shape[1:] != expected_3d:
        raise ValueError(
            "Shape mismatch for delayed_samples (B, E, Z, X). Expected trailing "
            f"shape {expected_3d}, got {delayed_np.shape[1:]}"
        )

    center_idx, border_limit, mx_per_z = _compute_nsi_geometry(
        x_coords=x_coords_np,
        z_coords=z_coords_np,
        x_elem_coords=x_elem_np,
        f_number=float(f_number),
    )

    if delayed_np.ndim == 3:
        img_sum, img_diff = _compute_nsi_sum_diff_single(
            delayed_samples_single=delayed_np,
            center_idx=center_idx,
            border_limit=border_limit,
            mx_per_z=mx_per_z,
            normalize_by_n_subap=bool(normalize_by_n_subap),
        )
    else:
        batch_size = delayed_np.shape[0]
        img_sum = np.zeros((batch_size, z_coords_np.shape[0], x_coords_np.shape[0]), dtype=np.complex64)
        img_diff = np.zeros_like(img_sum)
        for ib in range(batch_size):
            img_sum_b, img_diff_b = _compute_nsi_sum_diff_single(
                delayed_samples_single=delayed_np[ib],
                center_idx=center_idx,
                border_limit=border_limit,
                mx_per_z=mx_per_z,
                normalize_by_n_subap=bool(normalize_by_n_subap),
            )
            img_sum[ib] = img_sum_b
            img_diff[ib] = img_diff_b

    dc32 = np.float32(dc)
    nsi_img = (
        0.5
        * (np.abs(img_diff + dc32 * img_sum) + np.abs(-img_diff + dc32 * img_sum))
        - np.abs(img_diff)
    ).astype(np.float32, copy=False)

    return img_sum, img_diff, nsi_img


__all__ = [
    "compute_dynamic_apodizations_tf",
    "extract_map_for_x",
    "extract_profile_for_z",
    "compute_das_baseline_numpy",
    "compute_nsi_from_delayed_samples_numpy",
]
