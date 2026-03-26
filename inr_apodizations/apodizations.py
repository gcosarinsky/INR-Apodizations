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
    eps=1e-8,
):
    """Compute dynamic apodizations from CoordinateManager feature tensors.

    Args:
        cm: CoordinateManager instance.
        f_number: F-number used for dynamic aperture, where ap_radius = z / (2 * f_number).
        methods: Iterable containing any of "boxcar" and "hanning".
        scaled: Whether to use scaled coordinates/features from CoordinateManager.
        eps: Small epsilon used to avoid division by zero.

    Returns:
        Dictionary mapping method names to tensors with shape (n_elem, nz, nx)
        and dtype tf.float32.

    Raises:
        ZeroDivisionError: If f_number is zero.
    """
    features = cm.get_features_grid(scaled=scaled)
    dist = tf.cast(features[..., 0], tf.float32)
    depth = tf.cast(features[..., 1], tf.float32)

    ap_radius = depth / (2.0 * float(f_number))

    out = {}
    pi = tf.constant(math.pi, dtype=tf.float32)

    if "boxcar" in methods:
        w_box = tf.cast(dist <= ap_radius, tf.float32)
        out["boxcar"] = w_box

    if "hanning" in methods:
        safe_ap = ap_radius + tf.cast(eps, tf.float32)
        ratio = dist / safe_ap
        mask = ratio <= 1.0
        w_h = 0.5 * (1.0 + tf.cos(pi * ratio))
        w_h = tf.where(mask, w_h, tf.zeros_like(w_h))
        out["hanning"] = tf.cast(w_h, tf.float32)

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


__all__ = [
    "compute_dynamic_apodizations_tf",
    "extract_map_for_x",
    "extract_profile_for_z",
]
