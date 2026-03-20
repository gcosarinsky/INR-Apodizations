import os
import math
import numpy as np
import tensorflow as tf


def compute_dynamic_apodizations_tf(cm, bfd, methods=('boxcar', 'hanning'), scaled=False, eps=1e-8):
    """
    Compute dynamic apodizations using CoordinateManager tensors (TensorFlow).

    Args:
        cm: CoordinateManager instance
        bfd: twice f-number (float) used for dynamic aperture: ap_radius = z / bfd
        methods: iterable with any of 'boxcar' (rectangular) and 'hanning'
        scaled: whether to use scaled coordinates/features from `cm`
        eps: small epsilon to avoid division by zero

    Returns:
        dict: method -> tf.Tensor with shape (n_elem, nz, nx), dtype tf.float32
    """
    features = cm.get_features_grid(scaled=scaled)  # (n_elem, nz, nx, 3)
    # features[..., 0] == |x - x_elem|, features[..., 1] == z
    dist = tf.cast(features[..., 0], tf.float32)
    depth = tf.cast(features[..., 1], tf.float32)

    ap_radius = depth / float(bfd)

    out = {}
    pi = tf.constant(math.pi, dtype=tf.float32)

    if 'boxcar' in methods:
        w_box = tf.cast(dist <= ap_radius, tf.float32)
        out['boxcar'] = w_box

    if 'hanning' in methods:
        safe_ap = ap_radius + tf.cast(eps, tf.float32)
        ratio = dist / safe_ap
        mask = ratio <= 1.0
        w_h = 0.5 * (1.0 + tf.cos(pi * ratio))
        w_h = tf.where(mask, w_h, tf.zeros_like(w_h))
        out['hanning'] = tf.cast(w_h, tf.float32)

    return out


def extract_map_for_x(apod_tensor, cm, x_fixed, scaled=False):
    """
    Extract a map for a fixed lateral x value: returns array (nz, n_elem)

    Args:
        apod_tensor: tf.Tensor (n_elem, nz, nx)
        cm: CoordinateManager
        x_fixed: lateral x (same units as cm coordinates)
        scaled: whether to use scaled coords

    Returns:
        tf.Tensor with shape (nz, n_elem)
    """
    coords = cm.get_coordinates_1d(scaled=scaled)
    x_coords = np.asarray(coords['x'])
    idx = int(np.argmin(np.abs(x_coords - x_fixed)))
    # apod_tensor[:, :, idx] -> (n_elem, nz)
    slice_nelem_nz = apod_tensor[:, :, idx]
    # transpose to (nz, n_elem)
    return tf.transpose(slice_nelem_nz, perm=[1, 0])


def extract_profile_for_z(apod_tensor, cm, z_fixed, x_fixed=0.0, scaled=False):
    """
    Extract apodization profile at a given depth z for a specific lateral x.

    Args:
        apod_tensor: tf.Tensor (n_elem, nz, nx)
        cm: CoordinateManager
        z_fixed: depth value
        x_fixed: lateral x value (default 0.0). No averaging across x is performed.
        scaled: whether coords are scaled

    Returns:
        tf.Tensor with shape (n_elem,) : apod vs element
    """
    coords = cm.get_coordinates_1d(scaled=scaled)
    z_coords = np.asarray(coords['z'])
    z_idx = int(np.argmin(np.abs(z_coords - z_fixed)))

    x_coords = np.asarray(coords['x'])
    x_idx = int(np.argmin(np.abs(x_coords - x_fixed)))
    slice_nelem = apod_tensor[:, z_idx, x_idx]
    return tf.cast(slice_nelem, tf.float32)
