"""Reusable metrics for INR DAS experiments.

This module contains image-level metrics exported for use by training
scripts and experiments.

Note:
    The following implementations in this module are deprecated and kept
    for reference only: `rmse`, `ssim_metric`, and `GlobalRMSE`. They were
    experimental tests and are not used by the current pipelines. At the
    moment training and evaluation rely on TensorFlow/Keras built-in
    losses/metrics (e.g. `tf.keras.losses`, `tf.keras.metrics`). Custom
    losses and metrics may be added later with proper tests and integration.


"""
from __future__ import annotations

import tensorflow as tf

from inr_apodizations.utils import to_db_tensor


def mae_db(
    y_true,
    y_pred,
    ref: tuple[float, float] | None = None,
    eps: float = 1e-8,
):
    """Compute MAE between prediction and target in the dB domain.

    Args:
        y_true: Target image tensor (real or complex).
        y_pred: Predicted image tensor (real or complex).
        ref: Optional tuple ``(ref_image, ref_target)`` used as independent
            amplitude references for prediction and target dB conversion.
            If ``None``, uses a shared reference
            ``max(max(abs(y_true)), max(abs(y_pred)))``.
        eps: Small positive constant to stabilize dB conversion.

    Returns:
        Scalar tensor with mean absolute error in dB.

    Raises:
        tf.errors.InvalidArgumentError: If ``eps`` is not strictly positive.
    """
    mag_true = tf.cast(tf.abs(y_true), tf.float32)
    mag_pred = tf.cast(tf.abs(y_pred), tf.float32)

    if ref is None:
        shared_ref = tf.maximum(tf.reduce_max(mag_true), tf.reduce_max(mag_pred))
        ref_image = shared_ref
        ref_target = shared_ref
    else:
        if not isinstance(ref, tuple) or len(ref) != 2:
            raise ValueError("ref must be a tuple: (ref_image, ref_target)")
        ref_image, ref_target = ref

    y_true_db = to_db_tensor(y_true, ref=ref_target, eps=eps)
    y_pred_db = to_db_tensor(y_pred, ref=ref_image, eps=eps)
    return tf.reduce_mean(tf.abs(y_true_db - y_pred_db))


def mae_db_factory(
    ref: tuple[float, float] | None = None,
    eps: float = 1e-8,
    name: str = "mae_db",
):
    """Build a Keras-compatible callable metric for MAE in dB.

    Args:
        ref: Optional tuple ``(ref_image, ref_target)`` for dB conversion.
        eps: Small positive constant to stabilize dB conversion.
        name: Metric function name shown by Keras in logs/history.

    Returns:
        Callable metric ``metric(y_true, y_pred)`` that computes MAE in dB.
    """

    def metric(y_true, y_pred):
        return mae_db(y_true, y_pred, ref=ref, eps=eps)

    metric.__name__ = name
    return metric


def rmse(y_true, y_pred):
    """Compute RMSE between target and prediction images.

    Args:
        y_true: Target image tensor.
        y_pred: Predicted image tensor.

    Returns:
        Scalar RMSE tensor.
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


def ssim_metric(y_true, y_pred, max_val: float = 1.0):
    """Compute mean SSIM across a batch.

    Notes:
        - `tf.image.ssim` expects images in a known dynamic range; set
          ``max_val`` accordingly (e.g. 1.0 if images are normalized to [0, 1]).
        - Returned value is averaged across the batch.

    Args:
        y_true: Ground-truth image tensor.
        y_pred: Predicted image tensor.
        max_val: Maximum possible pixel value (for SSIM computation).

    Returns:
        Scalar tensor with mean SSIM over the batch.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    if tf.rank(y_true) == 3:
        y_true = y_true[..., tf.newaxis]
        y_pred = y_pred[..., tf.newaxis]
    ssim_per_example = tf.image.ssim(y_true, y_pred, max_val=max_val, filter_size=3)
    return tf.reduce_mean(ssim_per_example)


class GlobalRMSE(tf.keras.metrics.Metric):
    """Accumulate sum-of-squares and count to compute global RMSE per epoch.

    This metric accumulates SSE and total element count across batches and
    returns sqrt(sse / count) as the RMSE. It is robust to variable batch
    sizes and dataset sizes used for evaluation.
    """

    def __init__(self, name: str = "rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sse = self.add_weight(name="sse", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update accumulators with a new batch."""
        diff = tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32)
        sse_batch = tf.reduce_sum(tf.square(diff))
        n = tf.cast(tf.size(diff), tf.float32)
        self.sse.assign_add(sse_batch)
        self.count.assign_add(n)

    def result(self):
        """Return global RMSE from accumulated SSE and count."""
        return tf.sqrt(self.sse / (self.count + 1e-12))

    def reset_states(self):
        """Reset metric internal state."""
        self.sse.assign(0.0)
        self.count.assign(0.0)
