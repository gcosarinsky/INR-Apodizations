"""Custom loss utilities for INR‑Apodizations.

This module currently provides :class:`ScaledLoss`, a thin wrapper around any
TensorFlow/Keras loss that records the loss value observed on the first
training batch and divides all subsequent loss values by that initial scalar.

The purpose is to normalise the loss magnitude across experiments without
introducing an additional multiplicative factor – the loss is simply scaled by
``1 / initial_loss``.

The class inherits from ``tf.keras.losses.Loss`` so it can be saved and
restored together with a model checkpoint.
"""

from __future__ import annotations

import tensorflow as tf


def masked_mae_absolute_error(y_true, y_pred):
    """Return element-wise absolute error used by Masked MAE.

    Args:
        y_true: Ground-truth tensor.
        y_pred: Predicted tensor.

    Returns:
        Tensor with element-wise absolute error in float32.
    """
    return tf.abs(tf.cast(y_true, tf.float32) - tf.cast(y_pred, tf.float32))


class MaskedMAELoss(tf.keras.losses.Loss):
    """Pixel-wise MAE loss intended for per-pixel sample weighting.

    This loss returns element-wise absolute error and delegates reduction and
    weighting to Keras through ``sample_weight`` passed by ``Model.fit``.
    Combined with per-pixel sample weights, this produces the weighted MAE
    objective used by sandbox INR-DAS experiments.
    """

    def __init__(self, name: str = "masked_mae_loss"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, y_true, y_pred):  # type: ignore[override]
        """Compute element-wise absolute error for weighted reduction."""
        return masked_mae_absolute_error(y_true, y_pred)


class ScaledLoss(tf.keras.losses.Loss):
    """Wrap a base loss and divide by the first‑batch loss value.

    Parameters
    ----------
    base_loss : tf.keras.losses.Loss or callable
        The original loss to be wrapped (e.g. ``MeanAbsoluteError``).
    name : str, optional
        Optional name for the loss. If omitted, ``scaled_<base.name>`` is used.
    """

    def __init__(self, base_loss: tf.keras.losses.Loss | callable, name: str | None = None):
        super().__init__(name=name or f"scaled_{getattr(base_loss, 'name', 'loss')}")
        self.base_loss = base_loss
        # Use a non‑trainable variable to store the first‑batch loss so it is
        # available across tf.function calls. Initialise with NaN to detect the
        # first invocation.
        # Use a plain tf.Variable because tf.keras.losses.Loss does not expose
        # the ``add_weight`` helper that ``tf.keras.layers.Layer`` provides.
        self._initial = tf.Variable(
            initial_value=float('nan'),
            dtype=tf.float32,
            trainable=False,
            name="initial_loss",
        )

    def call(self, y_true, y_pred):  # type: ignore[override]
        """Compute the wrapped loss and scale it.

        The raw loss is first computed using ``self.base_loss``. On the very
        first call the mean of that loss is stored as ``_initial``. All later
        calls divide the raw loss by this stored scalar. A safeguard replaces
        a zero initial loss with ``1.0`` to avoid division‑by‑zero errors.
        """
        raw = self.base_loss(y_true, y_pred)
        raw_mean = tf.reduce_mean(raw)

        # Detect first call by checking for NaN (the variable is initialised with NaN)
        if tf.math.is_nan(self._initial):
            # Guard against zero division – replace 0 with 1.0
            init_val = tf.where(tf.equal(raw_mean, 0.0), tf.constant(1.0, dtype=raw_mean.dtype), raw_mean)
            self._initial.assign(init_val)
            tf.print("🔧 ScaledLoss – captured initial loss:", self._initial)

        divisor = tf.where(tf.equal(self._initial, 0.0), tf.constant(1.0, dtype=self._initial.dtype), self._initial)
        return raw / divisor
