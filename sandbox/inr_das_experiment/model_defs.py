"""Shared model and metric definitions for the sandbox INR DAS experiment.

This module is import-safe: it only defines functions and classes and does not
execute training pipelines at import time.
"""
from __future__ import annotations

import tensorflow as tf


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
        - `tf.image.ssim` expects images in a known dynamic range; set ``max_val``
          accordingly (e.g. 1.0 if images are normalized to [0, 1]).
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
    # tf.image.ssim expects images with a channels dimension; add one if missing.
    if tf.rank(y_true) == 3:
        y_true = y_true[..., tf.newaxis]
        y_pred = y_pred[..., tf.newaxis]
    # Use a small filter size to support small spatial dims (avoids negative output size).
    # 11 is the default but fails when height/width < 11; 3 is safer and still meaningful.
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


class DasInrTrainer(tf.keras.Model):
    """Keras model that wraps the INR and the physical DAS forward."""

    def __init__(self, apodization_model: tf.keras.Model, features_grid: tf.Tensor, feature_chunk_size: int):
        """Initialize trainer with INR model and feature grid.

        Args:
            apodization_model: INR model mapping geometry features to weights.
            features_grid: Feature tensor with shape (E, Z, X, 3).
            feature_chunk_size: Number of feature rows processed per forward chunk.
        """
        super().__init__(name="das_inr_trainer")
        self.apodization_model = apodization_model
        self.features_flat = tf.reshape(tf.cast(features_grid, tf.float32), (-1, 3))
        self.n_elem, self.nz, self.nx = [int(dim) for dim in features_grid.shape[:3]]
        self.feature_chunk_size = int(feature_chunk_size)

    def predict_weights_grid(self, training: bool = False) -> tf.Tensor:
        """Run the INR on geometry features and reshape to ``(E, Z, X)``."""
        n_features = int(self.features_flat.shape[0])
        predictions = []
        for start_idx in range(0, n_features, self.feature_chunk_size):
            end_idx = min(start_idx + self.feature_chunk_size, n_features)
            chunk_pred = self.apodization_model(self.features_flat[start_idx:end_idx], training=training)
            predictions.append(chunk_pred)

        weights_flat = tf.concat(predictions, axis=0)
        weights_grid = tf.reshape(weights_flat, (self.n_elem, self.nz, self.nx))
        return tf.cast(weights_grid, tf.float32)

    def reconstruct_image(self, delayed_batch: tf.Tensor, training: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
        """Apply weights to delayed samples and reconstruct ``abs(DAS)`` images."""
        weights_grid = self.predict_weights_grid(training=training)
        weighted_delayed = delayed_batch * tf.cast(weights_grid[tf.newaxis, ...], delayed_batch.dtype)
        predicted_complex = tf.reduce_sum(weighted_delayed, axis=1)
        predicted_image = tf.abs(predicted_complex)
        return predicted_image, weights_grid

    def call(self, delayed_batch: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Run forward reconstruction with chunked INR evaluation.

        Args:
            delayed_batch: Delayed samples with shape ``(B, E, Z, X)``.
            training: Whether to run the INR in training mode.

        Returns:
            Predicted magnitude image with shape ``(B, Z, X)``.
        """
        predicted_image, _ = self.reconstruct_image(delayed_batch, training=training)
        return predicted_image
