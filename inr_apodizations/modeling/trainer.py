"""Trainer wrapper that applies INR weights to delayed samples and reconstructs DAS.

Contains `DasInrTrainer`, a small Keras `Model` that runs the INR in chunks
and performs the element-wise weighting + summation physical forward.
"""
from __future__ import annotations

import tensorflow as tf


class DasInrTrainer(tf.keras.Model):
    """Keras model that wraps the INR and the physical DAS forward.

    The trainer evaluates the INR on a features grid (chunked to avoid large
    memory usage), reshapes the outputs into a `(E, Z, X)` weights grid, and
    applies those weights to delayed samples to produce a complex DAS image.
    The public API mirrors a typical Keras model so it can be compiled and
    trained with `model.fit`.
    """

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
