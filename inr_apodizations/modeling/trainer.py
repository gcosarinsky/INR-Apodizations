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

    def __init__(
        self,
        apodization_model: tf.keras.Model,
        features_grid: tf.Tensor,
        feature_chunk_size: int,
        weight_regularization_enabled: bool = False,
        weight_regularization_lambda: float = 1e-3,
        weight_regularization_tau: float = 0.30,
        weight_regularization_epsilon: float = 1e-8,
        weight_regularization_normalize: bool = True,
    ):
        """Initialize trainer with INR model and feature grid.

        Args:
            apodization_model: INR model mapping geometry features to weights.
            features_grid: Feature tensor with shape (E, Z, X, F).
            feature_chunk_size: Number of feature rows processed per forward chunk.
            weight_regularization_enabled: Whether to enable low-norm hinge regularization.
            weight_regularization_lambda: Multiplicative factor for regularization loss.
            weight_regularization_tau: Minimum target norm before hinge becomes active.
            weight_regularization_epsilon: Numerical epsilon used in norm normalization.
            weight_regularization_normalize: Whether to normalize norm by ``sqrt(E*Z*X)``.

        Raises:
            ValueError: If regularization hyperparameters are invalid.
        """
        super().__init__(name="das_inr_trainer")
        self.apodization_model = apodization_model
        self.n_features_per_point = int(features_grid.shape[-1])
        self.features_flat = tf.reshape(
            tf.cast(features_grid, tf.float32),
            (-1, self.n_features_per_point)
        )
        self.n_elem, self.nz, self.nx = [int(dim) for dim in features_grid.shape[:3]]
        self.feature_chunk_size = int(feature_chunk_size)

        self.weight_regularization_enabled = bool(weight_regularization_enabled)
        self.weight_regularization_lambda = float(weight_regularization_lambda)
        self.weight_regularization_tau = float(weight_regularization_tau)
        self.weight_regularization_epsilon = float(weight_regularization_epsilon)
        self.weight_regularization_normalize = bool(weight_regularization_normalize)
        if self.weight_regularization_lambda < 0.0:
            raise ValueError("weight_regularization_lambda must be >= 0")
        if self.weight_regularization_tau < 0.0:
            raise ValueError("weight_regularization_tau must be >= 0")
        if self.weight_regularization_epsilon <= 0.0:
            raise ValueError("weight_regularization_epsilon must be > 0")

        self._norm_denominator = float((self.n_elem * self.nz * self.nx) ** 0.5)
        self.reg_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")
        norm_metric_name = "w_norm_normalized" if self.weight_regularization_normalize else "w_norm"
        self.w_norm_tracker = tf.keras.metrics.Mean(name=norm_metric_name)
        self.reg_active_rate_tracker = tf.keras.metrics.Mean(name="reg_active_rate")

    @property
    def metrics(self):
        """Return Keras metrics including custom regularization trackers."""
        base_metrics = super().metrics
        base_names = {metric.name for metric in base_metrics}
        extra_metrics = [self.reg_loss_tracker, self.w_norm_tracker, self.reg_active_rate_tracker]
        for metric in extra_metrics:
            if metric.name not in base_names:
                base_metrics.append(metric)
        return base_metrics

    def compute_weight_regularization(self, weights_grid: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute hinge regularization that penalizes very low apodization norms.

        Args:
            weights_grid: INR apodizations with shape ``(E, Z, X)``.

        Returns:
            Tuple ``(reg_loss, norm_value, reg_active)`` where:
            - ``reg_loss`` is ``lambda * max(0, tau - norm)^2``.
            - ``norm_value`` is either normalized or raw norm depending on config.
            - ``reg_active`` is 1.0 when the hinge is active, else 0.0.
        """
        weights_grid = tf.cast(weights_grid, tf.float32)
        global_norm = tf.norm(weights_grid, ord="euclidean")
        if self.weight_regularization_normalize:
            norm_value = global_norm / tf.maximum(
                tf.cast(self._norm_denominator, tf.float32),
                tf.cast(self.weight_regularization_epsilon, tf.float32),
            )
        else:
            norm_value = global_norm

        tau = tf.cast(self.weight_regularization_tau, tf.float32)
        reg_lambda = tf.cast(self.weight_regularization_lambda, tf.float32)
        violation = tf.nn.relu(tau - norm_value)
        reg_loss = reg_lambda * tf.square(violation)
        reg_active = tf.cast(violation > 0.0, tf.float32)
        return reg_loss, norm_value, reg_active

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
        predicted_image, weights_grid = self.reconstruct_image(delayed_batch, training=training)
        if self.weight_regularization_enabled:
            reg_loss, norm_value, reg_active = self.compute_weight_regularization(weights_grid)
            if training:
                self.add_loss(reg_loss)
            self.reg_loss_tracker.update_state(reg_loss)
            self.w_norm_tracker.update_state(norm_value)
            self.reg_active_rate_tracker.update_state(reg_active)
        return predicted_image


def build_mlp_inr(
    input_dim: int,
    hidden_units_config: int | float | list | tuple,
    n_hidden_layers: int | None = None,
    activation: str = "relu",
    output_activation: str = "sigmoid",
) -> tf.keras.Model:
    """Build a flexible MLP that maps geometry features to apodization weights.

    The MLP takes `input_dim` features as input and outputs a single scalar weight
    in [0, 1] for each spatial point. Hidden layer sizes can be specified uniformly
    (int) or per-layer (list/tuple).

    Args:
        input_dim: Number of input features.
        hidden_units_config: Hidden layer size specification. Can be:
            - int or float: All hidden layers have this size. Requires `n_hidden_layers`.
            - list or tuple: Per-layer sizes. Length determines n_hidden_layers.
              If `n_hidden_layers` is also provided and differs, a warning is logged
              but the list length takes precedence.
        n_hidden_layers: Number of hidden layers (used only when `hidden_units_config`
            is int/float). Ignored if `hidden_units_config` is a list/tuple.
        activation: Activation function for hidden layers (e.g., "relu", "tanh").
        output_activation: Activation function for output layer (e.g., "sigmoid", "relu").

    Returns:
        Uncompiled tf.keras.Model that maps (batch, input_dim) -> (batch, 1).

    Raises:
        ValueError: If hidden_units_config is int but n_hidden_layers is None or <= 0.
        TypeError: If hidden_units_config has unsupported type.
    """
    import warnings

    # Normalize hidden_units_config to a list
    if isinstance(hidden_units_config, (list, tuple)):
        hidden_sizes = list(hidden_units_config)
        inferred_n_hidden = len(hidden_sizes)

        # Warn if n_hidden_layers conflicts with list length
        if n_hidden_layers is not None and n_hidden_layers != inferred_n_hidden:
            warnings.warn(
                f"hidden_units_config is a list with {inferred_n_hidden} layers, "
                f"but n_hidden_layers={n_hidden_layers} was also provided. "
                f"Using list length {inferred_n_hidden}.",
                UserWarning,
            )
    elif isinstance(hidden_units_config, (int, float)):
        if n_hidden_layers is None or n_hidden_layers <= 0:
            raise ValueError(
                f"When hidden_units_config is int/float ({hidden_units_config}), "
                f"n_hidden_layers must be a positive integer, got {n_hidden_layers}"
            )
        hidden_sizes = [int(hidden_units_config)] * int(n_hidden_layers)
    else:
        raise TypeError(
            f"hidden_units_config must be int, float, list, or tuple; "
            f"got {type(hidden_units_config)}"
        )

    # Build the model
    inputs = tf.keras.Input(shape=(input_dim,), name="coords")
    x = inputs
    for i, units in enumerate(hidden_sizes):
        x = tf.keras.layers.Dense(int(units), activation=activation, name=f"dense_{i + 1}")(x)
    outputs = tf.keras.layers.Dense(1, activation=output_activation, name="weight_out")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="inr_mlp")
    return model
