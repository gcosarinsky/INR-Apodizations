"""Trainer wrappers that apply INR weights to delayed samples and reconstruct DAS.

Contains `DasInrApod`, un pequeño modelo Keras que ejecuta la INR en chunks
y realiza el forward físico de ponderación y suma, y `DasInrApodMixer` para
combinación pixel-wise de múltiples apodizaciones.
"""
from __future__ import annotations

import tensorflow as tf


class DasInrApod(tf.keras.Model):
    """Keras model that wraps the INR and the physical DAS forward.

    Este modelo evalúa la INR sobre una grilla de features (en chunks para evitar
    uso excesivo de memoria), reordena la salida a una grilla de pesos `(E, Z, X)`
    y aplica esos pesos a los delayed samples para producir una imagen DAS compleja.
    La API pública es compatible con Keras (`model.fit`).
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
        super().__init__(name="das_inr_apod")
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

    def predict_weights_flat(self, training: bool = False) -> tf.Tensor:
        """Run the INR on geometry features in chunks and concatenate outputs.

        Args:
            training: Whether to run the INR in training mode.

        Returns:
            Flat INR predictions with shape ``(E * Z * X, C)`` where ``C`` is the
            number of output channels produced by the apodization model.
        """
        n_features = int(self.features_flat.shape[0])
        predictions = []
        for start_idx in range(0, n_features, self.feature_chunk_size):
            end_idx = min(start_idx + self.feature_chunk_size, n_features)
            chunk_pred = self.apodization_model(
                self.features_flat[start_idx:end_idx], training=training
            )
            predictions.append(chunk_pred)

        return tf.concat(predictions, axis=0)

    def predict_weights_grid(self, training: bool = False) -> tf.Tensor:
        """Run the INR on geometry features and reshape to ``(E, Z, X)``."""
        weights_flat = self.predict_weights_flat(training=training)
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


class DasInrApodMixer(DasInrApod):
    """Keras model that combines multiple DAS reconstructions pixel-wise.

    Este modelo evalúa una INR multi-salida sobre la grilla de features, interpreta
    cada canal de salida como una apodización diferente, reconstruye una imagen DAS
    de magnitud por apodización y las combina pixel a pixel con una capa lineal
    entrenable seguida de ReLU.
    """

    def __init__(
        self,
        apodization_model: tf.keras.Model,
        features_grid: tf.Tensor,
        feature_chunk_size: int,
        n_apodizations: int | None = None,
        weight_regularization_enabled: bool = False,
        weight_regularization_lambda: float = 1e-3,
        weight_regularization_tau: float = 0.30,
        weight_regularization_epsilon: float = 1e-8,
        weight_regularization_normalize: bool = True,
    ):
        """Initialize the ensemble trainer.

        Args:
            apodization_model: INR model mapping geometry features to multiple weights.
            features_grid: Feature tensor with shape ``(E, Z, X, F)``.
            feature_chunk_size: Number of feature rows processed per forward chunk.
            n_apodizations: Number of apodizations generated by the INR. If None, inferred from model.
            weight_regularization_enabled: Whether to enable low-norm hinge regularization.
            weight_regularization_lambda: Multiplicative factor for regularization loss.
            weight_regularization_tau: Minimum target norm before hinge becomes active.
            weight_regularization_epsilon: Numerical epsilon used in norm normalization.
            weight_regularization_normalize: Whether to normalize norm by ``sqrt(E*Z*X)``.

        Raises:
            ValueError: If ``n_apodizations`` is not positive or does not match model output.
        """
        # Infer output channels from model if not provided
        model_n_outputs = apodization_model.output_shape[-1]
        if n_apodizations is None:
            n_apodizations = model_n_outputs
        else:
            if int(n_apodizations) != int(model_n_outputs):
                raise ValueError(
                    f"n_apodizations ({n_apodizations}) does not match "
                    f"apodization_model output ({model_n_outputs})"
                )
        self.n_apodizations = int(n_apodizations)
        if self.n_apodizations <= 0:
            raise ValueError("n_apodizations must be > 0")

        super().__init__(
            apodization_model=apodization_model,
            features_grid=features_grid,
            feature_chunk_size=feature_chunk_size,
            weight_regularization_enabled=weight_regularization_enabled,
            weight_regularization_lambda=weight_regularization_lambda,
            weight_regularization_tau=weight_regularization_tau,
            weight_regularization_epsilon=weight_regularization_epsilon,
            weight_regularization_normalize=weight_regularization_normalize,
        )

        self.pixel_combiner = tf.keras.layers.Dense(
            units=1,
            activation=None,
            name="pixelwise_linear_combiner",
        )
        self._last_intermediate_images: tf.Tensor | None = None
        self._last_apodization_grids: tf.Tensor | None = None

    @property
    def intermediate_images(self) -> tf.Tensor | None:
        """Return the intermediate DAS magnitude images from the last forward pass.

        Returns:
            Tensor with shape ``(B, Z, X, N)`` or ``None`` if no forward pass has run yet.
        """
        return self._last_intermediate_images

    @property
    def apodization_grids(self) -> tf.Tensor | None:
        """Return the last predicted apodization grids.

        Returns:
            Tensor with shape ``(E, Z, X, N)`` or ``None`` if no forward pass has run yet.
        """
        return self._last_apodization_grids

    def predict_weights_grid(self, training: bool = False) -> tf.Tensor:
        """Run the INR on geometry features and reshape to ``(E, Z, X, N)``.

        Args:
            training: Whether to run the INR in training mode.

        Returns:
            INR apodizations with shape ``(E, Z, X, N)``.
        """
        weights_flat = self.predict_weights_flat(training=training)
        weights_grid = tf.reshape(
            weights_flat,
            (self.n_elem, self.nz, self.nx, self.n_apodizations),
        )
        return tf.cast(weights_grid, tf.float32)

    def reconstruct_image(
        self,
        delayed_batch: tf.Tensor,
        training: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Reconstruct multiple DAS images and combine them pixel-wise.

        Args:
            delayed_batch: Delayed samples with shape ``(B, E, Z, X)``.
            training: Whether to run the INR in training mode.

        Returns:
            Tuple ``(combined_image, weights_grid)`` where ``combined_image`` has
            shape ``(B, Z, X)`` and ``weights_grid`` has shape ``(E, Z, X, N)``.
        """
        weights_grid = self.predict_weights_grid(training=training)
        weighted_delayed = delayed_batch[..., tf.newaxis] * tf.cast(
            weights_grid[tf.newaxis, ...],
            delayed_batch.dtype,
        )
        predicted_complex = tf.reduce_sum(weighted_delayed, axis=1)
        intermediate_images = tf.abs(predicted_complex)

        intermediate_flat = tf.reshape(intermediate_images, (-1, self.n_apodizations))
        combined_flat = self.pixel_combiner(intermediate_flat, training=training)
        combined_flat = tf.nn.relu(combined_flat)
        batch_size = tf.shape(delayed_batch)[0]
        combined_image = tf.reshape(combined_flat, (batch_size, self.nz, self.nx))

        self._last_intermediate_images = intermediate_images
        self._last_apodization_grids = weights_grid
        return combined_image, weights_grid

    def call(self, delayed_batch: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Run forward reconstruction and pixel-wise ensemble combination.

        Args:
            delayed_batch: Delayed samples with shape ``(B, E, Z, X)``.
            training: Whether to run the INR in training mode.

        Returns:
            Predicted combined image with shape ``(B, Z, X)``.
        """
        predicted_image, weights_grid = self.reconstruct_image(delayed_batch, training=training)
        if self.weight_regularization_enabled:
            per_apodization_grids = tf.unstack(weights_grid, axis=-1)
            for apodization_grid in per_apodization_grids:
                reg_loss, norm_value, reg_active = self.compute_weight_regularization(
                    apodization_grid
                )
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
    n_apodizations: int = 1,
) -> tf.keras.Model:
    """Build a flexible MLP that maps geometry features to apodization weights.

    The MLP takes `input_dim` features as input and outputs one or more scalar
    weights in [0, 1] for each spatial point. Hidden layer sizes can be specified
    uniformly (int) or per-layer (list/tuple).

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
        n_apodizations: Number of output apodization channels.

    Returns:
        Uncompiled tf.keras.Model that maps ``(batch, input_dim)`` to
        ``(batch, n_apodizations)``.

    Raises:
        ValueError: If hidden_units_config is int but n_hidden_layers is None or <= 0,
            or if ``n_apodizations`` is not positive.
        TypeError: If hidden_units_config has unsupported type.
    """
    import warnings

    if int(n_apodizations) <= 0:
        raise ValueError(f"n_apodizations must be > 0, got {n_apodizations}")

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
    outputs = tf.keras.layers.Dense(
        int(n_apodizations),
        activation=output_activation,
        name="weight_out",
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="inr_mlp")
    return model
