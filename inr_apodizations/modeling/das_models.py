"""Trainer wrappers that apply INR weights to delayed samples and reconstruct DAS.

Contains `DasInrApod`, un pequeño modelo Keras que ejecuta la INR en chunks
y realiza el forward físico de ponderación y suma, y `DasInrApodMixer` para
combinación pixel-wise de múltiples apodizaciones.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


SIGNED_SYMMETRY_FEATURE_TOKENS = (
    "x",
    "xrel",
    "x_over_z",
    "xrel_over_z",
)


def _resolve_signed_flip_indices(
    feature_component_names: list[str] | tuple[str, ...],
    flip_feature_tokens: list[str] | tuple[str, ...] | None = None,
) -> tuple[int, ...]:
    """Resolve feature indices that must be sign-flipped for symmetry.

    Args:
        feature_component_names: Ordered feature token names used by the INR input.
        flip_feature_tokens: Optional token list to flip. If omitted, uses
            ``SIGNED_SYMMETRY_FEATURE_TOKENS``.

    Returns:
        Tuple with indices of features whose sign must be inverted.

    Raises:
        ValueError: If inputs are invalid or no flippable feature is found.
    """
    if not isinstance(feature_component_names, (list, tuple)):
        raise ValueError("feature_component_names must be a list/tuple of strings")
    normalized_features = tuple(str(name).strip() for name in feature_component_names)
    if len(normalized_features) == 0:
        raise ValueError("feature_component_names must not be empty")

    if flip_feature_tokens is None:
        tokens = set(SIGNED_SYMMETRY_FEATURE_TOKENS)
    else:
        if not isinstance(flip_feature_tokens, (list, tuple)):
            raise ValueError("flip_feature_tokens must be a list/tuple of strings")
        tokens = {str(token).strip() for token in flip_feature_tokens}
        if len(tokens) == 0:
            raise ValueError("flip_feature_tokens must not be empty")

    indices = tuple(i for i, token in enumerate(normalized_features) if token in tokens)
    if len(indices) == 0:
        raise ValueError(
            "No signed features available for symmetry flip. "
            f"feature_component_names={normalized_features}, flip_feature_tokens={sorted(tokens)}"
        )
    return indices


@tf.keras.utils.register_keras_serializable(package="INRApodizations")
class SymmetricApodizationWrapper(tf.keras.layers.Layer):
    """Enforce feature-sign symmetry on an apodization model output.

    The wrapper computes ``a_sym(f) = a(f) + a(f_flip)``, where ``f_flip`` is
    constructed by inverting the sign of selected feature channels.
    """

    def __init__(
        self,
        base_model: tf.keras.Model,
        feature_component_names: list[str] | tuple[str, ...],
        flip_feature_tokens: list[str] | tuple[str, ...] | None = None,
        combine_mode: str = "sum",
        **kwargs,
    ):
        """Initialize the symmetry-enforcing wrapper.

        Args:
            base_model: Base INR model that maps features to apodization channels.
            feature_component_names: Ordered input feature tokens.
            flip_feature_tokens: Optional subset of tokens to sign-flip.
            combine_mode: Combination strategy. Currently only ``"sum"``.
            **kwargs: Forwarded to base Keras layer.

        Raises:
            ValueError: If configuration is invalid.
        """
        super().__init__(**kwargs)
        self.base_model = base_model
        self.feature_component_names = tuple(
            str(name).strip() for name in feature_component_names
        )
        self.flip_feature_tokens = tuple(
            str(token).strip() for token in (
                SIGNED_SYMMETRY_FEATURE_TOKENS if flip_feature_tokens is None else flip_feature_tokens
            )
        )
        self.combine_mode = str(combine_mode).lower().strip()
        if self.combine_mode != "sum":
            raise ValueError("combine_mode must be 'sum'")

        self.flip_feature_indices = _resolve_signed_flip_indices(
            feature_component_names=self.feature_component_names,
            flip_feature_tokens=self.flip_feature_tokens,
        )

        flip_sign = np.ones((len(self.feature_component_names),), dtype=np.float32)
        for idx in self.flip_feature_indices:
            flip_sign[idx] = -1.0
        self._flip_sign_vector = tf.constant(flip_sign, dtype=tf.float32)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """Compute symmetric apodization predictions.

        Args:
            inputs: Feature tensor with shape ``(batch, F)`` or ``(..., F)``.
            training: Whether the base model runs in training mode.

        Returns:
            Symmetrized model output with the same shape as base model output.
        """
        outputs_pos = self.base_model(inputs, training=training)
        sign = tf.cast(self._flip_sign_vector, inputs.dtype)
        flipped_inputs = inputs * sign
        outputs_neg = self.base_model(flipped_inputs, training=training)
        return outputs_pos + outputs_neg

    def get_config(self) -> dict[str, object]:
        """Return serializable layer configuration."""
        config = super().get_config()
        config.update(
            {
                "base_model": tf.keras.utils.serialize_keras_object(self.base_model),
                "feature_component_names": list(self.feature_component_names),
                "flip_feature_tokens": list(self.flip_feature_tokens),
                "combine_mode": self.combine_mode,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "SymmetricApodizationWrapper":
        """Create a wrapper instance from serialized config."""
        config = dict(config)
        base_model_cfg = config.pop("base_model")
        base_model = tf.keras.utils.deserialize_keras_object(base_model_cfg)
        return cls(base_model=base_model, **config)


@tf.keras.utils.register_keras_serializable(package="INRApodizations")
class FeatureDrivenOutputMask(tf.keras.layers.Layer):
    """Apply a deterministic output mask derived from the input features.

    The layer is intended to sit after a base INR and before the downstream
    consumer. It currently supports a hard boxcar mask computed from a selected
    feature channel.
    """

    def __init__(
        self,
        mask_mode: str = "boxcar",
        feature_index: int = 0,
        threshold: float = 0.5,
        epsilon: float = 1e-8,
        output_channels: int = 1,
        **kwargs,
    ):
        """Initialize the feature-driven mask layer.

        Args:
            mask_mode: Mask family. Currently only ``"boxcar"`` is supported.
            feature_index: Input feature index used to derive the mask.
            threshold: Absolute threshold used by the hard boxcar mask.
            epsilon: Numerical tolerance for the threshold comparison.
            output_channels: Number of output channels to broadcast the mask to.
            **kwargs: Forwarded to the base Keras layer.

        Raises:
            ValueError: If the configuration is invalid.
        """
        super().__init__(**kwargs)
        self.mask_mode = str(mask_mode).lower().strip()
        self.feature_index = int(feature_index)
        self.threshold = float(threshold)
        self.epsilon = float(epsilon)
        self.output_channels = int(output_channels)

        if self.mask_mode != "boxcar":
            raise ValueError("mask_mode must be 'boxcar'")
        if self.feature_index < 0:
            raise ValueError("feature_index must be >= 0")
        if self.threshold < 0.0:
            raise ValueError("threshold must be >= 0")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if self.output_channels <= 0:
            raise ValueError("output_channels must be > 0")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Return a broadcastable hard mask for the selected feature.

        Args:
            inputs: Feature tensor with shape ``(batch, F)`` or ``(..., F)``.

        Returns:
            Mask tensor with the same leading dimensions as ``inputs`` and the
            last dimension broadcast to ``output_channels``.
        """
        feature = tf.cast(inputs[..., self.feature_index], tf.float32)
        mask = tf.cast(tf.abs(feature) <= (self.threshold + self.epsilon), tf.float32)
        mask = mask[..., tf.newaxis]
        if self.output_channels != 1:
            mask = tf.repeat(mask, repeats=self.output_channels, axis=-1)
        return mask

    def get_config(self) -> dict[str, object]:
        """Return the serializable layer configuration."""
        config = super().get_config()
        config.update(
            {
                "mask_mode": self.mask_mode,
                "feature_index": self.feature_index,
                "threshold": self.threshold,
                "epsilon": self.epsilon,
                "output_channels": self.output_channels,
            }
        )
        return config


class BaseApodizationRegularizer:
    """Abstract contract for apodization regularizers.

    Subclasses must implement ``compute`` and return a scalar loss tensor together
    with a flat dict of named scalar tensors for metric tracking.
    """

    def compute(self, apod_grid: tf.Tensor) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """Compute regularization loss and auxiliary metrics.

        Args:
            apod_grid: Apodization tensor with shape ``(E, Z, X)`` or ``(E, Z, X, N)``.

        Returns:
            Tuple ``(reg_loss, metrics)`` where ``metrics`` maps metric-tracker names
            to scalar tensors.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement compute()")


class LateralDistanceRegularizer(BaseApodizationRegularizer):
    """Regularize apodization amplitudes using q = (abs(x_rel) / z)^gamma.

    This helper caches ``q_grid`` once and computes a weighted amplitude penalty
    for both single-apodization and multi-apodization tensors.
    """

    def __init__(
        self,
        q_grid: tf.Tensor,
        reg_lambda: float,
        normalize_by_uniform: bool = True,
        channel_reduction: str = "mean",
        epsilon: float = 1e-8,
    ):
        """Initialize a lateral-distance weighted regularizer.

        Args:
            q_grid: Tensor with shape ``(E, Z, X)`` containing q values.
            reg_lambda: Multiplicative factor for regularization loss.
            normalize_by_uniform: Whether to normalize by uniform baseline.
            channel_reduction: Channel aggregation for 4D weights: ``"mean"`` or ``"sum"``.
            epsilon: Numerical epsilon for safe denominator handling.

        Raises:
            ValueError: If hyperparameters are invalid or q_grid rank is not 3.
        """
        self.q_grid = tf.cast(q_grid, tf.float32)
        self.reg_lambda = float(reg_lambda)
        self.normalize_by_uniform = bool(normalize_by_uniform)
        self.channel_reduction = str(channel_reduction).lower()
        self.epsilon = float(epsilon)

        if len(self.q_grid.shape) != 3:
            raise ValueError(
                "q_grid must have shape (E, Z, X); "
                f"got rank={len(self.q_grid.shape)}"
            )
        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda must be >= 0")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if self.channel_reduction not in {"mean", "sum"}:
            raise ValueError("channel_reduction must be 'mean' or 'sum'")

        self.q_mean = float(tf.reduce_mean(self.q_grid).numpy())
        self.uniform_reference = float(self.q_mean)

    @classmethod
    def from_features(
        cls,
        features_grid: tf.Tensor,
        reg_lambda: float,
        q_power: float = 1.0,
        q_epsilon: float = 1e-8,
        abs_xrel_feature_index: int = 0,
        z_feature_index: int = 1,
        normalize_by_uniform: bool = True,
        channel_reduction: str = "mean",
    ) -> "LateralDistanceRegularizer":
        """Create a regularizer from feature channels containing abs(x_rel) and z.

        Args:
            features_grid: Feature tensor with shape ``(E, Z, X, F)``.
            reg_lambda: Multiplicative factor for regularization loss.
            q_power: Exponent applied to q.
            q_epsilon: Numerical epsilon to avoid division by zero.
            abs_xrel_feature_index: Feature index containing abs(x_rel).
            z_feature_index: Feature index containing z.
            normalize_by_uniform: Whether to normalize by uniform baseline.
            channel_reduction: Channel aggregation for 4D weights.

        Returns:
            Initialized regularizer instance.

        Raises:
            ValueError: If feature indices are out of bounds or q_power is invalid.
        """
        n_features = int(features_grid.shape[-1])
        abs_xrel_idx = int(abs_xrel_feature_index)
        z_idx = int(z_feature_index)
        if not (0 <= abs_xrel_idx < n_features):
            raise ValueError(
                "abs_xrel_feature_index is out of bounds: "
                f"{abs_xrel_idx}, n_features={n_features}"
            )
        if not (0 <= z_idx < n_features):
            raise ValueError(
                "z_feature_index is out of bounds: "
                f"{z_idx}, n_features={n_features}"
            )
        if float(q_power) <= 0.0:
            raise ValueError("q_power must be > 0")
        if float(q_epsilon) <= 0.0:
            raise ValueError("q_epsilon must be > 0")

        features_grid = tf.cast(features_grid, tf.float32)
        abs_xrel = tf.abs(features_grid[..., abs_xrel_idx])
        z = tf.maximum(features_grid[..., z_idx], tf.cast(q_epsilon, tf.float32))
        q_grid = abs_xrel / z
        if float(q_power) != 1.0:
            q_grid = tf.pow(q_grid, tf.cast(q_power, tf.float32))

        return cls(
            q_grid=q_grid,
            reg_lambda=reg_lambda,
            normalize_by_uniform=normalize_by_uniform,
            channel_reduction=channel_reduction,
            epsilon=q_epsilon,
        )

    @classmethod
    def from_coords_grid(
        cls,
        coords_grid: tf.Tensor,
        reg_lambda: float,
        q_power: float = 1.0,
        q_epsilon: float = 1e-8,
        normalize_by_uniform: bool = True,
        channel_reduction: str = "mean",
    ) -> "LateralDistanceRegularizer":
        """Create a regularizer from coordinate grid [x, z, x_elem].

        Args:
            coords_grid: Coordinate tensor with shape ``(E, Z, X, 3)`` where
                channels are ``[x, z, x_elem]``.
            reg_lambda: Multiplicative factor for regularization loss.
            q_power: Exponent applied to q.
            q_epsilon: Numerical epsilon to avoid division by zero.
            normalize_by_uniform: Whether to normalize by uniform baseline.
            channel_reduction: Channel aggregation for 4D weights.

        Returns:
            Initialized regularizer instance.

        Raises:
            ValueError: If coords_grid shape is invalid or q_power is invalid.
        """
        if len(coords_grid.shape) != 4 or int(coords_grid.shape[-1]) < 3:
            raise ValueError(
                "coords_grid must have shape (E, Z, X, 3) with channels [x, z, x_elem]"
            )
        if float(q_power) <= 0.0:
            raise ValueError("q_power must be > 0")
        if float(q_epsilon) <= 0.0:
            raise ValueError("q_epsilon must be > 0")

        coords_grid = tf.cast(coords_grid, tf.float32)
        x = coords_grid[..., 0]
        z = tf.maximum(coords_grid[..., 1], tf.cast(q_epsilon, tf.float32))
        x_elem = coords_grid[..., 2]
        q_grid = tf.abs(x - x_elem) / z
        if float(q_power) != 1.0:
            q_grid = tf.pow(q_grid, tf.cast(q_power, tf.float32))

        return cls(
            q_grid=q_grid,
            reg_lambda=reg_lambda,
            normalize_by_uniform=normalize_by_uniform,
            channel_reduction=channel_reduction,
            epsilon=q_epsilon,
        )

    def compute(self, apod_grid: tf.Tensor) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """Compute lateral-distance weighted regularization loss and metrics.

        Args:
            apod_grid: Apodization tensor with shape ``(E, Z, X)`` or ``(E, Z, X, N)``.

        Returns:
            Tuple ``(reg_loss, metrics)`` where ``metrics`` contains keys
            ``"lateral_reg_loss"``, ``"lateral_reg_raw"``, ``"lateral_reg_scaled"``,
            and ``"lateral_q_mean"``.

        Raises:
            ValueError: If apod_grid rank is unsupported.
        """
        apod_grid = tf.cast(apod_grid, tf.float32)

        if len(apod_grid.shape) == 3:
            reg_raw = tf.reduce_mean(tf.abs(apod_grid) * self.q_grid)
            denom_scale = tf.cast(1.0, tf.float32)
        elif len(apod_grid.shape) == 4:
            weighted = tf.abs(apod_grid) * self.q_grid[..., tf.newaxis]
            if self.channel_reduction == "sum":
                reg_raw = tf.reduce_sum(tf.reduce_mean(weighted, axis=(0, 1, 2)))
                denom_scale = tf.cast(tf.shape(apod_grid)[-1], tf.float32)
            else:
                reg_raw = tf.reduce_mean(weighted)
                denom_scale = tf.cast(1.0, tf.float32)
        else:
            raise ValueError(
                "apod_grid must have shape (E, Z, X) or (E, Z, X, N); "
                f"got rank={len(apod_grid.shape)}"
            )

        if self.normalize_by_uniform:
            denom = tf.maximum(
                tf.cast(self.uniform_reference, tf.float32) * denom_scale,
                tf.cast(self.epsilon, tf.float32),
            )
            reg_scaled = reg_raw / denom
        else:
            reg_scaled = reg_raw

        reg_loss = tf.cast(self.reg_lambda, tf.float32) * reg_scaled
        return reg_loss, {
            "lateral_reg_loss": reg_loss,
            "lateral_reg_raw": reg_raw,
            "lateral_reg_scaled": reg_scaled,
            "lateral_q_mean": tf.constant(self.q_mean, dtype=tf.float32),
        }


class ApodNormHingeRegularizer(BaseApodizationRegularizer):
    """Hinge regularization penalizing low apodization norms.

    Applies a squared hinge penalty when the global Euclidean norm of the
    apodization grid falls below a minimum threshold ``tau``.  For 4D tensors
    ``(E, Z, X, N)`` (mixer mode), the penalty is applied independently per
    apodization channel and losses are summed; norms and active flags are averaged.
    """

    def __init__(
        self,
        n_points: int,
        reg_lambda: float = 1e-3,
        tau: float = 0.30,
        normalize: bool = True,
        epsilon: float = 1e-8,
    ):
        """Initialize hinge norm regularizer.

        Args:
            n_points: Total number of spatial points (E * Z * X). Used to compute
                the normalization denominator ``sqrt(n_points)``.
            reg_lambda: Multiplicative factor for regularization loss.
            tau: Minimum target norm before hinge becomes active.
            normalize: Whether to normalize the norm by ``sqrt(n_points)``.
            epsilon: Numerical epsilon for safe denominator handling.

        Raises:
            ValueError: If hyperparameters are invalid.
        """
        if float(reg_lambda) < 0.0:
            raise ValueError("reg_lambda must be >= 0")
        if float(tau) < 0.0:
            raise ValueError("tau must be >= 0")
        if float(epsilon) <= 0.0:
            raise ValueError("epsilon must be > 0")

        self.reg_lambda = float(reg_lambda)
        self.tau = float(tau)
        self.normalize = bool(normalize)
        self.epsilon = float(epsilon)
        self._norm_denominator = float(int(n_points) ** 0.5)
        self._norm_metric_key = "w_norm_normalized" if self.normalize else "w_norm"

    def _compute_single(self, apod_3d: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute hinge loss for a single 3D apodization grid.

        Args:
            apod_3d: Apodization tensor with shape ``(E, Z, X)``.

        Returns:
            Tuple ``(reg_loss, norm_value, reg_active)``.
        """
        apod_3d = tf.cast(apod_3d, tf.float32)
        global_norm = tf.norm(apod_3d, ord="euclidean")
        if self.normalize:
            norm_value = global_norm / tf.maximum(
                tf.cast(self._norm_denominator, tf.float32),
                tf.cast(self.epsilon, tf.float32),
            )
        else:
            norm_value = global_norm
        tau = tf.cast(self.tau, tf.float32)
        violation = tf.nn.relu(tau - norm_value)
        reg_loss = tf.cast(self.reg_lambda, tf.float32) * tf.square(violation)
        reg_active = tf.cast(violation > 0.0, tf.float32)
        return reg_loss, norm_value, reg_active

    def compute(self, apod_grid: tf.Tensor) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        """Compute hinge loss for 3D or 4D apodization grids.

        For 4D tensors ``(E, Z, X, N)``, the penalty is applied per channel; losses
        are summed and norm/active values are averaged across channels.

        Args:
            apod_grid: Apodization tensor with shape ``(E, Z, X)`` or ``(E, Z, X, N)``.

        Returns:
            Tuple ``(reg_loss, metrics)`` with keys ``"reg_loss"``,
            ``"w_norm_normalized"`` (or ``"w_norm"``), and ``"reg_active_rate"``.

        Raises:
            ValueError: If apod_grid rank is unsupported.
        """
        apod_grid = tf.cast(apod_grid, tf.float32)
        rank = len(apod_grid.shape)
        if rank == 3:
            reg_loss, norm_value, reg_active = self._compute_single(apod_grid)
        elif rank == 4:
            channels = tf.unstack(apod_grid, axis=-1)
            results = [self._compute_single(c) for c in channels]
            losses, norms, actives = zip(*results)
            reg_loss = tf.add_n(losses)
            norm_value = tf.reduce_mean(tf.stack(norms))
            reg_active = tf.reduce_mean(tf.stack(actives))
        else:
            raise ValueError(
                f"apod_grid must have rank 3 or 4; got rank={rank}"
            )
        return reg_loss, {
            "reg_loss": reg_loss,
            self._norm_metric_key: norm_value,
            "reg_active_rate": reg_active,
        }


class _RegularizationEngine:
    """Applies a list of regularizers and aggregates losses and metrics.

    Keeps regularization logic decoupled from the trainer's forward pass.
    """

    def __init__(self, regularizers: list[BaseApodizationRegularizer]):
        """Initialize the engine.

        Args:
            regularizers: Ordered list of regularizer instances to apply.
        """
        self.regularizers = list(regularizers)

    def apply(
        self,
        apod_grid: tf.Tensor,
        training: bool,
        model: tf.keras.Model,
        tracker_registry: dict[str, tf.keras.metrics.Metric],
    ) -> tf.Tensor:
        """Compute all regularization losses and update metric trackers.

        Args:
            apod_grid: Apodization tensor passed to each regularizer.
            training: Whether the model is in training mode. Only calls
                ``model.add_loss`` when ``True``.
            model: Keras model whose ``add_loss`` method is used in training.
            tracker_registry: Mapping from metric name to Keras ``Mean`` tracker.

        Returns:
            Total regularization loss scalar tensor.
        """
        total_loss = tf.constant(0.0, dtype=tf.float32)
        for reg in self.regularizers:
            loss, metrics = reg.compute(apod_grid)
            total_loss = total_loss + tf.cast(loss, tf.float32)
            for key, val in metrics.items():
                if key in tracker_registry:
                    tracker_registry[key].update_state(val)
        if training:
            model.add_loss(total_loss)
        return total_loss


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
        lateral_regularization_enabled: bool = False,
        lateral_regularization_lambda: float = 0.0,
        lateral_regularization_q_power: float = 1.0,
        lateral_regularization_q_epsilon: float = 1e-8,
        lateral_regularization_q_abs_xrel_feature_index: int = 0,
        lateral_regularization_q_z_feature_index: int = 1,
        lateral_regularization_normalize_by_uniform: bool = True,
        lateral_regularization_channel_reduction: str = "mean",
        lateral_regularization_q_grid: tf.Tensor | None = None,
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
            lateral_regularization_enabled: Whether to penalize large apodization values at
                large lateral-distance-to-depth ratio.
            lateral_regularization_lambda: Multiplicative factor for lateral regularization.
            lateral_regularization_q_power: Exponent applied to q = abs(x_rel) / z.
            lateral_regularization_q_epsilon: Numerical epsilon to avoid division by zero in q.
            lateral_regularization_q_abs_xrel_feature_index: Feature index containing abs(x_rel).
            lateral_regularization_q_z_feature_index: Feature index containing z.
            lateral_regularization_normalize_by_uniform: Whether to normalize the raw regularizer
                by its uniform-apodization reference.
            lateral_regularization_channel_reduction: Channel aggregation mode for multi-output
                apodizations: ``"mean"`` or ``"sum"``.
            lateral_regularization_q_grid: Optional precomputed q tensor with shape
                ``(E, Z, X)``. If provided, no feature-index assumption is required.

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
        self._weight_regularization_lambda = float(weight_regularization_lambda)
        self.weight_regularization_tau = float(weight_regularization_tau)
        self.weight_regularization_epsilon = float(weight_regularization_epsilon)
        self.weight_regularization_normalize = bool(weight_regularization_normalize)
        if self._weight_regularization_lambda < 0.0:
            raise ValueError("weight_regularization_lambda must be >= 0")
        if self.weight_regularization_tau < 0.0:
            raise ValueError("weight_regularization_tau must be >= 0")
        if self.weight_regularization_epsilon <= 0.0:
            raise ValueError("weight_regularization_epsilon must be > 0")

        self.lateral_regularization_enabled = bool(lateral_regularization_enabled)
        self._lateral_regularization_lambda = float(lateral_regularization_lambda)
        self.lateral_regularization_q_power = float(lateral_regularization_q_power)
        self.lateral_regularization_q_epsilon = float(lateral_regularization_q_epsilon)
        self.lateral_regularization_q_abs_xrel_feature_index = int(
            lateral_regularization_q_abs_xrel_feature_index
        )
        self.lateral_regularization_q_z_feature_index = int(
            lateral_regularization_q_z_feature_index
        )
        self.lateral_regularization_normalize_by_uniform = bool(
            lateral_regularization_normalize_by_uniform
        )
        self.lateral_regularization_channel_reduction = str(
            lateral_regularization_channel_reduction
        ).lower()
        if self._lateral_regularization_lambda < 0.0:
            raise ValueError("lateral_regularization_lambda must be >= 0")
        if self.lateral_regularization_q_power <= 0.0:
            raise ValueError("lateral_regularization_q_power must be > 0")
        if self.lateral_regularization_q_epsilon <= 0.0:
            raise ValueError("lateral_regularization_q_epsilon must be > 0")
        if self.lateral_regularization_channel_reduction not in {"mean", "sum"}:
            raise ValueError(
                "lateral_regularization_channel_reduction must be 'mean' or 'sum'"
            )

        # Build regularizer instances
        _regularizers: list[BaseApodizationRegularizer] = []

        if self.weight_regularization_enabled:
            self._apod_norm_regularizer: ApodNormHingeRegularizer | None = ApodNormHingeRegularizer(
                n_points=self.n_elem * self.nz * self.nx,
                reg_lambda=self._weight_regularization_lambda,
                tau=self.weight_regularization_tau,
                normalize=self.weight_regularization_normalize,
                epsilon=self.weight_regularization_epsilon,
            )
            _regularizers.append(self._apod_norm_regularizer)
        else:
            self._apod_norm_regularizer = None

        self._lateral_regularizer: LateralDistanceRegularizer | None = None
        if self.lateral_regularization_enabled:
            if lateral_regularization_q_grid is not None:
                self._lateral_regularizer = LateralDistanceRegularizer(
                    q_grid=lateral_regularization_q_grid,
                    reg_lambda=self.lateral_regularization_lambda,
                    normalize_by_uniform=self.lateral_regularization_normalize_by_uniform,
                    channel_reduction=self.lateral_regularization_channel_reduction,
                    epsilon=self.lateral_regularization_q_epsilon,
                )
            else:
                self._lateral_regularizer = LateralDistanceRegularizer.from_features(
                    features_grid=features_grid,
                    reg_lambda=self.lateral_regularization_lambda,
                    q_power=self.lateral_regularization_q_power,
                    q_epsilon=self.lateral_regularization_q_epsilon,
                    abs_xrel_feature_index=self.lateral_regularization_q_abs_xrel_feature_index,
                    z_feature_index=self.lateral_regularization_q_z_feature_index,
                    normalize_by_uniform=self.lateral_regularization_normalize_by_uniform,
                    channel_reduction=self.lateral_regularization_channel_reduction,
                )
            _regularizers.append(self._lateral_regularizer)

        if self._lateral_regularizer is not None:
            self._lateral_q_mean = float(self._lateral_regularizer.q_mean)
            self._lateral_uniform_reference = float(self._lateral_regularizer.uniform_reference)
        else:
            self._lateral_q_mean = 0.0
            self._lateral_uniform_reference = 0.0

        self._regularization_engine = _RegularizationEngine(_regularizers)

        # Prepare tracker registry only for enabled regularizers. If a
        # regularizer is disabled, the corresponding tracker attribute is set
        # to None to avoid accidental exposure to Keras or updates.
        self._tracker_registry: dict[str, tf.keras.metrics.Metric] = {}

        norm_metric_name = "w_norm_normalized" if self.weight_regularization_normalize else "w_norm"

        if self.weight_regularization_enabled:
            self.reg_loss_tracker = tf.keras.metrics.Mean(name="reg_loss")
            self.w_norm_tracker = tf.keras.metrics.Mean(name=norm_metric_name)
            self.reg_active_rate_tracker = tf.keras.metrics.Mean(name="reg_active_rate")
            self._tracker_registry.update(
                {
                    "reg_loss": self.reg_loss_tracker,
                    norm_metric_name: self.w_norm_tracker,
                    "reg_active_rate": self.reg_active_rate_tracker,
                }
            )
        else:
            self.reg_loss_tracker = None
            self.w_norm_tracker = None
            self.reg_active_rate_tracker = None

        if self.lateral_regularization_enabled:
            self.lateral_reg_loss_tracker = tf.keras.metrics.Mean(name="lateral_reg_loss")
            self.lateral_reg_raw_tracker = tf.keras.metrics.Mean(name="lateral_reg_raw")
            self.lateral_reg_scaled_tracker = tf.keras.metrics.Mean(name="lateral_reg_scaled")
            self.lateral_q_mean_tracker = tf.keras.metrics.Mean(name="lateral_q_mean")
            self._tracker_registry.update(
                {
                    "lateral_reg_loss": self.lateral_reg_loss_tracker,
                    "lateral_reg_raw": self.lateral_reg_raw_tracker,
                    "lateral_reg_scaled": self.lateral_reg_scaled_tracker,
                    "lateral_q_mean": self.lateral_q_mean_tracker,
                }
            )
        else:
            self.lateral_reg_loss_tracker = None
            self.lateral_reg_raw_tracker = None
            self.lateral_reg_scaled_tracker = None
            self.lateral_q_mean_tracker = None

    @property
    def metrics(self):
        """Return Keras metrics including custom regularization trackers."""
        base_metrics = super().metrics
        base_names = {metric.name for metric in base_metrics}

        extra_metrics: list[tf.keras.metrics.Metric] = []

        # Include weight-regularization trackers only when enabled
        if self.weight_regularization_enabled:
            extra_metrics.extend([
                self.reg_loss_tracker,
                self.w_norm_tracker,
                self.reg_active_rate_tracker,
            ])

        # Include lateral-regularization trackers only when enabled
        if self.lateral_regularization_enabled:
            extra_metrics.extend([
                self.lateral_reg_loss_tracker,
                self.lateral_reg_raw_tracker,
                self.lateral_reg_scaled_tracker,
                self.lateral_q_mean_tracker,
            ])

        for metric in extra_metrics:
            if metric.name not in base_names:
                base_metrics.append(metric)
        return base_metrics

    @property
    def lateral_q_mean(self) -> float:
        """Return the cached mean value of q = abs(x_rel) / z."""
        return self._lateral_q_mean

    @property
    def lateral_uniform_reference(self) -> float:
        """Return regularizer baseline for uniform apodization weights equal to one."""
        return self._lateral_uniform_reference

    @property
    def weight_regularization_lambda(self) -> float:
        """Return current lambda used by the apod norm hinge regularizer."""
        return self._weight_regularization_lambda

    @weight_regularization_lambda.setter
    def weight_regularization_lambda(self, value: float) -> None:
        """Update lambda and propagate it to the internal regularizer instance."""
        lambda_value = float(value)
        if lambda_value < 0.0:
            raise ValueError("weight_regularization_lambda must be >= 0")
        self._weight_regularization_lambda = lambda_value
        if getattr(self, "_apod_norm_regularizer", None) is not None:
            self._apod_norm_regularizer.reg_lambda = lambda_value

    @property
    def lateral_regularization_lambda(self) -> float:
        """Return current lambda used by the lateral distance regularizer."""
        return self._lateral_regularization_lambda

    @lateral_regularization_lambda.setter
    def lateral_regularization_lambda(self, value: float) -> None:
        """Update lambda and propagate it to the internal lateral regularizer instance."""
        lambda_value = float(value)
        if lambda_value < 0.0:
            raise ValueError("lateral_regularization_lambda must be >= 0")
        self._lateral_regularization_lambda = lambda_value
        if getattr(self, "_lateral_regularizer", None) is not None:
            self._lateral_regularizer.reg_lambda = lambda_value

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
        predicted_image, apod_grid = self.reconstruct_image(delayed_batch, training=training)
        self._regularization_engine.apply(apod_grid, training, self, self._tracker_registry)
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
        lateral_regularization_enabled: bool = False,
        lateral_regularization_lambda: float = 0.0,
        lateral_regularization_q_power: float = 1.0,
        lateral_regularization_q_epsilon: float = 1e-8,
        lateral_regularization_q_abs_xrel_feature_index: int = 0,
        lateral_regularization_q_z_feature_index: int = 1,
        lateral_regularization_normalize_by_uniform: bool = True,
        lateral_regularization_channel_reduction: str = "mean",
        lateral_regularization_q_grid: tf.Tensor | None = None,
        mixer_head_config: dict[str, object] | None = None,
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
            lateral_regularization_enabled: Whether to penalize large apodization values at
                large lateral-distance-to-depth ratio.
            lateral_regularization_lambda: Multiplicative factor for lateral regularization.
            lateral_regularization_q_power: Exponent applied to q = abs(x_rel) / z.
            lateral_regularization_q_epsilon: Numerical epsilon to avoid division by zero in q.
            lateral_regularization_q_abs_xrel_feature_index: Feature index containing abs(x_rel).
            lateral_regularization_q_z_feature_index: Feature index containing z.
            lateral_regularization_normalize_by_uniform: Whether to normalize the raw regularizer
                by its uniform-apodization reference.
            lateral_regularization_channel_reduction: Channel aggregation mode for multi-output
                apodizations: ``"mean"`` or ``"sum"``.
            lateral_regularization_q_grid: Optional precomputed q tensor with shape
                ``(E, Z, X)``. If provided, no feature-index assumption is required.
            mixer_head_config: Optional mixer-head configuration. When enabled,
                supports extra hidden layers for pixel-wise combination.

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
            lateral_regularization_enabled=lateral_regularization_enabled,
            lateral_regularization_lambda=lateral_regularization_lambda,
            lateral_regularization_q_power=lateral_regularization_q_power,
            lateral_regularization_q_epsilon=lateral_regularization_q_epsilon,
            lateral_regularization_q_abs_xrel_feature_index=(
                lateral_regularization_q_abs_xrel_feature_index
            ),
            lateral_regularization_q_z_feature_index=lateral_regularization_q_z_feature_index,
            lateral_regularization_normalize_by_uniform=lateral_regularization_normalize_by_uniform,
            lateral_regularization_channel_reduction=lateral_regularization_channel_reduction,
            lateral_regularization_q_grid=lateral_regularization_q_grid,
        )

        self._mixer_head_config = self._resolve_mixer_head_config(mixer_head_config)
        self._legacy_relu_after_combiner = not bool(self._mixer_head_config["enabled"])

        if self._legacy_relu_after_combiner:
            self._mixer_head_mode = "legacy"
            self.pixel_combiner = tf.keras.layers.Dense(
                units=1,
                activation="relu",
                name="pixelwise_linear_combiner",
            )
        else:
            self._mixer_head_mode = "mlp"
            self.pixel_combiner = self._build_configurable_mixer_head(
                hidden_units=list(self._mixer_head_config["hidden_units"]),
                activation=str(self._mixer_head_config["activation"]),
                output_activation=self._mixer_head_config["output_activation"],
            )

        self._last_intermediate_images: tf.Tensor | None = None
        self._last_apodization_grids: tf.Tensor | None = None

    @staticmethod
    def _resolve_mixer_head_config(
        mixer_head_config: dict[str, object] | None,
    ) -> dict[str, object]:
        """Normalize optional mixer-head config and validate its values."""
        if mixer_head_config is None:
            return {
                "enabled": False,
                "hidden_units": [],
                "activation": "relu",
                "output_activation": "relu",
            }
        if not isinstance(mixer_head_config, dict):
            raise ValueError("mixer_head_config must be a dict when provided")

        enabled = bool(mixer_head_config.get("enabled", False))
        hidden_units_raw = mixer_head_config.get("hidden_units", [])
        if hidden_units_raw is None:
            hidden_units = []
        elif isinstance(hidden_units_raw, (int, float)):
            hidden_units = [int(hidden_units_raw)]
        elif isinstance(hidden_units_raw, (list, tuple)):
            hidden_units = [int(unit) for unit in hidden_units_raw]
        else:
            raise ValueError("mixer_head_config.hidden_units must be int/list/tuple or null")
        if any(unit <= 0 for unit in hidden_units):
            raise ValueError("mixer_head_config.hidden_units entries must be > 0")

        activation = str(mixer_head_config.get("activation", "relu")).strip()
        if len(activation) == 0:
            raise ValueError("mixer_head_config.activation must be non-empty")

        output_activation_raw = mixer_head_config.get("output_activation", "relu")
        if output_activation_raw is None:
            output_activation = None
        else:
            output_activation = str(output_activation_raw).strip()
            if len(output_activation) == 0:
                output_activation = None

        return {
            "enabled": enabled,
            "hidden_units": hidden_units,
            "activation": activation,
            "output_activation": output_activation,
        }

    @staticmethod
    def _build_configurable_mixer_head(
        hidden_units: list[int],
        activation: str,
        output_activation: str | None,
    ) -> tf.keras.Sequential:
        """Build an optional multi-layer pixel-wise combiner head."""
        layers: list[tf.keras.layers.Layer] = []
        for idx, units in enumerate(hidden_units):
            layers.append(
                tf.keras.layers.Dense(
                    int(units),
                    activation=activation,
                    name=f"pixelwise_mixer_hidden_{idx + 1}",
                )
            )
        layers.append(
            tf.keras.layers.Dense(
                1,
                activation=output_activation,
                name="pixelwise_mixer_output",
            )
        )
        return tf.keras.Sequential(layers, name="pixelwise_mixer_head")

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

    @property
    def mixer_head_config(self) -> dict[str, object]:
        """Return resolved mixer-head configuration used by this model."""
        return {
            "enabled": bool(self._mixer_head_config["enabled"]),
            "hidden_units": list(self._mixer_head_config["hidden_units"]),
            "activation": str(self._mixer_head_config["activation"]),
            "output_activation": self._mixer_head_config["output_activation"],
        }

    @property
    def mixer_coefficients(self) -> dict[str, object]:
        """Return the learned coefficients of the pixel-wise linear combiner layer.

        The ``pixel_combiner`` Dense layer has a kernel of shape ``(N, 1)`` and a
        bias of shape ``(1,)``, where ``N`` is ``n_apodizations``.  This property
        exposes them as flat NumPy arrays for easy inspection and logging.

        Returns:
            Dict with keys:
            - ``"weights"``: 1-D array of shape ``(N,)`` — one scalar weight per
              apodization channel, squeezed from the kernel ``(N, 1)``.
            - ``"bias"``: 1-D array of shape ``(1,)`` — the combiner bias term.

        Raises:
            RuntimeError: If the ``pixel_combiner`` layer has not been built yet
                (i.e., no forward pass has been executed).
        """
        if not self.pixel_combiner.built:
            raise RuntimeError(
                "pixel_combiner has not been built yet. "
                "Run at least one forward pass before accessing mixer_coefficients."
            )
        if self._mixer_head_mode == "legacy":
            kernel, bias = self.pixel_combiner.get_weights()
            return {
                "weights": kernel.squeeze(axis=-1),  # (N, 1) -> (N,)
                "bias": bias,                         # (1,)
            }

        layers_payload: list[dict[str, object]] = []
        for layer in self.pixel_combiner.layers:
            if not isinstance(layer, tf.keras.layers.Dense):
                continue
            layer_weights = layer.get_weights()
            if len(layer_weights) != 2:
                continue
            kernel, bias = layer_weights
            layers_payload.append(
                {
                    "name": layer.name,
                    "kernel": np.asarray(kernel, dtype=np.float32),
                    "bias": np.asarray(bias, dtype=np.float32),
                }
            )
        return {"layers": layers_payload}

    def get_mixer_head_npz_payload(self) -> dict[str, np.ndarray]:
        """Export mixer-head weights to a backward-compatible NPZ payload."""
        if not self.pixel_combiner.built:
            raise RuntimeError(
                "pixel_combiner has not been built yet. "
                "Run at least one forward pass before exporting mixer-head weights."
            )

        payload: dict[str, np.ndarray] = {
            "mixer_head_mode": np.asarray(self._mixer_head_mode),
        }
        if self._mixer_head_mode == "legacy":
            kernel, bias = self.pixel_combiner.get_weights()
            payload["mixer_head_version"] = np.asarray(1, dtype=np.int32)
            payload["kernel"] = np.asarray(kernel, dtype=np.float32)
            payload["bias"] = np.asarray(bias, dtype=np.float32)
            return payload

        payload["mixer_head_version"] = np.asarray(2, dtype=np.int32)
        payload["mixer_head_hidden_units"] = np.asarray(
            list(self._mixer_head_config["hidden_units"]), dtype=np.int32
        )
        payload["mixer_head_activation"] = np.asarray(str(self._mixer_head_config["activation"]))
        output_activation = self._mixer_head_config["output_activation"]
        payload["mixer_head_output_activation"] = np.asarray(
            "" if output_activation is None else str(output_activation)
        )
        payload["mixer_head_num_layers"] = np.asarray(len(self.pixel_combiner.layers), dtype=np.int32)

        for idx, layer in enumerate(self.pixel_combiner.layers):
            if not isinstance(layer, tf.keras.layers.Dense):
                raise RuntimeError(
                    "Unsupported layer type in pixel_combiner serialization: "
                    f"{type(layer).__name__}"
                )
            layer_weights = layer.get_weights()
            if len(layer_weights) != 2:
                raise RuntimeError(
                    f"Dense layer {layer.name} does not expose kernel+bias weights"
                )
            kernel, bias = layer_weights
            payload[f"layer_{idx}_kernel"] = np.asarray(kernel, dtype=np.float32)
            payload[f"layer_{idx}_bias"] = np.asarray(bias, dtype=np.float32)
        return payload

    def load_mixer_head_from_npz(self, combiner_weights_npz: object) -> None:
        """Load mixer-head weights from legacy or versioned NPZ payload."""
        version = 1
        if "mixer_head_version" in combiner_weights_npz:
            version = int(np.asarray(combiner_weights_npz["mixer_head_version"]).reshape(()))

        if version <= 1:
            if self._mixer_head_mode != "legacy":
                raise ValueError(
                    "Legacy mixer combiner weights can only be loaded into legacy mixer head mode"
                )
            if "kernel" not in combiner_weights_npz or "bias" not in combiner_weights_npz:
                raise ValueError("Legacy mixer combiner NPZ must contain `kernel` and `bias`")

            combiner_kernel = np.asarray(combiner_weights_npz["kernel"], dtype=np.float32)
            combiner_bias = np.asarray(combiner_weights_npz["bias"], dtype=np.float32)
            expected_kernel_shape = (self.n_apodizations, 1)
            expected_bias_shape = (1,)
            if combiner_kernel.shape != expected_kernel_shape:
                raise ValueError(
                    "Mixer combiner kernel shape mismatch: "
                    f"got {combiner_kernel.shape}, expected {expected_kernel_shape}"
                )
            if combiner_bias.shape != expected_bias_shape:
                raise ValueError(
                    "Mixer combiner bias shape mismatch: "
                    f"got {combiner_bias.shape}, expected {expected_bias_shape}"
                )
            self.pixel_combiner.set_weights([combiner_kernel, combiner_bias])
            return

        if self._mixer_head_mode == "legacy":
            raise ValueError(
                "Versioned mixer-head weights require mixer_head.enabled=true in model config"
            )

        if "mixer_head_num_layers" not in combiner_weights_npz:
            raise ValueError("Versioned mixer combiner NPZ must contain `mixer_head_num_layers`")

        n_layers_saved = int(np.asarray(combiner_weights_npz["mixer_head_num_layers"]).reshape(()))
        n_layers_model = len(self.pixel_combiner.layers)
        if n_layers_saved != n_layers_model:
            raise ValueError(
                "Mixer head layer count mismatch: "
                f"saved={n_layers_saved}, model={n_layers_model}"
            )

        for idx, layer in enumerate(self.pixel_combiner.layers):
            if not isinstance(layer, tf.keras.layers.Dense):
                raise ValueError(
                    "Unsupported layer type in pixel_combiner deserialization: "
                    f"{type(layer).__name__}"
                )
            kernel_key = f"layer_{idx}_kernel"
            bias_key = f"layer_{idx}_bias"
            if kernel_key not in combiner_weights_npz or bias_key not in combiner_weights_npz:
                raise ValueError(
                    "Versioned mixer combiner NPZ is missing required layer keys: "
                    f"{kernel_key}, {bias_key}"
                )

            kernel = np.asarray(combiner_weights_npz[kernel_key], dtype=np.float32)
            bias = np.asarray(combiner_weights_npz[bias_key], dtype=np.float32)
            current_weights = layer.get_weights()
            if len(current_weights) != 2:
                raise ValueError(f"Layer {layer.name} does not expose kernel+bias weights")
            expected_kernel_shape = current_weights[0].shape
            expected_bias_shape = current_weights[1].shape
            if tuple(kernel.shape) != tuple(expected_kernel_shape):
                raise ValueError(
                    f"Mixer layer kernel shape mismatch for {layer.name}: "
                    f"got {kernel.shape}, expected {expected_kernel_shape}"
                )
            if tuple(bias.shape) != tuple(expected_bias_shape):
                raise ValueError(
                    f"Mixer layer bias shape mismatch for {layer.name}: "
                    f"got {bias.shape}, expected {expected_bias_shape}"
                )
            layer.set_weights([kernel, bias])

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
        batch_size = tf.shape(delayed_batch)[0]
        combined_image = tf.reshape(combined_flat, (batch_size, self.nz, self.nx))

        self._last_intermediate_images = intermediate_images
        self._last_apodization_grids = weights_grid
        return combined_image, weights_grid



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


def build_masked_mlp_inr(
    input_dim: int,
    hidden_units_config: int | float | list | tuple,
    n_hidden_layers: int | None = None,
    activation: str = "relu",
    output_activation: str = "sigmoid",
    n_apodizations: int = 1,
    mask_mode: str = "boxcar",
    mask_feature_index: int = 0,
    mask_threshold: float = 0.5,
    mask_epsilon: float = 1e-8,
) -> tf.keras.Model:
    """Build an INR whose outputs are multiplied by a deterministic feature mask.

    Args:
        input_dim: Number of input features.
        hidden_units_config: Hidden layer size specification for the base MLP.
        n_hidden_layers: Number of hidden layers when using an integer size.
        activation: Activation function for hidden layers.
        output_activation: Activation function for the base output layer.
        n_apodizations: Number of output apodization channels.
        mask_mode: Output mask family. Currently only ``"boxcar"``.
        mask_feature_index: Input feature index used to derive the mask.
        mask_threshold: Threshold used by the hard boxcar mask.
        mask_epsilon: Numerical tolerance for the mask boundary.

    Returns:
        Keras model with the same input interface as :func:`build_mlp_inr`.
    """
    inputs = tf.keras.Input(shape=(input_dim,), name="coords")
    base_model = build_mlp_inr(
        input_dim=input_dim,
        hidden_units_config=hidden_units_config,
        n_hidden_layers=n_hidden_layers,
        activation=activation,
        output_activation=output_activation,
        n_apodizations=n_apodizations,
    )
    base_outputs = base_model(inputs)
    output_mask = FeatureDrivenOutputMask(
        mask_mode=mask_mode,
        feature_index=mask_feature_index,
        threshold=mask_threshold,
        epsilon=mask_epsilon,
        output_channels=int(n_apodizations),
        name="feature_driven_output_mask",
    )(inputs)
    masked_outputs = tf.keras.layers.Multiply(name="masked_weight_out")([base_outputs, output_mask])
    return tf.keras.Model(inputs=inputs, outputs=masked_outputs, name="masked_inr_mlp")


def build_symmetrized_apodization_model(
    apodization_model: tf.keras.Model,
    feature_component_names: list[str] | tuple[str, ...],
    flip_feature_tokens: list[str] | tuple[str, ...] | None = None,
    combine_mode: str = "sum",
) -> tf.keras.Model:
    """Wrap an apodization model with deterministic feature-sign symmetry.

    Args:
        apodization_model: Base model to be wrapped.
        feature_component_names: Ordered list of input feature tokens.
        flip_feature_tokens: Optional subset of tokens to sign-flip.
        combine_mode: Combination strategy. Currently only ``"sum"``.

    Returns:
        Keras model with the same input/output interface as ``apodization_model``.
    """
    input_dim = len(feature_component_names)
    inputs = tf.keras.Input(shape=(input_dim,), name="coords")
    outputs = SymmetricApodizationWrapper(
        base_model=apodization_model,
        feature_component_names=feature_component_names,
        flip_feature_tokens=flip_feature_tokens,
        combine_mode=combine_mode,
        name="symmetry_wrapper",
    )(inputs)
    model_name = f"{apodization_model.name}_symmetrized"
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
