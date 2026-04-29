"""Utility functions for tuner candidate generation and live tuning plots.

This module centralizes generation of explicit candidate architectures and
shared helpers used by the Keras Tuner workflow.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any, Callable, List, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    import keras_tuner as kt
except Exception:  # pragma: no cover - only needed when tuning is unavailable.
    kt = None

try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional import when TF is unavailable.
    tf = None


def _build_discrete_values(min_value: int, max_value: int, step: int, name: str) -> List[int]:
    if step <= 0:
        raise ValueError(f"{name}_step must be > 0. Received: {step}.")
    if min_value > max_value:
        raise ValueError(f"{name}_min must be <= {name}_max. Received: {min_value} > {max_value}.")
    values = list(range(int(min_value), int(max_value) + 1, int(step)))
    if not values:
        raise ValueError(f"{name} interval produced an empty set of values.")
    return values


def generate_candidate_architectures(tuning_cfg: dict, fallback_seed: int) -> List[List[int]]:
    """Generate candidate INR architectures as explicit neuron-count lists.

    Parameters
    ----------
    tuning_cfg : dict
        Tuning section from the experiment config.
    fallback_seed : int
        Seed used when `candidate_seed` is not provided.

    Returns
    -------
    List[List[int]]
        Candidate architectures, each represented as `[units_layer_0, ..., units_layer_n]`.
    """
    layer_values = _build_discrete_values(
        min_value=int(tuning_cfg["n_layers_min"]),
        max_value=int(tuning_cfg["n_layers_max"]),
        step=int(tuning_cfg.get("n_layers_step", 1)),
        name="n_layers",
    )
    unit_values = _build_discrete_values(
        min_value=int(tuning_cfg["units_min"]),
        max_value=int(tuning_cfg["units_max"]),
        step=int(tuning_cfg["units_step"]),
        name="units",
    )

    non_increasing = bool(tuning_cfg.get("non_increasing", True))
    all_candidates: List[List[int]] = []
    seen: set[tuple[int, ...]] = set()

    for n_layers in layer_values:
        for architecture in itertools.product(unit_values, repeat=n_layers):
            if non_increasing and any(architecture[i] < architecture[i + 1] for i in range(n_layers - 1)):
                continue
            if architecture in seen:
                continue
            seen.add(architecture)
            all_candidates.append(list(architecture))

    if not all_candidates:
        raise ValueError("No candidate architectures were generated from the provided tuning ranges.")

    max_candidates = tuning_cfg.get("max_candidates")
    if max_candidates is not None:
        max_candidates = int(max_candidates)
        if max_candidates <= 0:
            raise ValueError(f"max_candidates must be > 0 when provided. Received: {max_candidates}.")
        if len(all_candidates) > max_candidates:
            # Prioritize candidates with fewer total neurons.
            # Sort by (total_neurons, architecture_tuple) to make selection deterministic
            # and stable for architectures with the same total.
            sorted_candidates = sorted(
                all_candidates, key=lambda arch: (sum(arch), tuple(arch))
            )
            all_candidates = sorted_candidates[:max_candidates]

    return all_candidates


class LiveTrialScorePlot:
    """Track completed trial scores and refresh a matplotlib figure.

    The figure shows trial scores as scatter points, the current best score as
    a horizontal line, and a text box with the hyperparameters of the best
    trial seen so far.
    """

    def __init__(
        self,
        output_path: str | Path,
        objective_name: str,
        architecture_lookup: Sequence[Sequence[int]] | None = None,
    ):
        self.output_path = Path(output_path)
        self.objective_name = str(objective_name)
        self.architecture_lookup = architecture_lookup
        self.trial_ids: list[str] = []
        self.trial_scores: list[float] = []
        self.best_so_far: list[float] = []
        self.best_trial_summary: list[str] = []
        self.figure = None
        self.axis = None
        self.enabled = MATPLOTLIB_AVAILABLE

        if self.enabled:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.figure, self.axis = plt.subplots(figsize=(10, 5))
            self.axis.set_xlabel("Completed trial")
            self.axis.set_ylabel(self.objective_name)
            self.axis.grid(True, alpha=0.3)

    def add_score(self, trial_id: str, score: float, trial: Any | None = None) -> None:
        """Record a completed trial score and refresh the plot."""
        if not self.enabled:
            return

        try:
            score_value = float(score)
        except Exception:
            return

        self.trial_ids.append(str(trial_id))
        self.trial_scores.append(score_value)
        if not self.best_so_far:
            self.best_so_far.append(score_value)
            self.best_trial_summary = self._format_trial_summary(trial, trial_id, score_value)
        else:
            current_best = min(self.best_so_far[-1], score_value)
            self.best_so_far.append(current_best)
            if score_value <= self.best_so_far[-2]:
                self.best_trial_summary = self._format_trial_summary(trial, trial_id, score_value)

        self._redraw()

    def _format_trial_summary(self, trial: Any | None, trial_id: str, score: float) -> list[str]:
        """Format the best trial hyperparameters as display lines."""
        if trial is None:
            return [f"trial_id: {trial_id}", f"score: {score:.6g}"]

        hp_values = dict(getattr(trial.hyperparameters, "values", {}))
        arch_index = hp_values.get("architecture_index")
        architecture = None
        if arch_index is not None and self.architecture_lookup is not None:
            try:
                architecture = self.architecture_lookup[int(arch_index)]
            except Exception:
                architecture = None

        lines = [f"trial_id: {trial_id}", f"score: {score:.6g}"]
        if architecture is not None:
            lines.append(f"architecture: {list(architecture)}")

        for key in ("architecture_index", "reg_lambda", "reg_tau", "lr", "pixel_weight_lambda"):
            if key in hp_values:
                value = hp_values[key]
                if isinstance(value, float):
                    lines.append(f"{key}: {value:.6g}")
                else:
                    lines.append(f"{key}: {value}")

        return lines

    def _redraw(self) -> None:
        """Redraw the live plot and persist it to disk."""
        if not self.enabled or self.figure is None or self.axis is None:
            return

        self.axis.clear()
        x_values = np.arange(1, len(self.trial_scores) + 1)
        self.axis.scatter(x_values, self.trial_scores, s=28, label="trial score")
        if self.best_so_far:
            self.axis.axhline(self.best_so_far[-1], linestyle="--", linewidth=1.4, label="best so far")
        self.axis.set_title(f"Hyperband progress - {self.objective_name}")
        self.axis.set_xlabel("Completed trial")
        self.axis.set_ylabel(self.objective_name)
        self.axis.grid(True, alpha=0.3)
        self.axis.legend(loc="best")

        if self.best_trial_summary:
            textbox = "\n".join(self.best_trial_summary)
            self.axis.text(
                0.98,
                0.98,
                textbox,
                transform=self.axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                family="monospace",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.4", "boxstyle": "round,pad=0.35"},
            )

        self.figure.tight_layout()
        try:
            self.figure.canvas.draw()
        except Exception:
            pass

        try:
            self.figure.savefig(self.output_path, dpi=150)
        except Exception:
            pass


class _RegularizationAutoInitCallback(
    tf.keras.callbacks.Callback if tf is not None else object
):
    """Initialize regularization lambda at trial start using the first train batch."""

    def __init__(
        self,
        train_ds: Any,
        trial_id: str,
        ratio: float,
        epsilon: float,
        norm_fraction: float,
        log_sink: dict[str, dict[str, Any]],
        persist_log: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.train_ds = train_ds
        self.trial_id = str(trial_id)
        self.ratio = float(ratio)
        self.epsilon = float(epsilon)
        self.norm_fraction = float(norm_fraction)
        self.log_sink = log_sink
        self.persist_log = persist_log

    def __deepcopy__(self, memo: dict) -> "_RegularizationAutoInitCallback":
        """Return a copy that shares non-copyable references (dataset, log dict, callable)."""
        import copy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # Attributes that must be shared across copies or are not deep-copyable.
        _shared = ("train_ds", "log_sink", "persist_log")
        for key, value in self.__dict__.items():
            if key in _shared:
                setattr(result, key, value)
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        return result

    def _write_log(self, payload: dict[str, Any]) -> None:
        self.log_sink[self.trial_id] = payload
        if self.persist_log is not None:
            try:
                self.persist_log()
            except Exception:
                pass

    def on_train_begin(self, logs=None) -> None:  # noqa: D401
        """Compute and apply trial-specific lambda before the first optimizer step."""
        del logs
        model = self.model
        if model is None:
            self._write_log({
                "trial_id": self.trial_id,
                "status": "skipped",
                "reason": "model_unavailable",
            })
            return

        if not bool(getattr(model, "weight_regularization_enabled", False)):
            self._write_log({
                "trial_id": self.trial_id,
                "status": "skipped",
                "reason": "weight_regularization_disabled",
            })
            return

        if not hasattr(model, "compute_weight_regularization") or not hasattr(model, "reconstruct_image"):
            self._write_log({
                "trial_id": self.trial_id,
                "status": "skipped",
                "reason": "model_missing_regularization_api",
            })
            return

        try:
            first_batch = next(iter(self.train_ds.take(1)))
        except StopIteration:
            self._write_log({
                "trial_id": self.trial_id,
                "status": "skipped",
                "reason": "empty_train_dataset",
            })
            return
        except Exception as exc:
            self._write_log({
                "trial_id": self.trial_id,
                "status": "skipped",
                "reason": f"failed_to_read_first_batch: {exc}",
            })
            return

        if isinstance(first_batch, (tuple, list)) and len(first_batch) == 3:
            x_init, y_init, sample_weight_init = first_batch
        elif isinstance(first_batch, (tuple, list)) and len(first_batch) == 2:
            x_init, y_init = first_batch
            sample_weight_init = None
        else:
            self._write_log({
                "trial_id": self.trial_id,
                "status": "skipped",
                "reason": "unsupported_batch_structure",
            })
            return

        try:
            y_pred_init, weights_grid_init = model.reconstruct_image(x_init, training=False)

            loss_fn = getattr(model, "loss", None)
            if callable(loss_fn):
                if sample_weight_init is None:
                    mae_initial = float(loss_fn(y_init, y_pred_init).numpy())
                else:
                    mae_initial = float(
                        loss_fn(y_init, y_pred_init, sample_weight=sample_weight_init).numpy()
                    )
            else:
                if sample_weight_init is None:
                    mae_initial = float(model.compiled_loss(y_init, y_pred_init).numpy())
                else:
                    mae_initial = float(
                        model.compiled_loss(
                            y_init,
                            y_pred_init,
                            sample_weight=sample_weight_init,
                            regularization_losses=None,
                        ).numpy()
                    )

            lambda_previous = float(getattr(model, "weight_regularization_lambda", 0.0))
            model.weight_regularization_lambda = 1.0
            reg_loss_initial_tensor, norm_initial_tensor, reg_active_tensor = (
                model.compute_weight_regularization(weights_grid_init)
            )
            reg_loss_initial = float(reg_loss_initial_tensor.numpy())
            norm_initial = float(norm_initial_tensor.numpy())
            reg_active_initial = bool(float(reg_active_tensor.numpy()) > 0.0)

            tau = float(getattr(model, "weight_regularization_tau", 0.0))
            hinge_active = reg_active_initial and (reg_loss_initial > self.epsilon)
            fallback_used = not hinge_active

            reg_loss_assumed = None
            if hinge_active:
                denominator = max(reg_loss_initial, self.epsilon)
            else:
                norm_assumed = float(self.norm_fraction * tau)
                violation_assumed = max(0.0, tau - norm_assumed)
                reg_loss_assumed = float(violation_assumed * violation_assumed)
                denominator = max(reg_loss_assumed, self.epsilon)

            lambda_applied = float(self.ratio * mae_initial / denominator)
            model.weight_regularization_lambda = lambda_applied

            self._write_log({
                "trial_id": self.trial_id,
                "status": "applied",
                "ratio": self.ratio,
                "epsilon": self.epsilon,
                "norm_fraction": self.norm_fraction,
                "mae_initial": mae_initial,
                "norm_initial": norm_initial,
                "tau": tau,
                "reg_loss_initial": reg_loss_initial,
                "reg_active_initial": reg_active_initial,
                "hinge_active": hinge_active,
                "fallback_used": fallback_used,
                "reg_loss_assumed": reg_loss_assumed,
                "lambda_previous_config": lambda_previous,
                "lambda_applied": lambda_applied,
            })
        except Exception as exc:
            try:
                model.weight_regularization_lambda = lambda_previous
            except Exception:
                pass
            self._write_log({
                "trial_id": self.trial_id,
                "status": "failed",
                "reason": f"autoinit_exception: {exc}",
            })


class _TrialEpochCounterCallback(
    tf.keras.callbacks.Callback if tf is not None else object
):
    """Count trained epochs per trial and accumulate across Hyperband promotions."""

    def __init__(
        self,
        trial_id: str,
        log_sink: dict[str, int],
        persist_log: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.trial_id = str(trial_id)
        self.log_sink = log_sink
        self.persist_log = persist_log
        self._epochs_this_run = 0

    def __deepcopy__(self, memo: dict) -> "_TrialEpochCounterCallback":
        """Return a copy that shares non-copyable references (log dict and callable)."""
        import copy

        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        _shared = ("log_sink", "persist_log")
        for key, value in self.__dict__.items():
            if key in _shared:
                setattr(result, key, value)
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        return result

    def on_train_begin(self, logs=None) -> None:  # noqa: D401
        """Reset per-run epoch counter at the beginning of each fit invocation."""
        del logs
        self._epochs_this_run = 0

    def on_epoch_end(self, epoch, logs=None) -> None:  # noqa: D401
        """Increase epoch count after each completed training epoch."""
        del epoch, logs
        self._epochs_this_run += 1

    def on_train_end(self, logs=None) -> None:  # noqa: D401
        """Accumulate run epochs into the trial-level total and optionally persist."""
        del logs
        self.log_sink[self.trial_id] = int(self.log_sink.get(self.trial_id, 0)) + int(
            self._epochs_this_run
        )
        if self.persist_log is not None:
            try:
                self.persist_log()
            except Exception:
                pass


class PlottingHyperband(kt.Hyperband if kt is not None else object):
    """Hyperband tuner that refreshes a live score plot after each trial."""

    def __init__(
        self,
        *args,
        live_plot: LiveTrialScorePlot | None = None,
        trial_data_builder: Callable[[Any], tuple[Any, Any]] | None = None,
        weight_regularization_autoinit_enabled: bool = False,
        weight_regularization_autoinit_ratio: float = 0.5,
        weight_regularization_autoinit_epsilon: float = 1e-12,
        weight_regularization_autoinit_norm_fraction: float = 0.9,
        autoinit_log_path: str | Path | None = None,
        trial_epoch_log_path: str | Path | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.live_plot = live_plot
        self.trial_data_builder = trial_data_builder
        self.weight_regularization_autoinit_enabled = bool(weight_regularization_autoinit_enabled)
        self.weight_regularization_autoinit_ratio = float(weight_regularization_autoinit_ratio)
        self.weight_regularization_autoinit_epsilon = float(weight_regularization_autoinit_epsilon)
        self.weight_regularization_autoinit_norm_fraction = float(
            weight_regularization_autoinit_norm_fraction
        )
        if self.weight_regularization_autoinit_ratio < 0.0:
            raise ValueError("weight_regularization_autoinit_ratio must be >= 0")
        if self.weight_regularization_autoinit_epsilon <= 0.0:
            raise ValueError("weight_regularization_autoinit_epsilon must be > 0")
        if self.weight_regularization_autoinit_norm_fraction <= 0.0:
            raise ValueError("weight_regularization_autoinit_norm_fraction must be > 0")

        self.autoinit_log_path = Path(autoinit_log_path) if autoinit_log_path else None
        self.trial_autoinit_log: dict[str, dict[str, Any]] = {}
        self.trial_epoch_log_path = Path(trial_epoch_log_path) if trial_epoch_log_path else None
        self.trial_epoch_log: dict[str, int] = {}

    def _persist_autoinit_log(self) -> None:
        """Persist auto-init diagnostics to disk when path is configured."""
        if self.autoinit_log_path is None:
            return
        self.autoinit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.autoinit_log_path, "w", encoding="utf-8") as file:
            json.dump(self.trial_autoinit_log, file, indent=2)

    def _persist_epoch_log(self) -> None:
        """Persist epoch-count diagnostics to disk when path is configured."""
        if self.trial_epoch_log_path is None:
            return
        self.trial_epoch_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trial_epoch_log_path, "w", encoding="utf-8") as file:
            json.dump(self.trial_epoch_log, file, indent=2)

    def on_trial_end(self, trial):
        """Handle the end of a trial and refresh the live score plot."""
        super().on_trial_end(trial)
        if self.live_plot is not None and getattr(trial, "score", None) is not None:
            self.live_plot.add_score(trial.trial_id, trial.score, trial=trial)

    def run_trial(self, trial, *args, **kwargs):
        """Run a trial with trial-specific datasets when a builder is provided."""
        kwargs = dict(kwargs)
        if tf is not None:
            callbacks = list(kwargs.get("callbacks", []))
            callbacks.append(
                _TrialEpochCounterCallback(
                    trial_id=trial.trial_id,
                    log_sink=self.trial_epoch_log,
                    persist_log=self._persist_epoch_log,
                )
            )
            kwargs["callbacks"] = callbacks

        if self.trial_data_builder is not None:
            train_ds, val_ds = self.trial_data_builder(trial)
            kwargs["x"] = train_ds
            kwargs["validation_data"] = val_ds
            if self.weight_regularization_autoinit_enabled and tf is not None:
                callbacks = list(kwargs.get("callbacks", []))
                callbacks.append(
                    _RegularizationAutoInitCallback(
                        train_ds=train_ds,
                        trial_id=trial.trial_id,
                        ratio=self.weight_regularization_autoinit_ratio,
                        epsilon=self.weight_regularization_autoinit_epsilon,
                        norm_fraction=self.weight_regularization_autoinit_norm_fraction,
                        log_sink=self.trial_autoinit_log,
                        persist_log=self._persist_autoinit_log,
                    )
                )
                kwargs["callbacks"] = callbacks
        return super().run_trial(trial, **kwargs)
