"""Utility functions for tuner candidate generation and live tuning plots.

This module centralizes generation of explicit candidate architectures and
shared helpers used by the Keras Tuner workflow.
"""
from __future__ import annotations

import itertools
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


class PlottingHyperband(kt.Hyperband if kt is not None else object):
    """Hyperband tuner that refreshes a live score plot after each trial."""

    def __init__(
        self,
        *args,
        live_plot: LiveTrialScorePlot | None = None,
        trial_data_builder: Callable[[Any], tuple[Any, Any]] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.live_plot = live_plot
        self.trial_data_builder = trial_data_builder

    def on_trial_end(self, trial):
        """Handle the end of a trial and refresh the live score plot."""
        super().on_trial_end(trial)
        if self.live_plot is not None and getattr(trial, "score", None) is not None:
            self.live_plot.add_score(trial.trial_id, trial.score, trial=trial)

    def run_trial(self, trial, *args, **kwargs):
        """Run a trial with trial-specific datasets when a builder is provided."""
        if self.trial_data_builder is not None:
            train_ds, val_ds = self.trial_data_builder(trial)
            kwargs = dict(kwargs)
            kwargs["x"] = train_ds
            kwargs["validation_data"] = val_ds
        return super().run_trial(trial, **kwargs)
