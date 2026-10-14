"""Keras Tuner trial score collection and persistence utilities."""

from __future__ import annotations

import csv
import json
import os
import statistics

import numpy as np


def save_tuner_architecture_scores(
    tuner,
    candidate_architectures,
    output_dir: str,
    objective_name: str = "val_mae",
    sort_by_best: bool = True,
) -> None:
    """Collect Keras Tuner trials and save per-architecture score summaries.

    Args:
        tuner: Keras Tuner instance after `.search()` has completed.
        candidate_architectures: List of architectures (indexed by `architecture_index`).
        output_dir: Folder where the artifacts will be saved (created if needed).
        objective_name: Metric name used as the tuner objective (default: "val_mae").
        sort_by_best: Whether to order the saved per-architecture summaries by
            `best_score` ascending (best/lowest first). Defaults to ``True``.

    Behavior:
        - Writes `tuner_trials.json` with one entry per trial (hyperparameters + score).
        - Writes `per_architecture_scores.json` mapping architecture index -> summary.
        - Writes `per_architecture_scores.csv` for quick inspection.
    """
    os.makedirs(output_dir, exist_ok=True)

    trials = getattr(getattr(tuner, "oracle", {}), "trials", {})
    trials_info = []

    arch_lookup = {}
    try:
        for i, arch in enumerate(candidate_architectures):
            arch_lookup[str(i)] = arch
    except Exception:
        arch_lookup = {}

    for trial_id, trial in (trials.items() if isinstance(trials, dict) else []):
        hp_obj = getattr(trial, "hyperparameters", None)
        hp_dict = {}
        if hp_obj is not None:
            if hasattr(hp_obj, "values"):
                try:
                    hp_dict = dict(hp_obj.values)
                except Exception:
                    hp_dict = {}
            else:
                try:
                    hp_dict = dict(getattr(hp_obj, "get_config", lambda: {})() or {})
                except Exception:
                    hp_dict = {}

        score = None
        try:
            if hasattr(trial, "score") and trial.score is not None:
                score = float(trial.score)
        except Exception:
            score = None

        if score is None:
            metrics_obj = getattr(trial, "metrics", None)
            if metrics_obj is not None:
                try:
                    if hasattr(metrics_obj, "get_last_value"):
                        val = metrics_obj.get_last_value(objective_name)
                        if val is not None:
                            score = float(val)
                except Exception:
                    try:
                        if hasattr(metrics_obj, "get_best_value"):
                            val = metrics_obj.get_best_value(objective_name)
                            if val is not None:
                                score = float(val)
                    except Exception:
                        score = None

        arch_index = None
        if isinstance(hp_dict, dict) and "architecture_index" in hp_dict:
            try:
                arch_index = int(hp_dict.get("architecture_index"))
            except Exception:
                arch_index = None

        hidden_units = None
        try:
            if arch_index is not None:
                hidden_units = arch_lookup.get(str(arch_index))
        except Exception:
            hidden_units = None

        trials_info.append({
            "trial_id": str(trial_id),
            "architecture_index": arch_index,
            "hidden_units": hidden_units,
            "hyperparameters": {
                k: (
                    int(v)
                    if isinstance(v, (np.integer,))
                    else (float(v) if isinstance(v, (np.floating,)) else v)
                )
                for k, v in hp_dict.items()
            },
            "score": (float(score) if score is not None else None),
        })

    per_arch = {}
    for trial_info in trials_info:
        ai = trial_info["architecture_index"]
        ai_key = str(ai) if ai is not None else "None"
        per_arch.setdefault(ai_key, {"n_trials": 0, "all_scores": [], "hidden_units": None})
        per_arch[ai_key]["n_trials"] += 1
        per_arch[ai_key]["all_scores"].append(trial_info["score"])
        if per_arch[ai_key]["hidden_units"] is None and trial_info.get("hidden_units") is not None:
            per_arch[ai_key]["hidden_units"] = trial_info.get("hidden_units")

    for ai_key, info in per_arch.items():
        scores = [s for s in info["all_scores"] if s is not None]
        if scores:
            info["best_score"] = float(min(scores))
            try:
                info["median_score"] = float(statistics.median(scores))
            except Exception:
                info["median_score"] = None
        else:
            info["best_score"] = None
            info["median_score"] = None

    if sort_by_best:

        def _best_score_key(item):
            info = item[1]
            score = info.get("best_score")
            return float(score) if score is not None else float("inf")

        ordered_items = sorted(per_arch.items(), key=_best_score_key)
    else:
        ordered_items = sorted(per_arch.items(), key=lambda x: (x[0] if x[0] != "None" else "zz"))

    ordered_per_arch = {k: v for k, v in ordered_items}

    trials_path = os.path.join(output_dir, "tuner_trials.json")
    with open(trials_path, "w", encoding="utf-8") as f:
        json.dump(trials_info, f, indent=2, default=str)

    arch_path = os.path.join(output_dir, "per_architecture_scores.json")
    with open(arch_path, "w", encoding="utf-8") as f:
        json.dump(ordered_per_arch, f, indent=2, default=str)

    csv_path = os.path.join(output_dir, "per_architecture_scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "architecture_index",
                "n_trials",
                "best_score",
                "median_score",
                "hidden_units",
                "all_scores",
            ]
        )
        for ai_key, info in ordered_items:
            writer.writerow(
                [
                    ai_key,
                    info.get("n_trials", 0),
                    info.get("best_score"),
                    info.get("median_score"),
                    json.dumps(info.get("hidden_units", None)),
                    json.dumps(info.get("all_scores", [])),
                ]
            )
