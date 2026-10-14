"""Collect Hyperband trial scores from a Keras Tuner `kt_hyperband` folder.

Reads `trial_*/trial.json` files and `oracle.json` when present and
produces `tuner_trials.json`, `per_architecture_scores.json` and
`per_architecture_scores.csv` in the same folder (or an explicit output
folder).

Usage:
    python collect_hyperband_scores.py --kt_dir path/to/kt_hyperband

The script is robust to variations in the structure of trial.json files
created by different Keras Tuner versions.
"""
from __future__ import annotations

import argparse
import json
import os
import csv
import statistics
from typing import Any
from pathlib import Path
import yaml
from collections import OrderedDict


def _safe_get_hp_values(hp_obj: Any) -> dict:
    """Extract hyperparameter values from a trial hyperparameters object/dict."""
    if hp_obj is None:
        return {}
    if isinstance(hp_obj, dict):
        if "values" in hp_obj and isinstance(hp_obj["values"], dict):
            return dict(hp_obj["values"])
        # fallback: maybe already a values dict
        return dict(hp_obj)
    # Unknown shape
    try:
        return dict(hp_obj.values)
    except Exception:
        try:
            return dict(getattr(hp_obj, "get_config", lambda: {})() or {})
        except Exception:
            return {}


def _extract_score_from_trial_json(data: dict, objective_name: str) -> float | None:
    """Robustly extract a scalar score for the trial from parsed JSON."""
    # Direct score field
    if "score" in data and data["score"] is not None:
        try:
            return float(data["score"])
        except Exception:
            pass

    # Some Keras Tuner versions store metrics under 'metrics'
    metrics = data.get("metrics") or data.get("history") or {}
    if isinstance(metrics, dict):
        # Pattern: metrics -> {objective_name: {observations: [{value: ...}, ...]}}
        if objective_name in metrics:
            obj = metrics[objective_name]
            if isinstance(obj, dict):
                obs = obj.get("observations") or []
                if obs:
                    last = obs[-1]
                    if isinstance(last, dict) and "value" in last:
                        try:
                            return float(last["value"])
                        except Exception:
                            pass
                # fallback to 'best' or 'best_value'
                for k in ("best", "best_value", "best_score"):
                    if k in obj and obj[k] is not None:
                        try:
                            return float(obj[k])
                        except Exception:
                            pass

        # Sometimes metrics contains 'metrics' key
        inner = metrics.get("metrics") if "metrics" in metrics else metrics
        if isinstance(inner, dict) and objective_name in inner:
            obj = inner[objective_name]
            obs = obj.get("observations") if isinstance(obj, dict) else None
            if obs:
                try:
                    return float(obs[-1].get("value"))
                except Exception:
                    pass

    # No scalar score found
    return None


def collect_from_kt_folder(kt_dir: str, objective_name: str = "val_mae") -> tuple[list, dict]:
    """Scan a kt_hyperband folder and return trials info and per-architecture summaries.

    Returns:
        trials_info: list of per-trial dicts
        per_arch: dict mapping architecture_index (str) -> summary dict
    """
    trials_info = []

    # Try to load candidate architectures from the parent tuning dir
    arch_lookup = {}
    try:
        cand_file = Path(kt_dir).parent / "candidate_architectures.yml"
        if cand_file.exists():
            with open(cand_file, "r", encoding="utf-8") as f:
                cand = yaml.safe_load(f) or {}
            # Expect either {'candidates': [...]} or a plain list
            candidates = cand.get("candidates") if isinstance(cand, dict) else cand
            if isinstance(candidates, list):
                for i, arch in enumerate(candidates):
                    arch_lookup[str(i)] = arch
    except Exception:
        arch_lookup = {}

    # List trial directories (trial_XXXX) containing trial.json
    for entry in sorted(os.listdir(kt_dir)):
        entry_path = os.path.join(kt_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        trial_json_path = os.path.join(entry_path, "trial.json")
        if not os.path.exists(trial_json_path):
            continue
        try:
            with open(trial_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # Skip unreadable files
            continue

        hp = _safe_get_hp_values(data.get("hyperparameters") or data.get("hyperparameters_values"))
        score = _extract_score_from_trial_json(data, objective_name)

        arch_index = None
        if isinstance(hp, dict) and "architecture_index" in hp:
            try:
                arch_index = int(hp.get("architecture_index"))
            except Exception:
                arch_index = None

        hidden_units = None
        if arch_index is not None:
            hidden_units = arch_lookup.get(str(arch_index))

        trials_info.append({
            "trial_folder": entry,
            "architecture_index": arch_index,
            "hidden_units": hidden_units,
            "hyperparameters": hp,
            "score": (float(score) if score is not None else None),
        })

    # Group by architecture index
    per_arch = {}
    for t in trials_info:
        ai = t["architecture_index"]
        ai_key = str(ai) if ai is not None else "None"
        per_arch.setdefault(ai_key, {"n_trials": 0, "all_scores": [], "hidden_units": None})
        per_arch[ai_key]["n_trials"] += 1
        per_arch[ai_key]["all_scores"].append(t["score"])
        if per_arch[ai_key]["hidden_units"] is None and t.get("hidden_units") is not None:
            per_arch[ai_key]["hidden_units"] = t.get("hidden_units")

    # Summarize
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

    return trials_info, per_arch


def write_outputs(kt_dir: str, trials_info: list, per_arch: dict) -> None:
    out_trials = os.path.join(kt_dir, "tuner_trials.json")
    out_arch = os.path.join(kt_dir, "per_architecture_scores.json")
    out_csv = os.path.join(kt_dir, "per_architecture_scores.csv")
    # Sort trials by score (ascending). Place None scores at the end.
    def _trial_score_key(t: dict):
        s = t.get("score")
        return (s is None, s if s is not None else float("inf"))

    trials_sorted = sorted(trials_info, key=_trial_score_key)

    with open(out_trials, "w", encoding="utf-8") as f:
        json.dump(trials_sorted, f, indent=2, default=str)

    # Sort architectures by best_score (ascending). None best_score go to the end.
    arch_items = list(per_arch.items())

    def _arch_best_key(item):
        info = item[1]
        bs = info.get("best_score")
        return (bs is None, bs if bs is not None else float("inf"))

    arch_sorted = sorted(arch_items, key=_arch_best_key)

    # Write per-architecture JSON preserving sorted order
    per_arch_ordered = OrderedDict()
    for k, info in arch_sorted:
        per_arch_ordered[k] = info

    with open(out_arch, "w", encoding="utf-8") as f:
        json.dump(per_arch_ordered, f, indent=2, default=str)

    # Write CSV with architectures ordered by best_score
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["architecture_index", "n_trials", "best_score", "median_score", "hidden_units", "all_scores"])
        for ai_key, info in arch_sorted:
            writer.writerow([
                ai_key,
                info.get("n_trials", 0),
                info.get("best_score"),
                info.get("median_score"),
                json.dumps(info.get("hidden_units", None)),
                json.dumps(info.get("all_scores", [])),
            ])


def main():
    parser = argparse.ArgumentParser(description="Collect Keras Tuner Hyperband trial scores from a kt_hyperband folder.")
    parser.add_argument("--kt_dir", required=True, help="Path to the kt_hyperband folder")
    parser.add_argument("--objective", default="val_mae", help="Objective metric name to extract (default: val_mae)")
    args = parser.parse_args()

    kt_dir = args.kt_dir
    if not os.path.isdir(kt_dir):
        print(f"Error: folder not found: {kt_dir}")
        return

    print(f"Scanning kt_hyperband folder: {kt_dir}")
    trials_info, per_arch = collect_from_kt_folder(kt_dir, objective_name=args.objective)
    write_outputs(kt_dir, trials_info, per_arch)
    print("Wrote:")
    print(" -", os.path.join(kt_dir, "tuner_trials.json"))
    print(" -", os.path.join(kt_dir, "per_architecture_scores.json"))
    print(" -", os.path.join(kt_dir, "per_architecture_scores.csv"))


if __name__ == "__main__":
    main()
