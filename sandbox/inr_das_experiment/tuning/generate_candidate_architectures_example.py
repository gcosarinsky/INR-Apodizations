"""Example: generate candidate INR architectures and print them.

This small script mirrors the generation logic used by the tuning script
but runs standalone so you can inspect the produced candidate lists.
"""
from __future__ import annotations

from pathlib import Path
import yaml
import sys

from inr_apodizations.tuning.utils import generate_candidate_architectures


cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("configs/tune_config.yml")
if not cfg_path.exists():
    print(f"Config file not found: {cfg_path}")
    sys.exit(2)

cfg = yaml.safe_load(cfg_path.read_text())
tuning = cfg.get("tuning", {})
seed = int(cfg.get("training", {}).get("seed", 42))

candidates = generate_candidate_architectures(tuning, fallback_seed=seed)
print(f"Generated {len(candidates)} candidate architectures from {cfg_path}")
for i, arch in enumerate(candidates):
    print(f"{i}: {arch}")



