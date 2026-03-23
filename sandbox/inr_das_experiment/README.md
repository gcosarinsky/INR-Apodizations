INR DAS Experiment (sandbox)

This folder contains an experimental sandbox to train an INR that predicts
apodization weights from geometry features and applies the physical forward:

1. Build weights for each `(element, z, x)` point.
2. Multiply `weights * delayed_samples`.
3. Sum over the element axis.
4. Compare `abs(image)` against the stored target.

Structure:
- `config.yml`: sandbox configuration for paths, model hyperparameters and training settings.
- `helpers.py`: utilities to load datasets, build the coordinate manager, split examples and save artifacts.
- `train_inr_das.py`: training script with the real forward `weights × delayed -> image`.
- `outputs/`: sandbox copy of run artifacts.

Run (from repo root):

```bash
python sandbox/inr_das_experiment/train_inr_das.py
```

Notes:
- Main artifacts are saved to `data/processed/inr_das_experiment/<timestamp>/`.
- A sandbox copy is also saved to `sandbox/inr_das_experiment/outputs/<timestamp>/`.
- If the dataset has a single example, the script reuses it for validation to keep the sandbox runnable.

Generated plots (saved in the sandbox output run directory):
- `training_loss.png`: training/validation loss curves (and RMSE metrics when available).
- `das_images_comparison_db.png`: DAS comparison panel with Uniform, INR before training,
  INR after training, and Target in shared dB scale.
- `apodization_map_before_after.png`: apodization maps before and after training at fixed x.
