# INR-Apodizations

Apodization functions for ultrasound beamforming learned with Implicit Neural
Representations (INR).

In phased-array ultrasound imaging, apodization weights are applied across
transducer elements to reduce side-lobes and improve lateral resolution. This
project explores replacing hand-crafted windows (Hanning, Tukey, and others)
with neural models that predict per-element weights from imaging geometry.

## Background

Delay-and-Sum (DAS) is the standard beamforming algorithm: for each image
pixel, each element signal is time-shifted and summed. Apodization weights
control the trade-off between main-lobe width (resolution) and side-lobe level
(contrast).

An INR model learns a continuous mapping from geometry features to a scalar
apodization weight:

$$
f_\theta(\mathbf{x}) \to w
$$

where $\mathbf{x}$ includes spatial and element-related coordinates.

## End-to-end pipeline

```text
configs/rf_dataset_simus.yml
            -> scripts/generate_rf_dataset_simus.py
            -> data/rf_dataset_simus/<timestamp>/

configs/delayed_samples_dataset.yml (rf_dataset_name = previous timestamp)
            -> scripts/create_delayed_samples_dataset.py
            -> data/delayed_samples_dataset/<timestamp>/

configs/train_mixer_config.yml
            -> scripts/train/train_inr_das_mixer.py
            -> scripts/outputs/train/<timestamp>/

evaluation configs + trained artifacts
            -> scripts/evaluation/*.py and scripts/evaluation/numeric_phantom/*.py
            -> scripts/outputs/evaluation/<...>/
```

## Environment setup

```bash
conda env create -f environment.yml
conda activate inr-apodizations
```

Validate TensorFlow + CuPy GPU availability:

```bash
python scripts/verify_env.py
```

## Quickstart

### 1. Generate RF dataset

Configure [configs/rf_dataset_simus.yml](configs/rf_dataset_simus.yml), then run:

```bash
python scripts/generate_rf_dataset_simus.py
```

Outputs are written to `data/rf_dataset_simus/<timestamp>/`.

### 2. Build delayed-samples dataset

Set `rf_dataset_name` in [configs/delayed_samples_dataset.yml](configs/delayed_samples_dataset.yml)
to the RF timestamp from step 1, then run:

```bash
python scripts/create_delayed_samples_dataset.py
```

Outputs are written to `data/delayed_samples_dataset/<timestamp>/`.

### 3. Train INR (primary workflow: mixer)

Configure [configs/train_mixer_config.yml](configs/train_mixer_config.yml), especially
dataset path and training hyperparameters, then run:

```bash
python scripts/train/train_inr_das_mixer.py
```

Main artifacts are stored in `scripts/outputs/train/<timestamp>/`.

Optional baseline training path (single-apodization):

```bash
python scripts/train/train_inr_das_by_indices.py
```

### 4. Evaluate and benchmark

Baseline classical apodizations:

```bash
python scripts/evaluation/das_standard_apodizations.py
```

Validation baseline metrics and references for training comparisons:

```bash
python scripts/evaluation/evaluate_baseline_apodizations.py
```

Apodization profile FFT analysis:

```bash
python scripts/evaluation/evaluate_apodization_profile_fft.py
```

Numeric phantom suite:

```bash
python scripts/evaluation/numeric_phantom/generate_evaluation_simulation.py
python scripts/evaluation/numeric_phantom/evaluate_apodizations.py
python scripts/evaluation/numeric_phantom/evaluate_mixer.py
python scripts/evaluation/numeric_phantom/compare_inr_models.py
```

## Configuration reference

All runtime parameters are defined in YAML files under [configs](configs). Scripts
are expected to consume configuration files instead of hardcoded execution
parameters.

| Config file | Purpose |
|---|---|
| [configs/rf_dataset_simus.yml](configs/rf_dataset_simus.yml) | RF simulation dataset generation (phantom, probe, acquisition). |
| [configs/delayed_samples_dataset.yml](configs/delayed_samples_dataset.yml) | Delayed-samples generation from an existing RF dataset. |
| [configs/train_config.yml](configs/train_config.yml) | Single-apodization INR training and related evaluation options. |
| [configs/train_mixer_config.yml](configs/train_mixer_config.yml) | Mixer training workflow (primary training path). |
| [configs/tune_config.yml](configs/tune_config.yml) | Hyperparameter tuning configuration. |
| [configs/tune_inr_das_manual.yml](configs/tune_inr_das_manual.yml) | Manual tuning experiments. |
| [configs/das_standard_apodizations.yml](configs/das_standard_apodizations.yml) | Classical DAS apodization visualization/evaluation script settings. |
| [configs/apodization_fft_profiles.yml](configs/apodization_fft_profiles.yml) | FFT profile analysis for apodization comparisons. |
| [configs/numeric_phantom_evaluation_config.yml](configs/numeric_phantom_evaluation_config.yml) | Numeric phantom evaluation for baseline/single-model scenarios. |
| [configs/numeric_phantom_evaluation_mixer_config.yml](configs/numeric_phantom_evaluation_mixer_config.yml) | Numeric phantom evaluation for mixer-based models. |
| [configs/picmus_beamforming.yml](configs/picmus_beamforming.yml) | PICMUS beamforming workflow configuration. |

## Makefile shortcuts

Useful targets from [Makefile](Makefile):

```bash
make create_environment
make requirements
make lint
make format
make run-delayed-samples
make run-phantom-eval
make run-fft-profile-eval
make help
```

## Repository structure

```text
INR-Apodizations/
|- configs/
|- data/
|  |- rf_dataset_simus/
|  |- delayed_samples_dataset/
|  |- raw/ interim/ processed/ external/
|- docs/
|  |- architecture.md
|- inr_apodizations/
|  |- config.py
|  |- coordinate_manager.py
|  |- apodizations.py
|  |- dataset.py
|  |- features.py
|  |- plots.py
|  |- training_console.py
|  |- interactive_navigator.py
|  |- utils.py
|  |- evaluation/
|  |- experiment_helpers/
|  |- kernels/
|  |  |- parameters.py
|  |  |- bf_cuda_kernels/
|  |- modeling/
|     |- das_models.py
|     |- losses.py
|     |- metrics.py
|     |- train.py
|     |- predict.py
|- scripts/
|  |- generate_rf_dataset_simus.py
|  |- create_delayed_samples_dataset.py
|  |- verify_env.py
|  |- train/
|  |- evaluation/
|  |- picmus/
|- models/
|- notebooks/
|- reports/
|- sandbox/
|- pyproject.toml
|- environment.yml
```

## Requirements and compatibility notes

- Python 3.10 is required.
- GPU acceleration is required for heavy delayed-samples and training workflows.
- Environment dependencies are defined in [environment.yml](environment.yml).
- If you change TensorFlow/CuPy/CUDA versions, validate compatibility first with
      [scripts/verify_env.py](scripts/verify_env.py).

## Operational notes

- Some scripts can require a display backend due to matplotlib interactive
      plotting. In headless environments, adapt backend or disable interactive
      display.
- RF and delayed-samples arrays can be large. Check dataset dimensions and GPU
      memory before increasing batch size or image grid size.

## Code style

- Linter and formatter: Ruff (line length 99, import sorting enabled).
- Docstrings: English for new or modified public functions/classes.

