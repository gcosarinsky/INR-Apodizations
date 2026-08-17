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

## PICMUS Pipeline

For real or experimental ultrasound data, the project includes a dedicated PICMUS workflow
that bypasses RF simulation and directly processes HDF5 recorded data.

**Configuration:** [configs/picmus_beamforming.yml](configs/picmus_beamforming.yml)

**Main Script:** [scripts/picmus/process_picmus.py](scripts/picmus/process_picmus.py)

The PICMUS workflow:
1. Loads real/experimental RF data and acquisition parameters from HDF5 files
2. Computes delayed samples on a regular imaging grid via GPU kernels
3. Applies and compares multiple apodization strategies simultaneously:
   - **Uniform:** Equal weights across all elements (baseline)
   - **Hanning:** Classical frequency-domain apodization
   - **INR:** Learned neural apodization (if model path configured)
   - **Mixer:** Multi-apodization combination via configurable mixer head (if model path configured)
   - **NSI:** Null Substraction Imaging for noise/clutter reduction
4. Extracts reflector profiles (lateral and axial) for quantitative comparison
5. Visualizes all apodizations side-by-side with extracted profiles

**Key configuration options:**
- `io`: Paths to HDF5 files (RF data, scan geometry, phantom scatterers)
- `probe`: Transducer parameters (pitch, element width, number of elements, frequency, bandwidth)
- `acquisition`: Sampling and timing (sample rate, number of pulses, start time)
- `angle_subset`: Select all angles or a specific subset for faster processing
- `apodizations`: Enable/disable each method and configure method-specific parameters

This workflow is useful for:
- Validating trained models on real data
- Comparing neural apodizations against classical baselines
- Benchmarking performance metrics (SNR, resolution, contrast)
- Debugging and analyzing per-element weight distributions

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

## Multi-Apodization Mixer Architecture

The primary training workflow (`train_inr_das_mixer.py`) uses a **multi-apodization model**
with a configurable **pixel-wise combiner head** that enables learned fusion of multiple
apodization strategies.

### Model Design

The `DasInrApodMixer` model architecture:

1. **INR Apodization Engine**: A neural network that predicts **N independent apodization channels**
   from geometry features (spatial coordinates, element positions, etc.)
   - Output shape: `(N_elements, N_z, N_x, N_apodizations)`
   - Each channel represents a distinct weighting strategy

2. **Per-Channel DAS Reconstruction**: For each of the N apodization channels:
   - Apply delayed-sample data weighted by predicted apodization values
   - Compute magnitude DAS image (phase-invariant beamforming)
   - Produces N intermediate images: `(N_z, N_x, N_apodizations)`

3. **Pixel-wise Combiner (Mixer Head)**: A configurable MLP that fuses N channels at each pixel
   - Input: Concatenated intermediate images at each `(z, x)` location
   - Output: Single combined image value per pixel
   - **Architecture options** (via `mixer_head` config):
     - **Legacy mode** (`enabled: false`): Single Dense layer with learnable weights
     - **MLP mode** (`enabled: true`): Multi-layer sequential network with configurable hidden units
       - Example: `hidden_units: [64, 32]` creates two hidden layers (64 and 32 units)
       - Activation functions configurable per layer

### Feature Handling & Training

**Physical Features**: The INR input uses domain-specific geometry features:
- Normalized distance from element: `|x - x_elem| / D` (where D is aperture)
- Axial depth: `z`
- Relative element position: `x_rel / z`
- Other combinations suitable for beamforming (scaled or unscaled)

**Feature Batching**: To avoid GPU memory overflow with large images:
- Features processed in chunks (`feature_batch_size`, typically 32K–65K features per batch)
- Model runs inference on chunk, accumulates predictions before combining

**Regularization**:
- **Weight norm regularizer**: Enforces minimum per-channel L2 norm to avoid degenerate solutions. 
  - Parameter: `weight_regularization_tau` (typical: 0.30)
  - Loss term: $L_{\text{norm}} = \sum_c \max(0, \tau - \|w_c\|_2)$ where $w_c$ is the weight vector for channel $c$ and $\tau$ is the threshold

- **Lateral regularizer**: Penalizes high apodization values at large lateral offsets.
  - Parameter: `lateral_regularization_q_power` (typical: 1.0)
  - Loss term: $L_{\text{lateral}} = \sum_{c,z,x} a_c(z,x) \cdot q(x,z)^p$ where $q(x,z) = |x_\text{rel}|/z$ is the normalized lateral offset, $p$ is the power exponent, and $a_c(z,x)$ is the apodization value
  - Encourages smooth lateral tapering

### Training Configuration

See [configs/train_mixer_config.yml](configs/train_mixer_config.yml) for full parameters:
- `model.n_apodizations`: Number of channels (e.g., 3–5)
- `model.mixer_head`: Combiner architecture settings
- `loss.weight_regularization_enabled`: Enable/disable norm penalty
- `loss.lateral_regularization_enabled`: Enable/disable lateral smoothing

### Backward Compatibility & Model Persistence

- Trained INR and mixer head weights stored separately:
  - `model.keras`: Keras-serialized INR (can be versioned)
  - `mixer_combiner_weights.npz`: Mixer head weights with version metadata
- Version tracking ensures old models load correctly even if architecture evolves
- Both `scripts/picmus/process_picmus.py` and evaluation scripts auto-detect and load versioned artifacts

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

### 3b. Alternative: Process Real Data (PICMUS)

Instead of steps 1–2 above, load and process real ultrasound data directly from HDF5:

Configure [configs/picmus_beamforming.yml](configs/picmus_beamforming.yml), then run:

```bash
python scripts/picmus/process_picmus.py
```

This script automatically:
- Loads RF data, acquisition parameters, and phantom scatterers from HDF5 files
- Computes delayed samples via GPU kernels
- Generates comparisons across all enabled apodizations (Uniform, Hanning, INR, NSI, Mixer)
- Extracts reflector profiles (lateral and axial slices)
- Produces visualizations with metrics

Outputs are saved to `scripts/outputs/picmus/<timestamp>/` and include:
- Combined image figure with all apodization methods overlaid
- Per-scatterer profile pages (lateral and axial)

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

## Training Modes

The project supports two primary training workflows:

### Single-Apodization Mode

Use [train_inr_das_by_indices.py](scripts/train/train_inr_das_by_indices.py) with
[configs/train_config.yml](configs/train_config.yml) to train a single apodization function:

```bash
python scripts/train/train_inr_das_by_indices.py
```

- Output: Single INR model predicting one apodization weight map
- Baseline for comparison; simpler architecture, lower memory overhead
- Useful for rapid prototyping or dataset validation

### Multi-Apodization + Mixer Mode (Primary)

Use [train_inr_das_mixer.py](scripts/train/train_inr_das_mixer.py) with
[configs/train_mixer_config.yml](configs/train_mixer_config.yml) to train multiple apodization
channels with a learned pixel-wise combiner:

```bash
python scripts/train/train_inr_das_mixer.py
```

- Output: Multi-channel INR + mixer head weights
- Enables more flexible beamforming by learning channel combinations
- Configuration options:
  - `model.n_apodizations`: Number of independent channels (3–5 typical)
  - `model.mixer_head.hidden_units`: Mixer combiner architecture (legacy Dense or configurable MLP)
  - `loss.weight_regularization_enabled`: Enforce minimum L2 norm per channel
  - `loss.lateral_regularization_enabled`: Smooth lateral tapering

### Hyperparameter Tuning (Optional)

Use [configs/tune_inr_das_manual.yml](configs/tune_inr_das_manual.yml) or
[configs/tune_config.yml](configs/tune_config.yml) for systematic exploration:

```bash
# Manual tuning with specific parameter overrides
python scripts/train/train_inr_das_mixer.py --config configs/tune_inr_das_manual.yml
```

### Resume & Checkpoint System

Both training modes support resuming interrupted runs:
- Set `resume.enabled: true` in your config
- Specify `resume.run_dir` pointing to a previous run directory
- Training will load saved weights, history, and continue from the last epoch
- Metadata stored in `train_config_info.yml` per run

## Post-Training Analysis

After training, use interactive tools to review and analyze results:

### Replay Training History

Regenerate training curves from a completed run:

```bash
# Edit RUN_DIR in the script to point to your training output folder
python scripts/train/replay_training_history.py
```

Loads `history_full.json` (or `history.json`) and regenerates:
- Loss curves (training, validation, regularization)
- Learning rate schedules
- Baseline (Hanning) reference lines
- All output figures in the run directory

Useful for:
- Re-plotting with different styling or axes
- Extracting metrics for reports
- Comparing multiple runs side-by-side

### Review Training Results Interactively

Inspect trained model predictions in detail:

```bash
# Edit RUN_DIR in the script to point to your training output folder
python scripts/train/review_training_results.py
```

Launches an interactive navigator (`InteractiveImageNavigator`) for:
- Pixel-by-pixel inspection of apodization maps
- Comparison: uniform DAS vs. trained model vs. target
- Per-channel analysis (for mixer models: inspect each intermediate image)
- Debug visualization of learned weight patterns

Outputs:
- INR comparison figures (before/after training)
- Per-channel breakdown (for multi-apodization models)
- Interactive terminal-based navigation

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
make create_environment      # Create conda environment from environment.yml
make requirements            # Update environment with latest dependencies
make lint                    # Check code style with ruff
make format                  # Auto-format code with ruff
make clean                   # Remove compiled Python files and caches
make run-delayed-samples     # Run delayed-samples dataset generation
make run-train-mixer         # Run multi-apodization mixer training
make run-phantom-create      # Generate numeric phantom evaluation simulation
make run-phantom-eval        # Run numeric phantom evaluation (baseline/single-model)
make run-phantom-eval-mixer  # Run numeric phantom evaluation (mixer-based models)
make run-fft-profile-eval    # Run FFT profile analysis for apodizations
make help                    # Show all available targets
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
|  |- hilbert_coef.py
|  |- reporting.py
|  |- utils.py
|  |- evaluation/
|  |  |- baseline.py
|  |  |- io_utils.py
|  |  |- metrics.py
|  |  |- profiles.py
|  |  |- scatterers.py
|  |  |- snr.py
|  |  |- summary.py
|  |- experiment_helpers/
|  |  |- _artifacts.py        (artifact persistence & history merging)
|  |  |- _config.py           (config parsing & resume logic)
|  |  |- _dataset_io.py       (dataset loading & validation)
|  |  |- _hardware.py         (GPU memory diagnostics)
|  |  |- _plotting.py         (training curves & visualizations)
|  |  |- _training.py         (training utilities & callbacks)
|  |  |- _tuning.py           (hyperparameter tuning integration)
|  |  |- __init__.py          (re-exports all public symbols)
|  |- kernels/
|  |  |- parameters.py
|  |  |- bf_cuda_kernels/
|  |- modeling/
|  |  |- das_models.py        (DasInrApod, DasInrApodMixer classes)
|  |  |- losses.py
|  |  |- metrics.py
|  |  |- train.py
|  |  |- predict.py
|  |- picmus/
|  |  |- core.py              (PICMUS pipeline functions)
|  |  |- __init__.py
|  |- tuning/
|  |  |- utils.py
|  |  |- __init__.py
|- scripts/
|  |- generate_rf_dataset_simus.py
|  |- create_delayed_samples_dataset.py
|  |- inr_hanning_fit.py      (fit MLP to Hanning apodization)
|  |- verify_env.py
|  |- train/
|  |  |- train_inr_das_mixer.py          (primary multi-apodization training)
|  |  |- train_inr_das_by_indices.py     (single-apodization training)
|  |  |- replay_training_history.py      (regenerate training curves)
|  |  |- review_training_results.py      (interactive result visualization)
|  |  |- legacy/
|  |  |  |- train_inr_das_by_examples.py (deprecated: memory inefficient)
|  |- evaluation/
|  |  |- das_standard_apodizations.py
|  |  |- evaluate_baseline_apodizations.py
|  |  |- evaluate_apodization_profile_fft.py
|  |  |- numeric_phantom/
|  |  |  |- generate_evaluation_simulation.py
|  |  |  |- evaluate_apodizations.py
|  |  |  |- evaluate_mixer.py
|  |  |  |- compare_inr_models.py
|  |- picmus/
|  |  |- process_picmus.py    (PICMUS data processing & comparison)
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

