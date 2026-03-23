# INR-Apodizations

> **Work in progress** — this README is provisional.

Apodization functions for ultrasound beamforming learned with **Implicit Neural Representations (INR)**.

In phased-array ultrasound imaging, apodization weights are applied across transducer elements to reduce side-lobes and improve lateral resolution. This project explores replacing hand-crafted apodization windows (Hanning, Tukey, …) with a small neural network that learns to predict per-element weights as a continuous function of the imaging geometry.

---

## Background

**Delay-and-Sum (DAS)** is the standard beamforming algorithm: for every image pixel, each element's received echo is time-shifted to compensate for the travel distance, and then summed. Applying *apodization* weights before the sum controls the trade-off between main-lobe width (resolution) and side-lobe level (contrast).

An **Implicit Neural Representation** is a network $f_\theta(\mathbf{x}) \to w$ that maps spatial coordinates $\mathbf{x}$ (lateral position, depth, element position) directly to a scalar weight. Training on simulated RF data allows the network to learn geometry-aware apodizations that outperform fixed analytical windows.

---

## Pipeline overview

```
configs/rf_dataset_simus.yml
         │
         ▼
scripts/generate_rf_dataset_simus.py   ──►  data/rf_dataset_simus/<timestamp>/
         │                                      rf.npy  scatterers.npy  config_rf_info.yml
         │
configs/delayed_samples_dataset.yml
         │
         ▼
scripts/create_delayed_samples_dataset.py  ──►  data/delayed_samples_dataset/<timestamp>/
                                                    delayed_samples_dataset.npy
                                                    targets_dataset.npy
                                                    cfg_delayed_samples.npy
         │
         ▼
   (INR training — in progress)
         │
         ▼
scripts/das_standard_apodizations.py    ──►  Baseline DAS + classical apodizations
```

**Step 1 — RF simulation** (`generate_rf_dataset_simus.py`):  
Simulates plane-wave acquisitions (multi-angle) using [PyMUST](https://www.biomecardio.com/MUST/). Generates raw RF signals for random point-scatterer phantoms.

**Step 2 — Delayed samples** (`create_delayed_samples_dataset.py`):  
Applies per-angle, per-element time delays using a custom CUDA kernel, producing complex-envelope *delayed samples* ready for summation. Also generates Gaussian-blob regression targets centred on each scatterer.

**Step 3 — INR training** *(in progress)*:  
A TensorFlow MLP is trained on the delayed samples to predict apodization weights as a function of imaging coordinates (`CoordinateManager` provides the input features).

**Step 4 — Evaluation** (`das_standard_apodizations.py`):  
Reconstructs images using uniform DAS and several classical apodization windows (Hanning, Tukey, …) as baselines for comparison.

---

## Repository structure

```
INR-Apodizations/
├── configs/                        # YAML configuration files (no hard-coded parameters)
│   ├── rf_dataset_simus.yml
│   ├── delayed_samples_dataset.yml
│   └── das_standard_apodizations.yml
├── data/
│   ├── rf_dataset_simus/           # Simulated RF data (per-run timestamped folders)
│   └── delayed_samples_dataset/    # Delayed + target arrays
├── docs/
│   └── architecture.md             # Mermaid architecture diagram
├── inr_apodizations/               # Main Python package
│   ├── config.py                   # Global paths and logging
│   ├── coordinate_manager.py       # Input-feature generation for the INR
│   ├── apodizations.py             # Classical and dynamic apodization functions
│   ├── dataset.py                  # Target generation utilities
│   ├── features.py                 # Feature engineering helpers
│   ├── utils.py                    # Config conversion and persistence utilities
│   ├── hilbert_coef.py             # Hilbert / analytic-signal FIR coefficients
│   ├── plots.py                    # Visualization helpers
│   ├── kernels/
│   │   ├── parameters.py           # KernelParameters2D/3D contract
│   │   └── bf_cuda_kernels/        # CUDA/C beamforming kernels
│   └── modeling/
│       ├── train.py                # INR training (stub)
│       └── predict.py              # INR inference (stub)
├── scripts/                        # Executable pipeline scripts
├── sandbox/                        # Quick exploration / smoke tests
├── models/                         # Saved model weights
├── reports/figures/                # Generated figures
└── pyproject.toml
```

---

## Requirements

| Dependency | Version / note |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.10 (GPU) |
| CuPy | `cupy-cuda11x` |
| CUDA toolkit | 11.8 |
| cuDNN | 8.9 |
| NumPy | 1.26 |
| PyMUST | via conda-forge |
| SciPy, Matplotlib, loguru, typer, tqdm, PyYAML | latest compatible |

A GPU is required for the delayed-samples pipeline and for INR training.

**NOTE: It is necessary to review if it is really required to choose those versions of TensorFlow, CuPy, etc. to ensure compatibility.**
---

## Environment setup

```bash
conda env create -f environment.yml
conda activate inr-apodizations
```

To verify the GPU environment (TensorFlow + CuPy):

```bash
python scripts/verify_env.py
```

---

## Running the pipeline

### 1. Generate RF dataset

Edit `configs/rf_dataset_simus.yml` as needed, then:

```bash
python scripts/generate_rf_dataset_simus.py
```

Output is saved to `data/rf_dataset_simus/<timestamp>/`.

### 2. Compute delayed samples

Set `rf_dataset_name` in `configs/delayed_samples_dataset.yml` to the timestamp produced in step 1, then:

```bash
python scripts/create_delayed_samples_dataset.py
```

Output is saved to `data/delayed_samples_dataset/<timestamp>/`.

### 3. Baseline DAS visualisation

```bash
python scripts/das_standard_apodizations.py
```

Reads the latest delayed-samples dataset and produces comparison figures for uniform DAS and classical apodizations.

---

## Configuration

All runtime parameters live in YAML files under `configs/`. No parameters are hard-coded in scripts.

Key fields in `rf_dataset_simus.yml`: probe geometry, acquisition angles, speed of sound, dataset size.  
Key fields in `delayed_samples_dataset.yml`: reference RF dataset name, bandpass filter, f-number, image grid, CUDA block sizes, target PSF sigmas.

---

## Code style

- **Linter**: [ruff](https://docs.astral.sh/ruff/) (`line-length = 99`, import sorting enabled).
- **Docstrings**: English, NumPy/Google style for public functions.
- Run checks with `ruff check .` from the project root.

