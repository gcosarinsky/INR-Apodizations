# Architecture diagram

Below is a high-level architecture diagram of the INR-Apodizations project. Open this file in VS Code and use Markdown Preview (Ctrl+Shift+V) or a Mermaid preview extension to render the diagram.

```mermaid
flowchart LR
  subgraph Data["Data folders"]
    RAW["data/raw"]
    INTERIM["data/interim"]
    PROCESSED["data/processed"]
  end

  subgraph Configs["Configuration"]
    CFG["configs/*.yml"]
  end

  subgraph Scripts["Pipelines / Scripts"]
    GEN_RF["scripts/generate_rf_dataset_simus.py"]
    CREATE_DS["scripts/create_delayed_samples_dataset.py"]
    DAS["sandbox/inr_das_experiment/evaluation/das_standard_apodizations.py"]
  end

  subgraph Core["inr_apodizations (library)"]
    CM["coordinate_manager.py"]
    FEAT["features.py"]
    DATA_MOD["dataset.py"]
    APOD["apodizations.py"]
    KERNELS["kernels/bf_cuda_kernels"]
    MODELING["modeling (train.py, predict.py)"]
    UTILS["utils.py / config.py"]
  end

  subgraph ModelsAndReports["Outputs"]
    MODELS["models (weights & checkpoints)"]
    REPORTS["reports/ figures & notebooks"]
  end

  %% Data flow
  GEN_RF --> RAW
  GEN_RF --> PROCESSED
  CREATE_DS --> PROCESSED
  CREATE_DS --> DATA_MOD
  PROCESSED --> MODELING
  CFG --> GEN_RF
  CFG --> CREATE_DS
  CFG --> DAS

  %% Core interactions
  DATA_MOD --> CM
  MODELING --> FEAT
  MODELING --> DATA_MOD
  MODELING --> KERNELS
  KERNELS -->|CUDA code| GPU["GPU runtime (CuPy / CUDA / TensorFlow)"]
  MODELING --> MODELS
  DAS --> KERNELS

  %% Other
  sandbox["sandbox/ experiments"] --> MODELING
  notebooks["notebooks/"] --> REPORTS
  MODELS --> PRED["modeling/predict.py"]
  PRED --> REPORTS

  classDef folder fill:#f9f,stroke:#333,stroke-width:1px;
  class Data,Configs,Scripts,Core,ModelsAndReports folder;
```

Notes
- To render Mermaid inside VS Code, you can install the "Markdown Preview Mermaid Support" extension or open the built-in Markdown preview.
- If you prefer, I can also export this diagram as a PNG/SVG and add it to `docs/figures/`.
