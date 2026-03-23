# INR-Apodizations - Copilot Instructions

  - INR: Implicit Neural Representations
  - Apodizations: weighting functions for ultrasound beamforming

## Istrucciones para agentes
  - No intentar ejecutar comandos ni scripts sin que el usuario lo solicite explícitamente.
    
## Idioma y estilo de interacción

- El chat con el usuario debe ser en castellano.
- El codigo, comentarios y docstrings deben estar en ingles.

## Reglas de desarrollo del proyecto

- No hardcodear parametros de ejecucion en scripts. Usar archivos de configuracion JSON o YAML.
- Agregar docstrings en ingles a funciones y clases nuevas o modificadas: proposito, parametros, retorno y excepciones relevantes.
- En scripts de exploracion o pruebas iniciales (normalmente en `sandbox/`), mantener logica simple y directa.
- Para `sandbox/`, evitar `main()` y evitar `argparse` salvo que el usuario lo pida explicitamente.

## Comandos de trabajo rapido

- Verificar entorno GPU (TensorFlow + CuPy): `python scripts/verify_env.py`

## Flujo principal del proyecto

- Generar dataset RF simulado: `python scripts/generate_rf_dataset_simus.py`
- Generar delayed samples desde RF: `python scripts/create_delayed_samples_dataset.py`
- El flujo es secuencial: primero RF, luego delayed samples.
- `configs/delayed_samples_dataset.yml` referencia un dataset RF existente por nombre (`rf_dataset_name`).

## Arquitectura (resumen operativo)

- `inr_apodizations/config.py`: rutas globales, entorno y logging.
- `inr_apodizations/kernels/parameters.py`: contrato de parametros para kernels CUDA (2D/3D).
- `inr_apodizations/coordinate_manager.py`: generacion de grillas y coordenadas para training/inferencia.
- `inr_apodizations/dataset.py`: utilidades de dataset/targets.
- `inr_apodizations/utils.py`: conversiones de configuracion y persistencia YAML.
- `inr_apodizations/kernels/bf_cuda_kernels/`: codigo CUDA/C asociado a beamforming.
- `scripts/`: pipelines ejecutables de generacion de datos.

## Convenciones tecnicas importantes

- Mantener compatibilidad con Python 3.10.
- Respetar ruff (`line-length = 99`, imports ordenados).
- Cuando se toquen enums/parametros de kernel, preservar consistencia entre Python y CUDA.
- Evitar refactors amplios en scripts de pipeline si no son requeridos por la tarea.

## Pitfalls frecuentes

- Este repositorio depende de GPU para pipelines pesados (CuPy + CUDA + TensorFlow).
- `matplotlib.use('TkAgg')` en scripts puede fallar en entornos sin display.
- Los arrays de RF/delayed samples pueden ocupar mucha memoria; revisar dimensiones antes de ampliar parametros.

