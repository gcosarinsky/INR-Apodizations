#%%
import matplotlib.pyplot as plt
import numpy as np
import pymust
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
from inr_apodizations.config import CONFIGS_DIR, DATA_DIR, RAW_DATA_DIR
from inr_apodizations.utils import cfg_to_must_param, save_config_yaml
plt.ion()  # Interactive mode on for plotting

# Set random seed for reproducibility
seed = 42
rng = np.random.default_rng(seed)
PLOT_RESULTS = True
config_file = CONFIGS_DIR / 'rf_dataset_simus.yml'

# Load configuration from YAML file
with open(config_file, 'r') as f:
    cfg = yaml.safe_load(f)

#%%
param_must = cfg_to_must_param(cfg)
angles = np.arange(*cfg['angles'])
angles = np.deg2rad(angles)
n_angles = len(angles)
n_elements = cfg['n_elements']

# Dataset parameters from cfg (with defaults)
n_examples = cfg['dataset_generation'].get('n_examples', 10)
n_scat = cfg['dataset_generation'].get('n_scat', 100)
# roi_simulation in mm from cfg, convert to meters. Default: [-20, 20, 1, 60] mm
roi_sim_mm = cfg['dataset_generation'].get('roi_simulation', [-20, 20, 1, 60])
roi = [x / 1000 for x in roi_sim_mm]  # [x_min, x_max, z_min, z_max] in meters

#%% Create output folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = DATA_DIR / 'rf_dataset_simus' / timestamp
output_folder.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {output_folder}")

# Estimar número de samples, con diagonal de media roi
d1 = 0.5*(roi[1] - roi[0])
d2 = (roi[3] - roi[2])
diagonal = np.hypot(d1, d2) * 1000  # en mm, factor de 0.8 para que no quede muy pesado

# Tiempo para recorrer 2*diagonal (ida y vuelta)
t_total = 2 * diagonal / cfg['c1']
n_samples = int(np.ceil(t_total * cfg['fs'] ))
print(f"Estimated n_samples: {n_samples}")

# Prealocar RF array
RF_array = np.zeros((n_examples, n_angles, n_elements, n_samples), dtype=np.int16)
print(f"Preallocated RF array shape: {RF_array.shape}")
print(f"Size of RF array in MB: {RF_array.nbytes / 1024**2:.2f} MB")
scatterers_list = []
txDelays = [pymust.txdelay(param_must, angle).reshape(1, -1) for angle in angles]

#%%
for i in tqdm(range(n_examples), desc="Generating examples"):
    # Randomly generate scatterer positions within ROI
    xs = rng.uniform(roi[0], roi[1], size=n_scat)
    zs = rng.uniform(roi[2], roi[3], size=n_scat)
    reflectivities = rng.uniform(0.1, 1, size=n_scat)
    scat = np.stack([xs, zs, reflectivities], axis=1)  # shape: (n_scat, 3)

    for j, txDelay in enumerate(txDelays):
        rf, _ = pymust.simus(scat[:, 0], scat[:, 1], scat[:, 2], txDelay, param_must)
        # Scale and convert to int16
        rf = (rf / np.max(np.abs(rf)) * 32767).astype(np.int16)
        # rf shape: (n_samples_actual, n_elements)
        # Ajustar tamaño de samples a n_samples
        n_samples_actual = rf.shape[0]
        if n_samples_actual < n_samples:
            # Pad with zeros at end
            rf_padded = np.pad(rf, ((0, n_samples - n_samples_actual), (0, 0)), mode='constant')
            rf_fixed = rf_padded
        else:
            # Recortar si es más grande
            rf_fixed = rf[:n_samples, :]
        # Guardar en RF_array
        RF_array[i, j, :, :] = rf_fixed.T  # Transponer para (n_elements, n_samples)
    scatterers_list.append(scat)

# Save RF dataset, scatterers and RF config (no beamforming params)
np.save(output_folder / 'rf.npy', RF_array)
np.save(output_folder / 'scatterers.npy', np.array(scatterers_list, dtype=object))
np.save(output_folder / 'cfg_rf.npy', cfg)

config_yaml_path = output_folder / 'config_rf_info.yml'
extra_params = {
    'dataset_generated': timestamp,
    'seed': seed,
    'n_samples': n_samples,
    'rf_array_shape': RF_array.shape,
}
save_config_yaml(config_yaml_path, cfg, extra_params)
print(f"RF configuration saved to: {config_yaml_path}")

print("Dataset generation complete.")
print(f"RF array shape: {RF_array.shape}")

# %% plot example, first angle
if PLOT_RESULTS:
    plt.figure()
    plt.imshow(np.abs(RF_array[0, 0, :, :].T), aspect='auto', cmap='gray')
    plt.title('Example RF Data (First Angle)')
    plt.xlabel('Element Index')
    plt.ylabel('Sample Index')
    plt.colorbar(label='Amplitude')
    plt.show()


