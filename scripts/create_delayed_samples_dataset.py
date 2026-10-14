#%%
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from pathlib import Path
from datetime import datetime
from scipy import signal
import inr_apodizations.hilbert_coef as hilb
from inr_apodizations.kernels import KernelParameters2D
from inr_apodizations.dataset import generate_das_modulated_target, generate_unit_gaussian_mask
from inr_apodizations.config import CONFIGS_DIR, DATA_DIR, CUDA_DIR
import yaml
from inr_apodizations.utils import save_config_yaml
plt.ion()

# Load beamforming/delayed samples config (independent of the RF dataset)
with open(CONFIGS_DIR / 'delayed_samples_dataset.yml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# Load RF dataset config 
dataset_path = DATA_DIR / "rf_dataset_simus" / cfg['rf_dataset_name']
rf_config_path = dataset_path / 'config_rf_info.yml'
with open(rf_config_path, 'r', encoding='utf-8') as f:
    rf_cfg = yaml.safe_load(f)
cfg.update(rf_cfg)  # Add RF config to main config

# Load RF data and RF config (probe/acquisition params saved with the dataset)
RF = np.load(dataset_path / 'rf.npy')  # shape: (n_examples, n_angles, n_elements, n_samples)
scatterers = np.load(dataset_path / 'scatterers.npy', allow_pickle=True)  # list of (n_scatterers, 3)
# Try to load noise.npy if present
noise_path = dataset_path / 'noise.npy'
NOISE = None
if cfg.get('process_noise', False) and noise_path.exists():
    NOISE = np.load(noise_path)
    print('Loaded noise array with shape:', NOISE.shape)

angles = np.arange(*cfg['angles'])
angles = np.deg2rad(angles)
cfg['n_angles'] = len(angles)  # Add n_angles to config for kernel use

kp = KernelParameters2D(cfg)

n_examples, n_angles, n_elements, n_samples = RF.shape
nz, nx = kp.nz, kp.nx

# Pre-allocate arrays for delayed samples, targets and gaussian masks
delayed_samples_all = np.zeros((n_examples, n_elements, nz, nx), dtype=np.complex64)
print(f'Pre-allocated delayed_samples_all with size in MB: {delayed_samples_all.nbytes / (1024*1024)} MB')
targets_all = np.zeros((n_examples, nz, nx), dtype=np.float32)
gaussian_masks_all = np.zeros((n_examples, nz, nx), dtype=np.float32)
# If noise processing is requested and noise exists, pre-allocate noise delayed samples
delayed_samples_noise_all = None
delayed_samples_combined_all = None
if NOISE is not None:
    delayed_samples_noise_all = np.zeros((n_examples, n_elements, nz, nx), dtype=np.complex64)
    if cfg.get('save_combined', False):
        delayed_samples_combined_all = np.zeros((n_examples, n_elements, nz, nx), dtype=np.complex64)

# Prepare grids for target generation
x = np.linspace(kp.roi_effective[0], kp.roi_effective[1], kp.nx)
z = np.linspace(kp.roi_effective[2], kp.roi_effective[3], kp.nz)
x_grid, z_grid = np.meshgrid(x, z)

# filter coefficients (using bf params for filter design, fs from RF config)
bandpass_coef = signal.firwin(cfg['taps'] + 1, [2 * cfg['f1'] / cfg['fs'], 2 * cfg['f2'] / cfg['fs']],
                              pass_zero=False)
bandpass_coef_gpu = cp.asarray(bandpass_coef, dtype=cp.float32)
# Hilbert coefficients
hilb_coef_gpu = cp.asarray(hilb.coef, dtype=cp.float32)

#%% load CUDA code
codepath = CUDA_DIR
codefiles = [
    'constants.h',
    'enum_parameters.c',
    'fir_filter.cu',
    'pwi_1pix_per_thread.cu'
]
code = ''
for codefile in codefiles:
    with open(codepath / codefile, encoding='utf-8') as f:
        code += f.read() + '\n'

kp.check_enum_consistency(code)  # Check consistency of enum names with kernel parameters
module = cp.RawModule(code=code, options=('--use_fast_math',))
filt_kernel = module.get_function('fir_filter')
# pwi_kernel = module.get_function('pwi_1pix_per_thread')
pwi_gather_kernel = module.get_function('pwi_gather_delayed_samples')

int_params = cp.asarray(kp.get_int_array(), dtype=cp.int32)
float_params = cp.asarray(kp.get_float_array(), dtype=cp.float32)
angles_gpu = cp.asarray(angles, dtype=cp.float32)

# CUDA filtering parameters
nblock = 128
n_ascans = RF.shape[1] * RF.shape[2]
grid_size = ((n_ascans + nblock - 1) // nblock,)
block_size = (nblock,)

#%% --- Main Loop: Compute delayed samples and targets ---
for idx in range(n_examples):
    print(f'Processing example {idx+1} / {n_examples}')
    RF_ex = RF[idx]  # shape: (n_angles, n_elements, n_samples)
    scat = scatterers[idx]
    # Transfer to GPU for processing
    RF_gpu = cp.asarray(RF_ex)
    RF_filt_gpu = cp.zeros_like(RF_gpu)
    RF_imag_gpu = cp.zeros_like(RF_gpu)
    filt_kernel(grid_size, block_size, (int_params, RF_gpu, bandpass_coef_gpu, RF_filt_gpu))
    filt_kernel(grid_size, block_size, (int_params, RF_filt_gpu, hilb_coef_gpu, RF_imag_gpu))
    delayed_samples_gpu = cp.zeros((kp.n_angles, kp.n_elements, kp.nz, kp.nx), dtype=cp.complex64)
    pwi_gather_kernel(
        kp.gridsize_img, kp.blocksize_img,
        (int_params, float_params, angles_gpu, RF_filt_gpu, RF_imag_gpu, delayed_samples_gpu)
    )
    delayed_samples = cp.asnumpy(delayed_samples_gpu.sum(axis=0))  # sum over angles
    delayed_samples_all[idx, ...] = delayed_samples  # (n_elements, nz, nx)

    # If noise is provided, process noise through the same filter+kernels
    if NOISE is not None:
        noise_ex = NOISE[idx]
        noise_gpu = cp.asarray(noise_ex)
        noise_filt_gpu = cp.zeros_like(noise_gpu)
        noise_imag_gpu = cp.zeros_like(noise_gpu)
        filt_kernel(grid_size, block_size, (int_params, noise_gpu, bandpass_coef_gpu, noise_filt_gpu))
        filt_kernel(grid_size, block_size, (int_params, noise_filt_gpu, hilb_coef_gpu, noise_imag_gpu))
        delayed_noise_gpu = cp.zeros((kp.n_angles, kp.n_elements, kp.nz, kp.nx), dtype=cp.complex64)
        pwi_gather_kernel(
            kp.gridsize_img, kp.blocksize_img,
            (int_params, float_params, angles_gpu, noise_filt_gpu, noise_imag_gpu, delayed_noise_gpu)
        )
        delayed_noise = cp.asnumpy(delayed_noise_gpu.sum(axis=0))
        delayed_samples_noise_all[idx, ...] = delayed_noise
        # Optionally create combined delayed samples
        if delayed_samples_combined_all is not None:
            if cfg.get('noise_operation', 'sum') == 'sum':
                delayed_samples_combined_all[idx, ...] = delayed_samples + delayed_noise
            else:
                delayed_samples_combined_all[idx, ...] = delayed_samples + delayed_noise

    # Uniform DAS image (sum over receive elements) modulated by unit-amplitude Gaussian mask
    das_uniform = delayed_samples.sum(axis=0)
    gaussian_mask = generate_unit_gaussian_mask(
        1000 * scat,
        x_grid,
        z_grid,
        sigma_x=cfg['target']['sigma_x'],
        sigma_z=cfg['target']['sigma_z'],
    )
    targets_all[idx] = np.abs(das_uniform).astype(np.float32) * gaussian_mask
    gaussian_masks_all[idx] = gaussian_mask

cp.cuda.Device().synchronize()  # Ensure all operations are complete

#%% Save results
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = DATA_DIR / "delayed_samples_dataset" / ts
output_folder.mkdir(parents=True, exist_ok=True)
# Save delayed samples (signal-only)
np.save(output_folder / 'delayed_samples_signal.npy', delayed_samples_all)
# If noise was processed, save noise-only delayed samples and optional combined
if delayed_samples_noise_all is not None:
    np.save(output_folder / 'delayed_samples_noise.npy', delayed_samples_noise_all)
    if delayed_samples_combined_all is not None:
        np.save(output_folder / 'delayed_samples_combined.npy', delayed_samples_combined_all)
np.save(output_folder / 'targets_dataset.npy', targets_all)
np.save(output_folder / 'gaussian_masks_dataset.npy', gaussian_masks_all)
np.save(output_folder / 'scatterers.npy', scatterers)
np.save(output_folder / 'cfg_delayed_samples.npy', cfg)

# Guardar la configuración de beamforming y metadatos en YAML
info_yaml_path = output_folder / 'delayed_samples_info.yaml'
info = {
    'generated': ts,
    'delayed_samples_shape': list(delayed_samples_all.shape),
    'delayed_samples_noise_saved': delayed_samples_noise_all is not None,
    'delayed_samples_combined_saved': delayed_samples_combined_all is not None,
    'gaussian_masks_shape': list(gaussian_masks_all.shape),
    'scatterers_saved': True,
    'target_sigma': dict(cfg['target']),
    'roi_effective': list(kp.roi_effective),
    'rf_dataset': dataset_path.name,
    'config': cfg,    
}

save_config_yaml(info_yaml_path, info, {})
print(f'Delayed samples dataset configuration saved to: {info_yaml_path}')

#%% plot example, do sum over elements
log_offset = 1e-6  # Pequeño valor para evitar log(0)
example_idx = 0
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Escala logarítmica para delayed samples
delayed_samples_log = 20 * np.log10(np.abs(delayed_samples_all[example_idx, ...].sum(axis=0)) /
                                    np.max(np.abs(delayed_samples_all[example_idx, ...].sum(axis=0))) + log_offset)
ax[0].imshow(delayed_samples_log, aspect='auto', cmap='gray', extent=kp.get_imshow_extent(), vmin=-60)
ax[0].set_title('Delayed Samples Log Scale (Example 0)')
ax[0].set_xlabel('Lateral [m]')
ax[0].set_ylabel('Axial [m]')

# Escala logarítmica para el target
targets_log = 20 * np.log10(targets_all[example_idx] / np.max(targets_all[example_idx]) + log_offset)
ax[1].imshow(targets_log, aspect='auto', cmap='gray', extent=kp.get_imshow_extent(), vmin=-60)
ax[1].set_title('Target Log Scale (Example 0)')

plt.tight_layout()
plt.show()

# If noise was processed, plot combined (signal + noise) delayed samples for the same example
if delayed_samples_noise_all is not None:
    combined_sum = delayed_samples_all[example_idx, ...].sum(axis=0) + delayed_samples_noise_all[example_idx, ...].sum(axis=0)
    combined_log = 20 * np.log10(np.abs(combined_sum) / np.max(np.abs(combined_sum)) + log_offset)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
    ax2.imshow(combined_log, aspect='auto', cmap='gray', extent=kp.get_imshow_extent(), vmin=-60)
    ax2.set_title('Delayed Samples + Noise Log Scale (Example 0)')
    ax2.set_xlabel('Lateral [m]')
    ax2.set_ylabel('Axial [m]')
    plt.tight_layout()
    plt.show()
