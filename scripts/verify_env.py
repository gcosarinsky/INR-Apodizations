import tensorflow as tf
import cupy as cp

print("=== TensorFlow ===")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {gpus}")

print("\n=== CuPy ===")
print(f"CUDA available: {cp.cuda.is_available()}")
print(f"Device count: {cp.cuda.runtime.getDeviceCount()}")
dev = cp.cuda.Device(0)
print(f"Device 0: {cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()}")