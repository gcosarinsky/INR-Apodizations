import os
import tensorflow as tf

# 1. Obtenemos la ruta y la limpiamos de inmediato usando barras normales
conda_path = os.environ.get('CONDA_PREFIX').replace('\\', '/')

if conda_path:
    # 2. Apuntamos a la carpeta Library de Conda
    # NOTA: Asegúrate de que NO haya comillas dobles internas si usas f-strings de esta forma
    # O mejor aún, usa la ruta limpia directamente:
    target_path = f"{conda_path}"
    
    os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={target_path}'
    
    # 3. También actualizamos el PATH para las DLLs
    os.environ['PATH'] = f"{target_path}/bin;" + os.environ['PATH']

print(f"Ruta de búsqueda XLA configurada en: {target_path}")

print("Verificando XLA...")

@tf.function(jit_compile=True)
def xla_test_func(a, b):
    return tf.matmul(a, b)

# Crear datos simples
size = 1000
with tf.device('/GPU:0'):
    x = tf.random.normal([size, size])
    y = tf.random.normal([size, size])

    try:
        # La primera ejecución dispara la compilación XLA
        print("Intentando compilar y ejecutar con XLA...")
        result = xla_test_func(x, y)
        print("✅ ¡ÉXITO! XLA está funcionando correctamente en tu GPU.")
    except Exception as e:
        print("\n❌ FALLO detectado en XLA:")
        print("-" * 30)
        print(e)
        print("-" * 30)
        print("\nSi el error menciona 'libdevice', confirma que XLA_FLAGS apunta a la carpeta")
        print("que contiene a la carpeta 'nvvm'.")