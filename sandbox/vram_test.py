import subprocess

def gpu_mem():
    try:
        out = subprocess.check_output("nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits".split()).decode().strip()
        return [dict(zip(['total', 'free'], map(int, line.split(',')))) for line in out.split('\n')]
    except:
        return "nvidia-smi no disponible"

print(gpu_mem())  # [{'total': 8192, 'free': 7000}, ...]
