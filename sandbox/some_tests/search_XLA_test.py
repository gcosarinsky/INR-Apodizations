import os, glob, sys

prefixes = [
    os.environ.get("CONDA_PREFIX") or sys.prefix,
    sys.prefix,
    os.environ.get("CUDA_PATH"),
    os.environ.get("CUDA_HOME"),
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
    "/usr/local/cuda",
]
checked = set()
found = False

for root in filter(None, prefixes):
    if root in checked:
        continue
    checked.add(root)
    candidates = [
        os.path.join(root, "nvvm", "libdevice"),
        os.path.join(root, "Library", "nvvm", "libdevice"),
        os.path.join(root, "Library", "lib", "nvvm", "libdevice"),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            for f in os.listdir(cand):
                if f.startswith("libdevice") and f.endswith(".bc"):
                    print("Found libdevice directory:", cand)
                    found = True
                    break
        if found:
            break
    if found:
        break

if not found:
    # last-resort recursive search (may be slow)
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    for match in glob.glob(os.path.join(prefix, "**", "libdevice.*.bc"), recursive=True):
        print("Found libdevice file:", match)
        found = True

if not found:
    print("libdevice not found in CONDA_PREFIX/sys.prefix or usual CUDA locations.")