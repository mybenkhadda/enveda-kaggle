"""Accelerator / host detection. Never raises and never initialises a GPU context it does not need: it reads
nvidia-smi, and imports CuPy / Numba only to ask whether they exist.

`DEVICE` is 'cuda' when an NVIDIA GPU is visible, else 'cpu'. That is the HARDWARE; whether a GPU BACKEND is used is
decided separately by `backends.select_backend` (frozen implementation + parity), never here.
"""
import os
import platform
import shutil
import subprocess
import sys


def _nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    gpus = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memory_total_mb": float(parts[1]) if parts[1].replace(".", "", 1).isdigit() else None,
                         "driver_version": parts[2]})
    return gpus or None


def gpu_memory_used_mb():
    """Current used memory of GPU 0 (MB) via nvidia-smi, or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        return float(r.stdout.strip().splitlines()[0]) if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _module_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _cupy():
    try:
        import cupy  # noqa: F401
        return True, getattr(cupy, "__version__", None)
    except Exception as e:  # missing, or installed without a usable CUDA runtime
        return False, f"unavailable ({type(e).__name__})"


def _numba_cuda():
    try:
        from numba import cuda
        return bool(cuda.is_available())
    except Exception:
        return False


def _ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024 ** 3, 2)
    except Exception:
        return None


def detect_accelerator():
    """Dict describing the host + accelerator. `device` is 'cuda' or 'cpu'."""
    gpus = _nvidia_smi()
    cupy_ok, cupy_ver = _cupy()
    lgbm = _module_version("lightgbm")
    info = {
        "python_version": sys.version.split()[0], "os": platform.platform(), "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(), "ram_gb": _ram_gb(),
        "cuda_available": bool(gpus), "gpu_count": len(gpus or []), "gpu_name": gpus[0]["name"] if gpus else None,
        "gpu_memory_total_mb": gpus[0]["memory_total_mb"] if gpus else None, "cuda_driver_version": gpus[0]["driver_version"] if gpus else None,
        "cupy_available": cupy_ok, "cupy_version": cupy_ver,
        "numba_available": _module_version("numba") is not None, "numba_version": _module_version("numba"),
        "numba_cuda_available": _numba_cuda() if gpus else False,
        "lightgbm_version": lgbm,
        # the frozen ranker predicts from LightGBM TEXT boosters on CPU (casmi_infer.ranker); LightGBM's GPU build only
        # accelerates TRAINING, which the runtime never does -- so this is reported, not used
        "lightgbm_prediction_device": "cpu",
    }
    info["device"] = "cuda" if info["cuda_available"] else "cpu"
    return info


DEVICE = None      # filled lazily by `device()`, so importing this module stays side-effect free


def device():
    global DEVICE
    if DEVICE is None:
        DEVICE = detect_accelerator()["device"]
    return DEVICE
