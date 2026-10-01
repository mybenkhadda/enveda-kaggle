"""Colab environment helpers: GPU / CUDA report, AMP dtype choice, seeds, RAM / disk report,
VRAM-aware batch-size recommendation. torch is imported lazily so CPU-only notebooks (validation,
candidate universe) work without it.

AMP policy:
    bf16 autocast   only when the GPU reports bf16 support (Ampere+: A100, L4, ...)
    fp16 autocast   other CUDA GPUs (T4, V100) -- needs a GradScaler
    off             CPU
Nothing assumes a particular GPU model.
"""
import os
import platform
import random
import shutil

import numpy as np


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def gpu_report():
    torch = _torch()
    out = {"torch_version": getattr(torch, "__version__", None), "cuda_available": False, "gpu_name": None,
           "vram_gb": None, "cuda_capability": None, "bf16_supported": False, "cuda_version": None}
    if torch is None or not torch.cuda.is_available():
        return out
    props = torch.cuda.get_device_properties(0)
    out.update(cuda_available=True, gpu_name=props.name, vram_gb=round(props.total_memory / 1024 ** 3, 2),
               cuda_capability=f"{props.major}.{props.minor}", bf16_supported=bool(torch.cuda.is_bf16_supported()),
               cuda_version=torch.version.cuda)
    return out


def system_report(paths_to_check=("/content", "/content/drive")):
    out = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        import psutil
        vm = psutil.virtual_memory()
        out.update(ram_total_gb=round(vm.total / 1024 ** 3, 2), ram_available_gb=round(vm.available / 1024 ** 3, 2))
    except ImportError:
        out.update(ram_total_gb=None, ram_available_gb=None)
    for p in paths_to_check:
        if os.path.exists(p):
            du = shutil.disk_usage(p)
            out[f"disk_free_gb[{p}]"] = round(du.free / 1024 ** 3, 2)
    return out


def choose_amp_dtype(preference="auto", report=None):
    """'bf16' | 'fp16' | None. `preference`: auto | bf16 | fp16 | off. An explicit bf16 request on a
    GPU without bf16 support raises instead of silently running slow/emulated."""
    rep = report or gpu_report()
    if preference == "off" or not rep["cuda_available"]:
        return None
    if preference == "bf16":
        if not rep["bf16_supported"]:
            raise RuntimeError(f"bf16 requested but {rep['gpu_name']} does not support it -- use amp: fp16 or auto")
        return "bf16"
    if preference == "fp16":
        return "fp16"
    return "bf16" if rep["bf16_supported"] else "fp16"


def autocast_context(amp_dtype, device_type="cuda"):
    """`torch.autocast` for the chosen dtype, or a null context when AMP is off."""
    import contextlib
    torch = _torch()
    if amp_dtype is None or torch is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16 if amp_dtype == "bf16" else torch.float16)


def needs_grad_scaler(amp_dtype):
    return amp_dtype == "fp16"


def set_seeds(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch = _torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed


# conservative per-GPU presets (examples per 16 GB of VRAM, scaled); the user can always override
_BATCH_PER_16GB = {"fingerprint": 128, "contrastive": 128, "default": 64}


def recommend_batch_size(task="default", report=None, override=None, safety=0.8, max_batch=1024):
    """A deliberately conservative batch size from available VRAM. `override` wins. Never returns more
    than `max_batch`; CPU -> a small batch for diagnostics only."""
    if override:
        return int(override)
    rep = report or gpu_report()
    base = _BATCH_PER_16GB.get(task, _BATCH_PER_16GB["default"])
    if not rep["cuda_available"]:
        return max(8, base // 8)
    scale = (rep["vram_gb"] or 16) / 16.0 * safety
    b = int(base * scale)
    b = 2 ** int(np.floor(np.log2(max(b, 8))))      # power of two, rounded DOWN
    return int(min(b, max_batch))


def ensure_packages(spec, run=True):
    """`spec`: {import_name: pip_requirement}. Installs MISSING packages with pip (Colab sessions start
    fresh; rdkit is not preinstalled there). Returns the list of requirements that were (or would be)
    installed. Meant for the Colab notebooks -- the user runs them."""
    import importlib.util
    import subprocess
    import sys
    missing = [req for mod, req in spec.items() if importlib.util.find_spec(mod) is None]
    if missing and run:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
    return missing


def oom_instructions():
    return ("CUDA OOM: (1) halve the batch size and double grad_accum_steps (same effective batch); "
            "(2) torch.cuda.empty_cache(); (3) restart the runtime and resume from last.pt; "
            "(4) disable torch_compile; (5) reduce max_peaks / sequence length.")
