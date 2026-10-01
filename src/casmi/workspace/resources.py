"""Resource monitoring for long Colab cells: RAM / GPU memory / elapsed time / throughput / ETA.

    MON = ResourceMonitor()
    with MON.stage('recall sweep', n_items=len(Q), unit='queries'):
        ...
    MON.progress('fold 3', done=1200, total=5000, unit='queries')   # rate + ETA line
    MON.table()                                                     # one row per stage (persist with the report)

torch / psutil are optional (CPU-only notebooks and minimal environments work without them).
"""
import gc
import os
import time
from contextlib import contextmanager


def _psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def memory_snapshot():
    """{'ram_used_gb', 'ram_available_gb', 'ram_total_gb', 'process_rss_gb', 'gpu_alloc_gb', 'gpu_reserved_gb', 'gpu_peak_gb'}
    (None when unavailable)."""
    out = {"ram_used_gb": None, "ram_available_gb": None, "ram_total_gb": None, "process_rss_gb": None,
           "gpu_alloc_gb": None, "gpu_reserved_gb": None, "gpu_peak_gb": None}
    ps = _psutil()
    if ps is not None:
        vm = ps.virtual_memory()
        out.update(ram_used_gb=round((vm.total - vm.available) / 1024 ** 3, 2), ram_available_gb=round(vm.available / 1024 ** 3, 2),
                   ram_total_gb=round(vm.total / 1024 ** 3, 2), process_rss_gb=round(ps.Process(os.getpid()).memory_info().rss / 1024 ** 3, 2))
    import sys
    torch = sys.modules.get("torch")            # never import torch just to measure it
    if torch is not None and torch.cuda.is_available():
        out.update(gpu_alloc_gb=round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
                   gpu_reserved_gb=round(torch.cuda.memory_reserved() / 1024 ** 3, 2),
                   gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2))
    return out


def free_memory(*names, namespace=None):
    """`del` the given global names (if present) + gc.collect() + CUDA cache release. Returns RAM after."""
    if namespace is not None:
        for n in names:
            namespace.pop(n, None)
    gc.collect()
    import sys
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return memory_snapshot()


def _fmt_s(s):
    s = int(max(s, 0))
    return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s" if s >= 3600 else f"{s // 60:d}m{s % 60:02d}s"


class ResourceMonitor:
    def __init__(self, log=print, ram_warn_fraction=0.85):
        self.log = log
        self.t0 = time.time()
        self.rows = []
        self.ram_warn_fraction = ram_warn_fraction

    def _warn_ram(self, snap):
        if snap["ram_total_gb"] and snap["ram_used_gb"] and snap["ram_used_gb"] / snap["ram_total_gb"] > self.ram_warn_fraction:
            self.log(f"[resources] WARNING RAM {snap['ram_used_gb']}/{snap['ram_total_gb']} GB used -- free large objects "
                     f"(casmi.workspace.resources.free_memory) or lower the chunk size before the kernel is killed")

    @contextmanager
    def stage(self, name, n_items=None, unit="items"):
        before = memory_snapshot()
        t = time.time()
        rec = {"stage": name, "n_items": n_items, "unit": unit}
        try:
            yield rec
        finally:
            dt = time.time() - t
            after = memory_snapshot()
            n = rec.get("n_items", n_items)
            rec.update(seconds=round(dt, 2), rate_per_s=round(n / dt, 2) if n and dt > 0 else None,
                       ram_used_gb=after["ram_used_gb"], process_rss_gb=after["process_rss_gb"],
                       rss_delta_gb=round(after["process_rss_gb"] - before["process_rss_gb"], 2) if after["process_rss_gb"] is not None else None,
                       gpu_peak_gb=after["gpu_peak_gb"])
            self.rows.append(rec)
            rate = f" | {rec['rate_per_s']:,} {unit}/s" if rec["rate_per_s"] else ""
            self.log(f"[resources] {name}: {_fmt_s(dt)}{rate} | RSS {after['process_rss_gb']} GB | RAM used {after['ram_used_gb']} GB"
                     + (f" | GPU peak {after['gpu_peak_gb']} GB" if after["gpu_peak_gb"] is not None else ""))
            self._warn_ram(after)

    def progress(self, label, done, total, unit="items", started_at=None):
        t = time.time() - (started_at or self.t0)
        rate = done / t if t > 0 and done else 0.0
        eta = (total - done) / rate if rate > 0 else float("nan")
        snap = memory_snapshot()
        self.log(f"[progress] {label}: {done:,}/{total:,} {unit} | {rate:,.1f} {unit}/s | ETA "
                 f"{_fmt_s(eta) if eta == eta else '?'} | RSS {snap['process_rss_gb']} GB")
        self._warn_ram(snap)
        return {"done": done, "total": total, "rate_per_s": rate, "eta_s": eta}

    def table(self):
        import pandas as pd
        return pd.DataFrame(self.rows)

    def elapsed(self):
        return time.time() - self.t0
