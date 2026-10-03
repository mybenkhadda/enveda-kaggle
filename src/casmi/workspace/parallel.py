"""CPU parallelism helpers for RDKit-bound work (Stage A of the candidate universe).

RDKit canonicalization is single-threaded CPU work; the speedup comes from N independent worker PROCESSES, each
processing batches of molecules. These helpers:

  * resolve the worker count (`resolve_n_jobs`): an explicit integer always wins; "auto" / -1 / None use all logical
    CPUs, capped by available RAM (a conservative per-worker budget) -- never more workers than logical CPUs by default;
  * prevent native thread oversubscription (`limit_native_threads`): OMP / MKL / OpenBLAS / numexpr pinned to 1 thread
    in the parent BEFORE workers start (inherited) and again in each worker initializer -- only when multiprocessing is
    actually used;
  * create ONE persistent pool (`make_pool`) that is reused for every chunk of a source (no pool per batch).
"""
import multiprocessing as mp
import os
import sys

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def limit_native_threads(n=1):
    """Pin native math libraries to `n` threads (environment, inherited by child processes). Returns previous values."""
    prev = {k: os.environ.get(k) for k in THREAD_ENV_VARS}
    for k in THREAD_ENV_VARS:
        os.environ[k] = str(n)
    return prev


def restore_env(prev):
    for k, v in (prev or {}).items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def available_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except ImportError:
        try:
            pages, page = os.sysconf("SC_AVPHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
            return pages * page / 1024 ** 3
        except (ValueError, OSError, AttributeError):
            return None


def resolve_n_jobs(requested="auto", ram_per_worker_gb=1.0, cpu_count=None, ram_gb=None):
    """Worker count. Positive integer -> used as is (explicit override, e.g. UCFG.n_jobs = 6). "auto" / -1 / None / 0 ->
    all logical CPUs (2 on standard Colab -> 2 workers), capped by available RAM / `ram_per_worker_gb`; always >= 1."""
    cpus = cpu_count or os.cpu_count() or 1
    if isinstance(requested, str) and requested.strip().lower() not in ("auto", ""):
        requested = int(requested)
    if isinstance(requested, int) and requested > 0:
        return requested
    n = cpus
    ram = available_ram_gb() if ram_gb is None else ram_gb
    if ram is not None and ram_per_worker_gb:
        n = min(n, max(int(ram // ram_per_worker_gb), 1))
    return max(int(n), 1)


def default_start_method():
    """'spawn' everywhere by default: it is the portable choice (Windows / macOS default) and avoids forking a parent that
    already runs pyarrow / tqdm threads. Workers import casmi from sys.path, which spawn passes to the children."""
    return "spawn"


def _init_worker(n_threads=1):
    limit_native_threads(n_threads)
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        pass


def make_pool(n_jobs, start_method=None, initializer=_init_worker):
    """A persistent multiprocessing pool (None when n_jobs <= 1: work runs inline, no thread limits are set).
    Use as a context manager or call `.close(); .join()` when the source is done."""
    if n_jobs is None or n_jobs <= 1:
        return None
    limit_native_threads(1)                 # inherited by the children at start
    ctx = mp.get_context(start_method or default_start_method())
    return ctx.Pool(processes=int(n_jobs), initializer=initializer)


def runtime_resources(paths=()):
    """Small resource snapshot: cpu count, process RSS, available RAM, free disk for `paths` (no heavy dependencies)."""
    out = {"cpu_count": os.cpu_count(), "python": sys.version.split()[0], "process_rss_gb": None,
           "ram_available_gb": round(available_ram_gb(), 2) if available_ram_gb() is not None else None}
    try:
        import psutil
        out["process_rss_gb"] = round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3, 2)
    except ImportError:
        try:
            import resource
            out["process_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 2)   # Linux: KiB -> peak
        except ImportError:
            pass
    import shutil
    for p in paths:
        try:
            out[f"free_gb[{p}]"] = round(shutil.disk_usage(p).free / 1024 ** 3, 1)
        except OSError:
            out[f"free_gb[{p}]"] = None
    return out
