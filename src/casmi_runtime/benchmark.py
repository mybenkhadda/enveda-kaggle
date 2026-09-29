"""Engineering benchmark of VALIDATED backends only (never used to choose a model, aggregator or backend policy).

A backend is benchmarked only after the bundle self-test PASSES on it; its outputs on the benchmark spectra are then
compared with the numpy reference (candidate sets / ranks exact, features + scores at the frozen 1e-9 tolerance).
"""
import time

import numpy as np
import pandas as pd

from casmi_runtime.accelerator import gpu_memory_used_mb

TOL = 1e-9


def _rss_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 3
    except Exception:
        return float("nan")


def available_backends(frozen, accel, bundle_config):
    """[(backend, reason)] in the production order; GPU only if the frozen code implements a declared one."""
    from casmi_runtime.backends import gpu_backend_status
    out = []
    ok, why = gpu_backend_status(frozen, bundle_config, accel)
    if ok:
        out.append((getattr(frozen.backend, "GPU", None) or frozen.backend.CUDA, why))
    if frozen.backend.numba_available():
        out.append((frozen.backend.NUMBA, "numba importable"))
    out.append((frozen.backend.NUMPY, "reference"))
    return out, (None if ok else why)


def time_backend(bundle, frozen, backend, rows, infer_fn=None, warmup=1):
    """Runs `infer_fn(bundle, row)` (default: the frozen self-test inference path) on every row under `backend`.
    Returns (metrics dict, concatenated outputs)."""
    infer_fn = infer_fn or frozen.selftest.infer_fixture_query
    with frozen.backend.use_backend(backend):
        for row in rows[:warmup]:                       # JIT compilation / caches excluded from the timing
            infer_fn(bundle, row)
        rss0, g0, t0 = _rss_gb(), gpu_memory_used_mb(), time.perf_counter()
        parts = [infer_fn(bundle, row).assign(spectrum_id=row["spectrum_id"]) for row in rows]
        dt = time.perf_counter() - t0
    out = pd.concat(parts, ignore_index=True)
    n_sim = int(out["n_selected_refs"].sum()) if "n_selected_refs" in out.columns else None
    return {"backend": backend, "n_spectra": len(rows), "runtime_seconds": dt, "spectra_per_sec": len(rows) / dt if dt else None,
            "candidate_pairs": int(len(out)), "candidate_pairs_per_sec": len(out) / dt if dt else None,
            "similarity_evaluations": n_sim, "similarity_evaluations_per_sec": (n_sim / dt) if (n_sim and dt) else None,
            "rss_gb_after": _rss_gb(), "rss_gb_delta": _rss_gb() - rss0, "gpu_memory_used_mb": gpu_memory_used_mb(), "gpu_memory_before_mb": g0}, out


def parity_vs_reference(out, ref, feature_cols):
    """Candidate sets + ranks exact; features + score within TOL (NaN == NaN)."""
    key = ["spectrum_id", "conn_idx"]
    m = ref.merge(out, on=key, how="outer", suffixes=("_ref", "_out"), indicator=True)
    both = (m["_merge"] == "both").to_numpy()
    res = {"candidate_set_mismatches": int((~both).sum())}
    for c in [*feature_cols, "score"]:
        a, b = m[f"{c}_ref"].to_numpy(float), m[f"{c}_out"].to_numpy(float)
        res[f"{c}_mismatches"] = int((both & ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= TOL))).sum())
    res["rank_mismatches"] = int((both & (m["rank_ref"].to_numpy() != m["rank_out"].to_numpy())).sum())
    res["parity"] = "PASS" if all(v == 0 for k, v in res.items() if k.endswith("mismatches")) else "FAIL"
    return res
