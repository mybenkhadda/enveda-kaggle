"""Validated backend selection (explicit, never silent):

    GPU     only if (a) a GPU is visible, (b) the FROZEN bundle code implements a GPU backend, (c) the bundle config
            declares it parity-validated (`gpu_backend.parity_validated` inside CONFIG_HASH), and (d) the bundled
            self-test passes on it at the frozen tolerances;
    else numba CPU if importable and the self-test passes;
    else numpy CPU if the self-test passes;
    else STOP (SelfTestFailed) -- no submission.

Tolerances are the frozen fixture's (features / fold scores / probabilities 1e-9; candidate sets, ranks, A7 choice and
Top-25 order exact); nothing here can loosen them. Every skipped or failed tier is recorded in `fallback_reason`.
"""

GPU_BACKEND = "cuda"


class BackendSelectionError(RuntimeError):
    pass


def gpu_backend_status(frozen, bundle_config, accel):
    """(eligible, reason). Eligible only when all of hardware / frozen implementation / declared parity hold."""
    if not accel.get("cuda_available"):
        return False, "no GPU visible on this runtime"
    decl = bundle_config.get("gpu_backend") or {}
    impl = getattr(frozen.backend, "GPU", None) or getattr(frozen.backend, "CUDA", None)
    if impl is None:
        return False, (f"the frozen bundle code ({bundle_config.get('bundle_version')}) implements no GPU backend; "
                       "GPU spectral evidence needs a parity-validated implementation shipped in a re-exported bundle")
    if decl.get("name") != impl or decl.get("parity_validated") is not True:
        return False, f"bundle config does not declare the GPU backend {impl!r} parity-validated (gpu_backend={decl!r})"
    avail = getattr(frozen.backend, "gpu_available", None)
    if callable(avail) and not avail():
        return False, "the frozen GPU backend reports its CUDA runtime unavailable"
    return True, "declared parity-validated GPU backend"


def select_backend(bundle, frozen, accel, device_preference="gpu", require_gpu=False, runner=None, log=print):
    """Runs the bundle self-test on each candidate backend in order; the first PASS is activated process-wide
    (`frozen.backend.set_backend`). Returns the record written into run_report.json."""
    runner = runner or (lambda b, be: frozen.selftest.run_selftest(b, be))
    be = frozen.backend
    reasons, attempts, chosen, final = [], [], None, None
    gpu_eligible, gpu_reason = gpu_backend_status(frozen, bundle.config, accel)
    candidates = []
    if device_preference == "gpu":
        if gpu_eligible:
            candidates.append(getattr(be, "GPU", None) or getattr(be, "CUDA"))
        else:
            reasons.append(f"GPU not used: {gpu_reason}")
    else:
        reasons.append("GPU not requested (device_preference='cpu')")
    if be.numba_available():
        candidates.append(be.NUMBA)
    else:
        reasons.append("numba not importable")
    candidates.append(be.NUMPY)

    for name in candidates:
        try:
            res = runner(bundle, name)
        except Exception as e:                       # a stale / broken fixture or backend is a failure, never swallowed
            res = {"backend": name, "passed": False, "error": f"{type(e).__name__}: {e}"}
        attempts.append(res)
        if res.get("passed"):
            chosen, final = name, res
            break
        reasons.append(f"{name} self-test FAILED" + (f" ({res['error']})" if res.get("error") else ""))
        log(f"BACKEND {name}: SELF-TEST FAILED -- trying the next validated backend")
    if chosen is None:
        raise BackendSelectionError(f"bundle self-test failed on every backend {candidates} -- STOP, no submission: {attempts}")
    gpu_used = chosen not in (be.NUMBA, be.NUMPY)
    if require_gpu and not gpu_used:
        raise BackendSelectionError(f"REQUIRE_GPU=True but no validated GPU backend: {'; '.join(reasons)}")
    be.set_backend(chosen)
    record = {"requested_device": device_preference, "require_gpu": bool(require_gpu), "actual_backend": chosen,
              "gpu_available": bool(accel.get("cuda_available")), "gpu_used": gpu_used, "gpu_name": accel.get("gpu_name"),
              "numba_available": bool(be.numba_available()),
              "numpy_fallback_used": chosen == be.NUMPY and be.numba_available(),
              "fallback_reason": "; ".join(reasons) if reasons else None, "self_test_attempts": attempts,
              "passed": True, "final": final}
    if record["numpy_fallback_used"] or (device_preference == "gpu" and not gpu_used):
        log("BACKEND FALLBACK: " + (record["fallback_reason"] or "") + f" -> using {chosen}")
    return record
