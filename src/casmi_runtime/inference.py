"""The production run sequence (Colab and Kaggle share it). ORCHESTRATION ONLY: every inference step is a call into
the FROZEN bundle code (`frozen.load_frozen_code`):

    discover inputs -> detect accelerator -> load frozen code -> Bundle(verify=True) [manifest sha256s, CONFIG_HASH,
    aggregator + frozen-calibration contract] -> identity checks (model / aggregator / bundle version / optional
    expected CONFIG_HASH) -> self-test + validated backend -> pipeline.run_inference [adducts, candidates, features,
    frozen V1 scores; per-spectrum nearest-mass fallback logged] -> pipeline.aggregate_molecules [frozen T, A7] ->
    connectivity dedup -> Top-25 (+ the existing nearest-mass padding) -> representative SMILES -> validation ->
    submission.csv (guarded writer) -> run_report.json.

Any failure raises; a `run_report_failed.json` records the stage and the error, and no submission.csv is written.
"""
import hashlib
import json
import threading
import time
import traceback
from pathlib import Path

from casmi_runtime.accelerator import detect_accelerator, gpu_memory_used_mb
from casmi_runtime.backends import select_backend
from casmi_runtime.frozen import load_frozen_code


class InferenceBlocked(RuntimeError):
    """A pre-inference check failed (identity, input discovery, validation) -- nothing is written."""


# ---------------------------------------------------------------------------------------------
# inputs + identity
# ---------------------------------------------------------------------------------------------

def discover_inputs(input_root, test_required):
    """Exactly one CASMI bundle (manifest.json with a casmi bundle_format), one test parquet with the required columns
    and one sample submission (molecule_id,smiles) under `input_root` -- found by schema, never by dataset name."""
    import pandas as pd
    import pyarrow.parquet as pq
    root = Path(input_root)
    bundles = []
    for p in root.rglob("manifest.json"):
        try:
            if str(json.loads(p.read_text(encoding="utf-8")).get("bundle_format", "")).startswith("casmi-modeA-bundle"):
                bundles.append(p.parent)
        except (ValueError, OSError):
            continue
    if len(bundles) != 1:
        raise InferenceBlocked(f"expected exactly one CASMI bundle under {root}, found {bundles}")
    bundle_dir = bundles[0]
    outside = lambda p: bundle_dir not in p.parents
    tests = [p for p in root.rglob("*.parquet") if outside(p) and set(test_required) <= set(pq.read_schema(p).names)]
    if len(tests) != 1:
        raise InferenceBlocked(f"expected exactly one test parquet with columns {list(test_required)}, found {tests}")
    samples = [p for p in root.rglob("*.csv") if outside(p) and list(pd.read_csv(p, nrows=0).columns) == ["molecule_id", "smiles"]]
    if len(samples) != 1:
        raise InferenceBlocked(f"expected exactly one sample submission (molecule_id,smiles), found {samples}")
    return bundle_dir, tests[0], samples[0]


def check_bundle_identity(config, manifest, model_info, cfg):
    """Blocks inference unless the bundle is the frozen one this runtime was built for."""
    problems = []
    if config.get("aggregator_name") != cfg.expected_aggregator:
        problems.append(f"aggregator {config.get('aggregator_name')!r} != {cfg.expected_aggregator!r}")
    if config.get("model_id") != cfg.expected_model_id or model_info.get("model_id") != cfg.expected_model_id:
        problems.append(f"model {config.get('model_id')!r}/{model_info.get('model_id')!r} != {cfg.expected_model_id!r}")
    if model_info.get("freeze_status") != "FROZEN":
        problems.append(f"model freeze_status {model_info.get('freeze_status')!r} != 'FROZEN'")
    if config.get("bundle_version") != cfg.expected_bundle_version or not str(manifest.get("bundle_version", "")).startswith(cfg.expected_bundle_version + "-"):
        problems.append(f"bundle version {config.get('bundle_version')!r}/{manifest.get('bundle_version')!r} != {cfg.expected_bundle_version!r}")
    if cfg.expected_config_hash and config.get("CONFIG_HASH") != cfg.expected_config_hash:
        problems.append(f"CONFIG_HASH {config.get('CONFIG_HASH')!r} != expected {cfg.expected_config_hash!r}")
    if config.get("calibration_temperature") is None and config.get("aggregator_name") == "MOST_CONFIDENT_SPECTRUM":
        problems.append("A7 bundle without a frozen calibration temperature")
    if problems:
        raise InferenceBlocked("bundle identity check failed -- inference blocked:\n  " + "\n  ".join(problems))
    return True


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

class GpuMemorySampler:
    """Background nvidia-smi sampler (only when a GPU backend is actually used)."""

    def __init__(self, enabled, every_s=5.0):
        self.enabled, self.every_s, self.peak, self._stop = enabled, every_s, None, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True) if enabled else None

    def _run(self):
        while not self._stop.is_set():
            v = gpu_memory_used_mb()
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
            self._stop.wait(self.every_s)

    def __enter__(self):
        if self._t:
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=10)


def _neutral_mass(bundle, row):
    """Neutral mass exactly as `pipeline.resolve_neutral_mass` derives it (polarity default for unsupported adducts),
    without re-logging the fallback run_inference already logged. Used only for the existing Top-k padding."""
    try:
        nm = bundle.adducts.neutral_mass(row["precursor_mz"], str(row["adduct"]))
        if nm is None:
            nm = bundle.adducts.neutral_mass(row["precursor_mz"], bundle.adducts.polarity_default(row.get("ionization_mode")))
        return nm
    except Exception:
        return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _rdkit_parse_fn():
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        return lambda s: Chem.MolFromSmiles(s) is not None
    except ImportError:
        return None


# ---------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------

def run_frozen_inference(cfg, log=print):
    """Returns the run report dict (also written to cfg.work_dir/run_report.json)."""
    t_start = time.time()
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stale = [n for n in ("submission.csv", "run_report.json", "run_report_failed.json") if (work / n).exists()]
    if stale:
        raise InferenceBlocked(f"{work} already holds {stale} from a previous run -- move them or use a fresh work_dir (never overwritten)")
    stage = "start"
    ctx = {}
    try:
        stage = "accelerator"
        accel = detect_accelerator()
        ctx["accelerator"] = accel
        log(f"[runtime] {cfg.runtime} | device hardware {accel['device']} ({accel['gpu_name']}) | cpu {accel['cpu_count']} | ram {accel['ram_gb']} GB")

        stage = "discover_inputs"
        # discovery needs only pyarrow/pandas; TEST_REQUIRED comes from the frozen code, so load it first
        bundle_probe = [p.parent for p in Path(cfg.input_root).rglob("manifest.json")
                        if str(json.loads(p.read_text(encoding="utf-8")).get("bundle_format", "")).startswith("casmi-modeA-bundle")]
        if len(bundle_probe) != 1:
            raise InferenceBlocked(f"expected exactly one CASMI bundle under {cfg.input_root}, found {bundle_probe}")

        stage = "load_frozen_code"
        frozen = load_frozen_code(bundle_probe[0])
        ctx["casmi_infer_source"] = frozen.source
        bundle_dir, test_path, sample_path = discover_inputs(cfg.input_root, frozen.pipeline.TEST_REQUIRED)

        stage = "verify_bundle"
        runlog = frozen.pipeline.RunLog()
        t = time.time()
        bundle = frozen.pipeline.Bundle(bundle_dir, verify=True)      # sha256 of every file, CONFIG_HASH, aggregator/calibration contract
        runlog.stage("load_bundle", time.time() - t)
        model_info = json.loads((bundle_dir / "models" / "model_info.json").read_text(encoding="utf-8"))
        check_bundle_identity(bundle.config, bundle.manifest, model_info, cfg)
        ctx["CONFIG_HASH"] = bundle.config["CONFIG_HASH"]
        log(f"[bundle] {bundle.manifest.get('bundle_version')} | CONFIG_HASH {bundle.config['CONFIG_HASH']} | model {bundle.ranker.model_id} | "
            f"aggregator {bundle.aggregator_contract['name']} (T={bundle.aggregator_contract['temperature']}) | code {frozen.source}")

        stage = "self_test_and_backend"
        t = time.time()
        backend = select_backend(bundle, frozen, accel, cfg.device_preference, cfg.require_gpu, log=log)
        runlog.stage("self_test", time.time() - t)
        fin = backend["final"]
        log(f"[self-test] PASS on {backend['actual_backend']} | gpu_used={backend['gpu_used']} | fallback: {backend['fallback_reason']}")

        stage = "load_test"
        import pandas as pd
        test = pd.read_parquet(test_path)
        sample = pd.read_csv(sample_path)
        missing = [c for c in frozen.pipeline.TEST_REQUIRED if c not in test.columns]
        if missing or test["spectrum_id"].duplicated().any() or not set(sample["molecule_id"]) <= set(test["molecule_id"]):
            raise InferenceBlocked(f"test/sample schema problem: missing={missing}, duplicate spectrum_id / sample molecules without spectra")
        unsupported = sorted(a for a in test["adduct"].astype(str).unique() if not bundle.adducts.supported(a))

        stage = "inference"
        with GpuMemorySampler(backend["gpu_used"]) as gpu_mem:
            per_spec = frozen.pipeline.run_inference(bundle, test, runlog, time_budget_s=cfg.time_budget_s,
                                                     emergency_cap_enabled=cfg.emergency_cap_enabled, emergency_cap_n=cfg.emergency_cap_n)

            stage = "aggregation"
            t = time.time()
            mol_rank = frozen.pipeline.aggregate_molecules(bundle, per_spec, counters=runlog.counters)
            selected = frozen.pipeline.selected_spectra(bundle, per_spec)
            top_k = int(bundle.config["top_k_submission"])
            ranked_conn = {m: list(dict.fromkeys(g.sort_values("mol_rank", kind="mergesort")["conn_idx"].astype(int).tolist()))[:top_k]
                           for m, g in mol_rank.groupby("molecule_id")}                  # connectivity dedup BEFORE top-k
            nm_by_spectrum = {r["spectrum_id"]: _neutral_mass(bundle, r) for r in test.to_dict("records")}
            conn_final, smiles_by_mol, n_padded = {}, {}, 0
            for mid in sample["molecule_id"]:
                conn = ranked_conn.get(mid, [])
                if bundle.config.get("pad_to_top_k_with_nearest_mass") and len(conn) < top_k:      # the EXISTING padding policy
                    nm = pd.Series([nm_by_spectrum.get(s) for s in test.loc[test["molecule_id"] == mid, "spectrum_id"]], dtype=float).dropna()
                    extra = bundle.search.nearest(float(nm.median()), top_k)[0].tolist() if len(nm) else []
                    padded = frozen.submission.pad_with_nearest(conn, extra, top_k)
                    n_padded += int(len(padded) > len(conn))
                    if not conn and padded:
                        runlog.bump("molecules_zero_candidates_nearest_mass_fallback")
                    conn = padded
                conn_final[mid] = conn
                smi = list(dict.fromkeys(bundle.smiles[c] for c in conn))
                if len(smi) < len(conn):
                    runlog.bump("duplicate_smiles_dropped", len(conn) - len(smi))
                smiles_by_mol[mid] = smi[:top_k]
            runlog.counters["molecules_padded_with_nearest_mass"] = n_padded
            runlog.stage("aggregation", time.time() - t)

        stage = "validation"
        V = frozen.validation
        sub = frozen.submission.build_submission(sample, smiles_by_molecule=smiles_by_mol)
        problems = V.validate_submission(sub, sample, top_k=top_k)
        problems += V.validate_rankings(conn_final, list(sample["molecule_id"]), top_k=top_k)
        export_smiles = bundle_dir / "smiles_validation.json"
        smiles_status, smiles_problems = V.check_smiles_parse([s for v in smiles_by_mol.values() for s in v],
                                                              json.loads(export_smiles.read_text(encoding="utf-8")) if export_smiles.exists() else None,
                                                              _rdkit_parse_fn())
        problems += smiles_problems
        problems += [f"{m}: output order differs from the aggregator ranking" for m, c in ranked_conn.items() if conn_final.get(m, [])[:len(c)] != c][:20]
        validation_status = "PASS" if not problems else "FAIL"
        log(f"[validation] {validation_status} | SMILES check {smiles_status} | problems: {problems[:10] or 'none'}")
        if problems:
            raise InferenceBlocked(f"{len(problems)} submission problems -- submission.csv NOT written: {problems[:20]}")

        stage = "write"
        sub_path = work / "submission.csv"
        frozen.submission.write_submission(sub, sub_path, backend)               # the guarded writer: refuses unless the self-test passed
        runlog.tick_rss()
        fallback_counts = {**pd.Series([f["fallback_type"] for f in runlog.fallbacks], dtype=object).value_counts().astype(int).to_dict(),
                           **{k: int(v) for k, v in runlog.counters.items() if "fallback" in k or k.startswith("retrieval_level_")}}
        c = bundle.aggregator_contract
        report = {
            **runlog.to_dict(),
            "runtime_platform": cfg.runtime, "python_version": accel["python_version"], "gpu_available": accel["cuda_available"],
            "gpu_name": accel["gpu_name"], "requested_device": backend["requested_device"], "actual_backend": backend["actual_backend"],
            "gpu_used": backend["gpu_used"], "numba_available": backend["numba_available"], "numpy_fallback_used": backend["numpy_fallback_used"],
            "fallback_reason": backend["fallback_reason"], "backend": backend["actual_backend"],
            "bundle_version": bundle.manifest.get("bundle_version"), "CONFIG_HASH": bundle.config["CONFIG_HASH"], "model_id": bundle.ranker.model_id,
            "aggregator": c["name"], "aggregator_config": bundle.config["aggregator"], "aggregator_artifact_hash": c["selection_artifact_sha256"],
            "calibration_hash": c["calibration_sha256"], "calibration_temperature": c["temperature"],
            "candidate_universe_version": bundle.config.get("candidate_universe_version"), "casmi_infer_source": frozen.source,
            "bundle_dir": str(Path(bundle_dir).resolve()), "frozen_code_dir": str(Path(frozen.code_dir).resolve()),
            "self_test_status": "PASS", "self_test_mismatches": {k: fin.get(k) for k in getattr(frozen.selftest, "MISMATCH_KEYS", ())},
            "self_test_attempts": backend["self_test_attempts"],
            "n_spectra": int(len(test)), "n_molecules": int(test["molecule_id"].nunique()),
            "candidate_pair_count": int(runlog.counters.get("candidates", 0)),
            "similarity_evaluations": int(runlog.counters.get("similarity_evaluations", 0)),
            "runtime_seconds": time.time() - t_start, "peak_ram": runlog.peak_rss_gb, "peak_gpu_memory": gpu_mem.peak,
            "fallback_counts": fallback_counts, "exception_count": len(runlog.exceptions), "unsupported_adducts": unsupported,
            "a7_selected_confidence": selected["confidence"].describe().to_dict() if selected is not None and len(selected) else None,
            "submission_rows": int(len(sub)), "submission_validation_status": validation_status, "smiles_parse_check": smiles_status,
            "submission_sha256": _sha256(sub_path), "accelerator": accel, "environment": V.environment_report(),
            "runtime_config": cfg.as_dict(), "mode": "CLOSED-WORLD / CLASS-1 ONLY",
            "submission_note": "submitting to Kaggle is MANUAL; nothing here submits",
        }
        V.assert_run_report_complete(report)
        (work / "run_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log(f"[done] submission.csv ({len(sub)} rows) + run_report.json in {work} | {report['runtime_seconds'] / 60:.1f} min")
        return report
    except Exception as e:
        (work / "run_report_failed.json").write_text(json.dumps({
            "failed_stage": stage, "error": f"{type(e).__name__}: {e}", "traceback_tail": traceback.format_exc().splitlines()[-8:],
            "runtime_platform": cfg.runtime, "runtime_seconds": time.time() - t_start, **ctx, "runtime_config": cfg.as_dict()},
            indent=2, default=str), encoding="utf-8")
        raise
