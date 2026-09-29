"""Direct-name Colab inference mode.

This module intentionally performs NO SHA256 verification. Runtime inputs are explicit paths,
not discovered by hashes or dataset IDs. Safety gates are filename presence, frozen identity,
aggregator/calibration metadata consistency, model identity, and the functional frozen self-test.

For the current Drive migration only: a self-test with feature-only drift is accepted when every
downstream invariant is exact (candidate sets, fold scores, scores, ranks, probabilities, selected
spectrum, confidence, and molecule Top-25 order). The original mismatch counts are preserved in
run_report.json.
"""
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from casmi_runtime.accelerator import detect_accelerator
from casmi_runtime.backends import select_backend
from casmi_runtime.frozen import load_frozen_code
from casmi_runtime.inference import (
    GpuMemorySampler,
    InferenceBlocked,
    _neutral_mass,
    _rdkit_parse_fn,
    check_bundle_identity,
)


def _require_named_files(bundle_dir, test_path, sample_path):
    bundle_dir, test_path, sample_path = map(Path, (bundle_dir, test_path, sample_path))
    required = [
        bundle_dir / "manifest.json",
        bundle_dir / "config.json",
        bundle_dir / "models" / "model_info.json",
        bundle_dir / "ppm_windows.json",
        bundle_dir / "adducts.json",
        bundle_dir / "structures.parquet",
        bundle_dir / "structures_mass.npy",
        bundle_dir / "connectivities.parquet",
        bundle_dir / "_ref_peaks_int.f8",
        bundle_dir / "_ref_peaks_mz.f8",
        bundle_dir / "ref_peaks_int.npy",
        bundle_dir / "ref_peaks_mz.npy",
        bundle_dir / "ref_peaks_offsets.npy",
        bundle_dir / "ref_index_ids.npy",
        bundle_dir / "ref_index_offsets.npy",
        bundle_dir / "ref_meta.parquet",
        bundle_dir / "aggregation" / "calibration.json",
        bundle_dir / "aggregation" / "selected_aggregator.json",
        bundle_dir / "selftest" / "fixture_manifest.json",
        test_path,
        sample_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Required named runtime files are missing:\n  " + "\n  ".join(missing))
    return bundle_dir, test_path, sample_path


def _install_filename_only_contract(frozen):
    """Replace only the aggregator contract checker with a filename/metadata checker."""
    def verify_contract(config, bundle_dir):
        agg_cfg = config.get("aggregator")
        if not isinstance(agg_cfg, dict) or not agg_cfg.get("name"):
            raise RuntimeError("config.json names no aggregator")
        if config.get("aggregator_name") not in (None, agg_cfg["name"]):
            raise RuntimeError("aggregator_name != aggregator.name")

        agg = frozen.aggregation.make_aggregator(agg_cfg)
        aid = frozen.aggregation.canonical_aggregator_id(agg_cfg["name"])
        out = {
            "name": agg_cfg["name"],
            "aggregator_id": aid,
            "needs_prob": bool(agg.needs_prob),
            "temperature": None,
            "calibration_file": None,
            "selection_artifact_file": None,
        }
        if not agg.needs_prob:
            return out

        t = agg_cfg.get("temperature")
        if t is None or float(t) <= 0:
            raise RuntimeError(f"{agg_cfg['name']} requires a positive frozen temperature")
        if config.get("calibration_temperature") != t:
            raise RuntimeError("calibration_temperature != aggregator.temperature")

        d = Path(bundle_dir)
        cal_rel = config.get("calibration_path")
        sel_rel = config.get("selection_artifact_path")
        if not cal_rel or not sel_rel:
            raise RuntimeError("calibration_path / selection_artifact_path missing from config")

        cal_path, sel_path = d / cal_rel, d / sel_rel
        if not cal_path.exists() or not sel_path.exists():
            raise RuntimeError(f"missing named aggregation artifact(s): {cal_path}, {sel_path}")

        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        if (cal.get("temperature_fit") or {}).get("temperature") != t:
            raise RuntimeError("calibration artifact temperature != config temperature")
        if frozen.aggregation.canonical_aggregator_id(sel.get("selected_aggregator", "")) != aid:
            raise RuntimeError("selection artifact aggregator != configured aggregator")
        if sel.get("temperature") != t or (sel.get("aggregator_config") or {}).get("temperature") != t:
            raise RuntimeError("selection artifact temperature != config temperature")
        if config.get("model_id") is not None and sel.get("spectrum_model") != config["model_id"]:
            raise RuntimeError("selection artifact spectrum_model != configured model")

        out.update(
            temperature=t,
            calibration_file=str(cal_rel),
            selection_artifact_file=str(sel_rel),
        )
        return out

    frozen.pipeline.verify_aggregator_contract = verify_contract
    frozen.validation.verify_aggregator_contract = verify_contract
    return verify_contract


def _filename_only_selftest_runner(frozen, bundle_dir, allow_feature_only_drift=True):
    """Load fixture files by filename only and run the normal functional parity path."""
    d = Path(bundle_dir) / "selftest"
    man = json.loads((d / "fixture_manifest.json").read_text(encoding="utf-8"))
    data = {}
    for name in frozen.selftest.FILES:
        p = d / name
        if not p.exists():
            raise FileNotFoundError(f"missing self-test file: {p}")
        data[name] = pd.read_parquet(p)

    # Remove the fixture's model-file SHA comparison. Keep identity that does not require file hashing.
    def check_fixture_matches_bundle(bundle, fixture_manifest):
        if fixture_manifest.get("CONFIG_HASH") != bundle.config.get("CONFIG_HASH"):
            raise frozen.selftest.SelfTestFailed("fixture CONFIG_HASH != bundle CONFIG_HASH")
        if list(fixture_manifest.get("feature_names") or []) != list(bundle.ranker.feature_names):
            raise frozen.selftest.SelfTestFailed("fixture feature order != frozen model feature order")
        if fixture_manifest.get("temperature") != bundle.temperature:
            raise frozen.selftest.SelfTestFailed("fixture temperature != bundle temperature")

    frozen.selftest.check_fixture_matches_bundle = check_fixture_matches_bundle
    fixture = (man, data)

    hard_keys = [
        "candidate_set_mismatches",
        "score_mismatches",
        "fold_score_mismatches",
        "rank_mismatches",
        "prob_mismatches",
        "selected_spectrum_mismatches",
        "confidence_mismatches",
        "molecule_ranking_mismatches",
    ]

    def runner(bundle, backend):
        result = frozen.selftest.run_selftest(bundle, backend, fixture=fixture)
        strict_passed = bool(result.get("passed"))
        downstream_exact = all(int(result.get(k, 0)) == 0 for k in hard_keys)
        accepted_feature_only_drift = (
            bool(allow_feature_only_drift)
            and downstream_exact
            and int(result.get("feature_mismatches", 0)) > 0
        )
        result["strict_passed"] = strict_passed
        result["downstream_exact"] = downstream_exact
        result["accepted_feature_only_drift"] = accepted_feature_only_drift
        if accepted_feature_only_drift:
            result["passed"] = True
        return result

    return runner


def _validate_named_test(frozen, test_path, sample_path):
    test = pd.read_parquet(test_path)
    sample = pd.read_csv(sample_path)
    missing = [c for c in frozen.pipeline.TEST_REQUIRED if c not in test.columns]
    if missing:
        raise InferenceBlocked(f"test is missing required columns: {missing}")
    if test["spectrum_id"].duplicated().any():
        raise InferenceBlocked("duplicate spectrum_id in test.parquet")
    if list(sample.columns) != ["molecule_id", "smiles"]:
        raise InferenceBlocked(f"sample submission columns are {list(sample.columns)}")
    if not set(sample["molecule_id"]) <= set(test["molecule_id"]):
        raise InferenceBlocked("sample submission contains molecule_id without test spectra")
    return test, sample


def run_named_inference(
    cfg,
    bundle_dir,
    test_path,
    sample_path,
    allow_feature_only_drift=True,
    log=print,
):
    """Run frozen inference from explicit filenames with no SHA256 checks."""
    t_start = time.time()
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stale = [n for n in ("submission.csv", "run_report.json", "run_report_failed.json") if (work / n).exists()]
    if stale:
        raise InferenceBlocked(f"{work} already holds {stale}; use a fresh work_dir")

    bundle_dir, test_path, sample_path = _require_named_files(bundle_dir, test_path, sample_path)
    stage = "start"
    ctx = {
        "bundle_integrity_mode": "FILENAMES_ONLY",
        "bundle_dir": str(bundle_dir.resolve()),
        "test_path": str(test_path.resolve()),
        "sample_submission_path": str(sample_path.resolve()),
    }

    try:
        stage = "accelerator"
        accel = detect_accelerator()
        log(f"[runtime] {cfg.runtime} | device {accel['device']} ({accel['gpu_name']})")

        stage = "load_frozen_code"
        frozen = load_frozen_code(bundle_dir)
        ctx["casmi_infer_source"] = frozen.source
        _install_filename_only_contract(frozen)

        stage = "load_bundle_by_filename"
        runlog = frozen.pipeline.RunLog()
        t = time.time()
        bundle = frozen.pipeline.Bundle(bundle_dir, verify=False)
        runlog.stage("load_bundle", time.time() - t)

        model_info = json.loads((bundle_dir / "models" / "model_info.json").read_text(encoding="utf-8"))
        check_bundle_identity(bundle.config, bundle.manifest, model_info, cfg)
        ctx["CONFIG_HASH"] = bundle.config["CONFIG_HASH"]
        log(
            f"[bundle] {bundle.config.get('bundle_version')} | CONFIG_HASH {bundle.config['CONFIG_HASH']} | "
            f"model {bundle.ranker.model_id} | aggregator {bundle.aggregator_contract['name']} "
            f"(T={bundle.aggregator_contract['temperature']}) | filenames only"
        )

        stage = "functional_self_test"
        runner = _filename_only_selftest_runner(
            frozen,
            bundle_dir,
            allow_feature_only_drift=allow_feature_only_drift,
        )
        t = time.time()
        backend = select_backend(
            bundle,
            frozen,
            accel,
            cfg.device_preference,
            cfg.require_gpu,
            runner=runner,
            log=log,
        )
        runlog.stage("self_test", time.time() - t)
        fin = backend["final"]
        log(
            f"[self-test] PASS on {backend['actual_backend']} | "
            f"strict={fin.get('strict_passed')} | feature_mismatches={fin.get('feature_mismatches')} | "
            f"downstream_exact={fin.get('downstream_exact')}"
        )

        stage = "load_named_test"
        test, sample = _validate_named_test(frozen, test_path, sample_path)
        unsupported = sorted(a for a in test["adduct"].astype(str).unique() if not bundle.adducts.supported(a))

        stage = "inference"
        with GpuMemorySampler(backend["gpu_used"]) as gpu_mem:
            per_spec = frozen.pipeline.run_inference(
                bundle,
                test,
                runlog,
                time_budget_s=cfg.time_budget_s,
                emergency_cap_enabled=cfg.emergency_cap_enabled,
                emergency_cap_n=cfg.emergency_cap_n,
            )

            stage = "aggregation"
            t = time.time()
            mol_rank = frozen.pipeline.aggregate_molecules(bundle, per_spec, counters=runlog.counters)
            selected = frozen.pipeline.selected_spectra(bundle, per_spec)
            top_k = int(bundle.config["top_k_submission"])
            ranked_conn = {
                m: list(dict.fromkeys(g.sort_values("mol_rank", kind="mergesort")["conn_idx"].astype(int).tolist()))[:top_k]
                for m, g in mol_rank.groupby("molecule_id")
            }
            nm_by_spectrum = {
                r["spectrum_id"]: _neutral_mass(bundle, r)
                for r in test.to_dict("records")
            }
            conn_final, smiles_by_mol, n_padded = {}, {}, 0
            for mid in sample["molecule_id"]:
                conn = ranked_conn.get(mid, [])
                if bundle.config.get("pad_to_top_k_with_nearest_mass") and len(conn) < top_k:
                    nm = pd.Series(
                        [nm_by_spectrum.get(s) for s in test.loc[test["molecule_id"] == mid, "spectrum_id"]],
                        dtype=float,
                    ).dropna()
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
        smiles_status, smiles_problems = V.check_smiles_parse(
            [s for v in smiles_by_mol.values() for s in v],
            json.loads(export_smiles.read_text(encoding="utf-8")) if export_smiles.exists() else None,
            _rdkit_parse_fn(),
        )
        problems += smiles_problems
        problems += [
            f"{m}: output order differs from aggregator ranking"
            for m, c in ranked_conn.items()
            if conn_final.get(m, [])[:len(c)] != c
        ][:20]
        validation_status = "PASS" if not problems else "FAIL"
        log(f"[validation] {validation_status} | SMILES {smiles_status} | problems: {problems[:10] or 'none'}")
        if problems:
            raise InferenceBlocked(f"{len(problems)} submission problems: {problems[:20]}")

        stage = "write"
        sub_path = work / "submission.csv"
        frozen.submission.write_submission(sub, sub_path, backend)
        runlog.tick_rss()
        fallback_counts = {
            **pd.Series([f["fallback_type"] for f in runlog.fallbacks], dtype=object).value_counts().astype(int).to_dict(),
            **{
                k: int(v)
                for k, v in runlog.counters.items()
                if "fallback" in k or k.startswith("retrieval_level_")
            },
        }
        c = bundle.aggregator_contract
        report = {
            **runlog.to_dict(),
            "runtime_platform": cfg.runtime,
            "python_version": accel["python_version"],
            "gpu_available": accel["cuda_available"],
            "gpu_name": accel["gpu_name"],
            "requested_device": backend["requested_device"],
            "actual_backend": backend["actual_backend"],
            "gpu_used": backend["gpu_used"],
            "numba_available": backend["numba_available"],
            "numpy_fallback_used": backend["numpy_fallback_used"],
            "fallback_reason": backend["fallback_reason"],
            "backend": backend["actual_backend"],
            "bundle_version": bundle.config.get("bundle_version"),
            "CONFIG_HASH": bundle.config["CONFIG_HASH"],
            "model_id": bundle.ranker.model_id,
            "aggregator": c["name"],
            "aggregator_config": bundle.config["aggregator"],
            "aggregator_artifact_file": c.get("selection_artifact_file"),
            "calibration_file": c.get("calibration_file"),
            "calibration_temperature": c["temperature"],
            "candidate_universe_version": bundle.config.get("candidate_universe_version"),
            "casmi_infer_source": frozen.source,
            "bundle_dir": str(bundle_dir.resolve()),
            "test_file": test_path.name,
            "sample_submission_file": sample_path.name,
            "bundle_integrity_mode": "FILENAMES_ONLY",
            "bundle_hash_verification": False,
            "fixture_hash_verification": False,
            "self_test_status": "PASS",
            "self_test_mode": "FUNCTIONAL_DOWNSTREAM_EXACT",
            "self_test_strict_passed": bool(fin.get("strict_passed")),
            "self_test_feature_mismatches": int(fin.get("feature_mismatches", 0)),
            "self_test_downstream_exact": bool(fin.get("downstream_exact")),
            "self_test_feature_only_drift_accepted": bool(fin.get("accepted_feature_only_drift")),
            "self_test_mismatches": {
                k: fin.get(k)
                for k in getattr(frozen.selftest, "MISMATCH_KEYS", ())
            },
            "self_test_attempts": backend["self_test_attempts"],
            "n_spectra": int(len(test)),
            "n_molecules": int(test["molecule_id"].nunique()),
            "candidate_pair_count": int(runlog.counters.get("candidates", 0)),
            "similarity_evaluations": int(runlog.counters.get("similarity_evaluations", 0)),
            "runtime_seconds": time.time() - t_start,
            "peak_ram": runlog.peak_rss_gb,
            "peak_gpu_memory": gpu_mem.peak,
            "fallback_counts": fallback_counts,
            "exception_count": len(runlog.exceptions),
            "unsupported_adducts": unsupported,
            "a7_selected_confidence": selected["confidence"].describe().to_dict() if selected is not None and len(selected) else None,
            "submission_rows": int(len(sub)),
            "submission_validation_status": validation_status,
            "smiles_parse_check": smiles_status,
            "submission_file": sub_path.name,
            "accelerator": accel,
            "environment": V.environment_report(),
            "runtime_config": cfg.as_dict(),
            "mode": "CLOSED-WORLD / CLASS-1 ONLY",
            "submission_note": "manual Kaggle submission; no SHA verification used in this run",
        }

        required_report = (
            "bundle_version",
            "CONFIG_HASH",
            "model_id",
            "aggregator",
            "self_test_status",
            "backend",
            "n_spectra",
            "n_molecules",
            "submission_rows",
            "submission_validation_status",
            "bundle_integrity_mode",
        )
        missing_report = [k for k in required_report if report.get(k) is None]
        if missing_report:
            raise RuntimeError(f"run_report incomplete: {missing_report}")

        (work / "run_report.json").write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
        log(
            f"[done] submission.csv ({len(sub)} rows) + run_report.json in {work} | "
            f"{report['runtime_seconds'] / 60:.1f} min"
        )
        return report

    except Exception as e:
        (work / "run_report_failed.json").write_text(
            json.dumps(
                {
                    "failed_stage": stage,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback_tail": traceback.format_exc().splitlines()[-8:],
                    "runtime_platform": cfg.runtime,
                    "runtime_seconds": time.time() - t_start,
                    **ctx,
                    "runtime_config": cfg.as_dict(),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        raise
