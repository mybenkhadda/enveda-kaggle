"""The Kaggle anchor run (internet OFF). ORCHESTRATION ONLY: every scientific step is a call into the FROZEN bundle code
(`frozen.load_frozen_code` -> <bundle>/code/casmi_infer); nothing here re-implements inference.

Inputs, resolved BY FILENAME under `input_root` (never by a Kaggle mount slug) -- `resolve_kaggle_model_inputs`:

    <payload>/runtime/src/casmi_runtime/named_inference.py   exactly one -> MODEL_PAYLOAD_ROOT, RUNTIME_SRC
    <payload>/bundle/config.json (+ code/casmi_infer/__init__.py) exactly one -> BUNDLE_ROOT == <payload>/bundle
    test.parquet, sample_submission.csv                       exactly one each, outside the model payload
    (train.parquet is never read)

Integrity mode FILENAMES_ONLY: no `verify_bundle`, no `Bundle(verify=True)`, no manifest-wide sha256 pass and no
fixture sha256 (the 7 self-test fixtures are loaded here by name and handed to the frozen `run_selftest(fixture=...)`,
bypassing the frozen hashing loader). The one hash check that remains is INSIDE the frozen `Bundle()` constructor:
its aggregator contract compares aggregation/calibration.json and aggregation/selected_aggregator.json with the
sha256s recorded in config.json -- frozen scientific code, not bundle verification, and not editable here.

Gate = the FUNCTIONAL self-test (format 2) before any hidden-test inference. With `allow_feature_only_drift=True`, a
backend whose only nonzero count is `feature_mismatches` is accepted iff every hard downstream key is exactly zero
(`backends.apply_selftest_policy`); strict_passed / downstream_exact / accepted_feature_only_drift are recorded. No
LightGBM version string is checked.

Run sequence (one pass over the whole test set, no smoke / reduced path):

    accelerator -> inputs -> load frozen code -> filename presence check -> Bundle(verify=False) -> frozen identity ->
    self-test + validated backend -> pipeline.run_inference [clean spectra, adduct neutral mass, candidate retrieval,
    reference selection, spectral similarity, 9 model features, 5 V1 folds -> fold-mean score, per-spectrum rank] ->
    pipeline.aggregate_molecules [frozen temperature, A7 MOST_CONFIDENT_SPECTRUM] -> connectivity dedup -> Top-25
    (+ the existing nearest-mass padding) -> representative SMILES -> validation -> submission.csv (guarded writer)
    -> run_report.json -> anchor comparison -> kaggle_anchor_report.json.

Outputs (work_dir): submission.csv, run_report.json, kaggle_anchor_report.json -- nothing else. On a failure only
kaggle_anchor_report.json is written (anchor_status FAIL, failed_stage, error) and no submission.csv exists. An anchor
COUNT mismatch never blocks a validated submission; it only sets anchor_status FAIL. Nothing here submits.
"""
import json
import sys
import time
import traceback
from pathlib import Path

from casmi_runtime.accelerator import detect_accelerator
from casmi_runtime.backends import HARD_MISMATCH_KEYS, select_backend
from casmi_runtime.frozen import FROZEN_PACKAGES, FrozenCodeError, _is_under, load_frozen_code
from casmi_runtime.inference import (GpuMemorySampler, InferenceBlocked, _neutral_mass, _rdkit_parse_fn, _sha256,
                                     check_bundle_identity)

TEST_FILE = "test.parquet"
SAMPLE_FILE = "sample_submission.csv"
BUNDLE_FORMAT_PREFIX = "casmi-modeA-bundle"
ANCHOR_REPORT = "kaggle_anchor_report.json"
OUTPUT_NAMES = ("submission.csv", "run_report.json", ANCHOR_REPORT)
STALE_NAMES = OUTPUT_NAMES + ("run_report_failed.json",)
ENTRY_RELPATH = ("runtime", "src", "casmi_runtime", "named_inference.py")
N_MODEL_FOLDS = 5
N_MODEL_FEATURES = 9
SELFTEST_FIXTURES = ("queries.parquet", "expected_features.parquet", "expected_scores.parquet", "expected_ranks.parquet",
                     "expected_probs.parquet", "expected_molecules.parquet", "expected_molecule_ranking.parquet")
INTEGRITY = {"bundle_integrity_mode": "FILENAMES_ONLY", "bundle_hash_verification": False, "fixture_hash_verification": False,
             "frozen_aggregator_contract_note": ("the frozen Bundle() constructor compares aggregation/calibration.json and "
                                                 "aggregation/selected_aggregator.json with the sha256s in config.json (frozen code)")}
# the frozen scientific identity this runtime was built for (values from bundle v2-A7 config.json / models/model_info.json)
FROZEN_IDENTITY = {"bundle_version": "v2-A7", "CONFIG_HASH": "60174e39a2a3c6b4", "model_id": "V1_TL_1K_TESTSIM_STRICT",
                   "aggregator": "MOST_CONFIDENT_SPECTRUM", "calibration_temperature": 1.1947045372735254}
# the last successful Colab anchor on the competition test.parquet; environment-dependent values are NOT compared
KAGGLE_ANCHOR_EXPECTED = {**{k: v for k, v in FROZEN_IDENTITY.items() if k != "bundle_version"},
                          "n_spectra": 1213, "n_molecules": 400, "candidate_pair_count": 471173, "similarity_evaluations": 2057028,
                          "exception_count": 0, "unsupported_adducts": [], "submission_rows": 400}
E2E_STAGES = ("load test.parquet", "spectrum cleaning / preprocessing", "adduct neutral mass", "candidate retrieval",
              "reference selection", "spectral similarity evidence", "9 frozen model features", "5 LightGBM fold predictions",
              "fold-mean score", "per-spectrum ranking", "frozen temperature calibration", "A7 MOST_CONFIDENT_SPECTRUM selection",
              "connectivity deduplication", "Top-25", "representative SMILES", "submission validation", "submission.csv")


# ---------------------------------------------------------------------------------------------
# inputs by filename
# ---------------------------------------------------------------------------------------------

def _exactly_one(hits, what, root):
    hits = sorted(set(hits))
    if len(hits) != 1:
        raise InferenceBlocked(f"expected exactly one {what} under {root}, found {len(hits)}: {hits[:10]}")
    return hits[0]


def _competition_files(root, exclude):
    out = {}
    for name in (TEST_FILE, SAMPLE_FILE):
        out[name] = _exactly_one([p for p in Path(root).rglob(name) if p.is_file() and not any(_is_under(p, e) for e in exclude)],
                                 f"{name} (outside the model payload / bundle)", root)
    return out


def resolve_kaggle_model_inputs(input_root):
    """The attached Kaggle Model + the competition, by filename. Returns a dict of Paths:
    MODEL_PAYLOAD_ROOT, RUNTIME_SRC, BUNDLE_ROOT, TEST_PATH, SAMPLE_SUBMISSION_PATH."""
    root = Path(input_root)
    entry = _exactly_one([p for p in root.rglob(ENTRY_RELPATH[-1]) if p.is_file() and tuple(p.parts[-4:]) == ENTRY_RELPATH],
                         "/".join(ENTRY_RELPATH), root)
    payload = entry.parents[3]
    bundle = _exactly_one([p.parent for p in root.rglob("config.json")
                           if p.parent.name == "bundle" and (p.parent / "code" / "casmi_infer" / "__init__.py").is_file()],
                          "bundle/config.json with bundle/code/casmi_infer/__init__.py", root)
    if bundle != payload / "bundle":
        raise InferenceBlocked(f"the bundle {bundle} is not the attached model's own bundle ({payload / 'bundle'}) -- attach ONE Kaggle Model")
    comp = _competition_files(root, exclude=(payload, bundle))
    return {"MODEL_PAYLOAD_ROOT": payload, "RUNTIME_SRC": payload / "runtime" / "src", "BUNDLE_ROOT": bundle,
            "TEST_PATH": comp[TEST_FILE], "SAMPLE_SUBMISSION_PATH": comp[SAMPLE_FILE]}


def _bundle_dirs(root):
    out = []
    for p in Path(root).rglob("manifest.json"):
        try:
            if str(json.loads(p.read_text(encoding="utf-8")).get("bundle_format", "")).startswith(BUNDLE_FORMAT_PREFIX):
                out.append(p.parent)
        except (ValueError, OSError):
            continue
    return out


def resolve_named_inputs(input_root):
    """Generic fallback (no Kaggle Model layout): (bundle_dir, test_path, sample_path) -- exactly one CASMI bundle
    (manifest bundle_format), one `test.parquet` and one `sample_submission.csv` outside it."""
    root = Path(input_root)
    bundle_dir = _exactly_one(_bundle_dirs(root), f"CASMI bundle (manifest.json with bundle_format {BUNDLE_FORMAT_PREFIX}*)", root)
    comp = _competition_files(root, exclude=(bundle_dir,))
    return bundle_dir, comp[TEST_FILE], comp[SAMPLE_FILE]


# ---------------------------------------------------------------------------------------------
# filename-only checks
# ---------------------------------------------------------------------------------------------

def check_bundle_presence(bundle_dir, manifest):
    """Presence (not hashing) of every file the manifest lists, plus the E2E-critical categories spelled out:
    5 V1 folds, 7 format-2 self-test fixtures, calibration + selected aggregator, frozen code. Returns counts."""
    d = Path(bundle_dir)
    listed = sorted((manifest.get("files") or {}).keys())
    if not listed:
        raise InferenceBlocked(f"{d / 'manifest.json'} lists no files -- incomplete bundle upload?")
    critical = ([f"models/v1_fold{k}.txt" for k in range(N_MODEL_FOLDS)] + [f"selftest/{f}" for f in SELFTEST_FIXTURES]
                + ["selftest/fixture_manifest.json", "aggregation/calibration.json", "aggregation/selected_aggregator.json",
                   "aggregation/aggregator_lock.json", "models/model_info.json", "models/feature_names.json",
                   "code/casmi_infer/__init__.py", "code/casmi_infer/pipeline.py", "code/casmi_infer/selftest.py"])
    missing = [f for f in dict.fromkeys(listed + critical) if not (d / f).is_file()]
    empty = [f for f in listed if (d / f).is_file() and (d / f).stat().st_size == 0 and Path(f).name != "__init__.py"]   # empty package inits are legitimate
    if missing or empty:
        raise InferenceBlocked(f"frozen bundle incomplete at {d}: missing {missing[:20]} (n={len(missing)}), empty {empty[:20]} "
                               "-- re-upload the WHOLE bundle directory")
    return {"n_manifest_files": len(listed), "n_critical_files": len(critical), **INTEGRITY}


def load_fixture_filenames_only(bundle_dir, selftest_mod):
    """(fixture_manifest, {file: DataFrame}) for the frozen `run_selftest(fixture=...)` -- the same files and format check
    as the frozen loader, WITHOUT its sha256 comparison (FILENAMES_ONLY mode)."""
    import pandas as pd
    d = Path(bundle_dir) / selftest_mod.FIXTURE_DIR
    files = tuple(selftest_mod.FILES)
    if set(files) != set(SELFTEST_FIXTURES):
        raise InferenceBlocked(f"frozen self-test expects fixtures {files}, runtime expects {SELFTEST_FIXTURES}")
    man_path = d / "fixture_manifest.json"
    if not man_path.is_file():
        raise InferenceBlocked(f"no self-test fixture manifest at {man_path}")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    if man.get("format") != selftest_mod.FIXTURE_FORMAT:
        raise InferenceBlocked(f"fixture format {man.get('format')!r} != {selftest_mod.FIXTURE_FORMAT!r} (no molecule-level parity)")
    missing = [f for f in files if not (d / f).is_file()]
    if missing:
        raise InferenceBlocked(f"self-test fixtures missing: {missing}")
    return man, {f: pd.read_parquet(d / f) for f in files}


def check_frozen_identity(config, model_info, contract, expected=FROZEN_IDENTITY):
    """Exact frozen identity (bundle version, CONFIG_HASH, model, FROZEN status, A7, calibration temperature)."""
    actual = {"bundle_version": config.get("bundle_version"), "CONFIG_HASH": config.get("CONFIG_HASH"), "model_id": config.get("model_id"),
              "aggregator": config.get("aggregator_name"), "calibration_temperature": config.get("calibration_temperature")}
    problems = [f"{k}: {actual.get(k)!r} != {v!r}" for k, v in expected.items() if actual.get(k) != v]
    if model_info.get("model_id") != expected["model_id"]:
        problems.append(f"models/model_info.json model_id {model_info.get('model_id')!r} != {expected['model_id']!r}")
    if model_info.get("freeze_status") != "FROZEN":
        problems.append(f"models/model_info.json freeze_status {model_info.get('freeze_status')!r} != 'FROZEN'")
    if contract.get("name") != expected["aggregator"] or contract.get("temperature") != expected["calibration_temperature"]:
        problems.append(f"frozen aggregator contract {contract.get('name')!r} T={contract.get('temperature')!r} != "
                        f"{expected['aggregator']!r} T={expected['calibration_temperature']!r}")
    if problems:
        raise InferenceBlocked("frozen identity check failed -- inference blocked:\n  " + "\n  ".join(problems))
    return actual


def assert_frozen_modules_only(code_dir):
    """Every imported casmi / casmi_infer module must come from <bundle>/code (checked before AND after inference)."""
    foreign = {n: getattr(m, "__file__", None) for n, m in list(sys.modules.items())
               if n.split(".")[0] in FROZEN_PACKAGES and getattr(m, "__file__", None) and not _is_under(m.__file__, code_dir)}
    if foreign:
        raise FrozenCodeError(f"non-frozen casmi / casmi_infer modules are loaded: {foreign}")


def compare_to_anchor(report, expected):
    """{key: {expected, actual}} for every anchor value that differs (lists compared as sorted lists)."""
    norm = lambda v: sorted(map(str, v)) if isinstance(v, (list, tuple)) else v
    return {k: {"expected": v, "actual": report.get(k)} for k, v in (expected or {}).items() if norm(report.get(k)) != norm(v)}


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------

def run_named_inference(cfg, bundle_dir=None, test_path=None, sample_path=None, allow_feature_only_drift=False,
                        anchor_expected=KAGGLE_ANCHOR_EXPECTED, log=print):
    """Full E2E Kaggle anchor. Returns the run report (also cfg.work_dir/run_report.json); writes
    cfg.work_dir/kaggle_anchor_report.json in every outcome. Inputs not passed are resolved by filename under
    cfg.input_root (`resolve_named_inputs`)."""
    t_start = time.time()
    work = Path(cfg.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stale = [n for n in STALE_NAMES if (work / n).exists()]
    if stale:
        raise InferenceBlocked(f"{work} already holds {stale} from a previous run -- move them or use a fresh work_dir (never overwritten)")
    stage = "start"
    anchor = {"anchor": "casmi_runtime.named_inference.run_named_inference", "anchor_status": "RUNNING", "runtime_platform": cfg.runtime,
              "input_resolution": "by filename", **INTEGRITY, "allow_feature_only_drift": bool(allow_feature_only_drift),
              "gate": "functional self-test (format 2, 7 fixtures)", "e2e_stages": list(E2E_STAGES), "stages_completed": [],
              "expected": anchor_expected, "actual": None, "feature_mismatches": None, "strict_self_test_passed": None,
              "downstream_exact": None, "mismatches": None, "network_used": False, "submission_performed": False}

    def done(name):
        anchor["stages_completed"].append(name)

    try:
        stage = "accelerator"
        accel = detect_accelerator()
        anchor["accelerator"] = accel
        log(f"[runtime] {cfg.runtime} | device hardware {accel['device']} ({accel['gpu_name']}) | cpu {accel['cpu_count']} | "
            f"ram {accel['ram_gb']} GB | lightgbm {accel['lightgbm_version']}")
        done(stage)

        stage = "resolve_inputs"
        if bundle_dir is None or test_path is None or sample_path is None:
            found = resolve_named_inputs(cfg.input_root)
            bundle_dir, test_path, sample_path = (Path(a) if a is not None else f for a, f in zip((bundle_dir, test_path, sample_path), found))
        bundle_dir, test_path, sample_path = Path(bundle_dir), Path(test_path), Path(sample_path)
        for p in (test_path, sample_path):
            if not p.is_file() or _is_under(p, bundle_dir):
                raise InferenceBlocked(f"competition file {p} missing or inside the bundle")
        anchor["inputs"] = {"bundle_dir": str(bundle_dir), "test_parquet": str(test_path), "sample_submission": str(sample_path)}
        log(f"[inputs] bundle {bundle_dir} | {test_path} | {sample_path}")
        done(stage)

        stage = "load_frozen_code"
        frozen = load_frozen_code(bundle_dir)
        assert_frozen_modules_only(frozen.code_dir)
        anchor["casmi_infer_source"], anchor["frozen_code_dir"] = frozen.source, str(frozen.code_dir)
        import pyarrow.parquet as pq
        missing_cols = [c for c in frozen.pipeline.TEST_REQUIRED if c not in pq.read_schema(test_path).names]
        if missing_cols:
            raise InferenceBlocked(f"{test_path} lacks the test columns {missing_cols}")
        done(stage)

        stage = "bundle_presence"
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        anchor["bundle_presence"] = check_bundle_presence(bundle_dir, manifest)
        done(stage)

        stage = "load_bundle"
        runlog = frozen.pipeline.RunLog()
        t = time.time()
        bundle = frozen.pipeline.Bundle(bundle_dir, verify=False)     # no manifest hashing; frozen aggregator contract still enforced
        runlog.stage("load_bundle", time.time() - t)
        model_info = json.loads((bundle_dir / "models" / "model_info.json").read_text(encoding="utf-8"))
        check_bundle_identity(bundle.config, bundle.manifest, model_info, cfg)
        anchor["frozen_identity"] = check_frozen_identity(bundle.config, model_info, bundle.aggregator_contract)
        if len(bundle.ranker.feature_names) != N_MODEL_FEATURES:
            raise InferenceBlocked(f"expected the {N_MODEL_FEATURES} frozen model features, the bundle ships {bundle.ranker.feature_names}")
        log(f"[bundle] {bundle.manifest.get('bundle_version')} | CONFIG_HASH {bundle.config['CONFIG_HASH']} | model {bundle.ranker.model_id} | "
            f"aggregator {bundle.aggregator_contract['name']} (T={bundle.aggregator_contract['temperature']}) | code {frozen.source}")
        done(stage)

        stage = "self_test_and_backend"
        t = time.time()
        fixture = load_fixture_filenames_only(bundle_dir, frozen.selftest)
        backend = select_backend(bundle, frozen, accel, cfg.device_preference, cfg.require_gpu, log=log,
                                 runner=lambda b, be: frozen.selftest.run_selftest(b, be, fixture=fixture),
                                 allow_feature_only_drift=allow_feature_only_drift)
        runlog.stage("self_test", time.time() - t)
        fin = backend["final"]
        st_mismatches = {k: fin.get(k) for k in getattr(frozen.selftest, "MISMATCH_KEYS", ())}
        anchor.update(feature_mismatches=backend["feature_mismatches"], strict_self_test_passed=backend["strict_passed"],
                      downstream_exact=backend["downstream_exact"], accepted_feature_only_drift=backend["accepted_feature_only_drift"],
                      self_test={"status": "PASS", "backend": backend["actual_backend"], "mismatches": st_mismatches,
                                 "hard_keys": list(HARD_MISMATCH_KEYS), "policy": backend["selftest_policy"],
                                 "n_fixture_queries": fin.get("n_fixture_queries"), "attempts": backend["self_test_attempts"]})
        log(f"[self-test] PASS on {backend['actual_backend']} | strict={backend['strict_passed']} | downstream_exact={backend['downstream_exact']} | "
            f"feature_mismatches={backend['feature_mismatches']} | gpu_used={backend['gpu_used']} | fallback: {backend['fallback_reason']}")
        done(stage)

        stage = "load_test"
        import pandas as pd
        test = pd.read_parquet(test_path)
        sample = pd.read_csv(sample_path)
        if list(sample.columns) != ["molecule_id", "smiles"]:
            raise InferenceBlocked(f"{sample_path} columns {list(sample.columns)} != ['molecule_id', 'smiles']")
        if sample["molecule_id"].duplicated().any():
            raise InferenceBlocked(f"{sample_path} has duplicate molecule_id rows")
        if test["spectrum_id"].duplicated().any() or not set(sample["molecule_id"]) <= set(test["molecule_id"]):
            raise InferenceBlocked("test/sample schema problem: duplicate spectrum_id / sample molecules without spectra")
        unsupported = sorted(a for a in test["adduct"].astype(str).unique() if not bundle.adducts.supported(a))
        done(stage)

        stage = "inference"
        with GpuMemorySampler(backend["gpu_used"]) as gpu_mem:
            per_spec = frozen.pipeline.run_inference(bundle, test, runlog, time_budget_s=cfg.time_budget_s,
                                                     emergency_cap_enabled=cfg.emergency_cap_enabled, emergency_cap_n=cfg.emergency_cap_n)
            done(stage)

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
            done(stage)

        stage = "validation"
        assert_frozen_modules_only(frozen.code_dir)
        V = frozen.validation
        sub = frozen.submission.build_submission(sample, smiles_by_molecule=smiles_by_mol)
        problems = V.validate_submission(sub, sample, top_k=top_k)         # columns, sample order/coverage, no empty, <= 25, no dup rows/SMILES
        problems += V.validate_rankings(conn_final, list(sample["molecule_id"]), top_k=top_k)   # once each, non-empty, connectivity-unique
        problems += [f"{m}: {len(v)} SMILES > {top_k}" for m, v in smiles_by_mol.items() if len(v) > top_k]
        problems += [f"{m}: duplicate SMILES in the row" for m, v in smiles_by_mol.items() if len(set(v)) != len(v)]
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
        done(stage)

        stage = "write"
        sub_path = work / "submission.csv"
        frozen.submission.write_submission(sub, sub_path, backend)               # the guarded writer: refuses unless the self-test was accepted
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
            "bundle_version": bundle.manifest.get("bundle_version"), "config_bundle_version": bundle.config.get("bundle_version"),
            "CONFIG_HASH": bundle.config["CONFIG_HASH"], "model_id": bundle.ranker.model_id,
            "aggregator": c["name"], "aggregator_config": bundle.config["aggregator"], "aggregator_artifact_hash": c["selection_artifact_sha256"],
            "calibration_hash": c["calibration_sha256"], "calibration_temperature": c["temperature"],
            "candidate_universe_version": bundle.config.get("candidate_universe_version"), "casmi_infer_source": frozen.source,
            "bundle_dir": str(Path(bundle_dir).resolve()), "frozen_code_dir": str(Path(frozen.code_dir).resolve()),
            **INTEGRITY, "bundle_presence": anchor["bundle_presence"], "input_resolution": "by filename",
            "test_parquet": str(test_path), "sample_submission": str(sample_path),
            "self_test_status": "PASS", "self_test_strict_passed": backend["strict_passed"], "self_test_downstream_exact": backend["downstream_exact"],
            "self_test_accepted_feature_only_drift": backend["accepted_feature_only_drift"],
            "self_test_feature_mismatches": backend["feature_mismatches"], "self_test_policy": backend["selftest_policy"],
            "self_test_mismatches": st_mismatches, "self_test_attempts": backend["self_test_attempts"],
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
        _write_json(work / "run_report.json", report)
        done(stage)

        stage = "anchor_comparison"
        mismatches = compare_to_anchor(report, anchor_expected)
        accepted = backend["passed"] and backend["downstream_exact"]
        anchor.update(actual={k: report.get(k) for k in (anchor_expected or {})}, mismatches=mismatches,
                      self_test_status="PASS", submission_validation_status=validation_status,
                      anchor_status="PASS" if accepted and not mismatches and validation_status == "PASS" else "FAIL",
                      runtime_seconds=time.time() - t_start, backend=backend["actual_backend"], gpu_used=backend["gpu_used"],
                      fallback_reason=backend["fallback_reason"], submission_sha256=report["submission_sha256"],
                      outputs=[str(sub_path), str(work / "run_report.json"), str(work / ANCHOR_REPORT)], environment=report["environment"])
        done(stage)
        _write_json(work / ANCHOR_REPORT, anchor)
        log(f"[done] submission.csv ({len(sub)} rows) + run_report.json + {ANCHOR_REPORT} (anchor_status {anchor['anchor_status']}) "
            f"in {work} | {report['runtime_seconds'] / 60:.1f} min")
        return report
    except Exception as e:
        anchor.update(anchor_status="FAIL", failed_stage=stage, error=f"{type(e).__name__}: {e}",
                      traceback_tail=traceback.format_exc().splitlines()[-8:], runtime_seconds=time.time() - t_start,
                      runtime_config=cfg.as_dict())
        _write_json(work / ANCHOR_REPORT, anchor)
        raise
