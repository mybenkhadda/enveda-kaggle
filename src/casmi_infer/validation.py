"""Bundle integrity + submission validation. Every check returns a list of problems (empty = OK) or
raises -- nothing is silently accepted."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_infer.submission import SEPARATOR


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()[:16]


def config_hash(config):
    """CONFIG_HASH = hash of config.json WITHOUT its own CONFIG_HASH field."""
    return stable_json_hash({k: v for k, v in config.items() if k != "CONFIG_HASH"})


def verify_bundle(bundle_dir):
    """Recompute sha256 of every file listed in manifest.json + CONFIG_HASH. Returns
    `(manifest, config, problems)`; the caller must stop if `problems` is non-empty."""
    d = Path(bundle_dir)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    for rel, expected in manifest["files"].items():
        p = d / rel
        if not p.exists():
            problems.append(f"missing bundle file {rel}")
        elif sha256_file(p) != expected:
            problems.append(f"sha256 mismatch: {rel}")
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    if config.get("CONFIG_HASH") != config_hash(config):
        problems.append("config.json CONFIG_HASH does not match its content")
    if manifest.get("CONFIG_HASH") != config.get("CONFIG_HASH"):
        problems.append("manifest CONFIG_HASH != config CONFIG_HASH")
    return manifest, config, problems


def check_bundle_invariants(structures, conn_table, ref_index_offsets, ref_peaks_offsets, n_refs):
    problems = []
    if not np.all(np.diff(structures["neutral_monoisotopic_mass"].to_numpy()) >= 0):
        problems.append("structures not sorted by mass")
    keys = conn_table["connectivity_key"].astype(str).to_numpy()
    if not (np.all(keys[:-1] < keys[1:]) and (conn_table["conn_idx"].to_numpy() == np.arange(len(conn_table))).all()):
        problems.append("connectivity table must be connectivity_key-sorted with conn_idx = 0..n-1")
    if len(ref_index_offsets) != len(conn_table) + 1 or ref_index_offsets[0] != 0 or np.any(np.diff(ref_index_offsets) < 0):
        problems.append("ref_index_offsets malformed")
    if len(ref_peaks_offsets) != n_refs + 1 or np.any(np.diff(ref_peaks_offsets) < 0):
        problems.append("ref_peaks_offsets malformed")
    return problems


def validate_submission(sub, sample_submission, top_k=25, min_predictions=1):
    """Problems list for: column set/order, molecule coverage + order, empty / null / 'nan' SMILES,
    duplicate SMILES within a row, > top_k predictions, duplicate molecule rows."""
    problems = []
    if list(sub.columns) != list(sample_submission.columns):
        problems.append(f"columns {list(sub.columns)} != sample {list(sample_submission.columns)}")
        return problems
    if sub["molecule_id"].duplicated().any():
        problems.append("duplicate molecule_id rows")
    if list(sub["molecule_id"]) != list(sample_submission["molecule_id"]):
        missing = set(sample_submission["molecule_id"]) - set(sub["molecule_id"])
        problems.append(f"molecule_id order/coverage differs from sample ({len(missing)} missing)")
    for mid, s in zip(sub["molecule_id"], sub["smiles"]):
        if not isinstance(s, str) or not s:
            problems.append(f"{mid}: empty/null prediction")
            continue
        parts = s.split(SEPARATOR)
        if len(parts) < min_predictions or len(parts) > top_k:
            problems.append(f"{mid}: {len(parts)} predictions (allowed {min_predictions}-{top_k})")
        if any((not p) or p.lower() in ("nan", "none") for p in parts):
            problems.append(f"{mid}: null SMILES entry")
        if len(set(parts)) != len(parts):
            problems.append(f"{mid}: duplicate SMILES within ranking")
    return problems


# ---------------------------------------------------------------------------------------------
# v6.3: aggregator + calibration contract (checked when a Bundle is loaded; never refit, never defaulted)
# ---------------------------------------------------------------------------------------------

class AggregatorContractError(RuntimeError):
    """The configured aggregator / its frozen calibration is missing, inconsistent or not the locked one."""


def verify_aggregator_contract(config, bundle_dir):
    """Returns {name, aggregator_id, needs_prob, temperature, calibration_sha256, selection_artifact_sha256}.

    * the config must NAME its aggregator (no implicit RRF); `aggregator_name` (if present) must agree;
    * a probability aggregator (A7 ...) needs `aggregator.temperature` == `calibration_temperature`, finite > 0,
      and the shipped calibration + selection artifacts whose sha256s the config records; the calibration file's
      fitted temperature and the selection record's aggregator / temperature / spectrum model must match exactly.
    Nothing is refit: a mismatch raises."""
    from casmi_infer.aggregation import canonical_aggregator_id, make_aggregator
    agg_cfg = config.get("aggregator")
    if not isinstance(agg_cfg, dict) or not agg_cfg.get("name"):
        raise AggregatorContractError("config.json names no aggregator -- RRF is never implied; re-export the bundle (11_00)")
    if config.get("aggregator_name") not in (None, agg_cfg["name"]):
        raise AggregatorContractError(f"aggregator_name {config.get('aggregator_name')!r} != aggregator.name {agg_cfg['name']!r}")
    agg = make_aggregator(agg_cfg)
    aid = canonical_aggregator_id(agg_cfg["name"])
    out = {"name": agg_cfg["name"], "aggregator_id": aid, "needs_prob": bool(agg.needs_prob), "temperature": None,
           "calibration_sha256": None, "selection_artifact_sha256": None}
    if not agg.needs_prob:
        return out
    t = agg_cfg.get("temperature")
    if t is None or not np.isfinite(float(t)) or float(t) <= 0:
        raise AggregatorContractError(f"{agg_cfg['name']} needs a frozen calibration temperature; config has {t!r}")
    if config.get("calibration_temperature") != t:
        raise AggregatorContractError(f"calibration_temperature {config.get('calibration_temperature')!r} != aggregator.temperature {t!r}")
    d = Path(bundle_dir)
    for key in ("calibration", "selection_artifact"):
        rel, sha = config.get(f"{key}_path"), config.get(f"{key}_sha256")
        if not rel or not sha:
            raise AggregatorContractError(f"config lacks {key}_path / {key}_sha256 (calibration metadata missing)")
        if not (d / rel).exists():
            raise AggregatorContractError(f"missing {rel} in the bundle (calibration metadata missing)")
        if sha256_file(d / rel) != sha:
            raise AggregatorContractError(f"{rel} sha256 != config {key}_sha256 (calibration metadata mismatched)")
    cal = json.loads((d / config["calibration_path"]).read_text(encoding="utf-8"))
    sel = json.loads((d / config["selection_artifact_path"]).read_text(encoding="utf-8"))
    if (cal.get("temperature_fit") or {}).get("temperature") != t:
        raise AggregatorContractError(f"calibration artifact temperature {(cal.get('temperature_fit') or {}).get('temperature')!r} != config {t!r}")
    if canonical_aggregator_id(sel.get("selected_aggregator", "")) != aid:
        raise AggregatorContractError(f"selection artifact locks {sel.get('selected_aggregator')!r}, config runs {aid!r}")
    if sel.get("temperature") != t or (sel.get("aggregator_config") or {}).get("temperature") != t:
        raise AggregatorContractError("selection artifact temperature != config temperature")
    if config.get("model_id") is not None and sel.get("spectrum_model") != config["model_id"]:
        raise AggregatorContractError(f"selection artifact was made for {sel.get('spectrum_model')!r}, bundle runs {config['model_id']!r}")
    out.update(temperature=t, calibration_sha256=config["calibration_sha256"], selection_artifact_sha256=config["selection_artifact_sha256"])
    return out


# ---------------------------------------------------------------------------------------------
# v6.3: final-output checks beyond the sample-shape validator
# ---------------------------------------------------------------------------------------------

def validate_rankings(conn_by_molecule, molecule_ids, top_k=25):
    """Per-molecule connectivity lists: every molecule exactly once, no null id, non-empty, <= top_k,
    connectivity-unique."""
    problems = []
    ids = list(molecule_ids)
    if any(m is None or (isinstance(m, float) and np.isnan(m)) for m in ids):
        problems.append("null molecule_id")
    if len(set(ids)) != len(ids):
        problems.append("duplicate molecule_id in the output order")
    for m in ids:
        c = conn_by_molecule.get(m)
        if not c:
            problems.append(f"{m}: empty ranking")
            continue
        if len(c) > top_k:
            problems.append(f"{m}: {len(c)} candidates > {top_k}")
        if len(set(c)) != len(c):
            problems.append(f"{m}: duplicate connectivity in ranking")
    extra = set(conn_by_molecule) - set(ids)
    if extra:
        problems.append(f"{len(extra)} ranked molecules not in the sample submission")
    return problems


def check_smiles_parse(smiles, export_validation=None, parse_fn=None):
    """`smiles`: every emitted SMILES. `parse_fn(s) -> bool` (the notebook passes an RDKit parser when RDKit is
    importable; casmi_infer itself stays RDKit-free) -> all must parse (status PARSER_PASS / FAIL). Without a
    parser the bundle's export-time RDKit validation is used: it must exist and no emitted SMILES may be in its
    failure list (status EXPORT_VALIDATED / FAIL / UNVERIFIED). Returns (status, problems)."""
    smiles = list(dict.fromkeys(smiles))
    if parse_fn is not None:
        bad = [s for s in smiles if not parse_fn(s)]
        return ("PARSER_PASS" if not bad else "FAIL"), [f"unparsable SMILES: {s}" for s in bad[:20]]
    if not export_validation or export_validation.get("n_checked", 0) <= 0:
        return "UNVERIFIED", ["RDKit unavailable here and the bundle carries no export-time SMILES validation"]
    failed = set(export_validation.get("failed_smiles", []))
    bad = [s for s in smiles if s in failed]
    return ("EXPORT_VALIDATED" if not bad else "FAIL"), [f"SMILES failed RDKit at export: {s}" for s in bad[:20]]


RUN_REPORT_REQUIRED = (
    "bundle_version", "CONFIG_HASH", "model_id", "aggregator", "aggregator_artifact_hash", "calibration_hash", "candidate_universe_version",
    "self_test_status", "backend", "numba_available", "numpy_fallback_used", "n_spectra", "n_molecules", "candidate_pair_count",
    "fallback_counts", "runtime_seconds", "peak_ram", "submission_rows", "submission_validation_status")


def assert_run_report_complete(report, required=RUN_REPORT_REQUIRED):
    """Every required key present; only hash fields of an aggregator without calibration may be None."""
    missing = [k for k in required if k not in report]
    nullable = {"aggregator_artifact_hash", "calibration_hash"} if report.get("aggregator") in ("RRF", "A6_rrf") else set()
    empty = [k for k in required if k in report and report[k] is None and k not in nullable]
    if missing or empty:
        raise ValueError(f"run_report incomplete: missing {missing}, null {empty}")
    return True


def environment_report():
    rep = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "pyarrow", "lightgbm", "numba", "psutil"):
        try:
            rep[mod] = __import__(mod).__version__
        except Exception as e:  # recorded, not swallowed
            rep[mod] = f"unavailable ({type(e).__name__})"
    return rep


def network_is_disabled(host="8.8.8.8", port=53, timeout=2.0):
    """P4 helper: True if an outbound TCP connection fails (used ONLY by the offline-environment
    check the user runs; the Kaggle notebook never calls it and makes no network call)."""
    import socket
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return False
    except OSError:
        return True


def kaggle_offline_environment_proven(env=None, input_root=None):
    """True ONLY inside a real Kaggle kernel run (Kaggle sets KAGGLE_KERNEL_RUN_TYPE; competition
    submissions run with internet disabled) reading the real /kaggle/input -- never for a local
    dry run (CASMI_KAGGLE_INPUT override) or a local preflight. No network call is made."""
    import os
    env = os.environ if env is None else env
    return bool(env.get("KAGGLE_KERNEL_RUN_TYPE")) and not env.get("CASMI_KAGGLE_INPUT") and str(input_root or "/kaggle/input").replace("\\", "/") == "/kaggle/input"


def p4_status(local_checks_passed, kaggle_offline_proven):
    """Separate labels: a local Windows preflight can never prove the Kaggle offline environment."""
    return {"P4_LOCAL_PREFLIGHT": "PASS" if local_checks_passed else "FAIL",
            "P4_KAGGLE_OFFLINE": "PROVEN" if kaggle_offline_proven else "NOT_PROVEN"}
