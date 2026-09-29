"""Offline Kaggle bundle export (local only; never shipped). Writes `bundle/` for `casmi_infer`.

Every numerical choice is SERIALIZED from the training source, not re-typed:
  structures     <- data/interim/candidate_generation/molecule_mass_variants (the exact library the
                    training MassIndex was built from; one row per MASS VARIANT, float64 masses)
  adducts.json   <- casmi.chemistry.adducts.parse_adduct over every train adduct + observed test adducts
  ppm_windows    <- candidate_generation_contract.json (the training retrieval tolerance) + the
                    documented inference-only fallbacks (100 ppm, 200 ppm, nearest 25)
  ref peaks      <- remove_invalid_peaks -> truncate_top_peaks(max_peaks) (deterministic top-N), float64
  ref_meta       <- train_spectrum_metadata fields exactly as `casmi.qcr.context.SpectrumLookups`
                    feeds compat ordering / eligibility (adduct, ionization_mode, ce_mean, instrument,
                    ingest_lib) + the T1 peak hash of the identity representation
  models         <- LightGBM text boosters + feature_names.json in the frozen training order
  code/          <- casmi_infer + the verbatim `casmi` subset in `casmi_infer.SHARED_CASMI_MODULES`

Swapping the model or the aggregator is a full re-export by src/11_00_export_bundle.ipynb (v6.3): the config
binds model id, feature order, aggregator, frozen calibration and the self-test fixture together.
"""
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.io.parquet import scan_batches
from casmi.spectra.deduplication import compute_peak_hash
from casmi.spectra.preprocessing import remove_invalid_peaks, truncate_top_peaks

PPM_FALLBACKS = (100.0, 200.0)
NEAREST_N = 25


def _sha(path):
    from casmi_infer.validation import sha256_file
    return sha256_file(path)


# ---------------------------------------------------------------------------------------------
# structures
# ---------------------------------------------------------------------------------------------

def export_structures(mass_variants, out_dir):
    mv = mass_variants.dropna(subset=["exact_mass"]).copy()
    conn = (mv.sort_values(["connectivity_key", "mass_variant_id"]).drop_duplicates("connectivity_key")
            [["connectivity_key", "representative_smiles", "molecular_formula"]].sort_values("connectivity_key").reset_index(drop=True))
    conn["conn_idx"] = np.arange(len(conn), dtype=np.int64)
    if conn["representative_smiles"].isna().any():
        raise ValueError(f"{int(conn['representative_smiles'].isna().sum())} connectivities lack a representative SMILES")
    idx_of = dict(zip(conn["connectivity_key"], conn["conn_idx"]))
    st = pd.DataFrame({"connectivity_key": mv["connectivity_key"].to_numpy(), "representative_smiles": mv["representative_smiles"].to_numpy(),
                       "molecular_formula": mv["molecular_formula"].to_numpy(), "neutral_monoisotopic_mass": mv["exact_mass"].to_numpy(np.float64),
                       "mass_variant_id": mv["mass_variant_id"].to_numpy()})
    st["conn_idx"] = st["connectivity_key"].map(idx_of).astype(np.int64)
    st = st.sort_values(["neutral_monoisotopic_mass", "connectivity_key", "mass_variant_id"], kind="mergesort").reset_index(drop=True)
    out_dir = Path(out_dir)
    st.to_parquet(out_dir / "structures.parquet", index=False)
    np.save(out_dir / "structures_mass.npy", st["neutral_monoisotopic_mass"].to_numpy(np.float64))
    conn.to_parquet(out_dir / "connectivities.parquet", index=False)
    return st, conn


# ---------------------------------------------------------------------------------------------
# adducts + windows
# ---------------------------------------------------------------------------------------------

def export_adducts(train_adducts, test_adducts, out_dir):
    from casmi_infer.adducts import export_adduct_rules
    payload = export_adduct_rules(train_adducts, observed_in_test=test_adducts)
    unsupported_test = [a for a, r in payload["rules"].items() if r["observed_in_test"] and not r["supported"]]
    payload["unsupported_observed_test_adducts"] = unsupported_test
    (Path(out_dir) / "adducts.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def export_ppm_windows(contract, adduct_payload, out_dir, candidate_generation_config=None):
    primary = float(contract["molecule_policy"]["tolerance_ppm"])
    max_std = float(getattr(candidate_generation_config, "max_tolerance_ppm", primary)) if candidate_generation_config else primary
    payload = {
        "source": "candidate_generation_contract.json molecule_policy.tolerance_ppm (global; the adaptive per-group policy was evaluated in notebook 03 but not adopted by the contract)",
        "default_primary_ppm": primary,
        "per_adduct_primary_ppm": {a: primary for a, r in adduct_payload["rules"].items() if r["supported"]},
        "max_standard_ppm": max_std,
        "fallback_ppm": list(PPM_FALLBACKS),
        "nearest_n": NEAREST_N,
        "fallback_sequence": ["primary", *[f"ppm_{p:g}" for p in PPM_FALLBACKS], f"nearest_{NEAREST_N}"],
        "fallback_note": "fallback levels are INFERENCE-ONLY (training never needed them: every training query had a non-empty primary window by construction); every use is logged",
    }
    (Path(out_dir) / "ppm_windows.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------------------------
# reference library
# ---------------------------------------------------------------------------------------------

def export_reference_library(train_meta, peak_store_path, conn_table, out_dir, max_peaks, batch_size=50_000,
                             compute_hash=True, log_every=250_000):
    """Streams the peak store once (bounded RAM): per spectrum, identity peaks -> T1 hash; similarity
    peaks appended to raw float64 files, then copied into .npy via memmap in chunks.
    `train_meta` must be in train.parquet row order with `train_spectrum_id == "train_<row>"`."""
    out_dir = Path(out_dir)
    tm = train_meta.reset_index(drop=True)
    expected = "train_" + pd.Series(np.arange(len(tm))).astype(str)
    if not (tm["train_spectrum_id"].astype(str).to_numpy() == expected.to_numpy()).all():
        raise ValueError("train_meta is not in train.parquet row order")
    tmp_mz, tmp_it = out_dir / "_ref_peaks_mz.f8", out_dir / "_ref_peaks_int.f8"
    counts = np.zeros(len(tm), dtype=np.int64)
    hashes = np.empty(len(tm), dtype=object)
    row = 0
    with open(tmp_mz, "wb") as fmz, open(tmp_it, "wb") as fit:
        for _, batch in scan_batches(peak_store_path, ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"], batch_size=batch_size):
            for mzs, ints, prec in zip(batch["ms2_mzs"], batch["ms2_normalized_intensities"], batch["precursor_mz"]):
                mz, it = remove_invalid_peaks(mzs if mzs is not None else [], ints if ints is not None else [])
                hashes[row] = compute_peak_hash(mz, it, prec) if compute_hash else None
                s_mz, s_it = truncate_top_peaks(mz, it, max_peaks=max_peaks)
                s_mz.astype(np.float64).tofile(fmz)
                s_it.astype(np.float64).tofile(fit)
                counts[row] = len(s_mz)
                row += 1
                if log_every and row % log_every == 0:
                    print(f"[export] {row:,}/{len(tm):,} reference spectra")
            del batch
    if row != len(tm):
        raise RuntimeError(f"peak store rows ({row}) != metadata rows ({len(tm)})")
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    np.save(out_dir / "ref_peaks_offsets.npy", offsets)
    for src, name in ((tmp_mz, "ref_peaks_mz.npy"), (tmp_it, "ref_peaks_int.npy")):
        raw = np.memmap(src, dtype=np.float64, mode="r")
        dst = np.lib.format.open_memmap(out_dir / name, mode="w+", dtype=np.float64, shape=(int(offsets[-1]),))
        step = 50_000_000
        for a in range(0, len(raw), step):
            dst[a:a + step] = raw[a:a + step]
        dst.flush()
        del dst, raw
        src.unlink()

    idx_of = dict(zip(conn_table["connectivity_key"], conn_table["conn_idx"]))
    ce = tm["ce_mean"].astype(float)
    meta = pd.DataFrame({
        "ref_row": np.arange(len(tm), dtype=np.int64), "ref_spectrum_id": tm["train_spectrum_id"].astype(str).to_numpy(),
        "connectivity_key": tm["connectivity_key"].astype(object).to_numpy(),
        "source": tm["ingest_lib"].astype(object).to_numpy(), "instrument": tm["instrument_type"].astype(object).to_numpy(),
        "adduct": tm["adduct"].astype(object).to_numpy(), "polarity": tm["ionization_mode"].astype(object).to_numpy(),
        "collision_energy": ce.to_numpy(), "collision_energy_units": np.where(ce.notna(), "eV", None),
        "precursor_mz": tm["precursor_mz"].astype(float).to_numpy(), "peak_hash": hashes, "n_peaks_similarity": counts,
    })
    meta["conn_idx"] = meta["connectivity_key"].map(idx_of).astype("Int64")
    meta["sid_rank"] = meta["ref_spectrum_id"].rank(method="first").astype(np.int64) - 1  # lexicographic id order = training's final tie-break
    meta.to_parquet(out_dir / "ref_meta.parquet", index=False)

    has = meta["conn_idx"].notna().to_numpy()
    ci = meta.loc[has, "conn_idx"].astype(np.int64).to_numpy()
    rows = meta.loc[has, "ref_row"].to_numpy(np.int64)
    order = np.lexsort((rows, ci))
    np.save(out_dir / "ref_index_ids.npy", rows[order].astype(np.int64))
    np.save(out_dir / "ref_index_offsets.npy", np.concatenate([[0], np.cumsum(np.bincount(ci, minlength=len(conn_table)))]).astype(np.int64))
    return {"n_refs": int(len(tm)), "n_peaks": int(offsets[-1]), "n_refs_without_library_connectivity": int((~has).sum())}


# ---------------------------------------------------------------------------------------------
# models, config, code, manifest
# ---------------------------------------------------------------------------------------------

def export_models(model_dir, bundle_dir, model_id, feature_names, tag="v1", extra_info=None):
    """Replace `bundle/models/` with the fold boosters of `model_dir` (fold_<k>.txt, v4b/v5 layout)."""
    src = Path(model_dir)
    dst = Path(bundle_dir) / "models"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    files = sorted(src.glob("fold_*.txt"), key=lambda p: int(p.stem.split("_")[1]))
    if len(files) != 5:
        raise ValueError(f"expected 5 fold boosters in {src}, found {len(files)}")
    hashes = {}
    for f in files:
        k = int(f.stem.split("_")[1])
        out = dst / f"{tag}_fold{k}.txt"
        shutil.copyfile(f, out)
        hashes[out.name] = _sha(out)
    (dst / "feature_names.json").write_text(json.dumps(list(feature_names)), encoding="utf-8")
    info = {"model_id": model_id, "source_dir": str(src), "n_folds": len(files), "file_sha256": hashes,
            "scoring": "mean of fold predictions", "exported_at": datetime.now(timezone.utc).isoformat(), **(extra_info or {})}
    (dst / "model_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return info


RRF_BASELINE_AGGREGATOR = {"name": "RRF", "k": 60, "tie_rule": ["RRF DESC", "best individual rank ASC", "connectivity_key ASC"]}
SPECTRUM_TIE_RULE = ["score DESC", "abs_mass_error_ppm ASC", "connectivity_key ASC"]


def build_config(similarity_config, aggregator, top_k=25, bundle_version=None, model=None, calibration=None, candidate_universe_version=None):
    """config.json. `aggregator` is REQUIRED (v6.3): there is no default, so RRF never enters a bundle unless it is
    passed explicitly (`RRF_BASELINE_AGGREGATOR`). v6.3 promotion fields (all inside CONFIG_HASH):
      model        {model_id, feature_order, feature_order_hash, protocol, protocol_hash}
      calibration  {temperature, calibration_path, calibration_sha256, selection_artifact_path, selection_artifact_sha256, lock_sha256}
    CONFIG_HASH is computed here, i.e. only when the user runs the export."""
    from casmi_infer.validation import config_hash
    from casmi.qcr.context import config_hash as sim_config_hash
    if not isinstance(aggregator, dict) or not aggregator.get("name"):
        raise ValueError("build_config needs an explicit aggregator config (no implicit RRF default)")
    cfg = {
        "bundle_format": "casmi-modeA-bundle-1",
        "mode": "CLOSED-WORLD / CLASS-1 ONLY (training structure library; no external candidates, no Mode B)",
        "similarity": {"bin_width_da": similarity_config["bin_width_da"], "peak_tol_da": similarity_config["peak_tol_da"],
                       "max_peaks_similarity": similarity_config["max_peaks_similarity"]},
        "similarity_config_hash": sim_config_hash(similarity_config),
        "intensity_transform": "none: ms2_normalized_intensities as provided; remove_invalid_peaks; deterministic top-N (intensity DESC, m/z ASC); peaks re-sorted by m/z; float64",
        "peak_hash": {"mz_decimals": 4, "intensity_decimals": 3, "precursor_decimals": 3, "representation": "identity (cleaned, untruncated)"},
        "reference_compatibility_ordering": "compat_sort_key v4a: same_adduct, same_ion_mode, ce_comparable, |ce_diff|, same_instrument, reference_spectrum_id",
        "reference_compatibility_version": _module_hash("casmi.spectra.reference_selection"),
        "protocol": "mirror_aware",
        "exclude_sources": [],
        "identity_tiers_applied": {"T1": True, "T2": False, "T2_note": "identity peaks of references are not shipped; see casmi_infer.compat"},
        "top_reference_count": 5,
        "rank_tie_rule": list(SPECTRUM_TIE_RULE),
        "candidate_retrieval": "ppm_windows.json; mass-variant rows deduplicated to connectivity by min abs ppm",
        "aggregator": aggregator,
        "top_k_submission": top_k,
        "pad_to_top_k_with_nearest_mass": True,
    }
    if bundle_version is not None:
        cfg["bundle_version"] = bundle_version
    cfg["aggregator_name"] = aggregator["name"]
    cfg["aggregator_params"] = {**aggregator.get("params", {}), **{k: aggregator[k] for k in ("k", "eps") if k in aggregator}}
    cfg["tie_rules"] = {"spectrum": list(SPECTRUM_TIE_RULE), "molecule": aggregator.get("tie_rule"),
                        "a7_spectrum_choice": ["top-1 calibrated probability DESC", "spectrum_id ASC"] if aggregator["name"] == "MOST_CONFIDENT_SPECTRUM" else None,
                        "submission": "connectivity dedup BEFORE top-k; representative SMILES per connectivity"}
    if model is not None:
        cfg.update(model_id=model["model_id"], feature_order=list(model["feature_order"]), feature_order_hash=model["feature_order_hash"],
                   protocol=model["protocol"], protocol_hash=model["protocol_hash"],
                   inference_eligibility={"exclude_sources": [], "T1": "excluded by peak hash", "T2": "not applied (deployment gap WAIVED_PENDING_EVIDENCE)",
                                          "T3_T4": "allowed", "note": "the production reading of the frozen training protocol for unpublished spectra"})
    if calibration is not None:
        cfg.update(calibration_temperature=calibration["temperature"], calibration_path=calibration["calibration_path"],
                   calibration_sha256=calibration["calibration_sha256"], selection_artifact_path=calibration["selection_artifact_path"],
                   selection_artifact_sha256=calibration["selection_artifact_sha256"], selection_lock_sha256=calibration.get("lock_sha256"))
    if candidate_universe_version is not None:
        cfg["candidate_universe_version"] = candidate_universe_version
    cfg["CONFIG_HASH"] = config_hash(cfg)
    return cfg


# ---------------------------------------------------------------------------------------------
# v6.3: promotion of the LOCKED molecule aggregator (A7) into a new, versioned bundle
# ---------------------------------------------------------------------------------------------

BUNDLE_VERSION = "v2-A7"
PROMOTED_AGGREGATOR_ID = "A7_most_confident_spectrum"
PROMOTED_AGGREGATOR_NAME = "MOST_CONFIDENT_SPECTRUM"
PROMOTION_PROTOCOL = "test_simulated_strict"
PROMOTION_MODEL_ID = "V1_TL_1K_TESTSIM_STRICT"
AGGREGATION_DIR = "aggregation"


class AggregatorPromotionError(RuntimeError):
    """The locked selection artifacts are missing / inconsistent / not for the frozen model -- nothing is exported."""


def _read_json(path, what):
    p = Path(path)
    if not p.exists():
        raise AggregatorPromotionError(f"{what} not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def load_locked_aggregator(selected_path, lock_path, calibration_path, freeze, expected_id=PROMOTED_AGGREGATOR_ID,
                           expected_protocol=PROMOTION_PROTOCOL, expected_model_id=PROMOTION_MODEL_ID):
    """Read the 11_02 artifacts and refuse unless they lock `expected_id` on `expected_protocol` (primary only) for
    the FROZEN `expected_model_id`, with ONE temperature everywhere (selection record, its aggregator config, the
    lock's alphas, the calibration fit) and a calibration fitted without HOST. Never falls back to RRF.
    Returns {"aggregator": bundle aggregator block, "temperature", "sha256": {...}, "paths": {...}}."""
    from casmi_infer.aggregation import canonical_aggregator_id
    sel = _read_json(selected_path, "selected_aggregator.json")
    lock = _read_json(lock_path, "aggregator lock")
    cal = _read_json(calibration_path, "calibration.json")
    problems = []
    try:
        sel_id = canonical_aggregator_id(sel.get("selected_aggregator", ""))
    except ValueError as e:
        sel_id = None
        problems.append(str(e))
    if sel_id != expected_id:
        problems.append(f"selected aggregator {sel.get('selected_aggregator')!r} != required {expected_id!r}")
    if list(sel.get("selection_protocols") or []) != [expected_protocol]:
        problems.append(f"selection protocols {sel.get('selection_protocols')!r} != [{expected_protocol!r}] (primary only)")
    if freeze.get("freeze_status") != "FROZEN" or freeze.get("model_id") != expected_model_id or freeze.get("protocol") != expected_protocol:
        problems.append(f"freeze record is {freeze.get('model_id')!r}/{freeze.get('freeze_status')!r}/{freeze.get('protocol')!r}, "
                        f"expected {expected_model_id!r}/FROZEN/{expected_protocol!r}")
    if sel.get("spectrum_model") != freeze.get("model_id"):
        problems.append(f"selection was made for spectrum model {sel.get('spectrum_model')!r}, frozen model is {freeze.get('model_id')!r}")
    t = sel.get("temperature")
    temps = {"selected_aggregator.temperature": t, "selected_aggregator.aggregator_config.temperature": (sel.get("aggregator_config") or {}).get("temperature"),
             "lock.alphas.temperature": (lock.get("alphas") or {}).get("temperature"), "calibration.temperature_fit.temperature": (cal.get("temperature_fit") or {}).get("temperature")}
    if t is None or not np.isfinite(float(t)) or float(t) <= 0 or any(v != t for v in temps.values()):
        problems.append(f"temperature missing or inconsistent across the locked artifacts: {temps}")
    if cal.get("host_used") is not False:
        problems.append("calibration.json does not certify host_used = false")
    if lock.get("selected_model_id") != sel.get("selected_aggregator"):
        problems.append(f"lock selects {lock.get('selected_model_id')!r}, selection record {sel.get('selected_aggregator')!r}")
    if lock.get("preregistration_sha256") != sel.get("preregistration_sha256"):
        problems.append("lock and selection record were made under different pre-registrations")
    if problems:
        raise AggregatorPromotionError("locked aggregator cannot be promoted:\n  " + "\n  ".join(problems))
    agg = {"name": PROMOTED_AGGREGATOR_NAME if expected_id == PROMOTED_AGGREGATOR_ID else expected_id, "aggregator_id": expected_id, "params": {},
           "temperature": t, "tie_rule": ["calibrated probability DESC (== frozen spectrum rank)", "abs_mass_error_ppm ASC", "connectivity_key ASC"],
           "spectrum_choice": ["top-1 calibrated probability DESC", "spectrum_id ASC"],
           "selection": {"population": "MOL_DEV", "protocol": expected_protocol, "locked_before_host": True, "rule_sha256": sel.get("preregistration_sha256")}}
    return {"aggregator": agg, "temperature": t,
            "sha256": {"selected_aggregator.json": _sha(selected_path), "calibration.json": _sha(calibration_path), "aggregator_lock.json": _sha(lock_path)},
            "paths": {"selected_aggregator.json": Path(selected_path), "calibration.json": Path(calibration_path), "aggregator_lock.json": Path(lock_path)}}


def verify_bundle_models_match_freeze(bundle_dir, freeze):
    """The exported fold boosters / feature order must be byte-identical to the freeze record."""
    from casmi.ranking.freeze import feature_order_hash
    d = Path(bundle_dir) / "models"
    info = _read_json(d / "model_info.json", "bundle models/model_info.json")
    names = _read_json(d / "feature_names.json", "bundle models/feature_names.json")
    problems = []
    if info.get("model_id") != freeze["model_id"]:
        problems.append(f"bundle model {info.get('model_id')!r} != frozen {freeze['model_id']!r}")
    fold = lambda k: int(re.search(r"(\d+)\.txt$", k).group(1))
    frozen = [v for _, v in sorted(freeze["model_hashes"].items(), key=lambda kv: fold(kv[0]))]
    shipped = [_sha(p) for p in sorted(d.glob("*_fold*.txt"), key=lambda p: fold(p.name))]
    if frozen != shipped:
        problems.append("bundle fold boosters are not byte-identical to the frozen model files")
    if list(names) != list(freeze["feature_names"]) or feature_order_hash(names) != freeze["feature_order_hash"]:
        problems.append("bundle feature order != frozen feature order")
    if problems:
        raise AggregatorPromotionError("bundle models do not match the frozen spectrum model:\n  " + "\n  ".join(problems))
    return {"model_id": freeze["model_id"], "n_folds": len(shipped), "feature_order_hash": freeze["feature_order_hash"]}


def export_aggregator_artifacts(bundle_dir, promotion):
    """Copy the locked selection + calibration artifacts into bundle/aggregation/ (byte-identical, verified)."""
    out = Path(bundle_dir) / AGGREGATION_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name, src in promotion["paths"].items():
        shutil.copyfile(src, out / name)
        if _sha(out / name) != promotion["sha256"][name]:
            raise AggregatorPromotionError(f"copied {name} differs from its source")
    return {"calibration_path": f"{AGGREGATION_DIR}/calibration.json", "calibration_sha256": promotion["sha256"]["calibration.json"],
            "selection_artifact_path": f"{AGGREGATION_DIR}/selected_aggregator.json",
            "selection_artifact_sha256": promotion["sha256"]["selected_aggregator.json"],
            "lock_sha256": promotion["sha256"]["aggregator_lock.json"], "temperature": promotion["temperature"]}


def candidate_universe_version(bundle_dir):
    """Content-derived id of the shipped candidate universe (connectivities + mass-variant structures)."""
    d = Path(bundle_dir)
    return f"closed_world_train_library:{_sha(d / 'connectivities.parquet')[:12]}:{_sha(d / 'structures.parquet')[:12]}"


def archive_bundle_metadata(bundle_dir, archive_root):
    """Before a NEW version overwrites bundle/, copy the old version's identity files (config, manifest, README,
    models/, selftest/, aggregation/, code/) to <archive_root>/<old bundle_version or CONFIG_HASH>. The large
    reference-library files are content-stamped (_reference_library_inputs.json) and reused, so nothing is lost.
    Returns the archive path, or None when bundle/ holds no previous export."""
    d = Path(bundle_dir)
    if not (d / "config.json").exists():
        return None
    old_cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    old_man = json.loads((d / "manifest.json").read_text(encoding="utf-8")) if (d / "manifest.json").exists() else {}
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(old_man.get("bundle_version") or old_cfg.get("bundle_version") or "v1")) + "__" + old_cfg.get("CONFIG_HASH", "nohash")
    dest = Path(archive_root) / tag
    if dest.exists():
        return dest
    dest.mkdir(parents=True)
    for name in ("config.json", "manifest.json", "README.md", "_reference_library_inputs.json"):
        if (d / name).exists():
            shutil.copyfile(d / name, dest / name)
    for sub in ("models", "selftest", AGGREGATION_DIR, "code"):
        if (d / sub).exists():
            shutil.copytree(d / sub, dest / sub, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def validate_representative_smiles(conn_table, out_path=None):
    """Export-time RDKit parse of every representative SMILES (local; inference stays RDKit-free). Failures are
    RECORDED (never silently dropped -- that would change the candidate universe); the Kaggle validator refuses
    to emit any of them. Returns {n_checked, n_failed, failed_smiles, rdkit_version}."""
    from rdkit import Chem, RDLogger, rdBase
    RDLogger.DisableLog("rdApp.*")
    smi = conn_table["representative_smiles"].astype(str).unique()
    failed = sorted(s for s in smi if Chem.MolFromSmiles(s) is None)
    rec = {"n_checked": int(len(smi)), "n_failed": len(failed), "failed_smiles": failed, "rdkit_version": rdBase.rdkitVersion}
    if out_path is not None:
        Path(out_path).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def _module_hash(modname):
    import hashlib
    import importlib
    import inspect
    return hashlib.sha256(inspect.getsource(importlib.import_module(modname)).encode("utf-8")).hexdigest()[:16]


def export_code(src_root, bundle_dir):
    from casmi_infer import SHARED_CASMI_MODULES
    src_root, code = Path(src_root), Path(bundle_dir) / "code"
    if code.exists():
        shutil.rmtree(code)
    for rel in SHARED_CASMI_MODULES:
        (code / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_root / rel, code / rel)
    (code / "casmi_infer").mkdir(parents=True)
    for f in sorted((src_root / "casmi_infer").glob("*.py")):
        shutil.copyfile(f, code / "casmi_infer" / f.name)
    return sorted(str(p.relative_to(bundle_dir)).replace("\\", "/") for p in code.rglob("*.py"))


def apply_aggregator_config(bundle_dir, aggregator_cfg, similarity_config):
    """v6.2: write a DEV-selected aggregator into bundle/config.json (new CONFIG_HASH; every other field is
    rebuilt by `build_config` exactly as the export does) and refresh the manifest. Refuses a probability
    aggregator without a temperature. Returns (old_config, new_config)."""
    from casmi_infer.aggregation import make_aggregator
    if make_aggregator(aggregator_cfg).needs_prob and not aggregator_cfg.get("temperature"):
        raise ValueError(f"{aggregator_cfg.get('name')} needs a fitted temperature in its bundle config")
    # v6.3: a config-only swap would ship a probability aggregator WITHOUT its calibration / selection artifacts,
    # model binding and molecule-level self-test fixture -- promotion goes through the full export instead.
    raise AggregatorPromotionError("config-only aggregator swaps are retired (v6.3): run src/11_00_export_bundle.ipynb "
                                   "(PROMOTE_LOCKED_AGGREGATOR = True), which ships calibration + selection artifacts and a new fixture")


def refresh_manifest(bundle_dir, source_artifacts=None, code_version=None, extra=None):
    """(Re)hash every file under bundle/ (except manifest.json itself)."""
    d = Path(bundle_dir)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    files = {str(p.relative_to(d)).replace("\\", "/"): _sha(p) for p in sorted(d.rglob("*"))
             if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts}
    old = json.loads((d / "manifest.json").read_text(encoding="utf-8")) if (d / "manifest.json").exists() else {}
    manifest = {"bundle_format": cfg["bundle_format"], "created_at": old.get("created_at", datetime.now(timezone.utc).isoformat()),
                "refreshed_at": datetime.now(timezone.utc).isoformat(), "code_version": code_version or old.get("code_version"),
                "CONFIG_HASH": cfg["CONFIG_HASH"], "source_artifacts": source_artifacts or old.get("source_artifacts", {}),
                "files": files, **(extra or {})}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


README = """# CASMI 2026 Mode-A inference bundle

CLOSED-WORLD / CLASS-1 ONLY. Candidates = training structure library. No external candidates, no Mode B.
Submission 1 is an anchor, not the final solution.

Load with: sys.path.insert(0, '<bundle>/code'); from casmi_infer.pipeline import Bundle; Bundle('<bundle>')
Integrity: manifest.json lists sha256 of every file; config.json carries CONFIG_HASH (hash of itself minus that key).
Swap the model: replace models/* (casmi.bundle_export.export_models) then refresh_manifest.
Aggregator (v6.3, bundle v2-A7): MOST_CONFIDENT_SPECTRUM (A7) locked on MOL_DEV before HOST, with the frozen
calibration temperature; aggregation/ ships the selection + calibration artifacts whose sha256s config.json records.
Loading refuses a missing / mismatched calibration and never substitutes RRF (RRF runs only if config names it).
Changing the aggregator = a full re-export (src/11_00_export_bundle.ipynb), which also rebuilds the self-test fixture.
Inference protocol: mirror_aware with exclude_sources = {} (hidden test spectra are unpublished); T1 applied via peak hash;
T2 NOT applied (no identity peaks shipped) -- see casmi_infer/compat.py and 11_01 P1 known-exception list.
"""
