"""Validation regimes v2: ELIGIBILITY-based C1 / C2 / C3 with natural counts, bound to an explicit identity.

Why v1 (`casmi.validation.regimes`) is retired for Gate A:
  * it split ELIGIBLE truths 34 / 33 / 33 by a seeded hash, so a truth with real reference evidence could be "C3" and
    the regime shares said nothing about the data;
  * C1 eligibility was `n_truth_spectra >= 2` under `test_simulated_strict`, not the mirror-aware protocol that is the
    project's reference-availability authority;
  * the persisted table was built 2026-10-01 against the TRAIN-only universe (274,195 candidates). Notebook 11 wrote a
    signature (universe_n_candidates, c2_mode) but notebook 13 never checked it, so a stale table was silently used.

v2 assignment (per truth CONNECTIVITY; first matching rule wins; no fractions, no hash, no random split):

    C1  has_mirror_aware_reference           >= 1 eligible mirror_aware reference of the truth (same-source references
                                             and T1/T2 mirrors excluded; `casmi.spectra.reference_selection`), from the
                                             EXACT lazy-walk QCR flags (`has_ge1_mirror_aware` of the truth row)
    C2  external_reachable                   no mirror-aware reference, but the truth is reachable from an EXTERNAL source
                                             (`external_sources`: COCONUT now; PUBCHEM only once its index exists, which
                                             changes the identity and forces a re-sign)
    C3  otherwise                            neither reference evidence nor an external source
    UNRESOLVED  MIRROR_AWARE_EVIDENCE_MISSING no QCR truth row for any query of the connectivity (query not walked, or truth
                                             outside the walked pool). Reported explicitly, never silently mapped to C2/C3

`test_simulated_strict` is a SENSITIVITY diagnostic only (`has_test_simulated_strict_reference`, NA when no evidence
file is configured); it never decides C1.

A connectivity's mirror-aware evidence is the OR over its evaluated queries (mirror-aware eligibility depends on the
query's source library); `n_queries_mirror_evaluated` / `n_queries_with_mirror_reference` are recorded.

The manifest is IDENTITY-BOUND: protocol version, dev-query identity, split identity, universe build id (modern Stage-C
or adopted legacy digest), external source set, membership identity, mirror-aware protocol + evidence identity, panel
manifest identity, assignment policy identity, seed, config. `check_regime_manifest` refuses (REGIME_REBUILD_REQUIRED)
on any mismatch. git_commit is recorded and compared; it blocks only with `strict_git_commit=True` (the assignment code
itself is part of the signature through `assignment_code_identity`).
"""
import hashlib
import inspect
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REGIME_PROTOCOL_VERSION = "casmi-validation-regime-2"
REGIMES = ("C1", "C2", "C3")
UNRESOLVED = "UNRESOLVED"
REASON_C1 = "C1_MIRROR_AWARE_REFERENCE"
REASON_C2 = "C2_NO_MIRROR_REFERENCE_EXTERNALLY_REACHABLE"
REASON_C3 = "C3_NO_MIRROR_REFERENCE_NOT_EXTERNALLY_REACHABLE"
REASON_UNRESOLVED = "UNRESOLVED_MIRROR_AWARE_EVIDENCE_MISSING"
REGIME_CURRENT, REGIME_REBUILD_REQUIRED, REGIME_MANIFEST_MISSING = "REGIME_CURRENT", "REGIME_REBUILD_REQUIRED", "REGIME_MANIFEST_MISSING"
SIGNATURE_EXCLUDED = ("git_commit", "built_at")
RECORD_COLUMNS = ["connectivity_key", "panel", "fold", "n_queries", "has_mirror_aware_reference", "n_queries_mirror_evaluated",
                  "n_queries_with_mirror_reference", "has_test_simulated_strict_reference", "coconut_reachable",
                  "pubchem_membership_estimate", "external_reachable", "external_sources", "regime", "regime_reason"]
QUERY_REGIME_COLUMNS = ["query_id", "true_connectivity_key", "fold", "population", "query_has_mirror_aware_reference", "regime",
                        "regime_reason"]
SOURCE_FLAG = {"COCONUT": "in_coconut", "PUBCHEM": "in_pubchem"}


class RegimeRebuildRequired(RuntimeError):
    """The persisted v2 regime manifest is missing or does not match the current identity."""


def _sha(obj, n=16):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:n]


def _file_identity(path):
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return {"name": p.name, "size_bytes": p.stat().st_size, "sha256": h.hexdigest()[:16]}


# ---------------------------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------------------------

def load_reference_evidence(files, flag_column, label="mirror_aware"):
    """Per-query truth evidence from QCR pair summaries: rows with is_true, `flag_column` (e.g. has_ge1_mirror_aware).
    Missing files are reported, never invented. Returns (DataFrame(query_id, flag), info)."""
    parts, info = [], {"label": label, "flag_column": flag_column, "files": [], "missing": []}
    for f in files or ():
        p = Path(f)
        if not p.is_file():
            info["missing"].append(str(f))
            continue
        t = pd.read_parquet(p, columns=["query_id", "is_true", flag_column])
        t = t[t["is_true"].astype(bool)]
        parts.append(pd.DataFrame({"query_id": t["query_id"].astype(str).to_numpy(), "flag": t[flag_column].astype(bool).to_numpy()}))
        info["files"].append({**(_file_identity(p) or {}), "n_truth_rows": int(len(t))})
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["query_id", "flag"])
    ev = ev.groupby("query_id", sort=True)["flag"].any().reset_index() if len(ev) else ev
    info["n_queries_with_evidence"] = int(len(ev))
    info["available"] = bool(len(info["files"]))
    return ev, info


def mirror_aware_protocol_identity(evidence_info):
    """Protocol definition (reference_selection source hash) + the evidence files actually used."""
    try:
        from casmi.ranking.freeze import mirror_aware_protocol_hash
        proto = mirror_aware_protocol_hash()
    except Exception as e:      # identity must never be silently empty
        proto = f"UNAVAILABLE:{type(e).__name__}"
    return {"protocol": "mirror_aware", "reference_selection_hash": proto,
            "evidence": [{k: f.get(k) for k in ("name", "size_bytes", "sha256", "n_truth_rows")} for f in evidence_info.get("files", [])],
            "flag_column": evidence_info.get("flag_column")}


# ---------------------------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------------------------

def external_reachability(membership, external_sources):
    """external_reachable = OR over the configured external sources (TRAIN never counts). `membership`: output of
    `external_coverage.build_external_membership` (in_coconut, in_pubchem, in_other_external, pubchem_available)."""
    m = membership.copy()
    flags = np.zeros(len(m), dtype=bool)
    for s in external_sources:
        col = SOURCE_FLAG.get(s, "in_other_external")
        if s == "PUBCHEM" and not bool(m.get("pubchem_available", pd.Series([False])).any()):
            raise ValueError("PUBCHEM listed in external_sources but no PubChem membership index is available")
        flags |= m[col].to_numpy(bool)
    m["coconut_reachable"] = m["in_coconut"].to_numpy(bool)
    m["pubchem_membership_estimate"] = m["in_pubchem"].astype(object).where(m["pubchem_available"].astype(bool), None)
    m["external_reachable"] = flags
    return m


def assign_regimes_v2(queries, mirror_evidence, membership, external_sources, strict_evidence=None):
    """`queries`: query_id, true_connectivity_key, fold, population (ALL_DEV / HOST rows; a query may appear once).
    `mirror_evidence` / `strict_evidence`: DataFrame(query_id, flag). `membership`: `external_reachability` output.
    Returns (query_regimes, molecules) -- one regime per connectivity, natural counts."""
    q = queries[["query_id", "true_connectivity_key", "fold", "population"]].copy()
    q["query_id"] = q["query_id"].astype(str)
    q["true_connectivity_key"] = q["true_connectivity_key"].astype(str)
    q = q.drop_duplicates("query_id")
    mev = mirror_evidence.set_index("query_id")["flag"] if len(mirror_evidence) else pd.Series(dtype=bool)
    q["query_has_mirror_aware_reference"] = q["query_id"].map(mev).astype("boolean")
    g = q.groupby("true_connectivity_key", sort=True)
    mol = pd.DataFrame({"n_queries": g.size(), "fold": g["fold"].first(),
                        "n_queries_mirror_evaluated": g["query_has_mirror_aware_reference"].count(),
                        "n_queries_with_mirror_reference": g["query_has_mirror_aware_reference"].sum().astype(int)})
    mol["has_mirror_aware_reference"] = pd.array(np.where(mol["n_queries_mirror_evaluated"] > 0, mol["n_queries_with_mirror_reference"] > 0,
                                                          pd.NA), dtype="boolean")
    if strict_evidence is not None and len(strict_evidence):
        sev = strict_evidence.set_index("query_id")["flag"]
        s = q.assign(_s=q["query_id"].map(sev).astype("boolean")).groupby("true_connectivity_key")["_s"]
        mol["has_test_simulated_strict_reference"] = pd.array(np.where(s.count() > 0, s.sum() > 0, pd.NA), dtype="boolean")
    else:
        mol["has_test_simulated_strict_reference"] = pd.array([pd.NA] * len(mol), dtype="boolean")
    mb = membership.assign(connectivity_key=membership["connectivity_key"].astype(str)).drop_duplicates("connectivity_key").set_index("connectivity_key")
    mol["coconut_reachable"] = mol.index.map(mb["coconut_reachable"]).fillna(False).astype(bool)
    mol["pubchem_membership_estimate"] = mol.index.map(mb["pubchem_membership_estimate"])
    mol["external_reachable"] = mol.index.map(mb["external_reachable"]).fillna(False).astype(bool)
    mol["external_sources"] = ";".join(external_sources)
    has_m = mol["has_mirror_aware_reference"]
    c1 = has_m.fillna(False).to_numpy(bool)
    unres = has_m.isna().to_numpy(bool)
    ext = mol["external_reachable"].to_numpy(bool)
    mol["regime"] = np.select([c1, unres, ext], ["C1", UNRESOLVED, "C2"], "C3")
    mol["regime_reason"] = np.select([c1, unres, ext], [REASON_C1, REASON_UNRESOLVED, REASON_C2], REASON_C3)
    mol = mol.reset_index().rename(columns={"true_connectivity_key": "connectivity_key"})
    qr = q.merge(mol[["connectivity_key", "regime", "regime_reason"]], left_on="true_connectivity_key", right_on="connectivity_key", how="left")
    return qr[QUERY_REGIME_COLUMNS].sort_values(["population", "fold", "query_id"], kind="mergesort").reset_index(drop=True), mol


def molecule_records(panel_queries, molecules):
    """§18 per-(panel, connectivity) record: panel membership from the panel query table; regime fields from `molecules`."""
    pm = panel_queries.groupby(["panel", "connectivity_key"], sort=True).size().rename("n_panel_queries").reset_index()
    out = pm.merge(molecules, on="connectivity_key", how="left", validate="many_to_one")
    missing = out["regime"].isna()
    out.loc[missing, "regime"] = UNRESOLVED
    out.loc[missing, "regime_reason"] = "UNRESOLVED_NOT_IN_REGIME_POPULATION"
    out["n_queries"] = out["n_panel_queries"]
    return out[RECORD_COLUMNS]


def assert_regime_v2_consistency(molecules):
    """Definitional invariants (raise): C1 <=> mirror evidence True; C2 => no mirror ref and externally reachable; C3 =>
    no mirror ref and not reachable; UNRESOLVED <=> evidence missing. Returns the check table."""
    m = molecules
    hm = m["has_mirror_aware_reference"]
    checks = [("C1 iff has_mirror_aware_reference", ((m["regime"] == "C1") == hm.fillna(False).astype(bool)).all()),
              ("UNRESOLVED iff mirror evidence missing", ((m["regime"] == UNRESOLVED) == hm.isna()).all()),
              ("C2 => externally reachable", m.loc[m["regime"] == "C2", "external_reachable"].astype(bool).all()),
              ("C3 => not externally reachable", not m.loc[m["regime"] == "C3", "external_reachable"].astype(bool).any()),
              ("one regime per connectivity", not m["connectivity_key"].duplicated().any()),
              ("every connectivity has a regime", m["regime"].isin(REGIMES + (UNRESOLVED,)).all())]
    t = pd.DataFrame([{"check": n, "passed": bool(ok)} for n, ok in checks])
    if not t["passed"].all():
        raise AssertionError("regime v2 consistency failed:\n" + t[~t["passed"]].to_string(index=False))
    return t


def regime_counts_by_panel(records):
    """Natural counts per panel and regime (UNRESOLVED shown explicitly; never filled as 'UNASSIGNED')."""
    c = records.groupby(["panel", "regime"]).size().rename("n_molecules").reset_index()
    c["share"] = c["n_molecules"] / c.groupby("panel")["n_molecules"].transform("sum")
    return c


# ---------------------------------------------------------------------------------------------
# identity + manifest
# ---------------------------------------------------------------------------------------------

def frame_content_identity(df, cols, path=None):
    """Content identity of a table: file identity (if a path is given) + digest of the sorted key columns."""
    d = df[list(cols)].astype(str)
    rows = sorted(map(tuple, d.to_numpy().tolist()))
    return {"file": _file_identity(path) if path else None, "n_rows": int(len(df)), "columns": list(cols), "content_sha256": _sha(rows, 20)}


def assignment_policy():
    return {"protocol_version": REGIME_PROTOCOL_VERSION, "order": ["C1 mirror_aware", "UNRESOLVED evidence missing", "C2 external", "C3"],
            "c1_authority": "mirror_aware (has_ge1_mirror_aware, truth row, exact lazy walk)",
            "test_simulated_strict": "sensitivity diagnostic only", "fractions": None, "aggregation": "OR over the connectivity's queries"}


def assignment_code_identity():
    import sys
    return hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode("utf-8")).hexdigest()[:16]


def build_regime_identity(*, git_commit, dev_identity, split_identity, universe_identity, external_sources, membership_identity,
                          mirror_identity, panel_manifest_identity, seed, config, host_identity=None):
    ident = {"protocol_version": REGIME_PROTOCOL_VERSION, "git_commit": git_commit, "dev_queries_identity": dev_identity,
             "split_identity": split_identity, "universe_build_id": universe_identity.get("build_id"),
             "universe_identity_kind": universe_identity.get("kind"), "external_sources": sorted(external_sources),
             "membership_identity": membership_identity, "mirror_aware_protocol_identity": mirror_identity,
             "panel_manifest_identity": panel_manifest_identity, "host_source_identity": host_identity,
             "policy_identity": _sha(assignment_policy()), "assignment_code_identity": assignment_code_identity(),
             "seed": seed, "config": config}
    return ident


DEV_IDENTITY_COLUMNS = ["query_id", "true_connectivity_key", "fold"]
SPLIT_IDENTITY_COLUMNS = ["connectivity_key", "fold"]


def membership_identity(pubchem_index):
    if pubchem_index is None:
        return None
    m = pubchem_index.manifest
    return {"membership_index_version": m.get("membership_index_version"), "identity_tag": m.get("identity_tag"),
            "n_unique_connectivity_keys": m.get("n_unique_connectivity_keys"), "label": m.get("label")}


def regime_identity_from_run(*, cfg, repo_root, dev, dev_path, folds, folds_path, universe_identity, mirror_info,
                             panel_manifest_identity, pubchem_index=None, host_path=None):
    """The ONE identity computation used by notebook 11b (to sign) and notebook 13 / 14 (to verify)."""
    from casmi.workspace.artifacts import git_commit
    v2 = cfg["validation_v2"]
    ext = list(v2["external_sources"])
    return build_regime_identity(
        git_commit=git_commit(repo_root), dev_identity=frame_content_identity(dev, DEV_IDENTITY_COLUMNS, dev_path),
        split_identity=frame_content_identity(folds, SPLIT_IDENTITY_COLUMNS, folds_path), universe_identity=universe_identity,
        external_sources=ext, membership_identity=membership_identity(pubchem_index) if "PUBCHEM" in ext else None,
        mirror_identity=mirror_aware_protocol_identity(mirror_info), panel_manifest_identity=panel_manifest_identity,
        seed=cfg["validation"]["split_seed"], config={"validation_v2": v2, "panels": cfg.get("panels")}, host_identity=_file_identity(host_path))


def regime_signature(identity):
    return _sha({k: v for k, v in identity.items() if k not in SIGNATURE_EXCLUDED}, 20)


def write_regime_manifest(path, identity, counts, extra=None):
    body = {"regime_protocol_version": REGIME_PROTOCOL_VERSION, "identity": identity, "regime_signature": regime_signature(identity),
            "counts": counts, "built_at": datetime.now(timezone.utc).isoformat(), **(extra or {})}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)
    return body


def check_regime_manifest(manifest_path, expected_identity, strict_git_commit=False):
    """Compare the persisted manifest with the CURRENT identity. Returns {status, mismatches, warnings, manifest}."""
    p = Path(manifest_path)
    if not p.is_file():
        return {"status": REGIME_MANIFEST_MISSING, "mismatches": [{"field": "manifest", "persisted": None, "expected": str(p)}],
                "warnings": [], "manifest": None}
    man = json.loads(p.read_text(encoding="utf-8"))
    got = man.get("identity") or {}
    mism, warns = [], []
    for k, v in expected_identity.items():
        if k == "git_commit":
            if got.get(k) != v:
                (mism if strict_git_commit else warns).append({"field": k, "persisted": got.get(k), "expected": v})
            continue
        if json.dumps(got.get(k), sort_keys=True, default=str) != json.dumps(v, sort_keys=True, default=str):
            mism.append({"field": k, "persisted": got.get(k), "expected": v})
    if man.get("regime_signature") != regime_signature(got):
        mism.append({"field": "regime_signature", "persisted": man.get("regime_signature"), "expected": regime_signature(got)})
    return {"status": REGIME_REBUILD_REQUIRED if mism else REGIME_CURRENT, "mismatches": mism, "warnings": warns, "manifest": man}


def require_current_regimes(manifest_path, expected_identity, strict_git_commit=False):
    chk = check_regime_manifest(manifest_path, expected_identity, strict_git_commit)
    if chk["status"] != REGIME_CURRENT:
        fields = [m["field"] for m in chk["mismatches"]]
        raise RegimeRebuildRequired(f"{REGIME_REBUILD_REQUIRED}: {chk['status']} -- mismatching {fields}. Run notebook "
                                    "11b_colab_validation_regimes_v2 (after the universe adoption) and re-run this notebook.")
    return chk


# ---------------------------------------------------------------------------------------------
# safe replacement of the stale v1 table + exclusion manifests
# ---------------------------------------------------------------------------------------------

def archive_legacy_regimes(processed_dir, legacy_dir=None, stem="validation_regimes", archive_stem="validation_regimes_pre_external_universe"):
    """COPY the v1 regime table (+ its metadata) to legacy/<archive_stem>.* and drop a STALE marker next to the original.
    Never deletes or rewrites the original; refuses to overwrite an existing, DIFFERENT archive."""
    src_dir = Path(processed_dir)
    dst_dir = Path(legacy_dir) if legacy_dir else src_dir / "legacy"
    report = {"archived": [], "skipped": [], "refused": []}
    for src, dst in ((src_dir / f"{stem}.parquet", dst_dir / f"{archive_stem}.parquet"),
                     (src_dir / f"{stem}.metadata.json", dst_dir / f"{archive_stem}.metadata.json")):
        if not src.is_file():
            report["skipped"].append(f"{src.name}: not present")
            continue
        si = _file_identity(src)
        if dst.is_file():
            if _file_identity(dst)["sha256"] == si["sha256"]:
                report["skipped"].append(f"{dst.name}: identical archive exists")
                continue
            report["refused"].append(f"{dst.name}: a DIFFERENT archive exists -- inspect it manually")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        report["archived"].append({"from": str(src), "to": str(dst), **si})
    if (src_dir / f"{stem}.parquet").is_file():
        marker = src_dir / f"{stem}.STALE.json"
        marker.write_text(json.dumps({"status": "STALE_PRE_EXTERNAL_UNIVERSE", "superseded_by": "validation_regimes_v2.parquet",
                                      "reason": "v1 table built against the TRAIN-only universe with 34/33/33 fractions; not valid for Gate A",
                                      "archive": [a["to"] for a in report["archived"]], "marked_at": datetime.now(timezone.utc).isoformat()},
                                     indent=2), encoding="utf-8")
        report["stale_marker"] = str(marker)
    return report


def write_exclusion_manifests(records, out_dir, regime_sig, panel_manifest_identity):
    """SUPPORT ONLY: connectivities a model must not train on when it is evaluated on that panel / regime.
    host / np_panel / c2 / c3 parquet (connectivity_key + why) + model_training_exclusions.metadata.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = {"host": records[records["panel"] == "HOST"], "np_panel": records[records["panel"] == "NP_PANEL"],
            "c2": records[records["regime"] == "C2"], "c3": records[records["regime"] == "C3"]}
    files = {}
    for name, df in sets.items():
        t = (df.groupby("connectivity_key", sort=True).agg(panels=("panel", lambda s: ";".join(sorted(set(s)))), regime=("regime", "first"))
             .reset_index() if len(df) else pd.DataFrame(columns=["connectivity_key", "panels", "regime"]))
        p = out / f"{name}_connectivities.parquet"
        t.to_parquet(p, index=False)
        files[p.name] = {"n_connectivities": int(len(t)), **(_file_identity(p) or {})}
    meta = {"purpose": "SUPPORT ONLY -- training-exclusion lists for leakage-free evaluation; they do not train or select anything",
            "regime_signature": regime_sig, "panel_manifest_identity": panel_manifest_identity, "files": files,
            "rule": "a model evaluated on panel P / regime R must exclude every connectivity of the matching file from its training data",
            "written_at": datetime.now(timezone.utc).isoformat()}
    (out / "model_training_exclusions.metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta
