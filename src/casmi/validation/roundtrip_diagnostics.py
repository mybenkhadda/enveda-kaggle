"""Actionable diagnosis of representative-SMILES round-trip FAILURES (the 22 COCONUT-only failures of the TRAIN ∪
COCONUT universe). Read-only: the universe is never modified, nothing is blacklisted here.

For every failing candidate (stored key K, stored representative S, recomputed key R = key(S)) the Stage-A STANDARDIZED
source records whose connectivity_key == K are collected (legacy `universe_work/standardized/<SRC>/chunk-*.parquet` or a
modern namespace), and the EXACT project canonicalizer (`smiles_roundtrip.project_keys`, training tautomer caps) is
re-applied to each record's raw_smiles, normalized_smiles and representative_smiles. Outcome (first rule that applies):

    UNRESOLVED                         no source record with key K found (or the canonicalizer errors on every input)
    SOURCE_CANONICALIZATION_FIX        no record's RAW SMILES maps to K any more: the stored key is not reproducible from
                                       the source -> fix / re-run source canonicalization (Stage A) for these records
    EXPECTED_CANONICALIZER_VARIATION   the raw SMILES of records sharing K map to BOTH K and R: the capped tautomer
                                       canonicalizer is not stable across equivalent inputs for this structure -> document
                                       as a canonicalizer limitation (deployment still needs a round-tripping SMILES)
    MERGE_REPRESENTATIVE_SELECTION_FIX another record of K has a representative / normalized SMILES that DOES round-trip
                                       -> choose it as the bucket representative (Stage-B selection rule)
    REPRESENTATIVE_SMILES_FIX_ONLY     K is reproducible from raw SMILES but no stored serialization round-trips -> emit a
                                       round-tripping serialization (e.g. the raw SMILES that maps to K)

`proposed_smiles` is a SMILES that maps to K when one exists (never applied automatically).
"""
from pathlib import Path

import pandas as pd

OUTCOMES = ("REPRESENTATIVE_SMILES_FIX_ONLY", "SOURCE_CANONICALIZATION_FIX", "MERGE_REPRESENTATIVE_SELECTION_FIX",
            "EXPECTED_CANONICALIZER_VARIATION", "UNRESOLVED")
DIAGNOSIS_COLUMNS = ["candidate_id", "stored_connectivity_key", "representative_smiles", "recomputed_connectivity_key", "stratum",
                     "n_source_records", "n_raw_maps_to_stored", "n_raw_maps_to_recomputed", "n_serializations_roundtrip",
                     "outcome", "proposed_smiles", "evidence"]


def collect_source_records(namespaces, keys, sources=None):
    """Standardized Stage-A rows (source, source_id, raw_smiles, normalized_smiles, representative_smiles,
    connectivity_key) whose connectivity_key is in `keys`. Reads one chunk file at a time with a pyarrow filter."""
    import pyarrow.parquet as pq
    keys = sorted({str(k) for k in keys})
    cols = ["source", "source_id", "raw_smiles", "normalized_smiles", "representative_smiles", "connectivity_key"]
    parts = []
    if not keys:
        return pd.DataFrame(columns=cols)
    for ns in [Path(n) for n in namespaces]:
        for f in sorted((ns / "standardized").glob("*/chunk-*.parquet")):
            if sources is not None and f.parent.name not in sources:
                continue
            have = [c for c in cols if c in pq.read_schema(f).names]
            t = pq.read_table(f, columns=have, filters=[("connectivity_key", "in", keys)])
            if t.num_rows:
                parts.append(pd.DataFrame({c: t.column(c).to_pylist() for c in have}))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    for c in cols:
        if c not in out:
            out[c] = None
    return out[cols]


def diagnose_roundtrip_failures(failures, source_records, keyfn_batch=None):
    """`failures`: smiles_roundtrip failures (candidate_id, stored_connectivity_key, representative_smiles,
    recomputed_connectivity_key, stratum). `source_records`: `collect_source_records` output. `keyfn_batch`: list of
    SMILES -> list of (key, error); default `smiles_roundtrip.project_keys` (RDKit, the exact project canonicalizer)."""
    if keyfn_batch is None:
        from casmi.validation.smiles_roundtrip import project_keys as keyfn_batch
    rows = []
    recs = source_records.assign(connectivity_key=source_records["connectivity_key"].astype(str))
    for f in failures.itertuples(index=False):
        k, r = str(f.stored_connectivity_key), (str(f.recomputed_connectivity_key) if f.recomputed_connectivity_key else None)
        g = recs[recs["connectivity_key"] == k]
        raw = [s for s in g["raw_smiles"].dropna().astype(str).unique()]
        ser = [s for s in pd.unique(pd.concat([g["representative_smiles"], g["normalized_smiles"]]).dropna().astype(str)) if s != f.representative_smiles]
        raw_keys = [x for x, _ in keyfn_batch(raw)] if raw else []
        ser_keys = [x for x, _ in keyfn_batch(ser)] if ser else []
        n_raw_k = sum(1 for x in raw_keys if x == k)
        n_raw_r = sum(1 for x in raw_keys if r is not None and x == r)
        ok_ser = [s for s, x in zip(ser, ser_keys, strict=True) if x == k]
        ok_raw = [s for s, x in zip(raw, raw_keys, strict=True) if x == k]
        if not len(g) or (raw_keys and all(x is None for x in raw_keys) and not ok_ser):
            outcome, proposed = "UNRESOLVED", None
            ev = "no standardized source record with this key" if not len(g) else "canonicalizer failed on every source SMILES"
        elif n_raw_k == 0:
            outcome, proposed, ev = "SOURCE_CANONICALIZATION_FIX", None, f"0/{len(raw)} raw SMILES map to the stored key"
        elif n_raw_r > 0:
            outcome, proposed = "EXPECTED_CANONICALIZER_VARIATION", (ok_ser or ok_raw or [None])[0]
            ev = f"raw SMILES map to both the stored ({n_raw_k}) and the recomputed key ({n_raw_r})"
        elif ok_ser:
            outcome, proposed, ev = "MERGE_REPRESENTATIVE_SELECTION_FIX", ok_ser[0], f"{len(ok_ser)} other stored serialization(s) round-trip"
        else:
            outcome, proposed, ev = "REPRESENTATIVE_SMILES_FIX_ONLY", ok_raw[0], f"{n_raw_k}/{len(raw)} raw SMILES map to the stored key"
        rows.append({"candidate_id": int(f.candidate_id), "stored_connectivity_key": k, "representative_smiles": f.representative_smiles,
                     "recomputed_connectivity_key": r, "stratum": getattr(f, "stratum", None), "n_source_records": int(len(g)),
                     "n_raw_maps_to_stored": n_raw_k, "n_raw_maps_to_recomputed": n_raw_r, "n_serializations_roundtrip": len(ok_ser),
                     "outcome": outcome, "proposed_smiles": proposed, "evidence": ev})
    return pd.DataFrame(rows, columns=DIAGNOSIS_COLUMNS)


def diagnosis_summary(diagnosis):
    counts = diagnosis["outcome"].value_counts().reindex(OUTCOMES, fill_value=0) if len(diagnosis) else pd.Series(0, index=OUTCOMES)
    return {"n_failures": int(len(diagnosis)), "by_outcome": {k: int(v) for k, v in counts.items()},
            "n_with_proposed_smiles": int(diagnosis["proposed_smiles"].notna().sum()) if len(diagnosis) else 0,
            "note": "diagnosis only: no representative is replaced and no candidate is blacklisted by this code"}
