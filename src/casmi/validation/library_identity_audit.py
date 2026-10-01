"""v6.1 EXACT-LIBRARY-IDENTITY AUDIT of the visible TEST against the TRAIN reference universe.

Why this exists
---------------
The executed 10v6_00 printed `exact_library_duplicate_share = 1.000`. That number was
`mean(n_t1_excluded_refs > 0)`. It is the share of TEST spectra for which AT LEAST ONE TRAIN reference
of ANY mass-window candidate carries the query's T1 peak hash. `compute_peak_hash` rounds m/z to 4
decimals and the max-normalised intensity to 3, drops zero-rounded peaks and rounds the precursor to
3 decimals. Hash equality is therefore a rounding-level equivalence, not array identity, and it
never proves that two spectra come from the same molecule. This module separates:

  hash_equal              T1 peak hash equal (the thing inference excludes as T1)
  mz_array_equal          m/z arrays identical (float64, after `remove_invalid_peaks`, (m/z, intensity)-sorted)
  intensity_array_equal   intensity arrays identical in that same order
  full_peak_array_equal   both, i.e. the same peak list
  precursor / adduct / CE / instrument equality   metadata only

Only `full_peak_array_equal & precursor_mz_equal` is called an exact library duplicate
(`possible_contamination_flag`): the visible-TEST measurement itself exists in the TRAIN library.
Even then this says nothing certain about the TEST molecule's identity. `train_connectivity_key` is
the TRAIN spectrum's annotation. It is reported for audit only and is never joined to TEST as a label.

TRAIN-only contract
-------------------
The hash index is built ONLY from the exported reference library: `ref_meta.peak_hash`, one row per
train.parquet row, with ids `train_<row>`. `assert_train_only_reference_index` refuses a TEST id, a
foreign id, a row misalignment or a CSR entry outside the library. `TrainHashIndex` has no argument
through which a TEST spectrum's hash could enter it. The TRAIN identity peaks used for array
comparison are re-read from train.parquet at `ref_row` offsets, and the bundle hash is re-derived
from them (`ref_hash_recomputed_equal`), so a row misalignment cannot pass silently.
"""
import hashlib
import re

import numpy as np
import pandas as pd

from casmi.spectra.deduplication import compute_peak_hash
from casmi.spectra.preprocessing import remove_invalid_peaks
from casmi.validation.test_match import assert_label_free

TRAIN_ID_RE = re.compile(r"^train_(\d+)$")
EXACT_DUPLICATE_BASIS = "full_peak_array_equal & precursor_mz_equal (NOT hash equality)"
MAX_COMPARE_PER_QUERY = 25          # hash-bucket members compared array-by-array per TEST spectrum (all are COUNTED)
IDENTITY_MAJORITY = 0.50            # verdict threshold, fixed before the audit is run

FULL_IDENTICAL, HASH_ONLY, MZ_ONLY, INT_ONLY, NO_IDENTITY = (
    "FULL_ARRAY_IDENTICAL", "HASH_EQUAL_ARRAYS_DIFFER", "SAME_MZ_ONLY", "SAME_INTENSITY_ONLY", "NO_IDENTITY")
BASIS_HASH, BASIS_BEST_COSINE, BASIS_NONE = "train_hash_match", "max_cosine_best_reference (no hash match)", "none"

INTEGRITY_FAILURE = "INTEGRITY_FAILURE"
VERDICTS = (INTEGRITY_FAILURE, "DEGENERATE_HASH_ARTIFACT", "VISIBLE_TEST_SPECTRA_PRESENT_IN_TRAIN_LIBRARY",
            "HASH_LEVEL_EQUIVALENCE_ONLY", "NO_TRAIN_IDENTITY", "MIXED")

AUDIT_COLUMNS = [
    "spectrum_id", "molecule_id", "query_peak_hash", "n_train_hash_matches", "best_train_reference_id", "best_train_source",
    "hash_equal", "mz_array_equal", "intensity_array_equal", "full_peak_array_equal", "precursor_mz_equal", "adduct_equal",
    "collision_energy_equal", "instrument_equal", "train_connectivity_key", "query_in_reference_index", "possible_contamination_flag",
    # audit extras
    "match_basis", "identity_relation", "n_train_hash_matches_compared", "n_full_array_matches",
    "n_distinct_train_connectivities_in_hash_matches", "n_train_hash_matches_in_candidate_pool", "legacy_n_t1_excluded_refs",
    "legacy_parity_ok", "query_hash_degenerate", "n_query_identity_peaks", "n_train_identity_peaks", "precursor_mz_abs_diff",
    "max_abs_mz_diff", "max_abs_intensity_diff", "ref_hash_recomputed_equal", "query_precursor_mz", "train_precursor_mz",
    "query_adduct", "train_adduct", "query_collision_energy", "train_collision_energy", "query_instrument", "train_instrument"]
MATCH_COLUMNS = ["spectrum_id", "ref_spectrum_id", "match_basis", "train_source", "train_connectivity_key", "hash_equal", "mz_array_equal",
                 "intensity_array_equal", "full_peak_array_equal", "precursor_mz_equal", "adduct_equal", "collision_energy_equal",
                 "instrument_equal", "ref_hash_recomputed_equal"]
# words an identity column / summary key may never use: hash (or array) equality is not molecule identity
FORBIDDEN_IDENTITY_CLAIMS = ("same_molecule", "same molecule", "true_structure", "truth", "true_connectivity")


class ReferenceIndexContaminationError(RuntimeError):
    """The reference universe searched for identity is not TRAIN-only."""


class IdentityWordingError(ValueError):
    """A metric is named as an exact duplicate / same molecule without the evidence for that claim."""


# ---------------------------------------------------------------------------------------------
# wording guard
# ---------------------------------------------------------------------------------------------

# the only metric names allowed to say "exact duplicate": each is computed from possible_contamination_flag
FULL_ARRAY_BACKED_KEYS = frozenset({"share_exact_library_duplicate", "among_exact_duplicates", "exact_duplicate_train_source_counts",
                                    "share_molecules_any_spectrum_exact_duplicate", "share_molecules_all_spectra_exact_duplicate"})


def assert_identity_wording(keys, full_array_verified=frozenset()):
    """`keys`: metric names (or a mapping). 'exact ... duplicate' is allowed only for the names in
    `full_array_verified` (metrics backed by full peak-array equality). A same-molecule / truth claim
    is never allowed."""
    keys = [str(k) for k in keys]
    claims = [k for k in keys if any(t in k.lower() for t in FORBIDDEN_IDENTITY_CLAIMS)]
    if claims:
        raise IdentityWordingError(f"identity metrics may not claim molecule identity / truth: {claims}")
    exact = [k for k in keys if "exact" in k.lower() and "duplicate" in k.lower() and k not in set(full_array_verified)]
    if exact:
        raise IdentityWordingError(f"'exact duplicate' wording needs full peak-array equality, but these are hash-level only: {exact}")
    return True


# ---------------------------------------------------------------------------------------------
# TRAIN-only reference universe
# ---------------------------------------------------------------------------------------------

def assert_train_only_reference_index(lib, test_ids, n_train_rows=None):
    """Refuses unless every reference id is `train_<row>` with row == its position (== ref_row ==
    train.parquet offset), the library has exactly `n_train_rows` rows (when given), no TEST id is
    present, the hash column is aligned, and the connectivity CSR points inside the library."""
    ids = np.asarray(lib.spectrum_id).astype(str)
    m = [TRAIN_ID_RE.match(s) for s in ids]
    bad = [s for s, mm in zip(ids, m) if mm is None]
    if bad:
        raise ReferenceIndexContaminationError(f"{len(bad)} reference ids are not TRAIN ids (train_<row>), e.g. {bad[:5]}")
    rows = np.fromiter((int(mm.group(1)) for mm in m), dtype=np.int64, count=len(ids))
    if not (rows == np.arange(len(ids))).all():
        raise ReferenceIndexContaminationError("reference ids are not aligned with train.parquet row offsets (train_<k> must sit at row k)")
    if n_train_rows is not None and len(ids) != int(n_train_rows):
        raise ReferenceIndexContaminationError(f"reference library has {len(ids)} rows but train.parquet has {int(n_train_rows)}: "
                                               "rows were added (e.g. TEST) or dropped")
    test_ids = set(map(str, test_ids))
    overlap = sorted(test_ids & set(ids))
    if overlap:
        raise ReferenceIndexContaminationError(f"{len(overlap)} TEST spectrum ids are present in the TRAIN reference index, e.g. {overlap[:5]}")
    if len(np.asarray(lib.peak_hash)) != len(ids):
        raise ReferenceIndexContaminationError("peak_hash column is not aligned with the reference ids")
    idx = np.asarray(lib.index_ids)
    if len(idx) and (int(idx.min()) < 0 or int(idx.max()) >= len(ids)):
        raise ReferenceIndexContaminationError("connectivity -> reference CSR points outside the TRAIN library")
    return {"reference_universe": "TRAIN_ONLY", "n_reference_spectra": int(len(ids)), "n_train_rows": None if n_train_rows is None else int(n_train_rows),
            "n_test_ids_checked": int(len(test_ids)), "n_test_ids_in_reference_index": 0, "id_pattern": TRAIN_ID_RE.pattern}


class TrainHashIndex:
    """peak hash -> TRAIN ref rows (ascending). Built from `lib.peak_hash` ONLY; `test_ids` is used
    solely to assert they are absent. Null hashes (a library exported with compute_hash=False) are
    counted. An all-null library is refused, because T1 would then have been a silent no-op."""
    provenance = "TRAIN_ONLY"

    def __init__(self, lib, test_ids, n_train_rows=None):
        self.integrity = assert_train_only_reference_index(lib, test_ids, n_train_rows)
        self.ref_ids = np.asarray(lib.spectrum_id).astype(str)
        h = pd.Series(np.asarray(lib.peak_hash, dtype=object))
        ok = h.notna().to_numpy()
        if not ok.any():
            raise ValueError("the reference library carries no peak hashes (exported with compute_hash=False): T1 was a no-op; audit cannot run")
        df = pd.DataFrame({"h": h.to_numpy()[ok].astype(str), "row": np.flatnonzero(ok).astype(np.int64)}).sort_values(["h", "row"], kind="mergesort")
        self._h = df["h"].to_numpy(dtype=object)
        self._rows = df["row"].to_numpy(np.int64)
        self.n_null_hash = int((~ok).sum())

    def lookup(self, query_hash):
        if query_hash is None:
            return self._rows[:0]
        a, b = np.searchsorted(self._h, query_hash, side="left"), np.searchsorted(self._h, query_hash, side="right")
        return self._rows[a:b]

    def bucket_size_summary(self, top=5):
        vc = pd.Series(self._h).value_counts()
        return {"n_hashed_refs": int(len(self._h)), "n_distinct_hashes": int(len(vc)), "n_null_hash": self.n_null_hash,
                "n_refs_in_multi_member_buckets": int(vc[vc > 1].sum()), "largest_bucket_sizes": [int(v) for v in vc.head(top)]}


def assert_audit_integrity(audit, index, test_ids):
    """Post-hoc structural assertions on a finished audit (raise, never warn)."""
    if index.provenance != "TRAIN_ONLY":
        raise ReferenceIndexContaminationError(f"hash index provenance is {index.provenance!r}, expected TRAIN_ONLY")
    leaked = sorted(set(map(str, test_ids)) & set(index.ref_ids))
    if leaked:
        raise ReferenceIndexContaminationError(f"TEST ids inside the hash index: {leaked[:5]}")
    if audit["query_in_reference_index"].astype(bool).any():
        raise ReferenceIndexContaminationError("a TEST spectrum id is present in the reference index")
    bad_ref = ~audit["best_train_reference_id"].dropna().astype(str).str.match(TRAIN_ID_RE.pattern)
    if bad_ref.any():
        raise ReferenceIndexContaminationError("a matched reference is not a TRAIN spectrum")
    hash_only = audit["possible_contamination_flag"].astype(bool) & ~audit["full_peak_array_equal"].astype(bool)
    if hash_only.any():
        raise IdentityWordingError("possible_contamination_flag set without full peak-array equality")
    assert_identity_wording(audit.columns)
    return True


# ---------------------------------------------------------------------------------------------
# peaks + pairwise identity comparison
# ---------------------------------------------------------------------------------------------

def query_identity(row, max_peaks):
    """TEST identity peaks + T1 hash through the PRODUCTION functions (`clean_query`, `query_peak_hash`)."""
    from casmi_infer.spectrum import clean_query, query_peak_hash
    identity, _ = clean_query(row["ms2_mzs"], row["ms2_normalized_intensities"], row["precursor_mz"], max_peaks)
    return identity, query_peak_hash(identity)


def load_train_identity_peaks(train_path, ref_rows, expected_precursor=None):
    """{ref_row: identity peaks} re-read from train.parquet (`remove_invalid_peaks`, as the export did).
    `expected_precursor`: the bundle's per-row precursor array; any disagreement means the bundle rows are
    not train.parquet rows, and the audit stops."""
    from casmi.io.parquet import load_rows_by_offset
    rows = np.unique(np.asarray(list(ref_rows), dtype=np.int64))
    if not len(rows):
        return {}
    df = load_rows_by_offset(train_path, ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"], rows)
    out = {}
    for r in df.to_dict("records"):
        mz, it = remove_invalid_peaks(r["ms2_mzs"] if r["ms2_mzs"] is not None else [], r["ms2_normalized_intensities"] if r["ms2_normalized_intensities"] is not None else [])
        out[int(r["_row_offset"])] = {"mzs": mz, "intensities": it, "precursor_mz": float(r["precursor_mz"])}
    missing = sorted(set(rows.tolist()) - set(out))
    if missing:
        raise RuntimeError(f"{len(missing)} TRAIN rows could not be read from train.parquet, e.g. {missing[:5]}")
    if expected_precursor is not None:
        exp = np.asarray(expected_precursor, dtype=float)
        off = [k for k, p in out.items() if not (np.isclose(p["precursor_mz"], exp[k], atol=1e-9, rtol=0) or (np.isnan(p["precursor_mz"]) and np.isnan(exp[k])))]
        if off:
            raise RuntimeError(f"train.parquet precursor != bundle ref_meta precursor at rows {off[:5]}: bundle rows are not train.parquet rows")
    return out


def _missing(v):
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _eq_str(a, b):
    return (not _missing(a)) and (not _missing(b)) and str(a) == str(b)


def _eq_num(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= 1e-9)


def index_hash(lib, ref_row):
    h = lib.peak_hash[int(ref_row)]
    return None if _missing(h) else str(h)


def _canonical(peaks):
    mz, it = np.asarray(peaks["mzs"], dtype=float), np.asarray(peaks["intensities"], dtype=float)
    o = np.lexsort((it, mz))
    return mz[o], it[o]


def compare_identity(q_peaks, q_hash, q_meta, r_peaks, r_hash, r_meta):
    """Every identity relation of one (TEST, TRAIN) pair. Arrays are compared EXACTLY (float64) after
    (m/z, intensity) sorting; `hash_equal` is reported separately and never implies the others."""
    qm, qi = _canonical(q_peaks)
    rm, ri = _canonical(r_peaks)
    same_len = len(qm) == len(rm)
    mz_eq = bool(same_len and np.array_equal(qm, rm))
    it_eq = bool(same_len and np.array_equal(qi, ri))
    qp, rp = float(q_peaks["precursor_mz"]), float(r_peaks["precursor_mz"])
    recomputed = compute_peak_hash(r_peaks["mzs"], r_peaks["intensities"], rp) == r_hash if r_hash is not None else np.nan
    return {"hash_equal": bool(q_hash is not None and r_hash is not None and q_hash == r_hash), "mz_array_equal": mz_eq, "intensity_array_equal": it_eq,
            "full_peak_array_equal": mz_eq and it_eq, "precursor_mz_equal": bool(qp == rp), "precursor_mz_abs_diff": abs(qp - rp),
            "adduct_equal": _eq_str(q_meta.get("adduct"), r_meta.get("adduct")),
            "collision_energy_equal": _eq_num(q_meta.get("collision_energy"), r_meta.get("collision_energy")),
            "instrument_equal": _eq_str(q_meta.get("instrument"), r_meta.get("instrument")),
            "max_abs_mz_diff": float(np.max(np.abs(qm - rm))) if same_len and len(qm) else (0.0 if same_len else np.nan),
            "max_abs_intensity_diff": float(np.max(np.abs(qi - ri))) if same_len and len(qi) else (0.0 if same_len else np.nan),
            "n_query_identity_peaks": int(len(qm)), "n_train_identity_peaks": int(len(rm)), "ref_hash_recomputed_equal": recomputed}


def identity_relation(c):
    if c is None:
        return NO_IDENTITY
    if c["full_peak_array_equal"]:
        return FULL_IDENTICAL
    if c["hash_equal"]:
        return HASH_ONLY
    if c["mz_array_equal"]:
        return MZ_ONLY
    if c["intensity_array_equal"]:
        return INT_ONLY
    return NO_IDENTITY


def _pick_key(c, ref_id):
    meta_eq = int(c["adduct_equal"]) + int(c["collision_energy_equal"]) + int(c["instrument_equal"])
    return (-int(c["full_peak_array_equal"]), -int(c["precursor_mz_equal"]), -int(c["hash_equal"]), -meta_eq, ref_id)


# ---------------------------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------------------------

def audit_test_identity(lib, test_df, train_path, max_peaks, *, n_train_rows=None, test_pool=None, legacy_t1=None, best_reference=None,
                        max_compare=MAX_COMPARE_PER_QUERY, progress=True):
    """One row per visible-TEST spectrum (spectrum_id ASC) against the TRAIN reference universe.

    lib             bundle `ReferenceLibrary` (TRAIN-only; asserted)
    test_df         visible test rows (observable fields only; truth columns are refused)
    train_path      train.parquet (TRAIN identity peaks at ref_row offsets)
    test_pool       optional label-free TEST pool (query_id, candidate_key) from `run_observed_test`; gives the
                    in-candidate-pool hash-match count, i.e. the executed diagnostic's T1 definition
    legacy_t1       optional Series query_id -> n_t1_excluded_refs (executed diagnostic) for the parity check
    best_reference  optional Series query_id -> best_reference_id (max-cosine ranker-visible reference); it is the
                    comparison partner when a spectrum has NO hash match, so the flag columns stay informative
    Returns (audit, matches, index)."""
    from casmi_infer.spectrum import ce_mean
    assert_label_free(test_df, "audit_test_identity(test_df)")
    index = TrainHashIndex(lib, test_df["spectrum_id"].astype(str), n_train_rows)
    ref_ids = index.ref_ids
    ref_set = set(ref_ids)
    row_of = {s: i for i, s in enumerate(ref_ids)}
    meta = lib.meta
    conn = meta["connectivity_key"].astype(object).to_numpy() if "connectivity_key" in meta.columns else np.full(len(meta), None, dtype=object)
    src = meta["source"].astype(object).to_numpy()
    r_add, r_ins = meta["adduct"].astype(object).to_numpy(), meta["instrument"].astype(object).to_numpy()
    r_ce = meta["collision_energy"].to_numpy(float)
    pool_keys = None
    if test_pool is not None:
        tp = test_pool.rename(columns={"candidate_connectivity_key": "candidate_key"})
        pool_keys = {str(q): set(g) for q, g in tp.groupby("query_id")["candidate_key"]}
    legacy = None if legacy_t1 is None else pd.Series(legacy_t1).rename(index=str)
    best_ref = None if best_reference is None else pd.Series(best_reference).rename(index=str)

    df = test_df.sort_values("spectrum_id", kind="mergesort").reset_index(drop=True)
    work, need = [], set()
    for row in df.to_dict("records"):
        sid = str(row["spectrum_id"])
        identity, qh = query_identity(row, max_peaks)
        hits = index.lookup(qh)
        basis, cmp_rows = (BASIS_HASH, [int(r) for r in hits[:max_compare]]) if len(hits) else (BASIS_NONE, [])
        if not len(hits) and best_ref is not None and not _missing(best_ref.get(sid)) and str(best_ref.get(sid)) in row_of:
            basis, cmp_rows = BASIS_BEST_COSINE, [row_of[str(best_ref.get(sid))]]
        need.update(cmp_rows)
        work.append((row, sid, identity, qh, hits, basis, cmp_rows))
    peaks = load_train_identity_peaks(train_path, need, expected_precursor=lib.precursor_mz)

    rows, matches = [], []
    for k, (row, sid, identity, qh, hits, basis, cmp_rows) in enumerate(work):
        qmeta = {"adduct": row.get("adduct"), "collision_energy": ce_mean(row.get("collision_energy_ev")), "instrument": row.get("instrument_type")}
        comps = []
        for r in cmp_rows:
            c = compare_identity(identity, qh, qmeta, peaks[r], index_hash(lib, r), {"adduct": r_add[r], "collision_energy": r_ce[r], "instrument": r_ins[r]})
            comps.append((r, c))
            matches.append({"spectrum_id": sid, "ref_spectrum_id": ref_ids[r], "match_basis": basis, "train_source": src[r], "train_connectivity_key": conn[r],
                            **{m: c[m] for m in MATCH_COLUMNS if m in c}})
        best_r, best_c = (min(comps, key=lambda rc: _pick_key(rc[1], ref_ids[rc[0]])) if comps else (None, None))
        in_pool = int(sum(1 for h in hits if not _missing(conn[h]) and conn[h] in pool_keys.get(sid, ()))) if pool_keys is not None else None
        leg = legacy.get(sid) if legacy is not None else None
        leg = None if _missing(leg) else int(leg)
        parity = np.nan if in_pool is None or leg is None else bool(in_pool == leg)
        hit_conns = {conn[h] for h in hits if not _missing(conn[h])}
        rec = {"spectrum_id": sid, "molecule_id": row.get("molecule_id"), "query_peak_hash": qh, "n_train_hash_matches": int(len(hits)),
               "best_train_reference_id": ref_ids[best_r] if best_r is not None else None, "best_train_source": src[best_r] if best_r is not None else None,
               "train_connectivity_key": conn[best_r] if best_r is not None else None,
               "query_in_reference_index": sid in ref_set, "match_basis": basis, "identity_relation": identity_relation(best_c),
               "n_train_hash_matches_compared": int(len(comps)) if basis == BASIS_HASH else 0,
               "n_full_array_matches": int(sum(c["full_peak_array_equal"] for _, c in comps)) if basis == BASIS_HASH else 0,
               "n_distinct_train_connectivities_in_hash_matches": int(len(hit_conns)),
               "n_train_hash_matches_in_candidate_pool": np.nan if in_pool is None else in_pool,
               "legacy_n_t1_excluded_refs": np.nan if leg is None else leg, "legacy_parity_ok": parity,
               "query_hash_degenerate": qh == compute_peak_hash([], [], identity["precursor_mz"]),
               "query_precursor_mz": float(identity["precursor_mz"]), "query_adduct": qmeta["adduct"], "query_collision_energy": qmeta["collision_energy"],
               "query_instrument": qmeta["instrument"],
               "train_precursor_mz": float(lib.precursor_mz[best_r]) if best_r is not None else np.nan,
               "train_adduct": r_add[best_r] if best_r is not None else None, "train_collision_energy": r_ce[best_r] if best_r is not None else np.nan,
               "train_instrument": r_ins[best_r] if best_r is not None else None}
        flags = ("hash_equal", "mz_array_equal", "intensity_array_equal", "full_peak_array_equal", "precursor_mz_equal", "adduct_equal",
                 "collision_energy_equal", "instrument_equal")
        for f in flags:
            rec[f] = bool(best_c[f]) if best_c is not None else False
        for f in ("precursor_mz_abs_diff", "max_abs_mz_diff", "max_abs_intensity_diff", "n_train_identity_peaks", "ref_hash_recomputed_equal"):
            rec[f] = best_c[f] if best_c is not None else np.nan
        rec["n_query_identity_peaks"] = int(len(identity["mzs"]))
        # exact library duplicate = the SAME measurement (full arrays + precursor), never hash equality alone
        rec["possible_contamination_flag"] = bool(rec["query_in_reference_index"] or (rec["full_peak_array_equal"] and rec["precursor_mz_equal"]))
        rows.append(rec)
        if progress and (k + 1) % 250 == 0:
            print(f"[identity audit] {k + 1}/{len(work)} TEST spectra")
    audit = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    m = pd.DataFrame(matches, columns=MATCH_COLUMNS)
    assert_identity_wording(audit.columns)
    return audit, m, index


# ---------------------------------------------------------------------------------------------
# summary, verdict, examples
# ---------------------------------------------------------------------------------------------

def _share(mask):
    mask = pd.Series(mask).astype(bool)
    return float(mask.mean()) if len(mask) else float("nan")


def identity_audit_summary(audit, index=None):
    """Honestly named shares. `share_exact_library_duplicate` is backed by full arrays + precursor;
    hash-level metrics say 'hash'."""
    a = audit
    hit = a["n_train_hash_matches"] > 0
    ex = a["possible_contamination_flag"].astype(bool)
    in_pool = pd.to_numeric(a["n_train_hash_matches_in_candidate_pool"], errors="coerce")
    par = a["legacy_parity_ok"].dropna().astype(bool)
    nm = pd.to_numeric(a["n_train_hash_matches"], errors="coerce")
    s = {
        "n_test_spectra": int(len(a)), "n_test_molecules": int(a["molecule_id"].nunique()),
        "definitions": {
            "legacy exact_library_duplicate_share (10v6_00, retired name)": "mean(n_t1_excluded_refs > 0): >=1 TRAIN reference of ANY mass-window "
                "candidate shares the query's T1 peak hash (4/3/3-decimal rounding). Hash-level; not array identity; not molecule identity.",
            "share_train_hash_match_in_candidate_pool": "the legacy definition re-derived here from the TRAIN-only hash index + the TEST pool",
            "share_train_hash_match_anywhere": "hash present anywhere in TRAIN (not restricted to the mass window)",
            "share_exact_library_duplicate": EXACT_DUPLICATE_BASIS,
            "share_hash_equal_arrays_differ": "hash equal, but the compared float64 peak arrays differ (rounding-level equivalence)"},
        "share_query_in_reference_index": _share(a["query_in_reference_index"]),
        "share_train_hash_match_anywhere": _share(hit),
        "share_train_hash_match_in_candidate_pool": float((in_pool.dropna() > 0).mean()) if in_pool.notna().any() else float("nan"),
        "legacy_parity": {"n_checked": int(len(par)), "n_mismatch": int((~par).sum())},
        "share_exact_library_duplicate": _share(ex),
        "share_hash_equal_arrays_differ": _share(a["identity_relation"] == HASH_ONLY),
        "share_degenerate_query_hash": _share(a["query_hash_degenerate"]),
        "share_degenerate_query_hash_among_hash_matches": _share(a.loc[hit, "query_hash_degenerate"]) if hit.any() else float("nan"),
        "share_hash_matches_span_multiple_train_connectivities": _share(a["n_distinct_train_connectivities_in_hash_matches"] > 1),
        "n_ref_hash_recomputed_mismatch": int((~a["ref_hash_recomputed_equal"].dropna().astype(bool)).sum()),
        "identity_relation_counts": a["identity_relation"].value_counts().sort_index().astype(int).to_dict(),
        "match_basis_counts": a["match_basis"].value_counts().sort_index().astype(int).to_dict(),
        "n_train_hash_matches_quantiles": {f"P{q}": float(nm.quantile(q / 100)) for q in (50, 90, 99)} | {"max": float(nm.max()) if len(nm) else float("nan")},
        "among_exact_duplicates": {k: _share(a.loc[ex, k]) if ex.any() else float("nan")
                                   for k in ("adduct_equal", "collision_energy_equal", "instrument_equal")},
        "exact_duplicate_train_source_counts": a.loc[ex, "best_train_source"].astype(str).value_counts().sort_index().astype(int).to_dict(),
        "molecule_level": {}}
    if a["molecule_id"].notna().any():
        g = a[a["molecule_id"].notna()].groupby("molecule_id")["possible_contamination_flag"]
        s["molecule_level"] = {"share_molecules_any_spectrum_exact_duplicate": float(g.any().mean()),
                               "share_molecules_all_spectra_exact_duplicate": float(g.all().mean())}
    if index is not None:
        s["train_hash_index"] = {"provenance": index.provenance, **index.integrity, **index.bucket_size_summary()}
    # metric names only (the `definitions` block quotes the retired name on purpose)
    assert_identity_wording([k for k in s if k != "definitions"] + list(s["molecule_level"]), full_array_verified=FULL_ARRAY_BACKED_KEYS)
    return s


def interpret_identity_audit(s, majority=IDENTITY_MAJORITY):
    """Runtime verdict from the summary (thresholds fixed in code before the run). A correctness
    failure is the only thing that reopens protocol work."""
    failures = []
    if s["share_query_in_reference_index"] > 0:
        failures.append("TEST spectrum ids are present in the TRAIN reference index")
    if s["legacy_parity"]["n_mismatch"]:
        failures.append(f"{s['legacy_parity']['n_mismatch']} TEST spectra: in-pool TRAIN hash matches != the executed diagnostic's n_t1_excluded_refs")
    if s["n_ref_hash_recomputed_mismatch"]:
        failures.append(f"{s['n_ref_hash_recomputed_mismatch']} bundle peak hashes do not reproduce from train.parquet (row misalignment / hash drift)")
    hit, ex, deg = s["share_train_hash_match_anywhere"], s["share_exact_library_duplicate"], s["share_degenerate_query_hash_among_hash_matches"]
    text = [f"legacy 'exact_library_duplicate_share' (retired name) == share_train_hash_match_in_candidate_pool = {s['share_train_hash_match_in_candidate_pool']:.4f}: "
            "a HASH-level statistic over every reference of every mass-window candidate"]
    verdict = INTEGRITY_FAILURE if failures else None
    if not failures and hit > 0 and np.isfinite(deg) and deg >= majority:
        # T1 exclusion at inference removed references on precursor alone: an inference-eligibility correctness bug
        failures.append(f"{deg:.3f} of the hash-matched TEST spectra have a DEGENERATE (peak-free) hash")
        verdict = "DEGENERATE_HASH_ARTIFACT"
    if failures:
        return {"verdict": verdict, "correctness_failure": True, "protocol_work_status": "REOPEN_REQUIRED", "failures": failures,
                "resemblance_scope": "UNDETERMINED", "text": text + failures}
    if hit == 0:
        verdict = "NO_TRAIN_IDENTITY"
    elif ex >= majority:
        verdict = "VISIBLE_TEST_SPECTRA_PRESENT_IN_TRAIN_LIBRARY"
    elif s["share_hash_equal_arrays_differ"] >= majority:
        verdict = "HASH_LEVEL_EQUIVALENCE_ONLY"
    else:
        verdict = "MIXED"
    text.append(f"exact library duplicates (full peak arrays + precursor identical): {ex:.4f} of TEST spectra; "
                f"hash-equal but arrays differ: {s['share_hash_equal_arrays_differ']:.4f}; hash match anywhere in TRAIN: {hit:.4f}")
    scope = "VISIBLE_AND_HIDDEN_TEST"
    if verdict == "VISIBLE_TEST_SPECTRA_PRESENT_IN_TRAIN_LIBRARY":
        scope = "VISIBLE_TEST_ONLY"
        text += ["the visible-TEST measurements themselves are in the TRAIN library; production inference excludes them as T1, so the production-path "
                 "TEST statistic is, by construction, 'own spectrum removed, other TRAIN spectra kept', which is what test_simulated_strict does",
                 "STRICT_LIKE therefore characterises the VISIBLE test. It transfers to the hidden (unpublished) test only if hidden molecules also "
                 "have other TRAIN spectra; this audit cannot establish that",
                 "the TRAIN annotation of a duplicate is NOT used as a TEST label anywhere; no protocol, model or aggregator choice may use it"]
    elif verdict == "HASH_LEVEL_EQUIVALENCE_ONLY":
        text.append("hash matches are rounding-level equivalents, not identical measurements; T1 exclusion at inference behaves as a tolerant mirror filter")
    return {"verdict": verdict, "correctness_failure": False, "protocol_work_status": "CLOSED", "failures": [], "resemblance_scope": scope, "text": text}


def deterministic_sample(audit, n=20):
    """Up to `n` rows, round-robin over identity_relation (sorted), inside each by sha256(spectrum_id):
    independent of row order and not biased to the start of the file."""
    a = audit.assign(_k=audit["spectrum_id"].astype(str).map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()))
    groups = [g.sort_values("_k", kind="mergesort") for _, g in a.groupby("identity_relation", sort=True)]
    picked, i = [], 0
    while len(picked) < n and any(i < len(g) for g in groups):
        for g in groups:
            if i < len(g) and len(picked) < n:
                picked.append(g.iloc[i])
        i += 1
    return pd.DataFrame(picked, columns=a.columns).drop(columns="_k").reset_index(drop=True)


def format_identity_example(r):
    def f(v, fmt="{:.4f}"):
        return "NA" if _missing(v) else (fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
    return "\n".join([
        f"TEST spectrum      {r['spectrum_id']}  (molecule {f(r['molecule_id'])})  relation={r['identity_relation']}  basis={r['match_basis']}",
        f"matched TRAIN      {f(r['best_train_reference_id'])}  (hash matches in TRAIN: {int(r['n_train_hash_matches'])}, "
        f"distinct TRAIN connectivities among them: {int(r['n_distinct_train_connectivities_in_hash_matches'])})",
        f"source             {f(r['best_train_source'])}",
        f"instrument         TEST {f(r['query_instrument'])} | TRAIN {f(r['train_instrument'])}  equal={r['instrument_equal']}",
        f"adduct             TEST {f(r['query_adduct'])} | TRAIN {f(r['train_adduct'])}  equal={r['adduct_equal']}",
        f"CE (eV)            TEST {f(r['query_collision_energy'], '{:.2f}')} | TRAIN {f(r['train_collision_energy'], '{:.2f}')}  equal={r['collision_energy_equal']}",
        f"precursor m/z      TEST {f(r['query_precursor_mz'], '{:.5f}')} | TRAIN {f(r['train_precursor_mz'], '{:.5f}')}  equal={r['precursor_mz_equal']}",
        f"peak counts        TEST {int(r['n_query_identity_peaks'])} | TRAIN {f(r['n_train_identity_peaks'], '{:.0f}')}",
        f"hash equal?        {r['hash_equal']}   degenerate query hash? {r['query_hash_degenerate']}",
        f"full arrays equal? {r['full_peak_array_equal']}  (m/z {r['mz_array_equal']}, intensity {r['intensity_array_equal']}; "
        f"max |dmz| {f(r['max_abs_mz_diff'], '{:.2e}')}, max |dint| {f(r['max_abs_intensity_diff'], '{:.2e}')})",
        f"possible contamination (same measurement in TRAIN): {r['possible_contamination_flag']}"])


def format_identity_report(s, verdict):
    lines = ["EXACT-LIBRARY-IDENTITY AUDIT (visible TEST vs TRAIN reference universe)", "",
             f"TEST spectra: {s['n_test_spectra']} | molecules: {s['n_test_molecules']}",
             f"reference universe: {s.get('train_hash_index', {}).get('reference_universe', 'TRAIN_ONLY')} "
             f"({s.get('train_hash_index', {}).get('n_reference_spectra', 'NA')} spectra); TEST ids in index: {s['share_query_in_reference_index']:.4f}",
             f"hash match in candidate pool (legacy definition): {s['share_train_hash_match_in_candidate_pool']:.4f}  "
             f"| parity vs executed n_t1: {s['legacy_parity']['n_checked'] - s['legacy_parity']['n_mismatch']}/{s['legacy_parity']['n_checked']}",
             f"hash match anywhere in TRAIN:                     {s['share_train_hash_match_anywhere']:.4f}",
             f"exact library duplicate ({EXACT_DUPLICATE_BASIS}): {s['share_exact_library_duplicate']:.4f}",
             f"hash equal, arrays differ:                        {s['share_hash_equal_arrays_differ']:.4f}",
             f"degenerate (peak-free) query hash:                {s['share_degenerate_query_hash']:.4f}",
             f"hash bucket spans >1 TRAIN connectivity:          {s['share_hash_matches_span_multiple_train_connectivities']:.4f}",
             f"identity relations: {s['identity_relation_counts']}",
             f"molecule level: {s['molecule_level']}", "",
             f"VERDICT: {verdict['verdict']}   | protocol work: {verdict['protocol_work_status']}   | resemblance scope: {verdict['resemblance_scope']}"]
    return "\n".join(lines + [f"  - {t}" for t in verdict["text"]])
