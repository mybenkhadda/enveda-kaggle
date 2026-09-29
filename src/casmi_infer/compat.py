"""Reference compatibility ordering + protocol eligibility at inference.

`compat_order` is a VECTORIZED re-implementation of training's
`casmi.spectra.reference_selection.compat_sort_key` (parity-tested):

    (same_adduct DESC, same_ion_mode DESC, ce_comparable DESC, |ce_diff| ASC (incomparable = inf),
     same_instrument DESC, reference_spectrum_id ASC)

with training's missing-value semantics: a missing (NaN) value is never "same" as anything, and CE
is comparable only when both sides have a finite collision energy (training sets ce_unit="eV"
exactly when ce is known).

Eligibility = `mirror_aware` evaluated with an explicit `exclude_sources` set:

    hidden test (default)  exclude_sources = {}            -- unpublished spectra: same-source exclusion is meaningless
    HOST benchmark mode    exclude_sources = {query source} and exclude_ids = {query spectrum id}
                           == training's mirror_aware + self-exclusion exactly, EXCEPT for T2 (below)

Identity tiers at inference:
    T1 (exact mirror) -- APPLIED: query peak hash vs the exported reference peak hash.
    T2 (tolerant mirror) -- NOT APPLIED: it needs the untruncated identity peaks of every reference,
        which the bundle does not ship. For unpublished test spectra a tolerant mirror of a library
        spectrum is not expected; in HOST benchmark mode the affected candidates are enumerated from
        QCR and reported as KNOWN T2 EXCEPTIONS by 11_01 (P1), never hidden.
    T3/T4 -- irrelevant to mirror_aware eligibility.
"""
import numpy as np

from casmi.spectra.reference_selection import MAX_REFS_PER_PROTOCOL, MIRROR_TIERS

T2_SUPPORTED = False


def compat_order(lib, ref_rows, qmeta):
    """Positions sorting `ref_rows` in training compat order for query metadata `qmeta`."""
    q_add, q_ion, q_inst = lib.code("adduct", qmeta.get("adduct")), lib.code("polarity", qmeta.get("ion_mode")), lib.code("instrument", qmeta.get("instrument"))
    r = ref_rows
    same_add = (lib.adduct_code[r] == q_add) & (q_add >= 0)
    same_ion = (lib.polarity_code[r] == q_ion) & (q_ion >= 0)
    same_inst = (lib.instrument_code[r] == q_inst) & (q_inst >= 0)
    q_ce = qmeta.get("ce")
    q_ce = float("nan") if q_ce is None else float(q_ce)
    r_ce = lib.ce[r]
    comparable = np.isfinite(r_ce) & np.isfinite(q_ce)
    ce_diff = np.where(comparable, np.abs(r_ce - q_ce), np.inf)
    return np.lexsort((lib.sid_rank[r], (~same_inst).astype(np.int8), ce_diff, (~comparable).astype(np.int8),
                       (~same_ion).astype(np.int8), (~same_add).astype(np.int8)))


def reference_tiers(lib, ref_rows, query_hash):
    """'T1' where the exported reference peak hash equals the query's; otherwise 'T4*' (T2/T3 not
    evaluated at inference -- see module docstring)."""
    if query_hash is None:
        return np.full(len(ref_rows), "T4*", dtype=object)
    return np.where(lib.peak_hash[ref_rows] == query_hash, "T1", "T4*").astype(object)


def select_references(lib, conn_idx, qmeta, query_hash, exclude_sources=frozenset(), exclude_ids=frozenset(),
                      max_refs=MAX_REFS_PER_PROTOCOL):
    """The first `max_refs` ELIGIBLE references of one candidate in compat order -- training's lazy
    walk collapsed to its mirror_aware outcome (walk order never depends on similarity values).
    Returns `(selected_ref_rows, n_refs_total, n_t1_excluded)`."""
    refs = lib.refs_of(conn_idx)
    if len(exclude_ids):
        refs = refs[~np.isin(lib.spectrum_id[refs], list(exclude_ids))]
    if len(refs) == 0:
        return refs, 0, 0
    ordered = refs[compat_order(lib, refs, qmeta)]
    tiers = reference_tiers(lib, ordered, query_hash)
    # set membership, not np.isin: the source column may hold None next to strings
    src_ok = (np.fromiter((s not in exclude_sources for s in lib.source[ordered]), dtype=bool, count=len(ordered))
              if len(exclude_sources) else np.ones(len(ordered), bool))
    elig = src_ok & ~np.isin(tiers, list(MIRROR_TIERS))
    return ordered[elig][:max_refs], len(refs), int((tiers == "T1").sum())
