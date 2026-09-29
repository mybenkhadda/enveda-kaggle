"""Reference protocols as FILTERS over the existing QCR (v5.2) -- no similarity is recomputed.

A protocol = (blanket source exclusion?, excluded identity tiers). Self-reference is always excluded
upstream: the walk never visits the query's own spectrum (`casmi.qcr.builder` drops it before the walk).

    standard                 no source exclusion, no tier exclusion
                             (STANDARD -- OPTIMISTIC / DUPLICATE-PERMITTING SENSITIVITY)
    mirror_aware             ref_source != query_source, exclude T1+T2    (== existing `is_eligible`)
    test_simulated_strict    no blanket source exclusion, exclude T1+T2, allow T3/T4
    test_simulated_relaxed   no blanket source exclusion, exclude T1 only, allow T2/T3/T4

Tiers are the QCR `tier` column, i.e. the output of the EXISTING classifier
`casmi.spectra.deduplication.classify_identity_tier` -- nothing is re-classified here.

Why the existing QCR suffices (no extension). `walk_references` visits references in compat order and
stops only when EVERY walked protocol (incl. mirror_aware) has 5 accepted references, or the list is
exhausted. Each protocol above has an eligible set that CONTAINS mirror_aware's (standard ⊇ relaxed ⊇
strict ⊇ mirror_aware). So when the walk stops early, mirror_aware already found 5 eligible
references inside the walked prefix, hence so did every superset protocol, and its first 5 eligible
references in compat order all lie in that prefix. When the walk exhausted the list, every reference
is present. `assert_walk_complete` re-checks this on the real data instead of trusting the argument.

Counts: exact when the walk exhausted the list; otherwise the protocol's walked eligible count
(>= 5 by the argument above) is a censored lower bound, reported as ">=5".
"""
import hashlib
import inspect
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from casmi.qcr.aggregate import _aggregate_selected
from casmi.spectra.reference_selection import MAX_REFS_PER_PROTOCOL

TOP_K = MAX_REFS_PER_PROTOCOL
_QCR_KEY = ["query_id", "candidate_key"]
_POOL_KEY = ["query_id", "candidate_connectivity_key"]
QCR_PROTOCOL_COLS = ["query_id", "candidate_key", "is_true", "ref_spectrum_id", "compat_rank", "tier", "ref_source", "query_source",
                     "cosine", "modified_cosine", "peak_overlap_frac", "neutral_loss_cosine", "eligible_mirror_aware", "eligible_standard"]


@dataclass(frozen=True)
class ProtocolDef:
    name: str
    source_exclusion: bool
    excluded_tiers: tuple
    exclude_self: bool = True
    label: str = ""

    @property
    def allow_t3(self):
        return "T3" not in self.excluded_tiers

    @property
    def allow_t4(self):
        return "T4" not in self.excluded_tiers


PROTOCOL_DEFS = {
    "standard": ProtocolDef("standard", False, (), label="STANDARD -- OPTIMISTIC / DUPLICATE-PERMITTING SENSITIVITY"),
    "mirror_aware": ProtocolDef("mirror_aware", True, ("T1", "T2"), label="benchmark mirror-aware (historical primary)"),
    "test_simulated_strict": ProtocolDef("test_simulated_strict", False, ("T1", "T2"), label="test-simulated STRICT: exclude T1+T2, allow same-source T3/T4"),
    "test_simulated_relaxed": ProtocolDef("test_simulated_relaxed", False, ("T1",), label="test-simulated RELAXED: exclude T1 only (T2 sensitivity)"),
}
TEST_SIM_PROTOCOLS = ("test_simulated_strict", "test_simulated_relaxed")


# ---------------------------------------------------------------------------------------------
# T2 rule (pre-registered; takes PROVENANCE inputs only -- no metric can reach it)
# ---------------------------------------------------------------------------------------------

T2_RULE = {
    "rule_id": "v5.2-t2-1",
    "inputs": "among TL_EVAL STANDARD_ONLY queries: q_t34 = share with >=1 SELECTED (standard top-5) truth reference of tier T3/T4; q_t12_only = share whose selected truth references are all T1/T2",
    "rule": "q_t34 >= 0.50 -> STRICT_ONLY (test_simulated primary = exclude T1+T2); else STRICT_AND_RELAXED (carry strict + relaxed as sensitivities; relaxed is never chosen by MRR)",
    "threshold": 0.50,
}


def t2_policy(q_t34, q_t12_only):
    """Pre-registered T2 decision from provenance shares ONLY (the signature admits nothing else)."""
    if not (0.0 <= q_t34 <= 1.0 and 0.0 <= q_t12_only <= 1.0):
        raise ValueError("q_t34 / q_t12_only must be shares in [0, 1]")
    if q_t34 >= T2_RULE["threshold"]:
        return {"t2_policy": "STRICT_ONLY", "primary": "test_simulated_strict", "protocols": ["test_simulated_strict"],
                "q_t34": q_t34, "q_t12_only": q_t12_only, "rule": T2_RULE}
    return {"t2_policy": "STRICT_AND_RELAXED", "primary": "test_simulated_strict", "protocols": list(TEST_SIM_PROTOCOLS),
            "q_t34": q_t34, "q_t12_only": q_t12_only, "rule": T2_RULE,
            "note": "relaxed is a T2 sensitivity carried until provenance or Kaggle evidence resolves T2; it is never chosen by score"}


# ---------------------------------------------------------------------------------------------
# eligibility, completeness, counts, selection, features
# ---------------------------------------------------------------------------------------------

def eligible_mask(qcr, pdef):
    """Boolean eligibility of every walked QCR row under `pdef` (self already excluded upstream)."""
    ok = ~qcr["tier"].astype(str).isin(list(pdef.excluded_tiers)).to_numpy()
    if pdef.source_exclusion:
        ok &= (qcr["ref_source"].astype(str) != qcr["query_source"].astype(str)).to_numpy()
    return ok


def check_existing_flag_parity(qcr):
    """The filter definitions of the two EXISTING protocols must reproduce the QCR's own eligibility
    flags exactly (proves the filters reuse, not re-define, the existing semantics)."""
    out = {}
    for name in ("mirror_aware", "standard"):
        col = f"eligible_{name}"
        if col in qcr.columns:
            out[name] = int((eligible_mask(qcr, PROTOCOL_DEFS[name]) != qcr[col].astype(bool).to_numpy()).sum())
    return out


def protocol_eligible_counts(qcr, pairs, pool, pdef):
    """Per pool row: `n_eligible` (exact or lower bound), `count_is_exact`, `count_censored`."""
    q = qcr[_QCR_KEY].assign(_e=eligible_mask(qcr, pdef))
    agg = q.groupby(_QCR_KEY, sort=False)["_e"].sum().rename("n_eligible_walked").reset_index()
    agg = agg.merge(pairs[_QCR_KEY + ["exhausted"]], on=_QCR_KEY, how="left", validate="one_to_one")
    agg = agg.rename(columns={"candidate_key": "candidate_connectivity_key"})
    out = pool[_POOL_KEY + ["is_true_candidate"]].merge(agg, on=_POOL_KEY, how="left", validate="one_to_one")
    out["n_eligible_walked"] = out["n_eligible_walked"].fillna(0).astype(int)
    out["exhausted"] = out["exhausted"].fillna(True).astype(bool)
    out["count_is_exact"] = out["exhausted"]
    out["count_censored"] = ~out["exhausted"]
    out["n_eligible"] = out["n_eligible_walked"]
    return out


def assert_walk_complete(counts, top_k=TOP_K):
    """Every non-exhausted (query, candidate) must already have >= top_k eligible walked references,
    otherwise its protocol top-k could lie beyond the walked prefix (QCR would need extending)."""
    bad = counts[counts["count_censored"] & (counts["n_eligible_walked"] < top_k)]
    if len(bad):
        raise AssertionError(f"{len(bad)} early-stopped walks have < {top_k} eligible references under this protocol -- "
                             "the existing QCR is insufficient for it; a QCR extension would be required")
    return True


def select_protocol_refs(qcr, pdef, k=TOP_K):
    """The protocol's first k eligible references per (query, candidate) in compat_rank order --
    the same deterministic top-k rule as `walk_references`."""
    e = qcr[eligible_mask(qcr, pdef)].sort_values(_QCR_KEY + ["compat_rank"], kind="mergesort")
    return e[e.groupby(_QCR_KEY, sort=False).cumcount() < k]


def aggregate_protocol(qcr, pool, pdef, k=TOP_K):
    """Candidate features under `pdef`: identical aggregation to V1 (max + top3_mean of the 4 metrics
    over the selected references; NaN only when none)."""
    return _aggregate_selected(select_protocol_refs(qcr, pdef, k), pool)


def features_from_shards(shard_paths, pool, pdef, k=TOP_K, columns=QCR_PROTOCOL_COLS):
    """Shard-by-shard protocol features (bounded RAM) + the selected-reference tier audit rows."""
    pool_by_q = {q: g for q, g in pool.groupby("query_id")}
    parts, sel_audit = [], []
    for p in shard_paths:
        q = pd.read_parquet(p, columns=columns)
        if not len(q):
            continue
        sub_pool = pd.concat([pool_by_q[i] for i in q["query_id"].unique() if i in pool_by_q], ignore_index=True)
        sel = select_protocol_refs(q, pdef, k)
        parts.append(_aggregate_selected(sel, sub_pool))
        sel_audit.append(sel[["query_id", "candidate_key", "is_true", "tier", "ref_source", "query_source"]])
    feats = pd.concat(parts, ignore_index=True) if parts else _aggregate_selected(pd.DataFrame(columns=_QCR_KEY), pool.iloc[0:0])
    miss = pool.merge(feats[_POOL_KEY], how="left", indicator=True)
    miss = miss[miss["_merge"] == "left_only"].drop(columns="_merge")
    if len(miss):
        feats = pd.concat([feats, _aggregate_selected(pd.DataFrame(columns=_QCR_KEY), miss)], ignore_index=True)
    audit = pd.concat(sel_audit, ignore_index=True) if sel_audit else pd.DataFrame(columns=["query_id", "candidate_key", "is_true", "tier"])
    return feats.sort_values(_POOL_KEY).reset_index(drop=True), audit


def forbidden_tier_counts(selected_audit, pdef):
    """T1 violations and any selected reference in a tier the protocol excludes (must both be 0)."""
    t = selected_audit["tier"].astype(str)
    return {"t1_violation_count": int((t == "T1").sum()) if "T1" in pdef.excluded_tiers else 0,
            "forbidden_tier_count": int(t.isin(list(pdef.excluded_tiers)).sum()),
            "same_source_violation_count": int((selected_audit["ref_source"].astype(str) == selected_audit["query_source"].astype(str)).sum())
            if pdef.source_exclusion else 0}


# ---------------------------------------------------------------------------------------------
# coverage, manifests, definition record, semantics
# ---------------------------------------------------------------------------------------------

def censored_median(values, censored):
    v = np.where(np.asarray(censored, bool), np.inf, np.asarray(values, float))
    if not len(v):
        return None
    s = np.sort(v)
    a, b = s[(len(s) - 1) // 2], s[len(s) // 2]
    return ">=5" if np.isinf(a) or np.isinf(b) else float((a + b) / 2)


def coverage_row(counts, all_query_ids, protocol_name):
    """Truth coverage over ALL queries (truth absent from pool counts as no evidence) + decoy coverage."""
    ids = pd.Index(list(all_query_ids))
    truth = counts[counts["is_true_candidate"]].drop_duplicates("query_id").set_index("query_id").reindex(ids)
    n_truth = truth["n_eligible"].fillna(0)
    decoys = counts[~counts["is_true_candidate"]]
    per_q = decoys.assign(_has=decoys["n_eligible"] >= 1).groupby("query_id")["_has"].mean()
    truth_share = float((n_truth >= 1).mean())
    decoy_share = float((decoys["n_eligible"] >= 1).mean()) if len(decoys) else float("nan")
    in_pool = truth["n_eligible"].notna()
    return {"protocol": protocol_name, "n_queries": int(len(ids)), "truth_in_pool": float(in_pool.mean()),
            "truth_ge1": truth_share, "truth_ge3": float((n_truth >= 3).mean()), "truth_ge5": float((n_truth >= 5).mean()),
            "median_eligible_truth_refs": censored_median(truth.loc[in_pool, "n_eligible"], truth.loc[in_pool, "count_censored"].astype(bool)),
            "decoy_ge1_share_pooled": decoy_share, "decoy_ge1_share_mean_per_query": float(per_q.mean()) if len(per_q) else float("nan"),
            "decoy_ge1_share_median_per_query": float(per_q.median()) if len(per_q) else float("nan"),
            "reference_availability_gap": truth_share - decoy_share}


def truth_evidence_ids(counts):
    """Queries whose truth is in the pool with >= 1 eligible reference under the protocol."""
    t = counts[counts["is_true_candidate"] & (counts["n_eligible"] >= 1)]
    return set(t["query_id"])


def derive_protocol_manifest(manifest, counts, suffix):
    """Protocol-matched Mode-A training subset (a copy; the input manifest is never mutated)."""
    keep_ids = truth_evidence_ids(counts)
    keep = manifest["query_id"].isin(keep_ids).to_numpy()
    out = manifest.loc[keep].copy()
    out["selection_reason"] = out["selection_reason"].astype(str) + f" | {suffix} filter: truth in pool with >=1 eligible reference under the protocol"
    return out, {"n_original": int(len(manifest)), "n_retained": int(keep.sum()), "retention_rate": float(keep.mean()) if len(manifest) else float("nan")}


def protocol_definition_record(pdef, t2_decision, code_version, config_hash, top_k=TOP_K):
    import casmi.spectra.reference_selection as rs
    rec = {"protocol_name": pdef.name, "label": pdef.label, "source_exclusion": "ref_source != query_source" if pdef.source_exclusion else "NONE",
           "exclude_self": pdef.exclude_self, "exclude_T1": "T1" in pdef.excluded_tiers, "exclude_T2": "T2" in pdef.excluded_tiers,
           "allow_T3": pdef.allow_t3, "allow_T4": pdef.allow_t4,
           "compat_rank_version": hashlib.sha256(inspect.getsource(rs).encode("utf-8")).hexdigest()[:16], "top_k_refs": top_k,
           "selection_rule": "first top_k eligible references in compat_sort_key order (walk_references); aggregation max + top3_mean (V1)",
           "provenance_decision_inputs": {k: t2_decision[k] for k in ("t2_policy", "q_t34", "q_t12_only")}, "t2_rule": T2_RULE,
           "code_version": code_version, "config_hash": config_hash}
    rec["definition_sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return rec


def protocol_semantics_table(selected_protocol="test_simulated_strict", kaggle_config=None):
    """Benchmark mirror vs test_simulated vs the CURRENT production hidden-test inference
    (`casmi_infer`, read from the bundle config when given). Mismatches are flagged, never fixed here."""
    kc = kaggle_config or {}
    kaggle_t2 = bool((kc.get("identity_tiers_applied") or {}).get("T2", False))
    pd_ = PROTOCOL_DEFS[selected_protocol]
    rows = [
        ("source exclusion", "ref_source != query_source", "NONE" if not pd_.source_exclusion else "ref_source != query_source",
         "NONE (exclude_sources = {})" if not kc.get("exclude_sources") else str(kc.get("exclude_sources"))),
        ("T1 exclusion", "yes", "yes" if "T1" in pd_.excluded_tiers else "no", "yes (peak hash)"),
        ("T2 exclusion", "yes", "yes" if "T2" in pd_.excluded_tiers else "no", "yes" if kaggle_t2 else "NO (identity peaks not shipped)"),
        ("T3/T4 allowed", "yes (other-source only)", "yes (any source)" if pd_.allow_t3 and pd_.allow_t4 else "partial", "yes (any source)"),
        ("self-reference possible?", "no (walk excludes query id)", "no (walk excludes query id)", "no (hidden query is not in the library)"),
        ("query published?", "yes (benchmark query is a library spectrum)", "yes (benchmark query is a library spectrum)", "no (unpublished)"),
        ("reference source known?", "yes", "yes", "yes (exported ref_meta)"),
    ]
    t = pd.DataFrame(rows, columns=["property", "benchmark_mirror_aware", selected_protocol, "hidden_kaggle_casmi_infer"])
    t["mismatch_testsim_vs_kaggle"] = [
        pd_.source_exclusion != bool(kc.get("exclude_sources")),
        "T1" not in pd_.excluded_tiers,
        ("T2" in pd_.excluded_tiers) != kaggle_t2,
        False,
        False,
        True,          # benchmark queries are published library spectra; hidden queries are not -- irreducible
        False,
    ]
    return t


def protocol_counts_from_shards(qcr_paths, pair_paths, pool, protocol_names):
    """Eligible-reference counts for several protocols in ONE bounded pass over QCR shards.
    Returns {protocol: counts frame as in `protocol_eligible_counts`}."""
    sums = {p: [] for p in protocol_names}
    for path in qcr_paths:
        q = pd.read_parquet(path, columns=["query_id", "candidate_key", "tier", "ref_source", "query_source"])
        if not len(q):
            continue
        for p in protocol_names:
            s = q[_QCR_KEY].assign(_e=eligible_mask(q, PROTOCOL_DEFS[p])).groupby(_QCR_KEY, sort=False)["_e"].sum()
            sums[p].append(s)
    pairs = pd.concat([pd.read_parquet(p, columns=["query_id", "candidate_key", "exhausted"]) for p in pair_paths], ignore_index=True) \
        if pair_paths else pd.DataFrame(columns=["query_id", "candidate_key", "exhausted"])
    out = {}
    for p in protocol_names:
        agg = (pd.concat(sums[p]).groupby(level=[0, 1]).sum().rename("n_eligible_walked").reset_index()
               if sums[p] else pd.DataFrame(columns=_QCR_KEY + ["n_eligible_walked"]))
        agg = agg.merge(pairs, on=_QCR_KEY, how="left").rename(columns={"candidate_key": "candidate_connectivity_key"})
        c = pool[_POOL_KEY + ["is_true_candidate"]].merge(agg, on=_POOL_KEY, how="left", validate="one_to_one")
        c["n_eligible_walked"] = c["n_eligible_walked"].fillna(0).astype(int)
        c["exhausted"] = c["exhausted"].fillna(True).astype(bool)
        c["count_is_exact"], c["count_censored"], c["n_eligible"] = c["exhausted"], ~c["exhausted"], c["n_eligible_walked"]
        out[p] = c
    return out


PRIMARY_PROTOCOL = "test_simulated_strict"      # v5.3: accepted after the provenance audit + 1k SCALE_GO


def protocol_semantic_hash(protocol):
    """Stable identity of a protocol's SEMANTICS (not of the code version): name, source exclusion,
    excluded tiers, self exclusion, top-k, and the compat/walk source. Changes only if the protocol
    definition or the reference-selection code changes."""
    import casmi.spectra.reference_selection as rs
    p = PROTOCOL_DEFS[protocol]
    payload = {"name": p.name, "source_exclusion": p.source_exclusion, "excluded_tiers": sorted(p.excluded_tiers), "exclude_self": p.exclude_self,
               "top_k": TOP_K, "compat_walk_source_sha": hashlib.sha256(inspect.getsource(rs).encode("utf-8")).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def verify_protocol_definition_file(path, protocol):
    """The persisted protocol artifact (10v5_04) exists, is internally consistent (recorded sha256 ==
    sha256 of its own content) and defines the same semantics as `PROTOCOL_DEFS[protocol]`."""
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return False, f"{path} missing"
    doc = json.loads(path.read_text(encoding="utf-8"))
    rec = (doc.get("definitions") or {}).get(protocol)
    if rec is None:
        return False, f"{protocol} not defined in {path.name}"
    body = {k: v for k, v in rec.items() if k != "definition_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest() != rec.get("definition_sha256"):
        return False, "definition_sha256 does not match the recorded definition"
    p = PROTOCOL_DEFS[protocol]
    same = (rec["exclude_T1"] == ("T1" in p.excluded_tiers) and rec["exclude_T2"] == ("T2" in p.excluded_tiers) and rec["allow_T3"] == p.allow_t3
            and rec["allow_T4"] == p.allow_t4 and (rec["source_exclusion"] == "NONE") == (not p.source_exclusion) and rec["top_k_refs"] == TOP_K)
    return (True, "ok") if same else (False, "persisted definition semantics differ from PROTOCOL_DEFS")
