"""Class-2 protocol validity -- the single place that decides whether local "C2" numbers simulate Class 2.

A hidden Class-2 molecule has NO reference spectrum; its structure can only be retrieved from an external database.
A local C2 evaluation is therefore valid only if ALL of the following hold:

  1. the candidate universe contains structures from at least one non-TRAIN source (COCONUT / PubChem / ...);
  2. every C2 truth exists in such an external source (enforced when regimes are built: `c2_require_external_source`);
  3. every reference spectrum of a C2 truth is hidden (`casmi.validation.regimes.assert_regime_integrity`);
  4. no TRAIN-provenance / reference-availability column is a ranking feature (`assert_no_train_provenance_features`);
  5. the regime table was built for THIS universe (identity match) and C2 has queries.

A TRAIN-only universe fails (1): hidden C2 truths would be the only reference-less TRAIN structures in their pools (a
perfect shortcut), and "external-only" recall is undefined. Such a universe is labelled `train_only`, its C2 mode is
`PRELIMINARY_train_only_universe`, and Gate A is recorded as `FAIL_PROTOCOL_INVALID` with the reason -- no candidate-
recall metric is reported as C2 evidence.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path

C2_MODE_EXTERNAL = "external_universe"
C2_MODE_PRELIM_TRAIN_ONLY = "PRELIMINARY_train_only_universe"
C2_MODE_PRELIM_NO_UNIVERSE = "PRELIMINARY_no_universe"
UNIVERSE_ABSENT, UNIVERSE_TRAIN_ONLY, UNIVERSE_EXTERNAL = "absent", "train_only", "external"
GATE_A_FAIL_PROTOCOL_INVALID = "FAIL_PROTOCOL_INVALID"
NO_EXTERNAL_REASON = ("No non-TRAIN candidate source contributed to the candidate universe; C2 external-universe recall is "
                      "undefined. Put a COCONUT export at data/external/coconut/coconut.csv (columns: configs/v6/coconut_source.json), "
                      "then run 12 -> 11 (REBUILD=True) -> 13.")

# reference-availability / TRAIN-provenance columns: never ranking features in a C2 evaluation
TRAIN_PROVENANCE_COLUMNS = ("train_present", "in_train", "is_train_structure", "has_reference_spectrum", "candidate_sources",
                            "source_ids", "n_mass_variants", "n_train_spectra")
# source flags that correlate with TRAIN membership: allowed only after an explicit provenance ablation
PROVENANCE_REVIEW_COLUMNS = ("source_count", "coconut_present", "pubchem_present", "n_source_records")


@dataclass(frozen=True)
class UniverseStatus:
    universe_present: bool
    universe_status: str                  # absent | train_only | external
    external_source_present: bool
    c2_protocol_valid: bool               # universe-level validity (conditions 1); regimes / Gate A add 2, 5
    c2_mode: str                          # mode notebook 11 must use with this universe
    n_candidates: int | None = None
    source_summary: dict | None = None
    identity: dict | None = None          # lightweight universe identity (no hashing)
    reason: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


def universe_identity(manifest, summary):
    """Lightweight identity of a finalized universe: enough to detect that a regime table / cache was built for a
    different universe. No content hashing (multi-GB data); a rebuild changes `finalized_at`."""
    if manifest is None:
        return None
    return {"universe_schema_version": manifest.get("universe_schema_version"), "n_candidates": manifest.get("n_candidates"),
            "n_buckets": manifest.get("n_buckets"), "finalized_at": manifest.get("finalized_at"),
            "external_any": (summary or {}).get("external_any")}


def universe_status_from_summary(summary, manifest=None, min_external_candidates=1, min_external_share=1e-4):
    """UniverseStatus from a source summary (`casmi.candidates.universe.universe_source_summary` /
    `universe_manifest.json['source_summary']`). `summary=None` -> absent."""
    if summary is None:
        return UniverseStatus(False, UNIVERSE_ABSENT, False, False, C2_MODE_PRELIM_NO_UNIVERSE,
                              reason="no finalized candidate universe (universe_manifest.json absent) -- run notebook 12")
    n = int(summary.get("n_candidates") or 0)
    ext = int(summary.get("external_any") or 0)
    share = ext / n if n else 0.0
    external = ext >= max(int(min_external_candidates), 1) and share >= float(min_external_share)
    ident = universe_identity(manifest, summary)
    if external:
        sources = sorted(k for k in (summary.get("per_source") or {}) if k != "TRAIN")
        return UniverseStatus(True, UNIVERSE_EXTERNAL, True, True, C2_MODE_EXTERNAL, n, summary, ident,
                              reason=f"{ext:,} of {n:,} candidates from non-TRAIN sources {sources}")
    reason = NO_EXTERNAL_REASON if ext == 0 else (f"only {ext:,} external candidates ({share:.2e} of {n:,}) -- below the configured "
                                                  f"minimum (c2_protocol in configs/casmi_v2_colab.yaml)")
    return UniverseStatus(True, UNIVERSE_TRAIN_ONLY, ext > 0, False, C2_MODE_PRELIM_TRAIN_ONLY, n, summary, ident, reason=reason)


def assess_universe(artifacts, cfg=None, recompute_if_missing=True):
    """UniverseStatus of the universe the registry points at. Reads `universe_manifest.json['source_summary']`
    (manifest v2); for older manifests falls back to manifests/universe_source_summary.json, then (if allowed) to a
    bucket-by-bucket recount. A manifest's existence alone NEVER makes the universe "external"."""
    from casmi.workspace.artifact_registry import read_json
    c2 = (cfg or {}).get("c2_protocol") or {}
    manifest = read_json(artifacts.universe_manifest)
    if manifest is None or not Path(artifacts.bucket_offsets).is_file():
        return universe_status_from_summary(None)
    summary = manifest.get("source_summary") or read_json(artifacts.universe_source_summary)
    if summary is None and recompute_if_missing:
        from casmi.candidates.universe import universe_source_summary
        summary = universe_source_summary(artifacts.universe_root)
    if summary is None:
        return UniverseStatus(True, UNIVERSE_TRAIN_ONLY, False, False, C2_MODE_PRELIM_TRAIN_ONLY, manifest.get("n_candidates"), None,
                              universe_identity(manifest, None),
                              reason="universe has no source summary (built by an older notebook 12) -- re-run notebook 12")
    return universe_status_from_summary(summary, manifest, c2.get("min_external_candidates", 1), c2.get("min_external_share", 1e-4))


def assess_c2_protocol(universe, regimes_meta, n_c2_queries):
    """Full C2 validity for an evaluation (notebooks 13 / 14). Returns {'protocol_valid', 'reasons', 'c2_mode'}."""
    reasons = []
    if not universe.c2_protocol_valid:
        reasons.append(universe.reason)
    mode = (regimes_meta or {}).get("c2_mode")
    if regimes_meta is None:
        reasons.append("no regime table -- run notebook 11")
    elif mode != C2_MODE_EXTERNAL:
        reasons.append(f"regime table is {mode}: C2 truths were not required to exist in an external source "
                       f"(reference-availability shortcut) -- run notebook 11 with REBUILD=True after a valid notebook 12")
    else:
        built_for = ((regimes_meta or {}).get("signature") or {}).get("universe_identity")
        if built_for != universe.identity:
            reasons.append("regime table was built for a different universe -- run notebook 11 with REBUILD=True")
    if not n_c2_queries:
        reasons.append("no C2 query in the regime table")
    return {"protocol_valid": not reasons, "reasons": reasons, "c2_mode": mode, "universe_status": universe.universe_status}


def gate_a_protocol_status(decision):
    """True only for a persisted Gate-A decision that was computed under a VALID C2 protocol."""
    return bool(decision) and decision.get("protocol_valid") is True and decision.get("decision") != GATE_A_FAIL_PROTOCOL_INVALID


def gate_a_metrics(summary, gate_cfg, regime="C2", mode="external_only"):
    """The Gate-A summary numbers from a `casmi.candidates.recall.recall_by_ppm(..., group_cols=('mode', 'regime'))`
    table. Only call when the protocol is valid. Missing ppm rows are reported as None (never invented)."""
    s = summary[(summary["mode"] == mode) & (summary["regime"] == regime)]

    def at(ppm, col):
        r = s[s["ppm"] == ppm]
        return float(r[col].iloc[0]) if len(r) and col in r else None
    out = {f"recall_all@{p}ppm": at(p, "recall_all") for p in gate_cfg["recall_all_ppm"]}
    k = gate_cfg["recall_at_k_ppm"]
    out.update({f"recall@100@{k}ppm": at(k, "recall_at_100"), f"recall@25@{k}ppm": at(k, "recall_at_25")})
    q = gate_cfg["pool_quantiles_ppm"]
    out.update({f"pool_median@{q}ppm": at(q, "pool_size_median"), f"pool_p90@{q}ppm": at(q, "pool_size_p90"),
                f"pool_p99@{q}ppm": at(q, "pool_size_p99")})
    return out


def gate_a_decision(protocol, summary, gate_cfg):
    """Persistable Gate-A record. Invalid protocol -> FAIL_PROTOCOL_INVALID with the reasons and NO metrics."""
    if not protocol["protocol_valid"]:
        return {"decision": GATE_A_FAIL_PROTOCOL_INVALID, "protocol_valid": False, "reason": " | ".join(protocol["reasons"]),
                "metrics": None, "c2_mode": protocol.get("c2_mode"), "universe_status": protocol.get("universe_status")}
    from casmi.candidates.recall import gate_a_assessment
    s = summary[(summary["mode"] == "external_only") & (summary["regime"] == "C2")]
    heuristic = gate_a_assessment(s, ppm_ref=gate_cfg["decision_ppm"])
    return {"decision": heuristic["decision"], "protocol_valid": True, "reason": heuristic.get("label", ""),
            "metrics": gate_a_metrics(summary, gate_cfg), "heuristic": heuristic, "c2_mode": protocol.get("c2_mode"),
            "universe_status": protocol.get("universe_status")}


def assert_no_train_provenance_features(feature_names):
    """Raise if a ranking feature reveals TRAIN provenance / reference availability (C2 shortcut). Returns the list
    of provenance-correlated columns that need an explicit ablation before use (a warning list, not an error)."""
    from casmi.candidates.provenance import assert_no_shortcut_features
    assert_no_shortcut_features(feature_names)
    bad = sorted(set(feature_names) & set(TRAIN_PROVENANCE_COLUMNS))
    if bad:
        raise AssertionError(f"TRAIN-provenance columns are not allowed as C2 ranking features: {bad}")
    return sorted(set(feature_names) & set(PROVENANCE_REVIEW_COLUMNS))
