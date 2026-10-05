"""Validation protocol v2: protocol validity (cases A-D), A8 decision thresholds, decision-panel policy, policy identity
and the notebook-14 preflight. Pure functions over already-computed notebook-13 / notebook-11b artifacts.

Protocol validity (universe identity from `casmi.candidates.legacy_adoption.effective_universe_identity`):

    A  no external source in the universe                       -> INVALID
    B  legacy universe, provenance VERIFIED in this run, no      -> VALID_FOR_READ_ONLY_COVERAGE (+ warning: adopt it to
       adoption sidecar                                             bind regimes / experiments to a build id)
    C  modern manifest v3, or legacy ADOPTED (sidecar matches)   -> VALID
    D  contradiction (sidecar vs files, manifest vs scan,        -> INVALID
       regime manifest bound to another universe)

A8 thresholds (rules B, C, D): the repository was searched for "Prompt A" / A8 threshold values. "A8" in this codebase is
the molecule-AGGREGATOR A8 (first spectrum, `casmi_infer.aggregation`), not a coverage threshold, and no value for "very
large pool", "high reachability", "meaningful rescue" or "immaterial loss" exists anywhere. So the thresholds stay null,
the policy reports BLOCKED_A8_THRESHOLDS and notebook 13 REFUSES a final decision (rules are still evaluated and shown).
Rule A keeps the project's 10 pp. Nothing is invented.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

VALID, VALID_READ_ONLY, INVALID = "VALID", "VALID_FOR_READ_ONLY_COVERAGE", "INVALID"
PROTOCOL_V2_VERSION = "casmi-protocol-2"
BLOCKED_A8_THRESHOLDS = "BLOCKED_A8_THRESHOLDS"
DECISION_PANEL_UNAVAILABLE = "DECISION_PANEL_UNAVAILABLE"
A8_THRESHOLD_FIELDS = ("rule_b_min_reachability", "rule_b_large_pool_threshold", "rule_c_min_rescued_fraction",
                       "rule_d_min_coconut_reachability", "rule_d_max_lost_fraction")
ROUNDTRIP_DEPLOYMENT_OK, ROUNDTRIP_DEPLOYMENT_BLOCKED = "ROUNDTRIP_DEPLOYMENT_OK", "ROUNDTRIP_DEPLOYMENT_BLOCKED"
RANKING_DECISIONS = ("PROCEED_TO_ANALOG_BASELINE",)


class ProtocolV2Error(RuntimeError):
    """Notebook-14 preflight failed (protocol, regimes, decision, round trip or panels)."""


def protocol_validity_v2(universe_identity, verification=None, regime_universe_build_id=None, n_population_rows=None):
    """Cases A-D above. `verification`: a `verify_legacy_universe` report computed in THIS run (case B) or None.
    `regime_universe_build_id`: the universe build id recorded in the regime manifest (contradiction if different)."""
    kind = universe_identity.get("kind")
    ext = list(universe_identity.get("external_sources") or [])
    reasons, warnings = [], []
    if kind == "contradiction":
        case, validity = "D", INVALID
        reasons += universe_identity.get("problems") or ["universe identity contradiction"]
    elif kind == "absent":
        case, validity = "A", INVALID
        reasons.append("no finalized universe")
    elif kind == "legacy_unadopted":
        v = verification or {}
        if v.get("status") == "LEGACY_EXTERNAL_VERIFIED":
            case, validity = "B", VALID_READ_ONLY
            ext = sorted(s for s, n in (v.get("per_source") or {}).items() if s != "TRAIN" and n)
            warnings.append("legacy universe verified in this run but NOT adopted: coverage is read-only; run the adoption "
                            "(notebook 11a / scripts/adopt_legacy_universe_manifest.py) before building regimes or ranking")
        elif v.get("status") == "LEGACY_TRAIN_ONLY":
            case, validity = "A", INVALID
            reasons.append("legacy universe has no external source")
        elif v.get("status") == "VERIFICATION_FAILED":
            case, validity = "D", INVALID
            reasons += [f"{c['check']}: {c['detail']}" for c in v.get("checks", []) if not c["passed"]]
        else:
            case, validity = "A", INVALID
            reasons.append("legacy universe without adoption and without verification: external provenance unknown")
    elif not ext:
        case, validity = "A", INVALID
        reasons.append(f"universe ({kind}) has no external source -- external reachability is undefined")
    else:
        case, validity = "C", VALID
    if validity != INVALID and not ext:
        case, validity = "A", INVALID
        reasons.append("no external source")
    bid = universe_identity.get("build_id")
    if validity != INVALID and regime_universe_build_id is not None and bid is not None and regime_universe_build_id != bid:
        case, validity = "D", INVALID
        reasons.append(f"regime manifest bound to universe {regime_universe_build_id}, current universe is {bid}")
    if n_population_rows is not None and not n_population_rows:
        validity = INVALID
        reasons.append("no validation population loaded")
    return {"protocol_version": PROTOCOL_V2_VERSION, "protocol_case": case, "protocol_validity": validity,
            "protocol_valid": validity == VALID, "protocol_valid_for_coverage": validity in (VALID, VALID_READ_ONLY),
            "protocol_reasons": reasons, "protocol_warnings": warnings, "universe_kind": kind, "universe_build_id": bid,
            "external_sources": ext}


def a8_threshold_status(policy):
    """BLOCKED_A8_THRESHOLDS while any rule-B/C/D threshold is null (no project value exists); else SET."""
    missing = [f for f in A8_THRESHOLD_FIELDS if getattr(policy, f) is None]
    return {"status": BLOCKED_A8_THRESHOLDS if missing else "SET", "missing": missing,
            "source": "no Prompt-A / A8 coverage threshold exists in the repository (A8 = aggregator); user must declare them before running",
            "rule_a_min_incremental_pp": policy.rule_a_min_incremental_pp}


def policy_identity_v2(policy, a8_status, decision_panel_order):
    blob = {"decision_policy": asdict(policy), "a8_status": a8_status["status"], "a8_missing": a8_status["missing"],
            "decision_panel_order": list(decision_panel_order), "protocol_version": PROTOCOL_V2_VERSION}
    return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def final_decision_v2(rule_evaluation, protocol, a8_status, decision_panel, roundtrip_status):
    """Overlay on `evaluate_decision_rules`: INVALID protocol -> PROTOCOL_INVALID; no decision panel ->
    DECISION_PANEL_UNAVAILABLE; null A8 thresholds -> BLOCKED_A8_THRESHOLDS (refuses a final decision). Otherwise the
    mechanical decision; a ranking decision additionally needs ROUNDTRIP_DEPLOYMENT_OK."""
    mech = rule_evaluation["decision"]
    blockers = []
    if protocol["protocol_validity"] == INVALID:
        final = "PROTOCOL_INVALID"
    elif not decision_panel.get("decision_panel"):
        final = DECISION_PANEL_UNAVAILABLE
    elif a8_status["status"] == BLOCKED_A8_THRESHOLDS:
        final = BLOCKED_A8_THRESHOLDS
    else:
        final = mech
    if protocol["protocol_validity"] == VALID_READ_ONLY:
        blockers.append("protocol VALID_FOR_READ_ONLY_COVERAGE: adopt the universe before ranking")
    if final in RANKING_DECISIONS and roundtrip_status.get("status") != ROUNDTRIP_DEPLOYMENT_OK:
        blockers.append(f"round trip: {roundtrip_status.get('status')} ({roundtrip_status.get('reason')})")
    permits_ranking = final in RANKING_DECISIONS and protocol["protocol_validity"] == VALID and not blockers
    return {"decision": final, "mechanical_decision": mech, "final_decision_refused": final != mech, "ranking_blockers": blockers,
            "permits_ranking": permits_ranking, "decision_panel": decision_panel.get("decision_panel"),
            "decision_panel_reason": decision_panel.get("reason")}


def roundtrip_deployment_check(failures, emitted_candidate_ids=None, blacklist_ids=None, exhaustive=True):
    """Deployment gate: every EMITTED candidate (default: every candidate) must round-trip, unless explicitly blacklisted.
    No blacklist is applied by this code (the caller must pass one deliberately). A SAMPLED round trip
    (`exhaustive=False`) can never certify deployment."""
    if not exhaustive:
        return {"status": ROUNDTRIP_DEPLOYMENT_BLOCKED, "n_failures_emitted": None, "n_blacklisted": 0, "n_unresolved": None,
                "unresolved_candidate_ids": [], "blacklist_applied": False,
                "reason": "round trip was SAMPLED (coverage.roundtrip.max_per_bucket set): run it on every candidate to certify deployment"}
    ids = set(int(x) for x in failures["candidate_id"]) if len(failures) else set()
    if emitted_candidate_ids is not None:
        ids &= set(int(x) for x in emitted_candidate_ids)
    bl = set(int(x) for x in (blacklist_ids or ()))
    remaining = sorted(ids - bl)
    return {"status": ROUNDTRIP_DEPLOYMENT_OK if not remaining else ROUNDTRIP_DEPLOYMENT_BLOCKED,
            "n_failures_emitted": len(ids), "n_blacklisted": len(ids & bl), "n_unresolved": len(remaining),
            "unresolved_candidate_ids": remaining[:50], "blacklist_applied": bool(bl),
            "reason": "all emitted candidates round-trip" if not remaining else f"{len(remaining)} emitted candidate(s) do not round-trip"}


def require_protocol_v2_ready(report_dir, regime_manifest_path, current_regime_identity=None, override=False):
    """Notebook-14 preflight (never recomputes anything). Blocks unless: gate completed; protocol VALID; regime manifest
    current and bound to the same universe as the gate; decision permits ranking; round-trip deployment OK; decision
    panel valid. `override=True` returns the problems instead of raising (the caller must log it)."""
    from casmi.validation.regimes_v2 import REGIME_CURRENT, check_regime_manifest
    p = Path(report_dir) / "coverage_gate_summary.json"
    s = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    problems = []
    if s is None:
        problems.append(f"{p} not found -- run notebook 13 first")
        s = {}
    if s and not s.get("completed"):
        problems.append("coverage gate did not complete")
    if s and s.get("protocol_validity") != VALID:
        problems.append(f"protocol validity {s.get('protocol_validity')!r} (needs VALID): {s.get('protocol_reasons')}")
    if s and not s.get("permits_ranking"):
        problems.append(f"decision {s.get('decision')!r} does not permit ranking; blockers {s.get('ranking_blockers')}")
    if s and (s.get("roundtrip_deployment") or {}).get("status") != ROUNDTRIP_DEPLOYMENT_OK:
        problems.append(f"round-trip deployment check: {(s.get('roundtrip_deployment') or {}).get('status')}")
    if s and not s.get("decision_panel"):
        problems.append("no valid decision panel (HOST / NP_PANEL)")
    man_p = Path(regime_manifest_path)
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.is_file() else None
    if man is None:
        problems.append(f"regime manifest {man_p} missing -- run notebook 11b")
    else:
        if s and man.get("regime_signature") != s.get("regime_signature"):
            problems.append(f"regime signature {man.get('regime_signature')} != gate's {s.get('regime_signature')} (regimes rebuilt after the gate: re-run 13)")
        if s and (man.get("identity") or {}).get("universe_build_id") != s.get("universe_build_id"):
            problems.append("regime manifest and coverage gate are bound to different universes")
        if current_regime_identity is not None:
            chk = check_regime_manifest(man_p, current_regime_identity)
            if chk["status"] != REGIME_CURRENT:
                problems.append(f"regimes not current: {[m['field'] for m in chk['mismatches']]}")
    if problems and not override:
        raise ProtocolV2Error("notebook 14 preflight failed:\n  - " + "\n  - ".join(problems))
    return {**s, "override_used": bool(problems and override), "preflight_problems": problems}
