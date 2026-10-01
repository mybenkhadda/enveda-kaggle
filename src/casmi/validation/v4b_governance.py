"""v4b governance: explicit waiver log, decisions document, Mode-A freeze conditions.

Nothing is silently dropped: every v4a/v4a.1 requirement not re-certified in v4b is listed here
with its reason, the risk that remains, and the concrete condition under which it must be revisited.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

WAIVERS = [
    {
        "id": "W1",
        "requirement": "strengthened C9 brute-force equivalence (reference ids AND aggregated features, HOST + DEV)",
        "decision": "WAIVED for starting v4b",
        "reason": ("historical v4a selection-equivalence test (300 HOST + 100 DEV pairs, 0 mismatches) already covers the "
                   "selection logic; C3 (fresh-vs-QCR similarity values) handles similarity correctness and is settled in "
                   "10v4b_00; the strengthened-C9 notebook-state failure (NameError / cache-only classify) is a notebook "
                   "defect, not evidence of wrong rankings"),
        "remaining_risk": "if C3 is materially wrong, the old selection equivalence alone is insufficient",
        "revisit_when": "only if the 10v4b_00 cache diagnostic shows material discrepancies (decision != ACCEPT_EXISTING and the one clean rebuild does not validate)",
    },
    {
        "id": "W2",
        "requirement": "<= 60 s warm Restart-Kernel -> Run-All runtime",
        "decision": "WAIVED",
        "reason": "engineering/performance target, not scientific validity",
        "remaining_risk": "slow iteration only; no effect on metric correctness",
        "revisit_when": "before any submission pipeline that must run end-to-end under a time budget",
        "v4b_requirement": ("10v4b_01 must load persisted feature/QCR artifacts and must NOT rebuild the full evidence layer "
                            "during normal model iteration (QCR is only rebuilt by 10v4b_00's one-time REBUILD_ONCE branch and "
                            "by the offline scale-feature script)"),
    },
    {
        "id": "W3",
        "requirement": "exhaustive C4/C5 gate certification (source-stratified tier audit, impostor audit)",
        "decision": "WAIVED as a gate",
        "reason": "not required to start ranking experiments",
        "remaining_risk": "T3 impostor decoys could be a systematic failure mode that no gate quantified",
        "revisit_when": "folded into the v4b near-miss analysis (T3 impostor / near-duplicate / same-formula / high-Tanimoto failures in 10v4b_01)",
    },
]

NOT_REQUIRED_FOR_FREEZE = ["60-second notebook runtime", "full v4a gate perfection", "old reference cap re-audit",
                           "notebook-state C9 NameError"]

HISTORICAL_REFERENCE = {
    "_label": "HISTORICAL REFERENCE -- v4a/v4a.1 values, NOT v4b measurements",
    "host_candidate_rows": 126573, "dev_candidate_rows": 248027, "dev_query_coverage": "996 / 1,000",
    "cap_only_selection_effect": "~0%", "ordering_effect_truth": "~98.31%", "ordering_effect_decoy": "~50.98%",
    "ordering_effect_row_weighted": "~51.42%", "peak_overlap_legacy_row_order_mrr": "~0.576",
    "peak_overlap_deterministic_mass_tiebreak_mrr": "~0.591", "v0_mirror_aware_mrr": "~0.713",
    "v0_k1": "~0.620", "v0_k3": "~0.684", "v0_k5": "~0.714",
    "c3_fresh_vs_qcr": "22 / 1,000 sampled pairs differed at atol=1e-12",
}


def write_waivers(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"waivers": WAIVERS, "not_required_for_mode_a_freeze": NOT_REQUIRED_FOR_FREEZE,
               "historical_reference": HISTORICAL_REFERENCE, "written_at": datetime.now(timezone.utc).isoformat()}
    path = out_dir / "waivers.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def mode_a_freeze_conditions(evidence):
    """`evidence`: dict of runtime facts gathered by 10v4b_01. Returns `(status, conditions)`
    where status is "FROZEN" only if every condition holds. Conditions (spec section 49):"""
    conditions = {
        "cache_settled": evidence.get("cache_decision") in ("ACCEPT_EXISTING", "REBUILT_AND_VALIDATED"),
        "dev_selected_headline_model_exists": bool(evidence.get("selected_model_id")),
        "host_confirmation_metrics_exist": bool(evidence.get("host_metrics_exist")),
        "connectivity_bootstrap_exists": bool(evidence.get("bootstrap_exists")),
        "k_stress_1_3_5_exists": bool(evidence.get("k_stress_exists")),
        "b2b_popularity_baseline_exists": bool(evidence.get("b2b_exists")),
        "near_miss_analysis_exists": bool(evidence.get("near_miss_exists")),
        "no_unresolved_missingness_shortcut": bool(evidence.get("top3_contract_holds")),
    }
    return ("FROZEN" if all(conditions.values()) else "NOT_FROZEN"), conditions
