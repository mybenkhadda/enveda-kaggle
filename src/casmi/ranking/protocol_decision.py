"""v5.2 pre-registered SCALE_GO decision for the test-simulated protocol (1k pilot -> 3k/10k).

Order is fixed: semantic validity FIRST (provenance rule, no T1 eligible, no forbidden tier selected),
then evidence size (>= 100 TL_EVAL queries with truth evidence), THEN performance (V1 MRR > B1
mass-only on TL_EVAL). A protocol that fails any semantic check is never rescued by its score.
`standard` is a sensitivity upper bound and can never be selected. HOST can never enter.
"""
from casmi.qcr.protocols import PROTOCOL_DEFS
from casmi.ranking.selection import HostLeakError, guard_no_host

MIN_EVAL_QUERIES_WITH_TRUTH_EVIDENCE = 100
NEXT_RUN, NEXT_T2_REVIEW, NEXT_INVALID = "RUN_3K_10K_SCALING", "NEED_T2_PROVENANCE_REVIEW", "PROTOCOL_INVALID"
SCALE_GO_RULE = {
    "rule_id": "v5.2-scale-go-1",
    "eligibility": ["protocol allowed by the pre-registered T2 policy (strict always; relaxed only as a sensitivity)",
                    ">= 100 TL_EVAL queries with >=1 truth reference under the protocol",
                    "V1 TL_EVAL_ALL MRR@25 > B1 mass-only TL_EVAL_ALL MRR@25 (point estimate; paired CI reported)",
                    "0 T1 references eligible/selected", "0 selected references in a tier the protocol excludes"],
    "preference": "stricter protocol; relaxed never selected automatically (a relaxed-only pass -> NEED_T2_PROVENANCE_REVIEW)",
    "never": "standard (sensitivity only); HOST metrics; MRR ranking between protocols",
}
ALLOWED_EVIDENCE_KEYS = ("protocol", "semantics_allowed", "n_eval_truth_evidence", "tl_eval_truth_coverage", "v1_mrr_all", "b1_mrr_all",
                         "t1_violation_count", "forbidden_tier_count")


def evaluate_protocol(ev):
    """One protocol's eligibility with every reason listed (semantic checks before MRR)."""
    guard_no_host(ev, "evaluate_protocol")
    extra = sorted(set(ev) - set(ALLOWED_EVIDENCE_KEYS))
    if extra:
        raise HostLeakError(f"evaluate_protocol: unexpected evidence keys {extra}")
    reasons, sem_ok = [], True
    if ev["protocol"] not in ("test_simulated_strict", "test_simulated_relaxed"):
        return {**ev, "eligible_for_scaling": False, "reasons": [f"{ev['protocol']} is not a test-simulated protocol (sensitivity/benchmark only)"]}
    if not ev["semantics_allowed"]:
        sem_ok = False
        reasons.append("not allowed by the pre-registered T2 policy")
    if ev["t1_violation_count"] != 0:
        sem_ok = False
        reasons.append(f"{ev['t1_violation_count']} T1 references eligible/selected")
    if ev["forbidden_tier_count"] != 0:
        sem_ok = False
        reasons.append(f"{ev['forbidden_tier_count']} selected references in a forbidden tier")
    size_ok = ev["n_eval_truth_evidence"] >= MIN_EVAL_QUERIES_WITH_TRUTH_EVIDENCE
    if not size_ok:
        reasons.append(f"only {ev['n_eval_truth_evidence']} TL_EVAL queries with truth evidence (< {MIN_EVAL_QUERIES_WITH_TRUTH_EVIDENCE})")
    perf_ok = sem_ok and size_ok and ev["v1_mrr_all"] is not None and ev["b1_mrr_all"] is not None and ev["v1_mrr_all"] > ev["b1_mrr_all"]
    if sem_ok and size_ok and not perf_ok:
        reasons.append("V1 MRR@25 does not exceed B1 mass-only on TL_EVAL")
    return {**ev, "semantic_ok": sem_ok, "size_ok": size_ok, "performance_ok": perf_ok, "eligible_for_scaling": bool(sem_ok and size_ok and perf_ok),
            "reasons": reasons or ["all pre-registered checks satisfied"]}


def scale_go_decision(evidence_by_protocol, t2_decision):
    """`evidence_by_protocol`: {protocol: evidence dict (ALLOWED_EVIDENCE_KEYS only)}."""
    guard_no_host(evidence_by_protocol, "scale_go_decision")
    results = {p: evaluate_protocol(ev) for p, ev in evidence_by_protocol.items() if p in PROTOCOL_DEFS}
    strict, relaxed = results.get("test_simulated_strict"), results.get("test_simulated_relaxed")
    if strict and strict["eligible_for_scaling"]:
        sel, nxt = strict, NEXT_RUN
    elif relaxed and relaxed["eligible_for_scaling"]:
        sel, nxt = relaxed, NEXT_T2_REVIEW              # never auto-select relaxed
    else:
        sel, nxt = strict or relaxed, NEXT_INVALID
    return {"selected_protocol": sel["protocol"] if sel and nxt == NEXT_RUN else None,
            "candidate_protocol": sel["protocol"] if sel else None,
            "protocol_status": "ELIGIBLE" if nxt == NEXT_RUN else ("RELAXED_ONLY_NEEDS_T2_REVIEW" if nxt == NEXT_T2_REVIEW else "NOT_ELIGIBLE"),
            "eligible_for_scaling": nxt == NEXT_RUN, "reasons": sel["reasons"] if sel else ["no test-simulated protocol evaluated"],
            "TL_EVAL_truth_reference_coverage": sel.get("tl_eval_truth_coverage") if sel else None,
            "n_eval_mode_a_queries": sel.get("n_eval_truth_evidence") if sel else None,
            "B1_MRR": sel.get("b1_mrr_all") if sel else None, "V1_MRR": sel.get("v1_mrr_all") if sel else None,
            "T1_violation_count": sel.get("t1_violation_count") if sel else None,
            "forbidden_tier_count": sel.get("forbidden_tier_count") if sel else None,
            "next_step": nxt, "t2_policy": t2_decision.get("t2_policy"), "per_protocol": results, "rule": SCALE_GO_RULE}
