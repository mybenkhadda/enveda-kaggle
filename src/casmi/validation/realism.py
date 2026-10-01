"""v6 validation-realism metadata: separates MODEL IMPLEMENTATION status (frozen, unchanged) from the
HEADLINE VALIDATION REALISM status (which TL_EVAL protocol resembles the hidden test), maps the
test-match resemblance decision to its interpretation and to the (protocol-specific) scaling reading,
and records the T2 deployment-gap waiver.

Every writer here is ADDITIVE: existing keys of the freeze / gap records are never rewritten; each
status change is appended to a history list, and the first annotation keeps a byte-exact backup of
the original file (`<name>.pre_v6.json`). Nothing here reruns scaling or touches model files.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PENDING = "PENDING_TEST_MATCH_DIAGNOSTIC"
REALISM_STATUSES = (PENDING, "STRICT_LIKE", "MIRROR_LIKE", "INTERMEDIATE_OR_INCONCLUSIVE")
GAP_WAIVED = "WAIVED_PENDING_EVIDENCE"

INTERPRETATION = {
    "STRICT_LIKE": ["keep test_simulated_strict as the primary direct-reference validation",
                    "the strict TL_EVAL score remains the Class-1 direct-reference headline (still a validation estimate, not hidden-test performance)",
                    "molecule aggregation: strict is the primary evaluation protocol; HOST remains confirmation only"],
    "MIRROR_LIKE": ["downgrade the strict TL_EVAL headline to an OPTIMISTIC SENSITIVITY",
                    "use mirror_aware / class-mixture validation for scientific claims",
                    "molecule aggregation: mirror_aware is the primary evaluation protocol; HOST remains confirmation only"],
    "INTERMEDIATE_OR_INCONCLUSIVE": ["report BOTH bounds (mirror_aware lower, test_simulated_strict upper)",
                                     "claim neither as the exact hidden-test performance",
                                     "molecule aggregation: report both protocols; HOST remains confirmation only"],
}

SCALING_INTERPRETATION = {
    "STRICT_LIKE": {"scaling_label": "PLATEAU_UNDER_REPRESENTATIVE_PROTOCOL",
                    "text": "the strict 1k/3k/10k scaling result may stand as the scaling status for direct-reference ranking"},
    "MIRROR_LIKE": {"scaling_label": "CEILING-LIMITED STRICT REGIME",
                    "text": "the strict scaling plateau is ceiling-limited; it must NOT be used to conclude that more training data is useless in general"},
    "INTERMEDIATE_OR_INCONCLUSIVE": {"scaling_label": "PROTOCOL-SPECIFIC",
                                     "text": "the scaling conclusion holds under test_simulated_strict only; no general claim about data scaling"},
}


def validation_interpretation(status):
    if status not in INTERPRETATION:
        raise ValueError(f"unknown resemblance status {status!r}")
    return list(INTERPRETATION[status])


def primary_eval_protocols(status):
    """Molecule-aggregation evaluation protocols implied by the diagnostic (first = primary)."""
    return {"STRICT_LIKE": ["test_simulated_strict"], "MIRROR_LIKE": ["mirror_aware"]}.get(status, ["test_simulated_strict", "mirror_aware"])


def scaling_interpretation(status, recorded_scaling_status):
    """The recorded strict scaling status is kept VERBATIM; only its interpretation is attached."""
    if status not in SCALING_INTERPRETATION:
        raise ValueError(f"unknown resemblance status {status!r}")
    return {"recorded_scaling_status_under_strict": recorded_scaling_status, "resemblance_status": status, **SCALING_INTERPRETATION[status],
            "note": "numerical scaling results are unchanged and were not rerun"}


def _backup_once(path):
    path = Path(path)
    bak = path.with_name(path.stem + ".pre_v6.json")
    if not bak.exists():
        shutil.copyfile(path, bak)
    return bak


def annotate_freeze_realism(freeze_path, status, evidence=None, set_by="10v6_00_test_match_diagnostic"):
    """Set `model_freeze_status` (mirrors the untouched `freeze_status`) and `validation_realism_status`,
    appending the change to `validation_realism_history`. Idempotent for an unchanged status."""
    if status not in REALISM_STATUSES:
        raise ValueError(f"unknown validation realism status {status!r}")
    path = Path(freeze_path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("freeze_status") != "FROZEN":
        raise RuntimeError(f"{path} is not FROZEN; validation-realism annotation applies to the frozen model only")
    hist = list(rec.get("validation_realism_history") or [])
    if rec.get("validation_realism_status") == status and rec.get("model_freeze_status") == "FROZEN" and hist:
        return rec
    _backup_once(path)
    rec["model_freeze_status"] = "FROZEN"
    rec["validation_realism_status"] = status
    hist.append({"validation_realism_status": status, "set_by": set_by, "set_at": datetime.now(timezone.utc).isoformat(),
                 **({"evidence": evidence} if evidence else {})})
    rec["validation_realism_history"] = hist
    path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


def waive_deployment_gap(gap_path, reasons, set_by="v6"):
    """`resolution_status` -> WAIVED_PENDING_EVIDENCE, previous status kept in `resolution_status_history`."""
    path = Path(gap_path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("resolution_status") == GAP_WAIVED:
        return rec
    _backup_once(path)
    hist = list(rec.get("resolution_status_history") or [])
    hist.append({"resolution_status": rec.get("resolution_status"), "until": datetime.now(timezone.utc).isoformat(), "replaced_by": GAP_WAIVED,
                 "set_by": set_by})
    rec.update(resolution_status=GAP_WAIVED, resolution_reason=list(reasons), resolution_status_history=hist)
    path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return rec


LEADERBOARD_POLICY = {
    "allowed_uses": ["catching a major validation mismatch", "checking pipeline direction", "evaluating a major candidate-universe change"],
    "forbidden_uses": ["tuning thresholds", "choosing between minor model variants", "tuning aggregator hyperparameters", "repeated score chasing"],
}
