"""Spectrum-model freeze record helpers + the guard every downstream notebook (11_02) applies.

`require_frozen_spectrum_model` refuses unless `freeze_status == "FROZEN"` exactly (a PROVISIONAL
freeze is refused), and re-verifies the recorded model-file sha256s, the feature-order hash and the
mirror_aware protocol hash against the files / source on disk.
"""
import hashlib
import inspect
import json
from pathlib import Path

NOT_FROZEN_MESSAGE = "Mode-A spectrum model is not frozen. Complete v5 scaling first."


class SpectrumModelNotFrozen(RuntimeError):
    pass


class FreezeIntegrityError(RuntimeError):
    pass


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_order_hash(feature_names):
    return hashlib.sha256(json.dumps(list(feature_names)).encode("utf-8")).hexdigest()[:16]


def mirror_aware_protocol_hash():
    import casmi.spectra.reference_selection as rs
    return hashlib.sha256((inspect.getsource(rs) + "|mirror_aware").encode("utf-8")).hexdigest()[:16]


def model_file_hashes(model_dir):
    return {p.name: sha256_file(p) for p in sorted(Path(model_dir).glob("fold_*.txt"))}


def require_frozen_spectrum_model(freeze_path, protocol_hash_fn=mirror_aware_protocol_hash):
    """Returns the freeze record, or raises `SpectrumModelNotFrozen` / `FreezeIntegrityError`."""
    freeze_path = Path(freeze_path)
    if not freeze_path.exists():
        raise SpectrumModelNotFrozen(NOT_FROZEN_MESSAGE + f" ({freeze_path} not found)")
    rec = json.loads(freeze_path.read_text(encoding="utf-8"))
    if rec.get("freeze_status") != "FROZEN":
        raise SpectrumModelNotFrozen(NOT_FROZEN_MESSAGE + f" (freeze_status={rec.get('freeze_status')!r})")
    problems = []
    recorded = rec.get("model_hashes") or {}
    on_disk = model_file_hashes(rec["model_dir"])
    if not recorded or recorded != on_disk:
        problems.append(f"model file hashes differ from the freeze record ({len(recorded)} recorded, {len(on_disk)} on disk)")
    if rec.get("feature_order_hash") != feature_order_hash(rec.get("feature_names", [])):
        problems.append("feature-order hash missing or does not match feature_names")
    if rec.get("protocol"):                       # v5.3+: the frozen evidence protocol is explicit
        from casmi.qcr.protocols import PROTOCOL_DEFS, protocol_semantic_hash
        if rec["protocol"] not in PROTOCOL_DEFS or rec.get("protocol_hash") != protocol_semantic_hash(rec["protocol"]):
            problems.append(f"protocol hash for {rec.get('protocol')!r} missing or does not match the current protocol definition")
    elif rec.get("mirror_aware_protocol_hash") != protocol_hash_fn():
        problems.append("mirror_aware protocol source changed since the freeze")
    if problems:
        raise FreezeIntegrityError("spectrum-model freeze failed verification: " + "; ".join(problems))
    return rec


# ---------------------------------------------------------------------------------------------
# v5.3 freeze conditions (all must hold for FROZEN; otherwise PROVISIONAL with the missing list)
# ---------------------------------------------------------------------------------------------

FREEZE_CONDITIONS = (
    "protocol_artifact_valid",          # persisted protocol definition exists, self-consistent, same semantics
    "protocol_hash_persisted",
    "training_manifests_persisted",     # TL_1K / TL_3K / TL_10K derived manifests all present
    "selected_model_artifacts_exist",   # 5 fold boosters + metadata
    "model_hashes_persisted",
    "selection_on_tl_eval_only",        # the lock's selection population is TL_EVAL and its inputs carry no HOST field
    "scale_decision_10k_vs_3k",         # CONTINUE_SCALING / PLATEAU decided on the 10k-vs-3k pair
    "selection_locked_before_host",
    "feature_contract_validated",
    "metrics_and_bootstrap_persisted",
)


def freeze_status(evidence):
    """`evidence`: {condition: bool}. Unknown/missing conditions count as not met."""
    missing = [c for c in FREEZE_CONDITIONS if not bool(evidence.get(c, False))]
    return ("FROZEN" if not missing else "PROVISIONAL"), missing
