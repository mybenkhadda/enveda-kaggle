"""Fingerprint DATASET manifest contract (no training here). A future fingerprint / structure model may only be trained on
a dataset whose manifest proves that the protocol-v2 exclusion lists were applied:

    exclusions        data/processed/exclusion_manifests/{host,np_panel,c2,c3}_connectivities.parquet
                      + model_training_exclusions.metadata.json (written by notebook 11b, bound to a regime signature)
    dataset manifest  identity of the training connectivity list, fingerprint spec, which exclusion files were applied
                      (with their sha256), the regime signature and panel manifest identity they came from, and the
                      counts removed per list. `assert_dataset_respects_exclusions` re-checks the invariant (raises).

Evaluation on panel P / regime R is valid only with a model whose dataset manifest excluded the matching list.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FINGERPRINT_CONTRACT_VERSION = "casmi-fingerprint-dataset-1"
EXCLUSION_LISTS = ("host", "np_panel", "c2", "c3")


@dataclass
class FingerprintSpec:
    kind: str = "morgan"
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = False
    extra: dict = field(default_factory=dict)


class ExclusionContractError(AssertionError):
    """The training connectivities intersect an exclusion list that the dataset claims to have applied."""


def load_exclusions(exclusion_dir, lists=EXCLUSION_LISTS):
    d = Path(exclusion_dir)
    meta_p = d / "model_training_exclusions.metadata.json"
    if not meta_p.is_file():
        raise FileNotFoundError(f"{meta_p} missing -- run notebook 11b (validation regimes v2) first")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    out = {}
    for name in lists:
        p = d / f"{name}_connectivities.parquet"
        if not p.is_file():
            raise FileNotFoundError(f"exclusion list {p} missing")
        out[name] = set(pd.read_parquet(p, columns=["connectivity_key"])["connectivity_key"].astype(str))
    return out, meta


def build_fingerprint_dataset_manifest(training_keys, exclusion_dir, spec=FingerprintSpec(), lists=EXCLUSION_LISTS):
    """Apply the exclusions to `training_keys` and describe the result. Returns (kept_keys sorted list, manifest)."""
    ex, meta = load_exclusions(exclusion_dir, lists)
    keys = sorted({str(k) for k in training_keys})
    removed = {n: len(set(keys) & s) for n, s in ex.items()}
    banned = set().union(*ex.values()) if ex else set()
    kept = [k for k in keys if k not in banned]
    man = {"contract_version": FINGERPRINT_CONTRACT_VERSION, "fingerprint_spec": asdict(spec),
           "n_input_connectivities": len(keys), "n_kept_connectivities": len(kept), "removed_per_list": removed,
           "applied_lists": list(lists), "exclusion_files": {k: v for k, v in (meta.get("files") or {}).items()},
           "regime_signature": meta.get("regime_signature"), "panel_manifest_identity": meta.get("panel_manifest_identity"),
           "kept_keys_sha256": hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()[:20],
           "created_at": datetime.now(timezone.utc).isoformat(), "training_performed": False}
    return kept, man


def assert_dataset_respects_exclusions(training_keys, manifest, exclusion_dir):
    ex, meta = load_exclusions(exclusion_dir, manifest["applied_lists"])
    if meta.get("regime_signature") != manifest.get("regime_signature"):
        raise ExclusionContractError("dataset manifest was built from another regime signature -- rebuild the dataset manifest")
    keys = {str(k) for k in training_keys}
    hits = {n: len(keys & s) for n, s in ex.items() if keys & s}
    if hits:
        raise ExclusionContractError(f"training connectivities intersect exclusion lists: {hits}")
    return True
