"""Public-baseline INTERFACE (no numbers). A public notebook's candidate pool / predictions can be compared with this
project's coverage only through a LOCAL, user-supplied artifact that is described here. No public baseline artifact
exists in the repository; without one every consumer reports NOT_AVAILABLE. Its leaderboard score is never typed into
code or config (scores are measured on the competition test set, not on this project's validation protocol).

Spec (config `coverage.public_baseline`):
    path                 parquet / csv relative to the Drive root (null -> NOT_AVAILABLE)
    query_id_column      default query_id
    key_column           connectivity_key column, or null with smiles_column (then the PROJECT canonicalizer keys it)
    smiles_column        default smiles
    rank_column          optional rank of the candidate in the public pool
    label / provenance   free text recorded with every report (where the artifact came from, which notebook version)
    ppm                  mass window used for the comparison
"""
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

NOT_AVAILABLE, AVAILABLE, INVALID = "NOT_AVAILABLE", "AVAILABLE", "INVALID"


@dataclass
class PublicBaselineSpec:
    path: str | None = None
    query_id_column: str = "query_id"
    key_column: str | None = "connectivity_key"
    smiles_column: str | None = "smiles"
    rank_column: str | None = None
    label: str = "public baseline"
    provenance: str | None = None
    ppm: float = 5.0

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_public_baseline(spec, root, keyfn=None):
    """(status dict, frame with query_id + connectivity_key [+ rank]) -- NOT_AVAILABLE without a file, INVALID when the
    declared columns are missing. Never fabricates rows."""
    if not spec.path:
        return {"status": NOT_AVAILABLE, "reason": "coverage.public_baseline.path is null", "spec": asdict(spec)}, None
    p = Path(root) / spec.path
    if not p.is_file():
        return {"status": NOT_AVAILABLE, "reason": f"{p} not found", "spec": asdict(spec)}, None
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, dtype=str)
    need = [spec.query_id_column] + ([spec.key_column] if spec.key_column and spec.key_column in df.columns else [spec.smiles_column])
    missing = [c for c in need if c not in df.columns]
    if missing or (spec.key_column not in df.columns and keyfn is None):
        return {"status": INVALID, "reason": f"missing columns {missing} (or smiles without a keyfn)", "spec": asdict(spec)}, None
    out = pd.DataFrame({"query_id": df[spec.query_id_column].astype(str)})
    out["connectivity_key"] = (df[spec.key_column].astype(str) if spec.key_column in df.columns
                               else [keyfn(s) for s in df[spec.smiles_column]])
    if spec.rank_column and spec.rank_column in df.columns:
        out["rank"] = pd.to_numeric(df[spec.rank_column], errors="coerce")
    ident = {"name": p.name, "size_bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}
    return {"status": AVAILABLE, "label": spec.label, "provenance": spec.provenance, "source_identity": ident, "n_rows": int(len(out)),
            "n_queries": int(out["query_id"].nunique()), "spec": asdict(spec)}, out
