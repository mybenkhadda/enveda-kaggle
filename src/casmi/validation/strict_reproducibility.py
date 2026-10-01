"""Strict C1 HOST/DEV reproducibility check: rebuild protocol features from the canonical QCR
table and compare against the existing/legacy table, enforcing key uniqueness and a full
one-to-one merge cardinality BEFORE any numeric comparison -- a merge that silently drops rows
down to just the intersection, or duplicates rows, must never be reported as a PASS (spec
section 5/20's D5 defect: "can PASS on only the intersection of tables").
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class ReproducibilityKeyError(Exception):
    """Raised when `rebuilt` or `existing` has duplicate keys -- a structural bug that makes a
    one-to-one merge meaningless, never just reported as a numeric "mismatch"."""


@dataclass
class ReproducibilityResult:
    passed: bool
    reason: str
    n_rebuilt: int
    n_existing: int
    n_merged: int
    mismatches: list = field(default_factory=list)
    merged: object = None


def strict_reproducibility_check(rebuilt, existing, keys, feature_cols, atol=1e-12, rtol=0):
    """Enforces, in order: (1) no duplicate keys in `rebuilt`; (2) no duplicate keys in
    `existing`; (3) an outer merge on `keys` covers every row of BOTH tables with no
    non-matching rows (never a silent intersection-only comparison); (4) every column in
    `feature_cols` matches within `atol`/`rtol`. (1)/(2) raise `ReproducibilityKeyError`
    immediately -- these are structural bugs, not mismatches to report and move past.
    Returns a `ReproducibilityResult` for (3)/(4)."""
    if rebuilt.duplicated(keys).any():
        raise ReproducibilityKeyError(f"rebuilt table has duplicate keys: {keys}")
    if existing.duplicated(keys).any():
        raise ReproducibilityKeyError(f"existing table has duplicate keys: {keys}")

    merged = rebuilt.merge(existing, on=keys, how="outer", validate="one_to_one", suffixes=("_rebuilt", "_existing"), indicator=True)

    if len(merged) != len(rebuilt) or len(merged) != len(existing):
        return ReproducibilityResult(
            passed=False, reason=f"merge is not a full one-to-one cover: rebuilt={len(rebuilt)}, existing={len(existing)}, merged={len(merged)}",
            n_rebuilt=len(rebuilt), n_existing=len(existing), n_merged=len(merged), merged=merged,
        )
    if (merged["_merge"] != "both").any():
        n_only_rebuilt = int((merged["_merge"] == "left_only").sum())
        n_only_existing = int((merged["_merge"] == "right_only").sum())
        return ReproducibilityResult(
            passed=False, reason=f"keys not present in both tables: {n_only_rebuilt} only in rebuilt, {n_only_existing} only in existing",
            n_rebuilt=len(rebuilt), n_existing=len(existing), n_merged=len(merged), merged=merged,
        )

    mismatches = []
    for col in feature_cols:
        a, b = merged[f"{col}_rebuilt"], merged[f"{col}_existing"]
        close = np.isclose(a, b, atol=atol, rtol=rtol, equal_nan=True)
        if not close.all():
            mismatches.append({"feature": col, "n_mismatches": int((~close).sum())})

    return ReproducibilityResult(
        passed=not mismatches, reason="feature mismatch" if mismatches else "exact reproduction",
        n_rebuilt=len(rebuilt), n_existing=len(existing), n_merged=len(merged), mismatches=mismatches, merged=merged,
    )


def persist_if_pass(result, rebuilt, canonical_path, mismatch_path):
    """Spec section 21's persistence rule: only write `canonical_path` when `result.passed`;
    otherwise write `rebuilt` to `mismatch_path` (conventionally `<name>__MISMATCH.parquet`)
    and raise `AssertionError` -- the existing canonical artifact at `canonical_path` is NEVER
    touched on failure."""
    if result.passed:
        rebuilt.to_parquet(canonical_path, index=False)
        return canonical_path
    mismatch_path.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.to_parquet(mismatch_path, index=False)
    raise AssertionError(f"C1 reproducibility FAILED ({result.reason}); rebuilt table saved to {mismatch_path}, canonical artifact NOT overwritten")
