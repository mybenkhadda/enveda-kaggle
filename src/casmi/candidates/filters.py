"""Configurable structure filters for the EXTERNAL candidate universe (v2).

Applied to `casmi.candidates.standardize.standardize_records` output. Nothing is dropped silently: every
row lands in `kept` or in `filtered_out` with a `filter_reason` ('a;b' when several apply).

TRAIN structures are NOT filtered by default (`apply_to_train=False`): the closed-world training contract
is not altered, and filtering a training truth would silently cap C1/C2 recall. `train_filter_audit`
reports how many TRAIN connectivities WOULD fail, so the cost of a filter is visible.
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")
DEFAULT_ORGANIC_ELEMENTS = ("C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B", "Se")


@dataclass(frozen=True)
class UniverseFilterConfig:
    apply_to_train: bool = False
    min_exact_mass: float = 150.0
    max_exact_mass: float = 1200.0
    organic_only: bool = True
    organic_elements: tuple = DEFAULT_ORGANIC_ELEMENTS
    neutral_only: bool = True
    single_component_only: bool = True

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        if "organic_elements" in d:
            d["organic_elements"] = tuple(d["organic_elements"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def formula_elements(formula):
    """Set of element symbols in a molecular formula string ('' / None -> empty set)."""
    return set(_ELEMENT_RE.findall(formula)) if isinstance(formula, str) else set()


def is_organic_formula(formula, organic_elements=DEFAULT_ORGANIC_ELEMENTS):
    els = formula_elements(formula)
    return bool(els) and "C" in els and els <= set(organic_elements)


def structure_flags(std, cfg=UniverseFilterConfig()):
    """`is_organic`, `is_single_component`, `is_neutral`, `in_mass_range` for standardized rows."""
    formulas = std["molecular_formula"]
    uniq = pd.unique(formulas.astype(object))
    organic = {f: is_organic_formula(f, cfg.organic_elements) for f in uniq}
    mass = pd.to_numeric(std["neutral_monoisotopic_mass"], errors="coerce")
    return pd.DataFrame({
        "is_organic": formulas.map(organic).fillna(False).astype(bool).to_numpy(),
        "is_single_component": (pd.to_numeric(std["n_fragments"], errors="coerce").fillna(0) == 1).to_numpy(),
        "is_neutral": (pd.to_numeric(std["formal_charge"], errors="coerce").fillna(1) == 0).to_numpy(),
        "in_mass_range": ((mass >= cfg.min_exact_mass) & (mass <= cfg.max_exact_mass)).fillna(False).to_numpy(),
    }, index=std.index)


def apply_filters(std, cfg=UniverseFilterConfig()):
    """Returns `(kept, filtered_out)`; both carry the four flag columns, `filtered_out` also `filter_reason`.
    TRAIN rows pass untouched unless `cfg.apply_to_train`."""
    flags = structure_flags(std, cfg)
    reasons = []
    for active, col, label in ((True, "in_mass_range", "mass_out_of_range"), (cfg.organic_only, "is_organic", "not_organic"),
                               (cfg.neutral_only, "is_neutral", "charged"), (cfg.single_component_only, "is_single_component", "multi_component")):
        if active:
            reasons.append(np.where(flags[col].to_numpy(), "", label))
    reason = pd.Series([";".join(r for r in rs if r) for rs in zip(*reasons)] if reasons else [""] * len(std), index=std.index)
    if not cfg.apply_to_train and "source" in std.columns:
        reason[std["source"].astype(str) == "TRAIN"] = ""
    out = pd.concat([std, flags], axis=1)
    bad = reason != ""
    filtered = out[bad].assign(filter_reason=reason[bad].to_numpy())
    kept = out[~bad]
    assert len(kept) + len(filtered) == len(std), "every row must be kept or filtered"
    return kept.reset_index(drop=True), filtered.reset_index(drop=True)


def train_filter_audit(train_table, cfg=UniverseFilterConfig(), formula_col="molecular_formula", mass_col="exact_mass",
                       charge_col="formal_charge", smiles_col="representative_smiles"):
    """How many TRAIN connectivities each filter WOULD remove (diagnostic only -- TRAIN is never filtered
    by default). `train_table`: one row per connectivity."""
    std = pd.DataFrame({"molecular_formula": train_table[formula_col].to_numpy(),
                        "neutral_monoisotopic_mass": train_table[mass_col].to_numpy(),
                        "formal_charge": train_table[charge_col].to_numpy() if charge_col in train_table else 0,
                        "n_fragments": train_table[smiles_col].astype(str).str.count(r"\.").to_numpy() + 1})
    f = structure_flags(std, cfg)
    n = len(f)
    return pd.DataFrame([{"filter": c, "n_train_would_fail": int((~f[c]).sum()), "share": float((~f[c]).mean()) if n else np.nan}
                         for c in f.columns])
