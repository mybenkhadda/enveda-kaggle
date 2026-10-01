"""Hidden-like validation regimes C1 / C2 / C3 (v2).

    C1  REFERENCE AVAILABLE      truth structure in the candidate universe AND other reference spectra of
                                 the truth remain (the query's own spectrum + T1/T2 identity duplicates are
                                 excluded by the frozen `test_simulated_strict` protocol downstream).
    C2  STRUCTURE KNOWN,         truth structure stays in the universe; EVERY reference spectrum of the truth
        SPECTRA HIDDEN           connectivity is removed.  <- the MAIN regime
    C3  STRUCTURE ABSENT         spectra removed AND the truth connectivity removed from the candidate pool
                                 (diagnostic only: no generator yet, so truth is unreachable by design).

Two separations, never confused:
  (1) MODEL LEAKAGE -- the connectivity-grouped folds (`casmi.validation.folds`): model f trains on
      connectivities of folds != f and is evaluated on fold f. Asserted: train ∩ val == ∅.
  (2) REGIME MASKS  -- what the evaluation of fold f may see: the reference library minus the hidden
      connectivities, the candidate universe minus the removed ones. Hiding is FOLD-scoped: all C2 + C3
      truths of fold f are hidden at once, so one query's truth references can never help another query
      of the same fold.

Regimes are assigned per TRUTH CONNECTIVITY (all spectra of a connectivity share one regime; a
connectivity lives in exactly one fold), by a seeded, platform-stable hash -- the assignment is a pure
function of (split_seed, connectivity_key, eligibility, fractions) and is persisted; it is never
re-randomized per run.

Reuses `casmi.validation.class2_split` (C2 eligibility via non-TRAIN provenance, TRAIN-provenance
stripping of hidden truths, reference masks). Implements no canonicalization and no retrieval.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from casmi.validation.class2_split import class2_candidate_view, class2_eligible, class2_reference_mask

REGIMES = ("C1", "C2", "C3")
REGIME_TABLE_VERSION = "casmi-v2-regimes-1"
REGIME_COLUMNS = ["query_id", "true_connectivity_key", "fold", "regime", "eligible_C1", "eligible_C2", "eligible_C3",
                  "n_truth_spectra", "regime_u", "split_seed", "regime_config_hash"]


@dataclass(frozen=True)
class RegimeConfig:
    split_seed: int = 20261001
    regime_fractions: dict = field(default_factory=lambda: {"C1": 0.34, "C2": 0.33, "C3": 0.33})
    hidden_scope: str = "fold"
    c1_reference_protocol: str = "test_simulated_strict"
    c1_min_truth_spectra: int = 2
    c2_require_external_source: bool = True
    max_queries_per_regime_per_fold: int | None = None
    version: str = REGIME_TABLE_VERSION

    def __post_init__(self):
        if self.hidden_scope != "fold":
            raise ValueError("only hidden_scope='fold' is supported (query scope lets same-fold truths leak references)")
        if set(self.regime_fractions) != set(REGIMES) or any(v < 0 for v in self.regime_fractions.values()):
            raise ValueError(f"regime_fractions must give non-negative weights for {REGIMES}")
        if sum(self.regime_fractions.values()) <= 0:
            raise ValueError("regime_fractions must not all be zero")

    @classmethod
    def from_dict(cls, d):
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in keys})

    def config_hash(self):
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def stable_unit_hash(keys, seed):
    """Deterministic u in [0, 1) per key (blake2b of 'seed|key'); identical on every platform/run."""
    out = np.empty(len(keys), dtype=np.float64)
    for i, k in enumerate(keys):
        h = hashlib.blake2b(f"{seed}|{k}".encode("utf-8"), digest_size=8).digest()
        out[i] = int.from_bytes(h, "big") / 2.0 ** 64
    return out


def _choose(u, eligible_row, fractions):
    """Pick among the ELIGIBLE regimes by u, with the configured fractions renormalized over them."""
    names = [r for r in REGIMES if eligible_row[r] and fractions[r] > 0]
    if not names:
        return None
    w = np.array([fractions[r] for r in names], dtype=float)
    edges = np.cumsum(w / w.sum())
    return names[int(np.searchsorted(edges, u, side="right").clip(0, len(names) - 1))]


def truth_eligibility(truth_keys, reference_connectivity, unified=None, cfg=RegimeConfig()):
    """One row per truth connectivity: n_truth_spectra, eligible_C1/C2/C3 and why C2 is ineligible.

    C2 needs the truth in the universe AND (by default) in a non-TRAIN source, so that a hidden truth looks
    like an ordinary external structure (`class2_split`). Without an external universe (unified=None) C2 is
    ineligible unless `c2_require_external_source=False` (then `c2_shortcut_risk=True` is recorded)."""
    keys = pd.Index(pd.unique(pd.Series(truth_keys, dtype=str)), name="true_connectivity_key")
    counts = pd.Series(reference_connectivity).astype(str).value_counts()
    t = pd.DataFrame(index=keys)
    t["n_truth_spectra"] = counts.reindex(keys).fillna(0).astype(int).to_numpy()
    t["eligible_C1"] = t["n_truth_spectra"] >= cfg.c1_min_truth_spectra
    t["eligible_C3"] = True
    if unified is not None:
        q = class2_eligible(pd.DataFrame({"connectivity_key": keys.to_numpy()}), unified)
        in_univ = pd.Series(keys).isin(set(unified["connectivity_key"])).to_numpy()
        ext = q["class2_eligible"].to_numpy(bool)
        t["eligible_C2"] = ext if cfg.c2_require_external_source else in_univ
        t["c2_ineligible_reason"] = np.where(t["eligible_C2"], "", np.where(~in_univ, "not_in_universe", "train_only_structure"))
    else:
        t["eligible_C2"] = not cfg.c2_require_external_source
        t["c2_ineligible_reason"] = "" if not cfg.c2_require_external_source else "no_external_universe"
    t["c2_shortcut_risk"] = bool(not cfg.c2_require_external_source)
    return t.reset_index()


def build_regime_table(queries, reference_connectivity, unified=None, cfg=RegimeConfig(), query_col="query_id",
                       truth_col="true_connectivity_key", fold_col="fold"):
    """Assign C1/C2/C3 to every query (via its truth connectivity). Queries with no eligible regime get
    `regime=None` (kept, reported, never silently dropped)."""
    q = queries[[query_col, truth_col, fold_col]].copy()
    q[truth_col] = q[truth_col].astype(str)
    multi_fold = q.groupby(truth_col)[fold_col].nunique()
    if (multi_fold > 1).any():
        raise AssertionError(f"{int((multi_fold > 1).sum())} truth connectivities span several folds -- folds are not connectivity-grouped")
    elig = truth_eligibility(q[truth_col], reference_connectivity, unified, cfg).set_index("true_connectivity_key")
    u = pd.Series(stable_unit_hash(elig.index.tolist(), cfg.split_seed), index=elig.index)
    regime = pd.Series([_choose(u[k], {"C1": r.eligible_C1, "C2": r.eligible_C2, "C3": r.eligible_C3}, cfg.regime_fractions)
                        for k, r in elig.iterrows()], index=elig.index, dtype=object)
    out = pd.DataFrame({"query_id": q[query_col].astype(str).to_numpy(), "true_connectivity_key": q[truth_col].to_numpy(),
                        "fold": q[fold_col].to_numpy()})
    k = out["true_connectivity_key"]
    out["regime"] = k.map(regime).to_numpy()
    for r in REGIMES:
        out[f"eligible_{r}"] = k.map(elig[f"eligible_{r}"]).astype(bool).to_numpy()
    out["n_truth_spectra"] = k.map(elig["n_truth_spectra"]).astype(int).to_numpy()
    out["regime_u"] = k.map(u).to_numpy()
    out["split_seed"] = cfg.split_seed
    out["regime_config_hash"] = cfg.config_hash()
    if cfg.max_queries_per_regime_per_fold:
        out = cap_queries(out, cfg.max_queries_per_regime_per_fold, cfg.split_seed)
    return out[REGIME_COLUMNS].sort_values(["fold", "regime", "query_id"], kind="mergesort", na_position="last").reset_index(drop=True)


def cap_queries(regimes, cap, seed):
    """Deterministic per-(fold, regime) subsample: keep the `cap` queries with the smallest query hash."""
    r = regimes.assign(_h=stable_unit_hash(regimes["query_id"].tolist(), f"{seed}|cap"))
    r["_rank"] = r.groupby(["fold", "regime"], dropna=False)["_h"].rank(method="first")
    return r[r["_rank"] <= cap].drop(columns=["_h", "_rank"])


# ---------------------------------------------------------------------------------------------
# fold-scoped masks
# ---------------------------------------------------------------------------------------------

def hidden_sets(regimes, fold):
    """`{'hidden_reference_keys', 'removed_structure_keys'}` for the evaluation of `fold`:
    C2 ∪ C3 truths lose every reference spectrum; C3 truths also leave the candidate pool."""
    f = regimes[regimes["fold"] == fold]
    return {"hidden_reference_keys": set(f.loc[f["regime"].isin(["C2", "C3"]), "true_connectivity_key"]),
            "removed_structure_keys": set(f.loc[f["regime"] == "C3", "true_connectivity_key"])}


def reference_allowed_mask(reference_connectivity, hidden_reference_keys):
    """ALLOWED mask over the reference library for one fold (self / T1 / T2 exclusion is per query and is
    applied by the evidence builder under `c1_reference_protocol`)."""
    return class2_reference_mask(reference_connectivity, hidden_reference_keys)


def regime_candidate_view(unified, hidden_reference_keys, removed_structure_keys):
    """Small/medium universes (pandas): hidden truths stripped of TRAIN provenance (`class2_candidate_view`),
    C3 truths removed. For huge universes use `removed_candidate_mask` on the id-indexed mass index."""
    v = class2_candidate_view(unified, hidden_reference_keys)
    return v[~v["connectivity_key"].isin(set(removed_structure_keys))].reset_index(drop=True)


def isin_sorted(sorted_keys, values):
    """Membership of `values` in a SORTED key array (searchsorted -- no Python set of the universe)."""
    sorted_keys = np.asarray(sorted_keys)
    values = np.asarray(values).astype(sorted_keys.dtype)
    pos = np.searchsorted(sorted_keys, values)
    pos = np.clip(pos, 0, max(len(sorted_keys) - 1, 0))
    return (len(sorted_keys) > 0) & (sorted_keys[pos] == values) if len(sorted_keys) else np.zeros(len(values), bool)


def removed_candidate_mask(sorted_candidate_keys, removed_structure_keys):
    """Boolean over candidate ids (key-sorted universe): True = removed for this fold (C3 truths)."""
    mask = np.zeros(len(sorted_candidate_keys), dtype=bool)
    if removed_structure_keys:
        rk = np.array(sorted(removed_structure_keys)).astype(np.asarray(sorted_candidate_keys).dtype)
        pos = np.searchsorted(sorted_candidate_keys, rk)
        ok = (pos < len(sorted_candidate_keys))
        ok[ok] = np.asarray(sorted_candidate_keys)[pos[ok]] == rk[ok]
        mask[pos[ok]] = True
    return mask


# ---------------------------------------------------------------------------------------------
# assertions (raise -- never warn)
# ---------------------------------------------------------------------------------------------

def assert_regime_integrity(regimes, fold_table, reference_connectivity, universe_keys_sorted, fold, cfg=RegimeConfig(),
                            fold_key_col="connectivity_key"):
    """All leakage / masking invariants for the evaluation of `fold`. Returns a check table; raises
    AssertionError on the first failing invariant group.

    universe_keys_sorted: SORTED array of every connectivity key in the (unmasked) candidate universe."""
    f = regimes[regimes["fold"] == fold]
    checks = []

    def check(name, ok, detail):
        checks.append({"fold": fold, "check": name, "passed": bool(ok), "detail": detail})

    # (1) model leakage: training connectivities (folds != f) vs validation truths (fold f)
    train_keys = set(fold_table.loc[fold_table["fold"] != fold, fold_key_col].astype(str))
    val_keys = set(f["true_connectivity_key"])
    overlap = train_keys & val_keys
    check("train ∩ val connectivities == ∅", not overlap, f"{len(overlap)} overlapping")
    n_folds_per_key = fold_table.groupby(fold_table[fold_key_col].astype(str))["fold"].nunique()
    check("fold table: one fold per connectivity", (n_folds_per_key <= 1).all(), f"{int((n_folds_per_key > 1).sum())} keys in >1 fold")
    fmap = fold_table.assign(_k=fold_table[fold_key_col].astype(str)).drop_duplicates("_k").set_index("_k")["fold"]
    wrong = f[f["true_connectivity_key"].map(fmap) != fold]
    check("regime fold == connectivity fold", wrong.empty, f"{len(wrong)} queries with mismatching fold")
    per_key = regimes.groupby("true_connectivity_key")["regime"].nunique(dropna=False)
    check("one regime per connectivity", (per_key <= 1).all(), f"{int((per_key > 1).sum())} connectivities with >1 regime")

    hs = hidden_sets(regimes, fold)
    allowed = reference_allowed_mask(reference_connectivity, hs["hidden_reference_keys"])
    ref = pd.Series(reference_connectivity).astype(str).to_numpy()
    allowed_keys = set(ref[allowed])
    removed = removed_candidate_mask(universe_keys_sorted, hs["removed_structure_keys"])
    view_keys = np.asarray(universe_keys_sorted)[~removed]

    for regime in REGIMES:
        t = f.loc[f["regime"] == regime, "true_connectivity_key"].unique()
        if not len(t):
            check(f"{regime}: present", True, "no queries in this regime")
            continue
        in_refs = np.array([k in allowed_keys for k in t])
        in_view = isin_sorted(view_keys, t)
        if regime == "C1":
            n_allowed = pd.Series(ref[allowed]).value_counts().reindex(t).fillna(0).to_numpy()
            check("C1: truth in candidate universe", in_view.all(), f"{int((~in_view).sum())} missing")
            check("C1: truth keeps >= 1 other reference", (n_allowed >= cfg.c1_min_truth_spectra).all(),
                  f"{int((n_allowed < cfg.c1_min_truth_spectra).sum())} truths with too few references")
        elif regime == "C2":
            check("C2: truth NOT in reference library", not in_refs.any(), f"{int(in_refs.sum())} truths still referenced")
            check("C2: truth still in candidate pool", in_view.all(), f"{int((~in_view).sum())} truths missing from pool")
        else:
            check("C3: truth NOT in reference library", not in_refs.any(), f"{int(in_refs.sum())} truths still referenced")
            check("C3: truth NOT in candidate pool", not in_view.any(), f"{int(in_view.sum())} truths still in pool")
    table = pd.DataFrame(checks)
    bad = table[~table["passed"]]
    if len(bad):
        raise AssertionError("regime integrity failed:\n" + bad.to_string(index=False))
    return table


def regime_counts(regimes):
    """Queries / truth connectivities per (fold, regime), incl. unassigned (regime None)."""
    r = regimes.assign(regime=regimes["regime"].fillna("UNASSIGNED"))
    return (r.groupby(["fold", "regime"]).agg(n_queries=("query_id", "size"), n_connectivities=("true_connectivity_key", "nunique"))
            .reset_index())
