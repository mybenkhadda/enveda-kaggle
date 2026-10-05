"""External candidate COVERAGE gate (Gate A, redesigned): could a held-out structure be found from EXTERNAL structure
databases if it had never appeared in training?

Why the previous Gate-A number could be trivial
-----------------------------------------------
Validation molecules are TRAINING molecules. The finalized universe is TRAIN u COCONUT, so every held-out truth is in
the universe through its TRAIN provenance: "truth in universe" (`recall.truth_reachable(mode='universe')`) is ~100% by
construction. And the C2 regime was SELECTED on external reachability (`class2_split.class2_eligible` requires a
non-TRAIN source), so C2 `external_only` Recall@All is conditioned on the very property it should measure -- it only
measures precursor mass accuracy for already-reachable truths. Neither answers the coverage question.

Leakage-free protocol (this module)
-----------------------------------
For every held-out truth connectivity k (ALL grouped-validation truths, not a reachability-selected subset):

    candidate_sources_without_train(k) = candidate_sources(k) - {"TRAIN"}       (`remove_train_provenance_for_truth`)
    COCONUT_REACHABLE  = "COCONUT" in candidate_sources_without_train(k)
    PUBCHEM_REACHABLE  = k in the PubChem key-membership index (PUBCHEM_KEY_MEMBERSHIP_ESTIMATE) or a finalized PUBCHEM source
    EXTERNAL_REACHABLE = any external source remains;  else EXTERNAL_UNREACHABLE
    TRAIN_REACHABLE    = "TRAIN" in candidate_sources(k)  -- reported as a diagnostic, NEVER counted as reachability

C2 (external protocol) := held-out truths that ARE externally reachable. A TRAIN-only truth is never C2.
No per-query universe rebuild: TRAIN provenance is masked per truth, and the external-only candidate pool is a single
boolean mask over candidate ids (`external_candidate_mask`) -- TRAIN-only candidates would not exist without training.

Coverage is separated from ranking: this module reports reachability, truth-in-mass-window, pool sizes, filter losses
and decision rules. No Recall@25 / @100 / MRR from any ranker is computed here (ranking starts in notebook 14).
"""
import hashlib
import importlib.util
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

COVERAGE_GATE_VERSION = "casmi-coverage-gate-1"
REACHABILITY_LABELS = ("TRAIN_REACHABLE", "COCONUT_REACHABLE", "PUBCHEM_REACHABLE", "EXTERNAL_REACHABLE", "EXTERNAL_UNREACHABLE")
EXTERNAL_SOURCES = ("COCONUT", "LOTUS", "NPATLAS", "PUBCHEM")
DECISIONS = ("BUILD_PUBCHEM_TIER1_NEXT", "REBUILD_COCONUT_WITH_RESCUE_POLICY", "BUILD_FORMULA_PRUNING_BEFORE_RANKING",
             "PROCEED_TO_ANALOG_BASELINE")
RULE_OUTCOME = {"A": "BUILD_PUBCHEM_TIER1_NEXT", "B": "BUILD_FORMULA_PRUNING_BEFORE_RANKING", "C": "REBUILD_COCONUT_WITH_RESCUE_POLICY",
                "D": "PROCEED_TO_ANALOG_BASELINE"}
LEGACY_DIAGNOSTIC_ONLY = "LEGACY_DIAGNOSTIC_ONLY"
ESTIMATE_ONLY = "ESTIMATE_ONLY"


class CoverageGateError(RuntimeError):
    """Notebook 14 precondition failed (gate missing / protocol invalid / decision does not permit the analog baseline)."""


def _as_list(s):
    if isinstance(s, (list, tuple, set, np.ndarray)):
        return [str(x) for x in s]
    return []


# ---------------------------------------------------------------------------------------------
# validation populations (never merged silently: every row carries its panel)
# ---------------------------------------------------------------------------------------------

@dataclass
class PanelSpec:
    name: str
    kind: str = "derived"                    # derived: subset of dev queries by a column filter | file: separate table
    column: str | None = None                # derived: column of dev_queries to filter on
    values: list = field(default_factory=list)
    path: str | None = None                  # file: parquet / csv with query_id, true_connectivity_key, neutral_mass
    description: str = ""

    @classmethod
    def from_dict(cls, name, d):
        d = dict(d or {})
        return cls(name=name, **{k: v for k, v in d.items() if k in cls.__dataclass_fields__ and k != "name"})


QUERY_COLUMNS = ["panel", "population_source", "query_id", "connectivity_key", "fold", "regime", "neutral_mass", "truth_exact_mass",
                 "adduct", "ionization_mode", "instrument_type"]


def build_population_table(dev, regimes=None, panels=(), panel_frames=None):
    """Query-level table of every validation population.

    dev     : grouped-validation dev queries (query_id, true_connectivity_key, fold, neutral_mass, [exact_mass_true, adduct,
              ionization_mode, instrument_type]) -> panel 'dev_grouped'.
    regimes : optional persisted C1/C2/C3 manifest (query_id, regime) -> `regime` column (else None). Regimes are an
              ANNOTATION of the same queries, not a separate population.
    panels  : PanelSpec list. kind='derived' filters dev queries (e.g. instrument_type == timsTOF -> host-like panel);
              kind='file' uses `panel_frames[name]` (a separately supplied table). A missing file panel raises.
    """
    d = dev.copy()
    d["query_id"] = d["query_id"].astype(str)
    if regimes is not None and len(regimes):
        r = regimes[["query_id", "regime"]].assign(query_id=regimes["query_id"].astype(str))
        d = d.merge(r, on="query_id", how="left", validate="one_to_one")
    else:
        d["regime"] = None

    def frame(df, panel, src):
        out = pd.DataFrame({"panel": panel, "population_source": src, "query_id": df["query_id"].astype(str).to_numpy(),
                            "connectivity_key": df["true_connectivity_key"].astype(str).to_numpy()})
        for c, srcc in (("fold", "fold"), ("regime", "regime"), ("neutral_mass", "neutral_mass"), ("truth_exact_mass", "exact_mass_true"),
                        ("adduct", "adduct"), ("ionization_mode", "ionization_mode"), ("instrument_type", "instrument_type")):
            out[c] = df[srcc].to_numpy() if srcc in df.columns else None
        for c in ("neutral_mass", "truth_exact_mass"):          # numeric (missing -> NaN), so per-molecule medians never fail
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        return out[QUERY_COLUMNS]

    parts = [frame(d, "dev_grouped", "dev_queries (connectivity-grouped folds)")]
    for p in panels:
        if p.kind == "derived":
            if p.column not in d.columns:
                raise KeyError(f"panel {p.name!r}: column {p.column!r} not in dev_queries ({list(d.columns)})")
            sub = d[d[p.column].astype(str).isin([str(v) for v in p.values])]
            parts.append(frame(sub, p.name, f"dev_queries[{p.column} in {list(p.values)}]"))
        elif p.kind == "file":
            f = (panel_frames or {}).get(p.name)
            if f is None:
                raise FileNotFoundError(f"panel {p.name!r}: table not supplied (configured path {p.path!r}) -- upload it or remove the panel")
            f = f.copy()
            if "regime" not in f.columns:
                f["regime"] = None
            parts.append(frame(f, p.name, f"file:{Path(p.path).name if p.path else '?'}"))
        else:
            raise ValueError(f"panel {p.name!r}: unknown kind {p.kind!r}")
    return pd.concat(parts, ignore_index=True)


def molecule_table(queries):
    """One row per (panel, connectivity_key): n_queries, regime (single by construction), truth mass (median)."""
    g = queries.groupby(["panel", "connectivity_key"], sort=True)
    out = g.agg(n_queries=("query_id", "size"), regime=("regime", "first"), fold=("fold", "first"),
                truth_exact_mass=("truth_exact_mass", "median"), median_query_neutral_mass=("neutral_mass", "median")).reset_index()
    out.insert(1, "molecule_id", out["connectivity_key"])
    return out


# ---------------------------------------------------------------------------------------------
# membership + reachability
# ---------------------------------------------------------------------------------------------

def remove_train_provenance_for_truth(candidate_sources, is_truth=True):
    """`candidate_sources` with TRAIN removed for held-out truths (vectorized over a list/Series). `is_truth` may be a
    bool or a boolean array aligned with `candidate_sources`."""
    cs = list(candidate_sources)
    mask = np.broadcast_to(np.asarray(is_truth, dtype=bool), (len(cs),))
    return [sorted(set(_as_list(s)) - {"TRAIN"}) if m else sorted(set(_as_list(s))) for s, m in zip(cs, mask, strict=True)]


def _pubchem_membership(keys, pubchem):
    """bool array or None (unavailable). `pubchem`: None | object with `.contains(keys)` | iterable of keys."""
    if pubchem is None:
        return None
    if hasattr(pubchem, "contains"):
        return np.asarray(pubchem.contains(keys), dtype=bool)
    s = set(map(str, pubchem))
    return np.array([k in s for k in keys], dtype=bool)


def build_external_membership(truth_keys, universe_rows, pubchem=None):
    """Per unique truth key: universe membership (FULL, TRAIN included -- `membership_in_full_train_union`) and the
    per-source flags. `universe_rows`: DataFrame(connectivity_key, candidate_sources[, candidate_id]) of the finalized
    universe rows of these truths (keys absent from the universe may simply be missing)."""
    keys = pd.unique(pd.Series(truth_keys, dtype="object").astype(str))
    u = universe_rows.assign(connectivity_key=universe_rows["connectivity_key"].astype(str)).drop_duplicates("connectivity_key")
    u = u.set_index("connectivity_key")
    srcs = [(_as_list(u.at[k, "candidate_sources"]) if k in u.index else []) for k in keys]
    cid = [(int(u.at[k, "candidate_id"]) if (k in u.index and "candidate_id" in u.columns) else -1) for k in keys]
    out = pd.DataFrame({"connectivity_key": keys, "candidate_id": np.asarray(cid, dtype=np.int64),
                        "membership_in_full_train_union": [k in u.index for k in keys], "candidate_sources": srcs})
    out["in_train"] = [("TRAIN" in s) for s in srcs]
    out["in_coconut"] = [("COCONUT" in s) for s in srcs]
    out["in_other_external"] = [bool(set(s) - {"TRAIN", "COCONUT", "PUBCHEM"}) for s in srcs]
    pc = _pubchem_membership(keys, pubchem)
    out["pubchem_available"] = pc is not None
    finalized_pubchem = np.array([("PUBCHEM" in s) for s in srcs], dtype=bool)
    out["in_pubchem"] = (pc | finalized_pubchem) if pc is not None else finalized_pubchem
    return out


def classify_external_reachability(membership):
    """Adds the leakage-free reachability columns. TRAIN membership never makes a truth reachable."""
    m = membership.copy()
    ext = remove_train_provenance_for_truth(m["candidate_sources"], True)
    m["candidate_sources_without_train"] = ext
    co, pc = m["in_coconut"].to_numpy(bool), m["in_pubchem"].to_numpy(bool)
    other = m["in_other_external"].to_numpy(bool)
    union = co | pc | other
    m["in_external_union"] = union
    m["coconut_only"] = co & ~pc
    m["pubchem_only"] = pc & ~co
    m["coconut_and_pubchem"] = co & pc
    m["neither"] = ~(co | pc)
    m["external_reachable"] = union
    m["reachability_status"] = np.where(union, "EXTERNAL_REACHABLE", "EXTERNAL_UNREACHABLE")
    m["reachability_labels"] = [";".join(lab for lab, f in (("TRAIN_REACHABLE", t), ("COCONUT_REACHABLE", c), ("PUBCHEM_REACHABLE", p))
                                         if f) or "NONE" for t, c, p in zip(m["in_train"], co, pc, strict=True)]
    return m


def c2_external_protocol(molecules):
    """C2 under the external protocol = held-out truths that ARE externally reachable (never via TRAIN). Adds
    `c2_external_eligible` and `legacy_c2_train_only` (a legacy C2 truth with no external source: must be zero)."""
    m = molecules.copy()
    m["c2_external_eligible"] = m["external_reachable"].to_numpy(bool)
    legacy_c2 = m["regime"].astype(str).eq("C2").to_numpy() if "regime" in m else np.zeros(len(m), bool)
    m["legacy_c2_train_only"] = legacy_c2 & ~m["external_reachable"].to_numpy(bool)
    return m


# ---------------------------------------------------------------------------------------------
# mass bins + summaries
# ---------------------------------------------------------------------------------------------

def mass_bin_labels(bins):
    b = [float(x) for x in bins]
    labels = [(f"<{b[1]:g}" if b[0] <= 0 else f"{b[0]:g}-{b[1]:g}")] + [f"{lo:g}-{hi:g}" for lo, hi in zip(b[1:-1], b[2:], strict=True)]
    return labels + [f">={b[-1]:g}"]


def stratify_by_mass_bin(masses, bins):
    """Labels per mass with the PROJECT bin edges (`validation.mass_bins_da`): [b_i, b_{i+1}) bins, '<b1' for the first
    when b0 <= 0, '>=b_last' above, 'unknown' for missing masses."""
    b = np.asarray([float(x) for x in bins])
    labels = mass_bin_labels(b)
    m = pd.to_numeric(pd.Series(masses), errors="coerce").to_numpy(float)
    idx = np.digitize(m, b[1:], right=False)                 # 0 .. len(b)-1
    out = np.array([labels[min(i, len(labels) - 1)] for i in idx], dtype=object)
    out[np.isnan(m)] = "unknown"
    if b[0] > 0:
        out[(~np.isnan(m)) & (m < b[0])] = f"<{b[0]:g}"
    return out


def reachability_summary(molecules, by=("panel",)):
    """Headline rates per group. TRAIN membership is reported ONLY as `train_membership_rate_DIAGNOSTIC`."""
    rows = []
    for keys, g in molecules.groupby(list(by), sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        n = len(g)
        pc_av = bool(g["pubchem_available"].any()) if "pubchem_available" in g else False

        def f(c, g=g, n=n):
            return float(g[c].mean()) if n else np.nan
        co = f("in_coconut")
        union = f("in_external_union")
        union_wo_pubchem = float((g["in_coconut"] | g["in_other_external"]).mean()) if n else np.nan
        rows.append({**dict(zip(by, keys, strict=True)), "n_molecules": n, "train_membership_rate_DIAGNOSTIC": f("in_train"),
                     "coconut_reachability": co, "pubchem_reachability": f("in_pubchem") if pc_av else np.nan,
                     "pubchem_label": "PUBCHEM_KEY_MEMBERSHIP_ESTIMATE" if pc_av else "NOT_AVAILABLE",
                     "union_external_reachability": union,
                     "incremental_pubchem_over_coconut_pp": (100.0 * (union - union_wo_pubchem)) if pc_av else np.nan,
                     "coconut_only_rate": f("coconut_only"), "pubchem_only_rate": f("pubchem_only") if pc_av else np.nan,
                     "coconut_and_pubchem_rate": f("coconut_and_pubchem") if pc_av else np.nan, "neither_rate": f("neither"),
                     "n_external_unreachable": int((~g["external_reachable"].astype(bool)).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# candidate masks + mass-window coverage + pool density
# ---------------------------------------------------------------------------------------------

def external_candidate_mask(universe_root):
    """Boolean masks over candidate ids from the finalized buckets: `external` (>= 1 non-TRAIN source), `train`,
    `coconut`. One pass over `candidate_sources` (pyarrow, pandas-3 safe)."""
    import pyarrow.parquet as pq
    root = Path(universe_root)
    off = json.loads((root / "bucket_offsets.json").read_text(encoding="utf-8"))
    n = int(off["n_candidates"])
    masks = {k: np.zeros(n, dtype=bool) for k in ("external", "train", "coconut")}
    for b, start in off["offsets"].items():
        cs = pq.read_table(root / "buckets" / f"bucket={b}.parquet", columns=["candidate_sources"]).column("candidate_sources").to_pylist()
        sl = slice(int(start), int(start) + len(cs))
        sets = [set(_as_list(s)) for s in cs]
        masks["external"][sl] = [bool(s - {"TRAIN"}) for s in sets]
        masks["train"][sl] = ["TRAIN" in s for s in sets]
        masks["coconut"][sl] = ["COCONUT" in s for s in sets]
    return masks


def evaluate_mass_window_coverage(index, queries, ppm_values=(2, 5, 10), external_mask=None, tier1_index=None):
    """Long table, one row per (query, ppm). Required query columns: query_id, panel, neutral_mass, truth_candidate_id
    (-1 when the truth is not in the finalized universe), external_reachable.

        truth_in_mass_window        external_reachable AND min |ppm| of the truth's mass variants <= ppm
        full_current_universe_pool_size   unique candidates of TRAIN u COCONUT (finalized) in the window
        external_only_pool_size     same, TRAIN-only candidates excluded (`external_mask`)
        projected_pubchem_tier1_pool_size  RAW Tier-1 CID rows in the window (NOT finalized connectivities), NaN if absent
        coverage_outcome            database_coverage_failure | mass_retrieval_failure | retrieved

    `database_coverage_failure` = not externally reachable (fix the database); `mass_retrieval_failure` = reachable but
    the truth is outside the ppm window (fix precursor / adduct / mass handling). No ranking is computed."""
    q = queries.reset_index(drop=True)
    m = q["neutral_mass"].to_numpy(float)
    t = q["truth_candidate_id"].to_numpy(np.int64)
    reach = q["external_reachable"].to_numpy(bool)
    tppm = index.truth_abs_ppm(m, t)
    excl = None if external_mask is None else ~np.asarray(external_mask, bool)
    rows = []
    for ppm in ppm_values:
        in_win_any = np.isfinite(tppm) & (tppm <= ppm)
        full = index.unique_counts(m, ppm)
        ext = index.unique_counts(m, ppm, exclude_mask=excl) if excl is not None else np.full(len(m), np.nan)
        proj = tier1_index.projected_counts(m, ppm).astype(float) if tier1_index is not None else np.full(len(m), np.nan)
        outcome = np.where(~reach, "database_coverage_failure", np.where(~in_win_any, "mass_retrieval_failure", "retrieved"))
        rows.append(pd.DataFrame({"query_id": q["query_id"].to_numpy(), "panel": q["panel"].to_numpy(), "ppm": ppm,
                                  "truth_abs_ppm": tppm, "external_reachable": reach, "truth_in_mass_window": reach & in_win_any,
                                  "full_current_universe_pool_size": full, "external_only_pool_size": ext,
                                  "projected_pubchem_tier1_pool_size": proj, "coverage_outcome": outcome}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


POOL_COLUMNS = ("full_current_universe_pool_size", "external_only_pool_size", "projected_pubchem_tier1_pool_size")


def summarize_pool_density(window_table, by=("panel", "ppm"), value_columns=POOL_COLUMNS):
    """median / p75 / p90 / p95 / p99 / max per group and pool definition (finalized connectivity counts vs the RAW
    projected PubChem Tier-1 count are separate rows -- never added together)."""
    rows = []
    for keys, g in window_table.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for c in value_columns:
            v = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy(float) if c in g else np.zeros(0)
            kind = "pubchem_projected_raw_count" if "pubchem" in c else "finalized_connectivity_count"
            stats = {"n": int(len(v))}
            if len(v):
                stats.update(median=float(np.median(v)), p75=float(np.percentile(v, 75)), p90=float(np.percentile(v, 90)),
                             p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)), max=float(v.max()))
            rows.append({**dict(zip(by, keys, strict=True)), "pool": c, "count_kind": kind, **stats})
    return pd.DataFrame(rows)


def window_outcome_summary(window_table, by=("panel", "ppm")):
    g = window_table.groupby(list(by), sort=True)
    out = g.agg(n_queries=("query_id", "size"), external_reachable_rate=("external_reachable", "mean"),
                truth_in_mass_window_rate=("truth_in_mass_window", "mean")).reset_index()
    oc = window_table.groupby(list(by) + ["coverage_outcome"]).size().unstack(fill_value=0).reset_index()
    return out.merge(oc, on=list(by), how="left")


# ---------------------------------------------------------------------------------------------
# filter-loss audit (current vs rescue policy)
# ---------------------------------------------------------------------------------------------

LOSS_CATEGORIES = {"mass_out_of_range": "mass", "prefilter_mass_out_of_range": "mass", "not_organic": "organic",
                   "prefilter_not_organic": "organic", "charged": "charge", "multi_component": "component",
                   "parse_failed": "parse", "no_connectivity_key": "parse", "missing_smiles": "parse", "no_mass": "parse"}


def loss_categories(reason):
    """'mass_out_of_range;charged' -> ['charge', 'mass']; parse reasons like 'parse_failed: ...' -> ['parse']."""
    out = set()
    for part in str(reason or "").split(";"):
        token = part.split(":")[0].strip()
        if token:
            out.add(LOSS_CATEGORIES.get(token, "other"))
    return sorted(out)


def collect_lost_records(namespaces, sources=None):
    """External source records LOST by the current Stage-A policy (from the given Stage-A build namespaces):
    prefilter + post-standardization filter drops (`filtered_out/`) and parse rejects (`rejected/`). Columns: source,
    source_id, raw_smiles, connectivity_key (NA when never canonicalized), loss_reason, loss_stage, loss_categories."""
    parts = []
    for ns in [Path(n) for n in namespaces]:
        for f in sorted((ns / "filtered_out").glob("*/chunk-*.parquet")):
            t = pd.read_parquet(f)
            if "connectivity_key" not in t:
                t["connectivity_key"] = pd.NA
            stage = np.where(t["filter_reason"].astype(str).str.startswith("prefilter_"), "prefilter", "filter")
            parts.append(pd.DataFrame({"source": t["source"].astype(str), "source_id": t["source_id"].astype(str),
                                       "raw_smiles": t["raw_smiles"], "connectivity_key": t["connectivity_key"],
                                       "loss_reason": t["filter_reason"].astype(str), "loss_stage": stage}))
        for f in sorted((ns / "rejected").glob("*/chunk-*.parquet")):
            t = pd.read_parquet(f)
            parts.append(pd.DataFrame({"source": t["source"].astype(str), "source_id": t["source_id"].astype(str),
                                       "raw_smiles": t["raw_smiles"], "connectivity_key": pd.NA,
                                       "loss_reason": t["failure_reason"].astype(str), "loss_stage": "parse"}))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["source", "source_id", "raw_smiles", "connectivity_key", "loss_reason", "loss_stage"])
    if sources is not None:
        out = out[out["source"].isin(set(sources))].reset_index(drop=True)
    out["loss_categories"] = out["loss_reason"].map(lambda r: ";".join(loss_categories(r)))
    return out


def key_unkeyed_lost_records(lost, engine):
    """Fill `connectivity_key` for lost records that were dropped BEFORE canonicalization (prefilter) with the SAME
    training canonicalizer (current policy, no cleanup) -- needed to attribute prefilter losses to truths. Parse rejects
    stay unkeyed. `engine`: `casmi.candidates.stage_a.StructureEngine` (caller opens / closes it)."""
    from casmi.candidates.stage_a import standardize_records_fast
    need = lost["connectivity_key"].isna() & lost["loss_stage"].eq("prefilter") & lost["raw_smiles"].notna()
    if not need.any():
        return lost.assign(key_from="stage_a")
    rec = lost.loc[need, ["source", "source_id", "raw_smiles"]].reset_index(drop=True)
    std, _, _ = standardize_records_fast(rec, engine)
    keymap = std.drop_duplicates(["source", "source_id"]).set_index(["source", "source_id"])["connectivity_key"]
    out = lost.copy()
    idx = pd.MultiIndex.from_frame(out.loc[need, ["source", "source_id"]])
    out.loc[need, "connectivity_key"] = keymap.reindex(idx).to_numpy()
    out["key_from"] = np.where(need, "recomputed_for_audit", "stage_a")
    return out


def rescue_lost_records(lost, filters, engine, policy="rescue_policy", categories=("charge", "component", "organic", "mass")):
    """Apply the cleanup policy to lost records whose categories intersect `categories`, canonicalize the cleaned
    structures with the training canonicalizer and re-apply the policy's effective filters. Returns one row per rescued
    attempt: source, source_id, original_connectivity_key, rescued_connectivity_key, salt_stripped, uncharged,
    permanently_charged, charge_cleanup_status, passes_after_rescue, rescue_filter_reason, rescue_kind.
    Nothing is written into the production universe."""
    from casmi.candidates.filters import apply_filters
    from casmi.candidates.rescue import CLEANUP_COLUMNS, apply_cleanup_to_records, effective_filters, get_policy
    from casmi.candidates.stage_a import standardize_records_fast
    p = get_policy(policy)
    sel = lost["loss_categories"].map(lambda s: bool(set(str(s).split(";")) & set(categories))) & lost["raw_smiles"].notna()
    rec = lost.loc[sel, ["source", "source_id", "raw_smiles", "connectivity_key"]].rename(columns={"connectivity_key": "original_connectivity_key"})
    rec = rec.reset_index(drop=True)
    cols = ["source", "source_id", "original_connectivity_key", "rescued_connectivity_key", "salt_stripped", "uncharged",
            "permanently_charged", "charge_cleanup_status", "passes_after_rescue", "rescue_filter_reason", "rescue_kind"]
    if rec.empty:
        return pd.DataFrame(columns=cols)
    cleaned = apply_cleanup_to_records(rec, p)
    std, _, _ = standardize_records_fast(cleaned, engine, extra_columns=tuple(CLEANUP_COLUMNS) + ("original_connectivity_key",))
    kept, filt = apply_filters(std, effective_filters(filters, p))
    kept = kept.assign(passes_after_rescue=True, rescue_filter_reason="")
    filt = filt.assign(passes_after_rescue=False, rescue_filter_reason=filt["filter_reason"])
    out = pd.concat([kept, filt], ignore_index=True).rename(columns={"connectivity_key": "rescued_connectivity_key"})
    from casmi.utils.booleans import as_bool
    ss, un, pcg = as_bool(out["salt_stripped"]), as_bool(out["uncharged"]), as_bool(out["permanently_charged"])
    # 'prefilter_relaxed': the structure itself needed no change -- it was lost only because the source FORMULA (e.g. with a
    # counter-ion) failed the current prefilter, which the rescue policy relaxes for non-organic formulas
    out["rescue_kind"] = [";".join(k for k, f in (("fragment_parent", a), ("uncharger", b), ("permanent_charge_retained", c)) if f)
                          or "prefilter_relaxed" for a, b, c in zip(ss, un, pcg, strict=True)]
    out["salt_stripped"], out["uncharged"], out["permanently_charged"] = ss.to_numpy(), un.to_numpy(), pcg.to_numpy()
    return out[cols]


def audit_filter_losses(molecules, lost, rescued=None, external_sources=("COCONUT",), mass_bins=None):
    """Molecule-level filter-loss table: one row per (panel, truth connectivity, external source).

        reachable_raw              the source has the truth before filters (kept in the universe OR a lost record whose
                                   ORIGINAL key is the truth)
        lost_by_current_filter     source records with the truth key exist but none survived the current policy
        loss_reason                union of loss categories (mass / organic / charge / component / parse)
        rescued_by_uncharger       a rescued record (Uncharger / retained permanent charge) now yields the truth and passes
        rescued_by_fragment_parent a rescued record (parent fragment) now yields the truth and passes
        newly_reachable_by_rescue  rescue reaches a truth no original record matched (e.g. a salt whose own key differed)
        reachable_after_rescue     kept OR rescued
    """
    rows = []
    lost = lost.assign(connectivity_key=lost["connectivity_key"].astype("string"))
    resc = rescued if rescued is not None else pd.DataFrame(columns=["source", "rescued_connectivity_key", "passes_after_rescue", "rescue_kind"])
    for src in external_sources:
        flag = f"in_{src.lower()}"
        ls = lost[lost["source"] == src]
        reasons = ls.dropna(subset=["connectivity_key"]).groupby("connectivity_key")["loss_categories"].agg(
            lambda s: ";".join(sorted({c for x in s for c in str(x).split(";") if c})))
        rs = resc[(resc["source"] == src) & resc["passes_after_rescue"].astype(bool)]
        by_kind = rs.groupby("rescued_connectivity_key")["rescue_kind"].agg(lambda s: ";".join(sorted(set(";".join(s).split(";")))))
        kept = molecules[flag].to_numpy(bool) if flag in molecules else np.zeros(len(molecules), bool)
        keys = molecules["connectivity_key"].astype(str).to_numpy()
        lost_hit = np.array([k in reasons.index for k in keys], dtype=bool)
        kinds = [by_kind.get(k, "") for k in keys]
        resc_un = np.array(["uncharger" in k or "permanent_charge_retained" in k for k in kinds], dtype=bool)
        resc_fr = np.array(["fragment_parent" in k for k in kinds], dtype=bool)
        any_resc = np.array([bool(k) for k in kinds], dtype=bool)
        rows.append(pd.DataFrame({"panel": molecules["panel"].to_numpy(), "connectivity_key": keys, "external_source": src,
                                  "reachable_raw": kept | lost_hit, "kept_current_policy": kept,
                                  "lost_by_current_filter": lost_hit & ~kept,
                                  "loss_reason": [reasons.get(k, "") if (lh and not kp) else "" for k, lh, kp in zip(keys, lost_hit, kept, strict=True)],
                                  "rescued_by_uncharger": resc_un & ~kept, "rescued_by_fragment_parent": resc_fr & ~kept,
                                  "newly_reachable_by_rescue": any_resc & ~kept & ~lost_hit,
                                  "reachable_after_rescue": kept | any_resc,
                                  "mass_bin": stratify_by_mass_bin(molecules["truth_exact_mass"], mass_bins) if mass_bins is not None else None}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def filter_loss_summary(loss_table, by=("panel", "external_source")):
    if loss_table is None or not len(loss_table):
        return pd.DataFrame(columns=list(by) + ["n_molecules", "reachable_before_filters", "kept_current_policy", "lost_count", "lost_fraction",
                                                "rescued_count", "rescue_fraction_of_lost", "newly_reachable_by_rescue", "reachable_after_rescue"])
    rows = []
    for keys, g in loss_table.groupby(list(by), sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        raw = int(g["reachable_raw"].sum())
        lost = int(g["lost_by_current_filter"].sum())
        resc = int((g["rescued_by_uncharger"] | g["rescued_by_fragment_parent"]).sum())
        newly = int(g["newly_reachable_by_rescue"].sum())
        cats = pd.Series([c for r in g.loc[g["lost_by_current_filter"], "loss_reason"] for c in str(r).split(";") if c]).value_counts()
        rows.append({**dict(zip(by, keys, strict=True)), "n_molecules": int(len(g)), "reachable_before_filters": raw, "kept_current_policy": int(g["kept_current_policy"].sum()),
                     "lost_count": lost, "lost_fraction": lost / raw if raw else np.nan, "rescued_count": resc,
                     "rescue_fraction_of_lost": resc / lost if lost else np.nan, "newly_reachable_by_rescue": newly,
                     "reachable_after_rescue": int(g["reachable_after_rescue"].sum()),
                     **{f"lost_by_{c}": int(cats.get(c, 0)) for c in ("mass", "organic", "charge", "component", "parse")}})
    return pd.DataFrame(rows)


def filter_policy_comparison(loss_table, by=("panel", "external_source")):
    """current_policy vs rescue_policy reachability side by side (never rebuilds anything)."""
    if loss_table is None or not len(loss_table):
        return pd.DataFrame(columns=list(by) + ["n_molecules", "reachable_current_policy", "reachable_rescue_policy", "rescue_gain_pp"])
    g = loss_table.groupby(list(by), sort=True)
    return g.agg(n_molecules=("connectivity_key", "size"), reachable_current_policy=("kept_current_policy", "mean"),
                 reachable_rescue_policy=("reachable_after_rescue", "mean")).reset_index().assign(
        rescue_gain_pp=lambda d: 100.0 * (d["reachable_rescue_policy"] - d["reachable_current_policy"]))


# ---------------------------------------------------------------------------------------------
# adduct / charge + mass-range audits
# ---------------------------------------------------------------------------------------------

def adduct_charge_audit(adducts):
    """One row per adduct string actually present: polarity, charge, multiplier, whether the grammar supports it, and
    whether M itself carries the charge ('[M]+' / '[M]-' style -> compatible with a PERMANENTLY charged candidate).
    Nothing is assumed: only adducts present in the data appear."""
    import re
    from casmi.chemistry.adducts import parse_adduct
    counts = pd.Series(adducts, dtype="object").value_counts(dropna=False)
    intrinsic = re.compile(r"^\[\d*M\]\d*[+-]$")
    rows = []
    for a, n in counts.items():
        p = parse_adduct(a)
        s = str(a).strip() if isinstance(a, str) else ""
        is_intrinsic = bool(p["supported"] and intrinsic.match(s))
        pol = None if not p["supported"] else ("positive" if p["charge"] > 0 else "negative")
        impl = ("unparsed (unsupported adduct grammar / element)" if not p["supported"] else
                "M is itself the ion (no proton / adduct gained or lost): requires a PERMANENTLY charged candidate" if is_intrinsic else
                "neutral M ionized by the adduct")
        rows.append({"adduct": a, "n_occurrences": int(n), "supported": bool(p["supported"]), "polarity": pol, "charge": p["charge"],
                     "molecule_multiplier": p["molecule_multiplier"], "charge_implication": impl,
                     "permanently_charged_compatible": is_intrinsic})
    return pd.DataFrame(rows)


def mass_range_audit(molecules, filters, by=("panel",)):
    """Validation truths outside the CURRENT external mass filter (reported, never changed automatically)."""
    m = pd.to_numeric(molecules["truth_exact_mass"], errors="coerce")
    d = molecules.assign(below=m < filters.min_exact_mass, above=m > filters.max_exact_mass, mass_missing=m.isna())
    out = d.groupby(list(by)).agg(n_molecules=("connectivity_key", "size"), n_below_min=("below", "sum"), n_above_max=("above", "sum"),
                                  n_mass_missing=("mass_missing", "sum")).reset_index()
    out["min_exact_mass_filter"], out["max_exact_mass_filter"] = filters.min_exact_mass, filters.max_exact_mass
    out["outside_fraction"] = (out["n_below_min"] + out["n_above_max"]) / out["n_molecules"].where(out["n_molecules"] > 0)
    return out


def inference_mass_range(table=None, precursor_col="precursor_mz", adduct_col="adduct"):
    """Neutral-mass statistics of an inference-safe query table (no labels used) or NOT_AVAILABLE when none is configured."""
    if table is None:
        return {"status": "NOT_AVAILABLE", "reason": "coverage.inference_metadata_path not configured"}
    from casmi.chemistry.adducts import neutral_mass_from_precursor
    nm = pd.Series([neutral_mass_from_precursor(float(p), a) if pd.notna(p) else None for p, a in zip(table[precursor_col], table[adduct_col], strict=True)],
                   dtype=float)
    v = nm.dropna().to_numpy()
    if not len(v):
        return {"status": "NO_PARSEABLE_ROWS", "n_rows": int(len(table))}
    return {"status": "AVAILABLE", "n_rows": int(len(table)), "n_parsed": int(len(v)), "min": float(v.min()), "p01": float(np.percentile(v, 1)),
            "median": float(np.median(v)), "p99": float(np.percentile(v, 99)), "max": float(v.max())}


# ---------------------------------------------------------------------------------------------
# official-metric parity + public baseline (local artifacts only)
# ---------------------------------------------------------------------------------------------

def official_metric_parity(module_path=None, function_name=None, smiles=(), project_keyfn=None):
    """Compare the project canonicalizer with a COMPETITION-PROVIDED canonicalizer (a local .py file the user supplies).
    No such file exists in the repository; without one -> OFFICIAL_METRIC_PARITY = NOT_AVAILABLE. Nothing is downloaded."""
    if not module_path or not Path(module_path).is_file() or not function_name:
        return {"OFFICIAL_METRIC_PARITY": "NOT_AVAILABLE",
                "reason": "no competition-provided metric / canonicalizer file configured (coverage.official_metric.module_path)"}
    spec = importlib.util.spec_from_file_location("casmi_official_metric_user", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # the USER's local file, executed only when the user runs notebook 13
    fn = getattr(mod, function_name)
    rows = []
    for s in smiles:
        a = project_keyfn(s) if project_keyfn else None
        try:
            b = fn(s)
        except Exception as exc:                    # report, never hide
            b = f"error: {exc}"
        rows.append({"smiles": s, "project_key": a, "official_key": b, "equal": a == b})
    t = pd.DataFrame(rows)
    return {"OFFICIAL_METRIC_PARITY": "AVAILABLE", "module": Path(module_path).name, "function": function_name, "n": int(len(t)),
            "n_equal": int(t["equal"].sum()) if len(t) else 0, "mismatches": t[~t["equal"]].head(50).to_dict(orient="records") if len(t) else []}


def public_baseline_coverage(public_pool=None, queries=None, window_table=None, ppm=5, keyfn=None):
    """Coverage gap vs ranking gap against a LOCAL public-baseline candidate pool (query_id + connectivity_key, or
    query_id + smiles -> project canonicalizer). Without one -> PUBLIC_BASELINE_COVERAGE = NOT_AVAILABLE."""
    if public_pool is None or queries is None:
        return {"PUBLIC_BASELINE_COVERAGE": "NOT_AVAILABLE",
                "reason": "no local public-baseline candidate pool configured (coverage.public_baseline.path)"}, pd.DataFrame()
    p = public_pool.copy()
    if "connectivity_key" not in p:
        if keyfn is None or "smiles" not in p:
            raise ValueError("public baseline needs connectivity_key, or smiles + keyfn")
        p["connectivity_key"] = [keyfn(s) for s in p["smiles"]]
    p["query_id"] = p["query_id"].astype(str)
    pub = set(zip(p["query_id"], p["connectivity_key"].astype(str), strict=True))
    q = queries[["panel", "query_id", "connectivity_key"]].assign(query_id=queries["query_id"].astype(str))
    q["truth_in_public_pool"] = [(a, b) in pub for a, b in zip(q["query_id"], q["connectivity_key"], strict=True)]
    w = window_table[window_table["ppm"] == ppm][["query_id", "panel", "truth_in_mass_window"]] if window_table is not None else None
    q = q.merge(w, on=["panel", "query_id"], how="left") if w is not None else q.assign(truth_in_mass_window=np.nan)
    from casmi.utils.booleans import as_bool
    cur = as_bool(q["truth_in_mass_window"], missing=False).to_numpy()
    pubv = q["truth_in_public_pool"].to_numpy(bool)
    q["coverage_overlap"] = np.select([pubv & cur, pubv & ~cur, ~pubv & cur], ["both", "public_only", "current_only"], "neither")
    comp = q.groupby("panel")["coverage_overlap"].value_counts().unstack(fill_value=0).reset_index()
    status = {"PUBLIC_BASELINE_COVERAGE": "AVAILABLE", "ppm": ppm, "n_queries": int(len(q)),
              "public_pool_truth_coverage": float(pubv.mean()) if len(q) else None,
              "current_external_truth_coverage": float(cur.mean()) if len(q) else None}
    return status, comp


# ---------------------------------------------------------------------------------------------
# decision rules (declared BEFORE results; evaluated mechanically)
# ---------------------------------------------------------------------------------------------

@dataclass
class DecisionPolicy:
    """Thresholds come from `coverage.decision` in the config. Only Rule A's 10 pp threshold is given by the project
    brief; every other threshold defaults to None = NOT EVALUATED (no 'meaningful' / 'large' / 'high' value is invented)."""
    natural_product_panel: str | None = "HOST"           # set at run time to the selected decision panel (HOST, else NP_PANEL); never TIMSTOF_DEV
    rule_a_min_incremental_pp: float = 10.0
    rule_b_ppm: float = 5.0
    rule_b_pool: str = "full_current_universe_pool_size"
    rule_b_statistic: str = "median"
    rule_b_min_reachability: float | None = None
    rule_b_large_pool_threshold: float | None = None
    rule_c_min_rescued_fraction: float | None = None      # share of externally reachable-before-filters truths recovered by rescue
    rule_d_min_coconut_reachability: float | None = None
    rule_d_max_lost_fraction: float | None = None
    priority: tuple = ("A", "C", "B", "D")
    stage_b_invalidation_fixed: bool = True               # code fact of this branch (STAGE_B_VERSION 3); Rule C requires it

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        if "priority" in d:
            d["priority"] = tuple(d["priority"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _rule(rule, status, reason, **inputs):
    return {"rule": rule, "outcome_if_fired": RULE_OUTCOME[rule], "status": status, "fired": status == "FIRED", "reason": reason, "inputs": inputs}


def evaluate_decision_rules(metrics, policy, protocol_valid=True):
    """`metrics` keys (None when unavailable): np_panel_present, pubchem_incremental_pp, np_reachability_union,
    pool_statistic, np_coconut_reachability, lost_fraction, rescued_fraction. Returns {'decision', 'rules', ...}."""
    p = policy
    rules = {}
    if not metrics.get("np_panel_present"):
        rules["A"] = _rule("A", "NOT_EVALUATED", f"natural-product panel {p.natural_product_panel!r} not present")
    elif metrics.get("pubchem_incremental_pp") is None:
        rules["A"] = _rule("A", "NOT_EVALUATED", "PubChem membership not available")
    else:
        v = float(metrics["pubchem_incremental_pp"])
        rules["A"] = _rule("A", "FIRED" if v >= p.rule_a_min_incremental_pp else "NOT_FIRED",
                           f"PubChem adds {v:.2f} pp over COCONUT (threshold {p.rule_a_min_incremental_pp} pp)", pubchem_incremental_pp=v)
    if p.rule_b_large_pool_threshold is None or p.rule_b_min_reachability is None:
        rules["B"] = _rule("B", "NOT_EVALUATED", "rule_b_large_pool_threshold / rule_b_min_reachability not set in coverage.decision")
    elif metrics.get("pool_statistic") is None or metrics.get("np_reachability_union") is None:
        rules["B"] = _rule("B", "NOT_EVALUATED", "pool statistic or reachability unavailable")
    else:
        big = float(metrics["pool_statistic"]) >= p.rule_b_large_pool_threshold
        high = float(metrics["np_reachability_union"]) >= p.rule_b_min_reachability
        rules["B"] = _rule("B", "FIRED" if (big and high) else "NOT_FIRED",
                           f"{p.rule_b_statistic} {p.rule_b_pool} @{p.rule_b_ppm} ppm = {metrics['pool_statistic']}, reachability {metrics['np_reachability_union']}",
                           pool_statistic=metrics["pool_statistic"], reachability=metrics["np_reachability_union"])
    if not p.stage_b_invalidation_fixed:
        rules["C"] = _rule("C", "NOT_EVALUATED", "Stage-B/C invalidation not fixed -- a rescue rebuild could reuse stale buckets")
    elif p.rule_c_min_rescued_fraction is None:
        rules["C"] = _rule("C", "NOT_EVALUATED", "rule_c_min_rescued_fraction not set (no project threshold for 'meaningful')")
    elif metrics.get("rescued_fraction") is None:
        rules["C"] = _rule("C", "NOT_EVALUATED", "rescue comparison not computed")
    else:
        v = float(metrics["rescued_fraction"])
        rules["C"] = _rule("C", "FIRED" if v >= p.rule_c_min_rescued_fraction else "NOT_FIRED",
                           f"rescue recovers {v:.4f} of externally reachable truths (threshold {p.rule_c_min_rescued_fraction})", rescued_fraction=v)
    if p.rule_d_min_coconut_reachability is None or p.rule_d_max_lost_fraction is None:
        rules["D"] = _rule("D", "NOT_EVALUATED", "rule_d thresholds not set in coverage.decision")
    elif metrics.get("np_coconut_reachability") is None or metrics.get("lost_fraction") is None:
        rules["D"] = _rule("D", "NOT_EVALUATED", "COCONUT reachability or filter-loss fraction unavailable")
    else:
        hi = float(metrics["np_coconut_reachability"]) >= p.rule_d_min_coconut_reachability
        lo = float(metrics["lost_fraction"]) <= p.rule_d_max_lost_fraction
        rules["D"] = _rule("D", "FIRED" if (hi and lo) else "NOT_FIRED",
                           f"COCONUT reachability {metrics['np_coconut_reachability']}, lost fraction {metrics['lost_fraction']}",
                           coconut_reachability=metrics["np_coconut_reachability"], lost_fraction=metrics["lost_fraction"])
    fired = [r for r in p.priority if rules[r]["fired"]]
    if not protocol_valid:
        decision = "PROTOCOL_INVALID"
    elif fired:
        decision = RULE_OUTCOME[fired[0]]
    elif any(rules[r]["status"] == "NOT_EVALUATED" for r in rules):
        decision = "UNDETERMINED_THRESHOLDS_OR_INPUTS_MISSING"
    else:
        decision = "NO_RULE_FIRED"
    return {"decision": decision, "fired_rules": fired, "priority": list(p.priority), "rules": rules, "protocol_valid": bool(protocol_valid),
            "policy": asdict(p), "policy_identity": p.identity(), "evaluated_at": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------------------------
# gate summary + notebook-14 precondition
# ---------------------------------------------------------------------------------------------

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def require_coverage_gate(report_dir, allowed_decisions=("PROCEED_TO_ANALOG_BASELINE",), override=False):
    """Notebook 14 precondition: `coverage_gate_summary.json` exists, `completed`, `protocol_valid`, and its decision
    is allowed. `override=True` (explicit, logged by the caller) returns the summary with `override_used=True` instead
    of raising. Never recomputes coverage."""
    p = Path(report_dir) / "coverage_gate_summary.json"
    problems = []
    s = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    if s is None:
        problems.append(f"{p} not found -- run notebook 13 (13_candidate_coverage) first")
    else:
        if not s.get("completed"):
            problems.append("candidate coverage gate did not complete")
        if not s.get("protocol_valid"):
            problems.append(f"external protocol invalid: {s.get('protocol_reasons')}")
        if s.get("decision") not in allowed_decisions:
            problems.append(f"decision {s.get('decision')!r} does not permit the analog baseline (allowed: {list(allowed_decisions)})")
    if problems and not override:
        raise CoverageGateError("notebook 14 precondition failed:\n  - " + "\n  - ".join(problems))
    return {**(s or {}), "override_used": bool(problems and override), "precondition_problems": problems}


# ---------------------------------------------------------------------------------------------
# notebook-13 orchestration helpers (kept here so the notebook stays orchestration-sized)
# ---------------------------------------------------------------------------------------------

def universe_truth_rows(universe_root, keys):
    """Finalized-universe rows (candidate_id, connectivity_key, candidate_sources, molecular_formula, exact_mass,
    representative_smiles) of the given truth keys; keys absent from the universe are simply missing. Reads only the
    buckets involved."""
    import pyarrow.parquet as pq
    from casmi.candidates.universe import keys_to_ids, load_candidate_keys
    cols = ["connectivity_key", "candidate_sources", "molecular_formula", "exact_mass", "representative_smiles"]
    root = Path(universe_root)
    uk = load_candidate_keys(root)
    k = pd.unique(pd.Series(keys, dtype="object").astype(str))
    ids = np.unique(keys_to_ids(uk, k))
    ids = ids[ids >= 0]
    if not len(ids):
        return pd.DataFrame(columns=["candidate_id"] + cols)
    off = json.loads((root / "bucket_offsets.json").read_text(encoding="utf-8"))["offsets"]
    names = sorted(off, key=lambda b: off[b])
    starts = np.array([off[b] for b in names])
    which = np.searchsorted(starts, ids, side="right") - 1
    parts = []
    for bi in np.unique(which):
        local = ids[which == bi] - starts[bi]
        t = pq.read_table(root / "buckets" / f"bucket={names[bi]}.parquet", columns=cols).take(local)   # pyarrow: pandas-3 safe lists
        d = pd.DataFrame({c: t.column(c).to_pylist() for c in cols})
        parts.append(d.assign(candidate_id=local + starts[bi]))
    return pd.concat(parts, ignore_index=True)[["candidate_id"] + cols]


def protocol_validity(universe_status, manifest_status, populations):
    """The external protocol is valid only with an external source in the finalized universe and >= 1 validation
    population. A legacy manifest is readable (reported) and does not by itself invalidate coverage measurements."""
    reasons = []
    if universe_status.get("status") != "external":
        reasons.append(f"universe status {universe_status.get('status')!r}: no external source -- external reachability is undefined")
    srcs = universe_status.get("sources") or {}
    if not (srcs.get("COCONUT") or {}).get("n_candidates"):
        reasons.append("COCONUT has no candidates in the finalized universe")
    if populations is None or not len(populations):
        reasons.append("no validation population loaded")
    return {"protocol_valid": not reasons, "protocol_reasons": reasons,
            "legacy_manifest_detected": bool(manifest_status.get("legacy_manifest_detected"))}


def decision_metrics(reach_by_panel, pool_summary, loss_summary, policy):
    """Inputs of `evaluate_decision_rules` from the notebook-13 summary tables (None where unavailable)."""
    np_panel = policy.natural_product_panel
    r = reach_by_panel[reach_by_panel["panel"] == np_panel] if np_panel else reach_by_panel.iloc[0:0]
    present = bool(len(r)) and int(r["n_molecules"].iloc[0]) > 0
    inc = float(r["incremental_pubchem_over_coconut_pp"].iloc[0]) if present else None
    inc = None if inc is None or not np.isfinite(inc) else inc
    pool_stat = None
    if present and pool_summary is not None and len(pool_summary):
        ps = pool_summary[(pool_summary["panel"] == np_panel) & (pool_summary["ppm"] == policy.rule_b_ppm) & (pool_summary["pool"] == policy.rule_b_pool)]
        if len(ps) and policy.rule_b_statistic in ps:
            v = ps[policy.rule_b_statistic].iloc[0]
            pool_stat = float(v) if pd.notna(v) else None
    lost_frac = rescued_frac = None
    if present and loss_summary is not None and len(loss_summary):
        ls = loss_summary[loss_summary["panel"] == np_panel]
        raw = float(ls["reachable_before_filters"].sum()) if len(ls) else 0.0
        if raw > 0:
            lost_frac = float(ls["lost_count"].sum()) / raw
            rescued_frac = float(ls["rescued_count"].sum() + ls["newly_reachable_by_rescue"].sum()) / raw
    return {"np_panel_present": present, "pubchem_incremental_pp": inc,
            "np_reachability_union": float(r["union_external_reachability"].iloc[0]) if present else None,
            "np_coconut_reachability": float(r["coconut_reachability"].iloc[0]) if present else None,
            "pool_statistic": pool_stat, "lost_fraction": lost_frac, "rescued_fraction": rescued_frac}


def observed_stage_a_throughput(namespace, source="COCONUT"):
    """Observed Stage-A throughput from the per-chunk done markers of a COMPLETED build (records/s over the chunks that
    were actually processed in some session). Used only to feed `estimate_runtime` (ESTIMATE_ONLY)."""
    recs = []
    for f in sorted((Path(namespace) / "_done" / source).glob("chunk-*.json")):
        try:
            recs.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    n = sum(int(r.get("n_input", 0)) for r in recs if r.get("seconds"))
    s = sum(float(r.get("seconds", 0)) for r in recs if r.get("seconds"))
    return {"n_chunks": len(recs), "n_records_timed": n, "seconds": s, "records_per_second": (n / s) if s > 0 else None}


def read_optional_table(path):
    """parquet / csv / tsv table or None when `path` is None / missing (optional user-supplied inputs)."""
    if not path or not Path(path).is_file():
        return None
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, sep="\t" if p.suffix in (".tsv", ".txt") else ",", dtype=str)


# ---------------------------------------------------------------------------------------------
# planning helper
# ---------------------------------------------------------------------------------------------

def estimate_runtime(n_records, observed_records_per_second, observed_workers=1, target_workers=None, parallel_efficiency=1.0):
    """ESTIMATE_ONLY: linear extrapolation of an OBSERVED throughput to `n_records` (and optionally another worker count,
    scaled by `parallel_efficiency`). Never a measurement."""
    if not observed_records_per_second or observed_records_per_second <= 0:
        raise ValueError("observed_records_per_second must be > 0")
    tw = target_workers or observed_workers
    rate = float(observed_records_per_second) * (float(tw) / float(observed_workers)) * (float(parallel_efficiency) if tw != observed_workers else 1.0)
    sec = float(n_records) / rate
    return {"label": ESTIMATE_ONLY, "n_records": int(n_records), "assumed_records_per_second": rate, "estimated_seconds": sec,
            "estimated_hours": sec / 3600.0, "observed_workers": observed_workers, "target_workers": tw,
            "parallel_efficiency": parallel_efficiency}
