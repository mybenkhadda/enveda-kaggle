"""v5.1 query-regime audit: where does the TRUE candidate stand on reference evidence?

For every query the truth candidate's eligible-reference count is read from the EXISTING QCR evidence
(no similarity is recomputed) under two protocols, using the existing implementations:

    standard      -- `casmi.spectra.reference_selection.is_eligible("standard", ...)`: every reference
                     except the query itself (same-library evidence allowed)
    mirror_aware  -- same-source references and T1/T2 mirrors excluded (the PRIMARY protocol)

Counts come from `casmi.qcr.aggregate.eligible_reference_counts`, i.e. the censored representation of
`casmi.qcr.censored`: exact when the lazy walk exhausted the list (or accepted < 5), otherwise the lower
bound 5 flagged `*_count_censored`. Exact counts are never inferred from a censored walk.

Regimes (exhaustive; every query gets exactly one):

    POOL_ABSENT        truth not in the mass-window candidate pool
    MODE_A_MIRROR      truth in pool AND >= 1 eligible mirror_aware reference
    STANDARD_ONLY      truth in pool AND >= 1 standard reference AND 0 mirror_aware references
    REF_ABSENT         truth in pool AND 0 references under both protocols
    INCONSISTENT       truth in pool AND mirror_aware >= 1 but standard 0 -- impossible by construction
                       (standard ⊇ mirror_aware); kept as an explicit label so it can never be merged silently
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi.qcr.aggregate import eligible_reference_counts

POOL_ABSENT, MODE_A_MIRROR, STANDARD_ONLY, REF_ABSENT, INCONSISTENT = "POOL_ABSENT", "MODE_A_MIRROR", "STANDARD_ONLY", "REF_ABSENT", "INCONSISTENT"
REGIMES = (MODE_A_MIRROR, STANDARD_ONLY, REF_ABSENT, POOL_ABSENT, INCONSISTENT)
AUDIT_PROTOCOLS = ("standard", "mirror_aware")
ROUTE_THRESHOLD = 0.50          # pre-registered: TL_EVAL MODE_A_MIRROR share below this -> PRIORITIZE_REFERENCE_ABSENT
ROUTE_CONTINUE, ROUTE_REF_ABSENT = "CONTINUE_MODE_A_SEQUENCE", "PRIORITIZE_REFERENCE_ABSENT"
NOT_BUILT = "NOT_BUILT"
KEY = "candidate_connectivity_key"
QCR_TRUTH_COLS = ["query_id", "candidate_key", "is_true"] + [f"{a}_{p}" for a in ("eligible", "accepted") for p in AUDIT_PROTOCOLS]


class ArtifactNotBuilt(RuntimeError):
    """A dataset's QCR / pool artifacts do not exist yet (reported as NOT_BUILT, never a crash)."""


# ---------------------------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------------------------

def classify_regime(truth_in_pool, n_standard, n_mirror):
    """Vectorized. Counts are the exact value or the censored lower bound (both >= 1 test correctly)."""
    tip = np.asarray(truth_in_pool, dtype=bool)
    s = np.asarray(n_standard, dtype=float)
    m = np.asarray(n_mirror, dtype=float)
    return np.select(
        [~tip, (m >= 1) & (s >= 1), (s >= 1) & (m < 1), (s < 1) & (m < 1), (m >= 1) & (s < 1)],
        [POOL_ABSENT, MODE_A_MIRROR, STANDARD_ONLY, REF_ABSENT, INCONSISTENT], default=INCONSISTENT).astype(object)


def truth_reference_counts(qcr_truth, pairs_truth, truth_pool):
    """`qcr_truth` / `pairs_truth`: QCR and pair-summary rows of TRUTH candidates only; `truth_pool`: one
    pool row per query whose truth is in the pool. Returns per-query counts for both protocols."""
    out = truth_pool[["query_id", KEY]].copy()
    for p in AUDIT_PROTOCOLS:
        c = eligible_reference_counts(qcr_truth, pairs_truth, truth_pool[["query_id", KEY]], protocol=p)
        short = "standard" if p == "standard" else "mirror"
        out = out.merge(c.rename(columns={"eligible_reference_count": f"n_{short}_eligible_exact_or_lower_bound",
                                          "eligible_reference_count_is_exact": f"{short}_count_is_exact",
                                          "eligible_reference_saturated": f"{short}_ge5"})[
            ["query_id", KEY, f"n_{short}_eligible_exact_or_lower_bound", f"{short}_count_is_exact", f"{short}_ge5"]],
            on=["query_id", KEY], how="left", validate="one_to_one")
        out[f"{short}_count_censored"] = ~out[f"{short}_count_is_exact"]
    return out


def build_query_regimes(manifest, pool, qcr_truth, pairs_truth):
    """One row per MANIFEST query (the denominator), whatever its pool / evidence status."""
    pool = pool[["query_id", KEY, "is_true_candidate"]]
    truth_pool = pool[pool["is_true_candidate"].astype(bool)].drop_duplicates("query_id")
    counts = truth_reference_counts(qcr_truth, pairs_truth, truth_pool)
    meta_cols = [c for c in ("query_id", "connectivity_key", "source", "instrument", "adduct", "fold") if c in manifest.columns]
    r = manifest[meta_cols].merge(counts.drop(columns=[KEY]), on="query_id", how="left", validate="one_to_one")
    r["truth_in_pool"] = r["query_id"].isin(set(truth_pool["query_id"]))
    for short in ("standard", "mirror"):
        n = f"n_{short}_eligible_exact_or_lower_bound"
        r[n] = r[n].fillna(0).astype(int)
        r[f"{short}_count_censored"] = r[f"{short}_count_censored"].fillna(False).astype(bool)
        r[f"{short}_count_is_exact"] = r[f"{short}_count_is_exact"].fillna(True).astype(bool)
        r[f"{short}_ge5"] = r[f"{short}_ge5"].fillna(False).astype(bool)
    r["has_standard_ref"] = r["n_standard_eligible_exact_or_lower_bound"] >= 1
    r["has_mirror_ref"] = r["n_mirror_eligible_exact_or_lower_bound"] >= 1
    r["regime"] = classify_regime(r["truth_in_pool"], r["n_standard_eligible_exact_or_lower_bound"], r["n_mirror_eligible_exact_or_lower_bound"])
    return r


# ---------------------------------------------------------------------------------------------
# artifact loading (existing QCR only; ArtifactNotBuilt when absent)
# ---------------------------------------------------------------------------------------------

def _truth_rows(parquet_paths, columns):
    parts = []
    for p in parquet_paths:
        df = pd.read_parquet(p, columns=columns)
        parts.append(df[df["is_true"].astype(bool)])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def load_v5_truth_evidence(v5_dirs, units):
    """Truth-only QCR / pairs / pool for the build units of a v5 manifest (bounded RAM: shard by shard)."""
    qcr_paths, pair_paths, pools = [], [], []
    for u in units:
        qd, pd_, pool_p = v5_dirs["qcr"] / u / "qcr" / "shards", v5_dirs["qcr"] / u / "qcr_pairs" / "shards", v5_dirs["features"] / "units" / f"{u}_pool.parquet"
        if not (qd.exists() and pool_p.exists() and any(qd.glob("part-*.parquet"))):
            raise ArtifactNotBuilt(f"unit {u}: QCR shards / pool not found under {v5_dirs['qcr'] / u}")
        qcr_paths += sorted(qd.glob("part-*.parquet"))
        pair_paths += sorted(pd_.glob("part-*.parquet"))
        pools.append(pd.read_parquet(pool_p, columns=["query_id", KEY, "is_true_candidate"]))
    qcr = _truth_rows(qcr_paths, QCR_TRUTH_COLS)
    pairs = _truth_rows(pair_paths, ["query_id", "candidate_key", "is_true", "exhausted"])
    return pd.concat(pools, ignore_index=True), qcr, pairs


def load_host_truth_evidence(evidence_artifacts):
    arts = evidence_artifacts
    pool = pd.read_parquet(arts["host_features_mirror_aware"], columns=["query_id", KEY, "is_true_candidate"])
    qcr = _truth_rows([arts["host_qcr"]], QCR_TRUTH_COLS)
    pairs = _truth_rows([arts["host_qcr_pairs"]], ["query_id", "candidate_key", "is_true", "exhausted"])
    return pool, qcr, pairs


# ---------------------------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------------------------

def censored_median(n_lower, censored):
    """Median of a censored-at-5 count sample: a number when the median element is exact, the string
    '>=5' when it falls in the censored block (never a fabricated exact value)."""
    lo = np.asarray(n_lower, dtype=float)
    cen = np.asarray(censored, dtype=bool)
    if len(lo) == 0:
        return None
    key = np.where(cen, np.inf, lo)
    order = np.sort(key)
    mid = order[(len(order) - 1) // 2], order[len(order) // 2]
    if np.isinf(mid[0]) or np.isinf(mid[1]):
        return ">=5"
    return float((mid[0] + mid[1]) / 2)


def summary_row(regimes):
    r = regimes
    n = len(r)
    share = lambda mask: float(np.mean(mask)) if n else float("nan")
    row = {"n_queries": int(n), "truth_in_pool_share": share(r["truth_in_pool"])}
    for short, name in (("standard", "standard"), ("mirror", "mirror")):
        cnt = r[f"n_{short}_eligible_exact_or_lower_bound"]
        row[f"{name}_ge1_share"] = share(cnt >= 1)
        row[f"{name}_ge3_share"] = share(cnt >= 3)
        row[f"{name}_ge5_share"] = share(cnt >= 5)
    for reg, key in ((MODE_A_MIRROR, "mode_a_share"), (STANDARD_ONLY, "standard_only_share"), (REF_ABSENT, "ref_absent_share"),
                     (POOL_ABSENT, "pool_absent_share"), (INCONSISTENT, "inconsistent_share")):
        row[key] = share(r["regime"] == reg)
    in_pool = r[r["truth_in_pool"]]
    row["median_mirror_refs_censored"] = censored_median(in_pool["n_mirror_eligible_exact_or_lower_bound"], in_pool["mirror_count_censored"])
    row["median_standard_refs_censored"] = censored_median(in_pool["n_standard_eligible_exact_or_lower_bound"], in_pool["standard_count_censored"])
    return row


def source_breakdown(regimes, source_col="source"):
    rows = []
    for src, g in regimes.groupby(source_col, dropna=False, sort=True):
        rows.append({"source": src, **summary_row(g)})
    return pd.DataFrame(rows)


def metrics_by_regime(per_query, regimes, groups=(MODE_A_MIRROR, STANDARD_ONLY, REF_ABSENT, POOL_ABSENT)):
    """`per_query`: rank_eval.per_query_metrics output over ALL queries. Returns one row per
    population: ALL (every query, RR=0 for pool-absent) and each regime subset (n always shown)."""
    d = per_query.merge(regimes[["query_id", "regime"]], on="query_id", how="left", validate="one_to_one")
    if d["regime"].isna().any():
        raise ValueError(f"{int(d['regime'].isna().sum())} evaluated queries have no regime label")
    pops = [("ALL", d)] + [(g, d[d["regime"] == g]) for g in groups]
    out = []
    for name, sub in pops:
        rec = {"population": name, "n_queries": int(len(sub))}
        for m, c in (("mrr_at_25", "rr"), ("hit_at_1", "hit_at_1"), ("hit_at_5", "hit_at_5"), ("hit_at_25", "hit_at_25")):
            rec[m] = float(sub[c].mean()) if len(sub) else float("nan")
        out.append(rec)
    return pd.DataFrame(out)


def molecule_regime_mix(regimes, molecule_col="connectivity_key"):
    """Per molecule: ALL_MODE_A / MIXED / ALL_REF_ABSENT / OTHER (any STANDARD_ONLY or POOL_ABSENT
    spectrum that makes it neither pure case)."""
    g = regimes.groupby(molecule_col)["regime"]
    n, n_a, n_r = g.size(), g.apply(lambda s: int((s == MODE_A_MIRROR).sum())), g.apply(lambda s: int((s == REF_ABSENT).sum()))
    label = np.where(n_a == n, "ALL_MODE_A", np.where(n_r == n, "ALL_REF_ABSENT", np.where((n_a > 0) & (n_r > 0) & (n_a + n_r == n), "MIXED", "OTHER")))
    return pd.DataFrame({"n_spectra": n, "n_mode_a": n_a, "n_ref_absent": n_r, "molecule_regime_mix": label}).reset_index()


def project_route(tl_eval_mode_a_share, threshold=ROUTE_THRESHOLD):
    """Pre-registered routing (NOT a gate: existing builds and the Mode-A comparison always finish)."""
    route = ROUTE_REF_ABSENT if tl_eval_mode_a_share < threshold else ROUTE_CONTINUE
    meaning = {ROUTE_REF_ABSENT: "finish TL_3K/TL_10K builds and the Mode-A scale comparison; the NEXT major research effort is reference-absent / Mode-B evidence",
               ROUTE_CONTINUE: "complete the scaling freeze, run molecule aggregation, then move to Mode B"}[route]
    return {"project_route": route, "mirror_mode_a_coverage": float(tl_eval_mode_a_share), "threshold": threshold,
            "rule": "PRIORITIZE_REFERENCE_ABSENT iff TL_EVAL MODE_A_MIRROR share < 0.50", "meaning": meaning, "is_gate": False}


def save_regimes(regimes, name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_query_regimes.parquet"
    regimes.to_parquet(path, index=False)
    content = regimes[["query_id", "regime"]].astype(str).sort_values("query_id")
    fp = hashlib.sha256("\n".join("|".join(r) for r in content.itertuples(index=False, name=None)).encode("utf-8")).hexdigest()
    meta = {"name": name, "n_queries": int(len(regimes)), "regime_counts": regimes["regime"].value_counts().to_dict(), "sha256": fp}
    (out_dir / f"{name}_query_regimes.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_regimes(name, out_dir):
    path = Path(out_dir) / f"{name}_query_regimes.parquet"
    if not path.exists():
        raise ArtifactNotBuilt(f"regime table {path} not built -- run 10v5_03_tl_regime_audit.ipynb")
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------------------------
# v5.2 routing: the largest regime decides the next question (supersedes the 0.50 mirror-coverage rule)
# ---------------------------------------------------------------------------------------------

ROUTE_V2_MODE_A, ROUTE_V2_RESOLVE, ROUTE_V2_REF_ABSENT = "MODE_A", "RESOLVE_TEST_PROTOCOL", "REFERENCE_ABSENT"
ROUTE_V2_RULE = ("largest of MODE_A_MIRROR / STANDARD_ONLY / REF_ABSENT TL_EVAL shares: REF_ABSENT -> REFERENCE_ABSENT; "
                 "STANDARD_ONLY -> RESOLVE_TEST_PROTOCOL; otherwise MODE_A. Supersedes the v5.1 'mirror coverage < 0.50' rule, "
                 "which could not distinguish protocol-induced absence from genuine reference absence.")


def project_route_v2(mode_a_share, standard_only_share, ref_absent_share):
    shares = {MODE_A_MIRROR: mode_a_share, STANDARD_ONLY: standard_only_share, REF_ABSENT: ref_absent_share}
    dominant = max(shares, key=lambda k: (shares[k], k == MODE_A_MIRROR))     # exact tie -> MODE_A (no new route on a tie)
    route = {REF_ABSENT: ROUTE_V2_REF_ABSENT, STANDARD_ONLY: ROUTE_V2_RESOLVE}.get(dominant, ROUTE_V2_MODE_A)
    return {"project_route": route, "dominant_regime": dominant, "regime_shares": shares, "rule": ROUTE_V2_RULE, "is_gate": False,
            "supersedes": "v5.1 project_route (mirror_mode_a_coverage < 0.50)"}
