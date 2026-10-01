"""Stratified reporting for the v2 regimes: one long table per metric family, broken down by
regime x {ALL, ion mode, adduct, mass bin, instrument, number of query spectra, candidate-pool-size bin}.

The metric functions are passed in (`casmi.validation.metrics.candidate_stage_metrics` /
`ranking_metrics`), so this module contains no metric definitions of its own."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_STRATA = ("ionization_mode", "adduct", "mass_bin", "instrument_type", "n_query_spectra_bin", "pool_size_bin")
N_SPECTRA_BINS = [0, 1, 2, 3, 5, 10, 10 ** 9]


def _bin_labels(edges):
    return [f"[{edges[i]}, {edges[i + 1]})" for i in range(len(edges) - 1)]


def add_strata(per_query, query_meta, mass_bins, pool_bins=None, query_col="query_id", mass_col="neutral_mass",
               pool_col="pool_size", group_cols=("fold", "true_connectivity_key")):
    """Join query metadata (ion mode, adduct, instrument, mass) and add `mass_bin`, `pool_size_bin` and
    `n_query_spectra_bin` (#evaluated spectra sharing the query's truth inside its fold -- the size of the
    molecule-level query a spectrum belongs to)."""
    wanted = [mass_col, "ionization_mode", "adduct", "instrument_type", *group_cols]
    meta_cols = [query_col] + [c for c in dict.fromkeys(wanted) if c in query_meta.columns and c not in per_query.columns]
    d = per_query.merge(query_meta[meta_cols].drop_duplicates(query_col), on=query_col, how="left", validate="many_to_one")
    d["mass_bin"] = pd.cut(d[mass_col].astype(float), bins=mass_bins, right=False, labels=_bin_labels(mass_bins)).astype(str)
    if pool_bins is not None and pool_col in d:
        d["pool_size_bin"] = pd.cut(d[pool_col].astype(float), bins=pool_bins, right=False, labels=_bin_labels(pool_bins)).astype(str)
    gc = [c for c in group_cols if c in d.columns]
    if gc:
        n = d.groupby(gc)[query_col].transform("size")
        d["n_query_spectra"] = n.to_numpy()
        d["n_query_spectra_bin"] = pd.cut(n, bins=N_SPECTRA_BINS, right=False, labels=_bin_labels(N_SPECTRA_BINS)).astype(str)
    return d


def stratified_table(per_query, metric_fn, strata=DEFAULT_STRATA, regime_col="regime", min_queries=1, **metric_kwargs):
    """Long table: one row per (regime, stratum, value) + an 'ALL' stratum per regime and an 'ALL' regime.
    Strata with fewer than `min_queries` queries are dropped from the table (counts stay in 'ALL')."""
    rows = []
    regimes = ["ALL"]
    if regime_col in per_query.columns:
        regimes += sorted(per_query[regime_col].dropna().unique().tolist())
    for reg in regimes:
        d = per_query if reg == "ALL" else per_query[per_query[regime_col] == reg]
        rows.append({"regime": reg, "stratum": "ALL", "value": "ALL", **metric_fn(d, **metric_kwargs)})
        for s in strata:
            if s not in d.columns:
                continue
            for v, g in d.groupby(d[s].astype(str), sort=True, observed=True):
                if len(g) >= min_queries:
                    rows.append({"regime": reg, "stratum": s, "value": v, **metric_fn(g, **metric_kwargs)})
    return pd.DataFrame(rows)


def regime_summary(per_query, metric_fn, regime_col="regime", **metric_kwargs):
    t = stratified_table(per_query, metric_fn, strata=(), regime_col=regime_col, **metric_kwargs)
    return t.drop(columns=["stratum", "value"]).reset_index(drop=True)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def save_report(table, summary, reports_dir, name):
    """`<reports_dir>/<name>.parquet` (+ `<name>.json` with `summary`). NaN -> null in JSON."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(reports_dir / f"{name}.parquet", index=False)
    (reports_dir / f"{name}.json").write_text(json.dumps(_jsonable(summary), indent=2, default=str), encoding="utf-8")
    return reports_dir / f"{name}.parquet", reports_dir / f"{name}.json"
