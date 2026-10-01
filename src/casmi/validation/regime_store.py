"""Persisted regime table (notebook 11): signature-checked load-or-build + composition tables.

The signature records everything the assignment depends on -- regime config hash, C2 mode, universe status and the
universe identity (`casmi.validation.c2_protocol.universe_identity`), number of dev queries. A persisted table with a
different signature is NEVER reused silently: the notebook must be re-run with REBUILD=True.
"""
import pandas as pd

from casmi.validation.regimes import build_regime_table


class RegimeSignatureError(RuntimeError):
    pass


def regime_signature(regime_cfg, c2_mode, universe, n_dev_queries):
    return {"regime_config_hash": regime_cfg.config_hash(), "c2_mode": c2_mode, "universe_status": universe.universe_status,
            "universe_identity": universe.identity, "n_dev_queries": int(n_dev_queries)}


def load_or_build_regimes(dev, reference_connectivity, unified_truths, regime_cfg, artifacts, signature, rebuild=False):
    """Returns (regime table, action in {'loaded', 'built'})."""
    from casmi.workspace.artifact_registry import read_json
    path = artifacts.validation_regimes
    meta = read_json(artifacts.validation_regimes_meta)
    old = (meta or {}).get("signature")
    if path.exists() and old == signature and not rebuild:
        return pd.read_parquet(path), "loaded"
    if path.exists() and not rebuild:
        raise RegimeSignatureError(
            f"ERROR: the persisted regime table was built with signature\n  {old}\nbut the current inputs give\n  {signature}\n"
            f"    Fix: set REBUILD = True and re-run notebook 11 (then every downstream notebook: 13, 14).")
    return build_regime_table(dev, reference_connectivity, unified=unified_truths, cfg=regime_cfg), "built"


def composition_tables(regimes, dev, mass_bins, top=8):
    """Row-normalized regime composition by ion mode / instrument / mass bin / #truth spectra / adduct. The regimes
    should differ only by what is hidden, not by population (otherwise C1-vs-C2 comparisons are confounded)."""
    d = regimes.merge(dev[["query_id", "neutral_mass", "ionization_mode", "adduct", "instrument_type"]], on="query_id", how="left",
                      validate="one_to_one")
    d["regime"] = d["regime"].fillna("UNASSIGNED")
    d["mass_bin"] = pd.cut(d["neutral_mass"], bins=mass_bins, right=False).astype(str)
    d["n_truth_spectra_bin"] = pd.cut(d["n_truth_spectra"], bins=[0, 1, 2, 3, 5, 10, 50, 10 ** 9], right=False).astype(str)
    out = {}
    for col in ("ionization_mode", "instrument_type", "mass_bin", "n_truth_spectra_bin", "adduct"):
        t = pd.crosstab(d["regime"], d[col].astype(str), normalize="index").round(3)
        if col in ("adduct", "instrument_type"):
            t = t[t.sum().sort_values(ascending=False).index[:top]]
        out[col] = t
    return out, d
