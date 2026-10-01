"""v4b training-set construction: source / structure hold-out filters and the nested 1k -> 3k ->
10k training-query sets for the scale experiment.

Training-set filters only ever remove TRAINING queries; the DEV evaluation query set is fixed
across every variant so DEV OOF numbers stay comparable.

Test-like enrichment uses ONLY observable, label-free test-side metadata (instrument type, adduct,
ionization mode, neutral-mass distribution of the test spectra). No hidden test label, and no
test structure, is ever read. Structure-level "natural-product-likeness" enrichment is NOT
implemented: the test side has no label-free structural descriptor to match against, and
inventing one from candidate pools would smuggle candidate-generation choices into the
training distribution.
"""
import numpy as np
import pandas as pd

TEST_LIKE_DIMS = ("instrument_type", "adduct", "ionization_mode", "mass_bin")


def v2_filter(train_queries, host_source):
    """V2 -- source held out: drop training queries acquired by the HOST source/library."""
    return train_queries[train_queries["source"] != host_source]


def v3_filter(train_queries, host_source, host_connectivity_keys):
    """V3 -- source AND structure held out: V2 plus drop every training query whose TRUE
    connectivity is any HOST connectivity (the ranker never sees a HOST structure as a positive)."""
    keys = set(host_connectivity_keys)
    return v2_filter(train_queries, host_source).loc[lambda d: ~d["true_connectivity_key"].isin(keys)]


def _mass_bins(test_mass, n_bins=10):
    qs = np.quantile(np.asarray(test_mass, dtype=float), np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    return np.unique(qs)


def test_like_weights(pool_queries, test_queries, dims=TEST_LIKE_DIMS, n_mass_bins=10, clip=(0.05, 20.0),
                      restrict_to_test_adducts=True, floor_unseen=0.05):
    """Importance weight per pool query = product over `dims` of the marginal density ratio
    p_test(level) / p_pool(level) (each clipped to `clip`). Levels that never occur in test get
    weight `floor_unseen` (or 0 for adducts when `restrict_to_test_adducts`: only the adducts
    observed in the test set are eligible). Returns `(weights, report_df)`."""
    pool = pool_queries.copy()
    test = test_queries.copy()
    edges = _mass_bins(test["neutral_mass"].dropna(), n_mass_bins)
    pool["mass_bin"] = pd.cut(pool["neutral_mass"], edges, labels=False, include_lowest=True).astype("float")
    test["mass_bin"] = pd.cut(test["neutral_mass"], edges, labels=False, include_lowest=True).astype("float")
    w = np.ones(len(pool))
    report = []
    for d in dims:
        p_test = test[d].astype(str).value_counts(normalize=True)
        p_pool = pool[d].astype(str).value_counts(normalize=True)
        lv = pool[d].astype(str)
        ratio = (lv.map(p_test) / lv.map(p_pool)).to_numpy(float)
        unseen = ~lv.isin(p_test.index).to_numpy()
        ratio = np.clip(np.nan_to_num(ratio, nan=0.0), *clip)
        if d == "adduct" and restrict_to_test_adducts:
            ratio[unseen] = 0.0
        else:
            ratio[unseen] = floor_unseen
        w *= ratio
        for level in sorted(set(p_test.index) | set(p_pool.index)):
            report.append({"dim": d, "level": level, "p_test": float(p_test.get(level, 0.0)), "p_pool": float(p_pool.get(level, 0.0))})
    return w, pd.DataFrame(report)


def nested_scale_sets(pool_queries, base_ids, sizes=(3000, 10000), strategy="test_like", weights=None,
                      seed=42, exclude_ids=()):
    """Nested training-query sets: `base_ids` (the existing DEV-1k) is always included; each
    larger set adds new queries drawn WITHOUT replacement from the remaining pool (weighted by
    `weights` for strategy="test_like", uniform for "random"). Returns
    `{"1k": ids, "3k": ids, "10k": ids}` (keys from `sizes`), each a superset of the previous."""
    base = list(dict.fromkeys(base_ids))
    rng = np.random.default_rng(seed)
    excluded = set(exclude_ids) | set(base)
    pool = pool_queries.reset_index(drop=True)
    avail_mask = ~pool["query_id"].isin(excluded).to_numpy()
    if strategy == "test_like":
        if weights is None:
            raise ValueError("strategy='test_like' needs weights")
        p = np.asarray(weights, dtype=float).copy()
    elif strategy == "random":
        p = np.ones(len(pool))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    p[~avail_mask] = 0.0
    need_total = max(sizes) - len(base)
    if need_total > int((p > 0).sum()):
        raise ValueError(f"not enough eligible pool queries ({int((p > 0).sum())}) for {need_total} additions")
    # one weighted draw of the full ordered addition sequence -> prefixes are the nested sets
    order = rng.choice(len(pool), size=need_total, replace=False, p=p / p.sum())
    additions = pool["query_id"].to_numpy()[order].tolist()
    out = {size_name(len(base)): base}
    for s in sorted(sizes):
        out[size_name(s)] = base + additions[: s - len(base)]
    return out


def size_name(n):
    """1000 -> "1k", 10000 -> "10k"; non-multiples of 1000 keep the exact count ("250")."""
    return f"{n // 1000}k" if n % 1000 == 0 and n > 0 else str(n)


def distribution_report(sets_by_name, pool_queries, test_queries, dims=("instrument_type", "adduct", "ionization_mode")):
    """Share of each level per training set, next to the test share -- documents the sampling."""
    rows = []
    q = pool_queries.set_index("query_id")
    for name, ids in sets_by_name.items():
        sub = q.loc[list(ids)]
        for d in dims:
            for level, share in sub[d].astype(str).value_counts(normalize=True).items():
                rows.append({"set": name, "dim": d, "level": level, "share": float(share)})
    for d in dims:
        for level, share in test_queries[d].astype(str).value_counts(normalize=True).items():
            rows.append({"set": "TEST (observable metadata)", "dim": d, "level": level, "share": float(share)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# v5.1: Mode-A training filter (applied AFTER QCR construction; no evidence is rebuilt)
# ---------------------------------------------------------------------------------------------

MODE_A_SUFFIX = "_MODE_A"


def filter_mode_a_training_queries(manifest, regimes):
    """Mode-A ranker training keeps a query iff its truth is in the candidate pool AND has >= 1
    eligible mirror_aware reference (regime MODE_A_MIRROR). Returns `(filtered_copy, audit)`; the
    input manifest is never mutated. Every manifest query must have a regime label."""
    from casmi.validation.regime_audit import MODE_A_MIRROR
    lab = regimes.set_index("query_id")["regime"]
    missing = set(manifest["query_id"]) - set(lab.index)
    if missing:
        raise ValueError(f"{len(missing)} manifest queries have no regime label -- run the regime audit first")
    keep = manifest["query_id"].map(lab).eq(MODE_A_MIRROR).to_numpy()
    out = manifest.loc[keep].copy()
    out["selection_reason"] = out["selection_reason"].astype(str) + " | MODE_A filter: truth in pool with >=1 mirror_aware reference"
    audit = {"n_original": int(len(manifest)), "n_mode_a_retained": int(keep.sum()),
             "retention_rate": float(keep.mean()) if len(manifest) else float("nan"),
             "removed_by_regime": manifest.loc[~keep, "query_id"].map(lab).value_counts().to_dict()}
    return out, audit


def derive_mode_a_manifests(manifests, regimes_by_name, nested_chain=("TL_1K", "TL_3K", "TL_10K")):
    """`{name + "_MODE_A": filtered}` for every available manifest, plus audits. Because the filter
    is a per-query decision, prefixes stay prefixes: TL_1K_MODE_A ⊂ TL_3K_MODE_A ⊂ TL_10K_MODE_A
    (asserted, never repaired by resampling)."""
    out, audits = {}, {}
    for name, m in manifests.items():
        out[name + MODE_A_SUFFIX], audits[name + MODE_A_SUFFIX] = filter_mode_a_training_queries(m, regimes_by_name[name])
    chain = [n + MODE_A_SUFFIX for n in nested_chain if n + MODE_A_SUFFIX in out]
    for a, b in zip(chain[:-1], chain[1:]):
        assert set(out[a]["query_id"]) <= set(out[b]["query_id"]), f"{a} is not nested in {b} after the Mode-A filter"
    return out, audits
