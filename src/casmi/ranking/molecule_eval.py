"""Molecule-level development evaluation for 11_02: test-like multiplicity resampling, molecule
MRR@25 per aggregator, molecule-level bootstrap, and the DEV-only aggregator selection rule.

A development "molecule" = one connectivity with >= 3 spectra (MOL_DEV manifest). For each of R
resamples, every molecule draws m spectra, m ~ the VISIBLE test spectra-per-molecule distribution
(capped at the molecule's available spectra), without replacement; the aggregators rank the union
of the drawn spectra's candidates; the truth's molecule rank gives RR (0 if absent / beyond 25).
"""
import numpy as np
import pandas as pd

from casmi.ranking.selection import HostLeakError, guard_no_host

MRR_K = 25


def multiplicity_resamples(per_spectrum, test_n_spectra, n_resamples=20, seed=42, molecule_col="molecule_id", all_spectra=None):
    """Yields `(r, subset_df)`; subset keeps only the drawn spectra of every molecule.
    `all_spectra` (v6.2, optional): the population table (molecule_col, spectrum_id) to DRAW from --
    e.g. the MOL_DEV manifest, so a spectrum with an empty candidate pool still consumes a draw and the
    draws are identical for every protocol / feature table evaluated with the same seed."""
    rng = np.random.default_rng(seed)
    dist = np.asarray(test_n_spectra, dtype=int)
    src = per_spectrum if all_spectra is None else all_spectra
    spectra = src[[molecule_col, "spectrum_id"]].drop_duplicates().sort_values([molecule_col, "spectrum_id"])
    by_mol = {m: g["spectrum_id"].to_numpy() for m, g in spectra.groupby(molecule_col, sort=True)}
    for r in range(n_resamples):
        keep = []
        for m, sids in by_mol.items():
            k = int(min(rng.choice(dist), len(sids)))
            keep.extend(rng.choice(sids, size=k, replace=False).tolist())
        yield r, per_spectrum[per_spectrum["spectrum_id"].isin(set(keep))]


def molecule_rr(mol_ranking, truth_of_molecule, cand_col, k=MRR_K):
    """Per-molecule RR@k over EVERY molecule in `truth_of_molecule` (absent -> 0)."""
    t = mol_ranking.merge(pd.Series(truth_of_molecule, name="_truth").rename_axis("molecule_id").reset_index(), on="molecule_id")
    hit = t[t[cand_col] == t["_truth"]].set_index("molecule_id")["mol_rank"]
    r = hit.reindex(list(truth_of_molecule)).astype(float)
    return pd.Series(np.where(np.isfinite(r) & (r <= k), 1.0 / r, 0.0), index=r.index, name="rr")


def select_aggregator_dev_only(summary, simplicity, ci_col_low="delta_ci_low", ci_col_high="delta_ci_high"):
    """`summary`: one row per aggregator with `aggregator`, `dev_mrr_mean`, and the paired molecule-
    bootstrap CI of (aggregator - best). Rule: best DEV MRR; every aggregator whose CI vs best
    includes 0 is 'within CI'; the SIMPLEST of those wins. HOST-named columns are refused."""
    guard_no_host(summary, "select_aggregator_dev_only")
    s = summary.copy()
    best = s.sort_values(["dev_mrr_mean", "aggregator"], ascending=[False, True]).iloc[0]["aggregator"]
    s["within_ci_of_best"] = (s["aggregator"] == best) | ((s[ci_col_low] <= 0) & (s[ci_col_high] >= 0))
    s["simplicity"] = s["aggregator"].map(simplicity)
    s = s.sort_values(["within_ci_of_best", "simplicity", "dev_mrr_mean", "aggregator"], ascending=[False, True, False, True])
    s["selection_order"] = np.arange(1, len(s) + 1)
    return str(s.iloc[0]["aggregator"]), best, s.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# v6: explicit absent-candidate policy, cross-protocol selection, MOL_DEV requirements
# ---------------------------------------------------------------------------------------------

ABSENT_CANDIDATE_POLICY = {
    "universe": "molecule candidate universe = UNION of its spectra's pools, deduplicated by connectivity BEFORE the top-25 cut",
    "A1_max_score": "max raw score over the spectra where the candidate is PRESENT (absent spectra contribute nothing)",
    "A2_max_prob": "max calibrated probability over PRESENT spectra",
    "A3_mean_prob": "absent -> probability 0; mean over ALL the molecule's (drawn) spectra",
    "A4_sum_log_prob": "absent -> log(eps) (eps fixed before any result); sum over ALL spectra",
    "A5_logmeanexp_present": "absent spectra IGNORED; log of the mean probability over PRESENT spectra",
    "A6_rrf": "absent -> 0 contribution; RRF = sum over PRESENT spectra of 1/(k + rank)",
    "A7_most_confident_spectrum": "only the spectrum with the highest top-1 probability is ranked; candidates absent from it are not ranked",
    "A8_first_spectrum": "only the first spectrum (spectrum_order, else spectrum_id) is ranked; candidates absent from it are not ranked",
    "tie_rule": "aggregate score DESC, secondary ASC (mean abs ppm over present spectra; RRF: best individual rank), candidate key ASC",
}


def absent_policy_fixture(cand_col="candidate_connectivity_key"):
    """One molecule, two spectra: A in both, B only in s1, C only in s2."""
    return pd.DataFrame({"molecule_id": "M", "spectrum_id": ["s1", "s1", "s2", "s2"], "spectrum_order": [0, 0, 1, 1],
                         cand_col: ["A", "B", "A", "C"], "score": [2.0, 1.0, 0.5, 3.0], "rank": [1, 2, 2, 1],
                         "prob": [0.6, 0.4, 0.3, 0.7], "abs_mass_error_ppm": [1.0, 2.0, 1.0, 3.0]})


def expected_absent_policy_scores(eps, k=60):
    """Hand-derived aggregate scores on `absent_policy_fixture` (None = candidate not ranked)."""
    L = np.log
    return {"A1_max_score": {"A": 2.0, "B": 1.0, "C": 3.0}, "A2_max_prob": {"A": 0.6, "B": 0.4, "C": 0.7},
            "A3_mean_prob": {"A": 0.45, "B": 0.2, "C": 0.35}, "A4_sum_log_prob": {"A": L(0.6) + L(0.3), "B": L(0.4) + L(eps), "C": L(eps) + L(0.7)},
            "A5_logmeanexp_present": {"A": L(0.45), "B": L(0.4), "C": L(0.7)}, "A6_rrf": {"A": 1 / (k + 1) + 1 / (k + 2), "B": 1 / (k + 2), "C": 1 / (k + 1)},
            "A7_most_confident_spectrum": {"A": 0.3, "B": None, "C": 0.7}, "A8_first_spectrum": {"A": -1.0, "B": -2.0, "C": None}}


def verify_absent_candidate_policy(aggregators, cand_col="candidate_connectivity_key", atol=1e-12):
    """Runs every aggregator on the fixture and checks it against `expected_absent_policy_scores`, so
    the documented policy -- not an implementation default -- is what the notebook uses. Raises on any
    deviation; returns the per-aggregator check rows."""
    fx = absent_policy_fixture(cand_col)
    rows = []
    for a in aggregators:
        exp = expected_absent_policy_scores(getattr(a, "eps", 1e-6), getattr(a, "k", 60))[a.name]
        got = a.scores(fx).reset_index().set_index(cand_col)["agg_score"]
        for c, v in exp.items():
            ok = (c not in got.index) if v is None else (c in got.index and abs(float(got.loc[c]) - v) <= atol)
            rows.append({"aggregator": a.name, "candidate": c, "expected": v, "got": float(got.loc[c]) if c in got.index else None, "ok": bool(ok)})
    out = pd.DataFrame(rows)
    if not out["ok"].all():
        raise AssertionError(f"absent-candidate policy violated:\n{out[~out['ok']]}")
    return out


def select_across_protocols(selected_by_protocol, simplicity):
    """Pre-registered: one evaluation protocol -> its DEV selection. Several (diagnostic INTERMEDIATE or
    not yet run) -> the common choice if they agree, otherwise the SIMPLEST of the per-protocol choices
    (ties: aggregator id ASC). HOST never enters."""
    guard_no_host(dict(selected_by_protocol), "select_across_protocols")
    picks = sorted(set(selected_by_protocol.values()))
    if len(selected_by_protocol) == 1:
        # only the PRIMARY protocol selects; sensitivity protocols are never passed in here
        return picks[0], "selected on primary protocol; sensitivity protocol did not influence selection"
    if len(picks) == 1:
        return picks[0], "selection protocols agree; sensitivity protocols did not influence selection"
    best = sorted(picks, key=lambda a: (simplicity[a], a))[0]
    return best, f"protocols disagree ({dict(selected_by_protocol)}); simplest per-protocol choice wins"


def moldev_build_instructions(qcr_ready, missing_protocols):
    cmds = []
    if not qcr_ready:
        cmds.append("PYTHONPATH=src PYTHONNOUSERSITE=1 python scripts/v5_build_scale_features.py --manifest MOL_DEV")
    for p in missing_protocols:
        cmds.append(f"PYTHONPATH=src PYTHONNOUSERSITE=1 python scripts/v5_build_protocol_features.py --manifest MOL_DEV --protocol {p}")
    return ("MOL_DEV QCR / protocol features are missing -- this notebook does NOT generate a different development population.\n"
            "Build them, then re-run:\n  " + "\n  ".join(cmds))


class MolDevNotReady(RuntimeError):
    pass


def host_connectivity_keys(host_processed_dir, filename="host_holdout_spectra.parquet"):
    """HOST truth connectivities, used ONLY to prove MOL_DEV excludes them (no HOST metric is read)."""
    from pathlib import Path
    return set(pd.read_parquet(Path(host_processed_dir) / filename, columns=["true_connectivity_key"])["true_connectivity_key"].astype(str))


def mol_dev_requirements(mol_manifest, per_spectrum, training_keys, excluded_keys, host_keys, host_source, molecule_col="molecule_id"):
    """MOL_DEV contract: connectivity-disjoint from the frozen model's training set and from TL_EVAL,
    no HOST source / structure, >= 2 spectra per molecule in the evaluated table."""
    keys = set(mol_manifest["connectivity_key"].astype(str))
    n_spec = per_spectrum.groupby(molecule_col)["spectrum_id"].nunique()
    checks = {"disjoint_from_training": not (keys & set(map(str, training_keys))), "disjoint_from_excluded": not (keys & set(map(str, excluded_keys))),
              "no_host_structures": not (keys & set(map(str, host_keys))), "no_host_source": bool((mol_manifest["source"].astype(str) != host_source).all()),
              "multi_spectrum_molecules": bool(len(n_spec) and (n_spec >= 2).all()), "n_molecules": int(len(n_spec)),
              "n_spectra": int(n_spec.sum()) if len(n_spec) else 0}
    bad = [k for k, v in checks.items() if v is False]
    if bad:
        raise AssertionError(f"MOL_DEV requirements violated: {bad}")
    return checks


def molecule_top_k(mol_ranking, k=MRR_K):
    """Top-k per molecule of an `Aggregator.rank` output (already one row per (molecule, candidate))."""
    return mol_ranking[mol_ranking["mol_rank"] <= k].reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# v6.2: one clean molecule-level experiment (protocol = configuration, not a code path)
# ---------------------------------------------------------------------------------------------

SPECTRUM_RANK_KEYS = [("score", False), ("abs_mass_error_ppm", True), ("candidate_connectivity_key", True)]   # == inference rank_tie_rule
PAIRED_BASELINES = ("A6_rrf", "A1_max_score", "A7_most_confident_spectrum")      # required paired comparisons vs the selected aggregator
SPECTRUM_BASELINE = "A8_first_spectrum"


def score_spectra(models, feats, feature_names, temperature, cand_col="candidate_connectivity_key"):
    """Frozen fold-mean scores -> per-spectrum rank (inference tie rule) -> calibrated prob (softmax(score/T)
    over the spectrum's FULL pool, before any multiplicity resampling). Development molecule = its truth
    connectivity. Returns the long per-(spectrum, candidate) table the aggregators consume."""
    from casmi.ranking.rank_eval import rank_by_keys
    from casmi.ranking.scaling import fold_mean_scores
    from casmi_infer.aggregation import softmax_by_group
    keys = [(cand_col if c == "candidate_connectivity_key" else c, a) for c, a in SPECTRUM_RANK_KEYS]
    f = feats.assign(score=fold_mean_scores(models, feats, feature_names))
    f = rank_by_keys(f, keys)
    per = f.rename(columns={"query_id": "spectrum_id"}).assign(spectrum_id=lambda d: d["spectrum_id"].astype(str),
                                                              molecule_id=lambda d: d["true_connectivity_key"].astype(str))
    per["prob"] = softmax_by_group(per["score"], per["spectrum_id"], temperature)
    return per


def evaluate_aggregators(per_spectrum, truth_of_molecule, aggregators, agg_id, test_n_spectra, all_spectra, n_resamples=20, seed=42,
                         cand_col="candidate_connectivity_key", k=MRR_K):
    """Long RR table (resample, aggregator, molecule_id, rr). Every aggregator sees the SAME drawn spectra
    in a resample; draws depend only on `all_spectra` + seed (identical across protocols)."""
    rows = []
    for r, sub in multiplicity_resamples(per_spectrum, test_n_spectra, n_resamples=n_resamples, seed=seed, all_spectra=all_spectra):
        for a in aggregators:
            rr = molecule_rr(a.rank(sub), truth_of_molecule, cand_col, k=k)
            rows.append(pd.DataFrame({"resample": r, "aggregator": agg_id(a), "molecule_id": rr.index.astype(str), "rr": rr.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def aggregator_metrics(rr_long, n_boot=2000, seed=42):
    """One row per aggregator: mean molecule MRR over resamples, SD over resamples, molecule-bootstrap CI
    (per-molecule RR averaged over resamples, clusters = molecules = connectivities), and the paired
    molecule-bootstrap delta vs the best aggregator (the selection rule's input)."""
    from casmi.validation.cluster_bootstrap import cluster_bootstrap_mean, paired_cluster_bootstrap
    per_mol = rr_long.groupby(["aggregator", "molecule_id"])["rr"].mean().unstack(0).sort_index()
    by_res = rr_long.groupby(["aggregator", "resample"])["rr"].mean().unstack(0)
    s = pd.DataFrame({"dev_mrr_mean": by_res.mean(), "dev_mrr_sd_over_resamples": by_res.std(ddof=1)})
    best = s.sort_values("dev_mrr_mean", ascending=False, kind="mergesort").index[0]
    mols = per_mol.index.to_numpy()
    rows = []
    for a in per_mol.columns:
        ci = cluster_bootstrap_mean(per_mol[a].to_numpy(), mols, n_boot=n_boot, seed=seed)
        d = paired_cluster_bootstrap(per_mol[a].to_numpy(), per_mol[best].to_numpy(), mols, n_boot=n_boot, seed=seed)
        rows.append({"aggregator": a, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "delta_vs_best": d["delta"], "delta_ci_low": d["ci_low"],
                     "delta_ci_high": d["ci_high"], "n_molecules": int(len(mols))})
    out = s.rename_axis("aggregator").reset_index().merge(pd.DataFrame(rows), on="aggregator")
    return out.sort_values(["dev_mrr_mean", "aggregator"], ascending=[False, True], kind="mergesort").reset_index(drop=True), per_mol


def paired_deltas(per_mol, selected, others=PAIRED_BASELINES, n_boot=2000, seed=42):
    """selected - other, paired molecule-level bootstrap (never spectrum-level)."""
    from casmi.validation.cluster_bootstrap import paired_cluster_bootstrap
    rows = []
    for o in others:
        if o not in per_mol.columns:
            continue
        d = paired_cluster_bootstrap(per_mol[selected].to_numpy(), per_mol[o].to_numpy(), per_mol.index.to_numpy(), n_boot=n_boot, seed=seed)
        rows.append({"comparison": f"{selected} - {o}", "selected": selected, "other": o, "delta": d["delta"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                     "n_molecules": int(len(per_mol))})
    return pd.DataFrame(rows, columns=["comparison", "selected", "other", "delta", "ci_low", "ci_high", "n_molecules"])


def spectrum_level_mrr(per_spectrum, all_spectra, k=MRR_K):
    """Context only: mean per-spectrum RR@k of the frozen ranker (empty-pool spectra -> 0)."""
    hit = per_spectrum[per_spectrum["is_true_candidate"].astype(bool)].set_index("spectrum_id")["rank"]
    r = hit.reindex(all_spectra["spectrum_id"].astype(str).unique()).astype(float)
    return float(np.where(np.isfinite(r) & (r <= k), 1.0 / r, 0.0).mean()) if len(r) else float("nan")


def molecule_matchability(per_spectrum, all_spectra, high=0.80, low=0.50):
    """LABEL-FREE matchability proxy per development molecule, analogous to the TEST diagnostic: spectrum
    max library cosine = max `cosine_max` over the spectrum's candidates (== max over ranker-visible pairs;
    0 when the pool / eligible set is empty), molecule = max over its spectra. Reads no truth column."""
    s = per_spectrum.groupby("spectrum_id")["cosine_max"].max()
    a = all_spectra[["molecule_id", "spectrum_id"]].drop_duplicates().astype(str)
    a["max_cosine"] = a["spectrum_id"].map(s).astype(float).fillna(0.0)
    m = a.groupby("molecule_id")["max_cosine"].max().rename("best_spectrum_max_cosine").to_frame()
    m["matchability_tier"] = np.where(m["best_spectrum_max_cosine"] >= high, "HIGH", np.where(m["best_spectrum_max_cosine"] >= low, "MEDIUM", "LOW"))
    return m


def improvement_by_matchability(per_mol, tiers, selected, baselines=(SPECTRUM_BASELINE, "A6_rrf"), n_boot=2000, seed=42):
    """Aggregation gain per label-free matchability tier (reporting only; the tier is never a target)."""
    from casmi.validation.cluster_bootstrap import paired_cluster_bootstrap
    j = per_mol.join(tiers["matchability_tier"], how="left")
    rows = []
    for t, g in j.groupby("matchability_tier", sort=True):
        for b in baselines:
            if b not in g.columns or len(g) < 2:
                continue
            d = paired_cluster_bootstrap(g[selected].to_numpy(), g[b].to_numpy(), g.index.to_numpy(), n_boot=n_boot, seed=seed)
            rows.append({"matchability_tier": t, "n_molecules": int(len(g)), "selected": selected, "baseline": b, "mrr_selected": float(g[selected].mean()),
                         "mrr_baseline": float(g[b].mean()), "delta": d["delta"], "ci_low": d["ci_low"], "ci_high": d["ci_high"]})
    return pd.DataFrame(rows)


def with_representative_smiles(mol_ranking, conn_table, cand_col="candidate_connectivity_key"):
    """Attach the bundle's representative SMILES (one per connectivity) to a molecule ranking."""
    smi = conn_table.set_index("connectivity_key")["representative_smiles"]
    return mol_ranking.assign(representative_smiles=mol_ranking[cand_col].map(smi))


def verify_aggregator_roundtrip(selected_obj, exported_cfg, per_spectrum, cand_col="candidate_connectivity_key"):
    """The exported bundle config must reproduce the DEV ranking EXACTLY on the inference path:
    `make_aggregator(cfg)` + prob = softmax(score / cfg['temperature']) as `casmi_infer.pipeline.aggregate_molecules`
    computes it. Raises on any difference; returns the number of compared (molecule, candidate) rows."""
    from casmi_infer.aggregation import make_aggregator, softmax_by_group
    rebuilt = make_aggregator(exported_cfg, cand_col=cand_col)
    df = per_spectrum.drop(columns=["prob"], errors="ignore")
    if rebuilt.needs_prob:
        df = df.assign(prob=softmax_by_group(df["score"], df["spectrum_id"], exported_cfg.get("temperature", 1.0)))
    a = selected_obj.rank(per_spectrum)[["molecule_id", cand_col, "mol_rank"]].reset_index(drop=True)
    b = rebuilt.rank(df)[["molecule_id", cand_col, "mol_rank"]].reset_index(drop=True)
    if type(rebuilt) is not type(selected_obj) or not a.equals(b):
        raise AssertionError(f"exported aggregator config {exported_cfg} does not reproduce the DEV ranking of {selected_obj.name}")
    return int(len(a))


def aggregator_export_config(selected_obj, aggregator_id, temperature):
    """Bundle `config.json['aggregator']` payload; temperature only when the aggregator uses probabilities."""
    cfg = {**selected_obj.config(), "aggregator_id": aggregator_id, "temperature": float(temperature) if selected_obj.needs_prob else None}
    cfg["tie_rule"] = (["RRF DESC", "best individual rank ASC", "connectivity_key ASC"] if selected_obj.name == "A6_rrf"
                       else ["aggregate score DESC", "mean abs ppm over present spectra ASC", "connectivity_key ASC"])
    return cfg
