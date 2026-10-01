"""Class-2 simulation foundation (split / manifest code only -- no fingerprint model, no training).

Class 2 = KNOWN STRUCTURE, NO DIRECT SPECTRUM: the truth structure is in the open candidate universe,
but no reference spectrum of it exists. A development query simulates this when:

    * its truth connectivity stays IN the candidate universe;
    * EVERY library spectrum of the truth connectivity (any source) is inaccessible as a reference;
    * nothing about the truth reveals that its references were removed.

The last point is the subtle one. In the unified universe a TRAIN structure carries `in_train=True`,
`has_reference_spectrum=True` and TRAIN source ids, while a real Class-2 truth is an external-only
structure. If the simulation merely hid the spectra, "in_train but no reference" would single out the
truth. Therefore:

    * eligible Class-2 queries are those whose truth ALSO occurs in a non-TRAIN source (so it can be
      presented as an ordinary external structure) -- `class2_eligible`;
    * the simulated candidate VIEW strips the TRAIN provenance of every hidden connectivity
      (`in_train=False`, `has_reference_spectrum=False`, TRAIN removed from sources / ids, the external
      representative SMILES) -- `class2_candidate_view`;
    * hiding is FOLD-scoped by default: every Class-2 truth of the evaluation fold is hidden at once, so
      one query's truth references cannot serve another query of the same fold;
    * reference-availability encodings stay forbidden as features
      (`casmi.candidates.provenance.FORBIDDEN_SHORTCUT_FEATURES`).

Class 3 (NOT implemented; kept separate): the truth structure itself is REMOVED from the candidate
universe (novel structure). It needs structure generation / biotransformation and is out of scope.
"""
import numpy as np
import pandas as pd

from casmi.candidates.provenance import FORBIDDEN_SHORTCUT_FEATURES

CLASS3_DEFINITION = ("Class 3: the truth connectivity is REMOVED from the candidate universe (novel structure); requires a generator "
                     "(e.g. biotransformations) -- not implemented, never mixed with Class-2 splits")
MANIFEST_COLUMNS = ["query_id", "truth_connectivity_key", "fold", "hidden_scope", "n_hidden_reference_spectra", "truth_open_sources"]


def class2_eligible(queries, unified, truth_col="connectivity_key"):
    """Development queries whose truth is in the unified universe AND in at least one non-TRAIN source."""
    u = unified.set_index("connectivity_key")
    open_src = u["candidate_sources"].map(lambda s: sorted(set(s) - {"TRAIN"}))
    q = queries.copy()
    q["_in_universe"] = q[truth_col].isin(u.index)
    q["truth_open_sources"] = q[truth_col].map(open_src)
    q["class2_eligible"] = q["_in_universe"] & q["truth_open_sources"].map(lambda s: isinstance(s, list) and len(s) > 0)
    return q.drop(columns="_in_universe")


def class2_manifest(eligible_queries, reference_connectivity, truth_col="connectivity_key", fold_col="fold", hidden_scope="fold"):
    """One row per Class-2 query. `reference_connectivity`: connectivity of every library reference
    (e.g. `train_spectrum_metadata.connectivity_key`), used to COUNT hidden spectra."""
    if hidden_scope not in ("fold", "query"):
        raise ValueError("hidden_scope must be 'fold' or 'query'")
    q = eligible_queries[eligible_queries["class2_eligible"]].copy()
    counts = pd.Series(reference_connectivity).astype(str).value_counts()
    return pd.DataFrame({"query_id": q["query_id"].astype(str).to_numpy(), "truth_connectivity_key": q[truth_col].astype(str).to_numpy(),
                         "fold": q[fold_col].to_numpy() if fold_col in q else 0, "hidden_scope": hidden_scope,
                         "n_hidden_reference_spectra": q[truth_col].astype(str).map(counts).fillna(0).astype(int).to_numpy(),
                         "truth_open_sources": q["truth_open_sources"].to_numpy()})[MANIFEST_COLUMNS]


def hidden_connectivities(manifest, query_id=None, fold=None):
    """Connectivities whose references are hidden for one query (query scope) or one fold (fold scope)."""
    m = manifest
    if (m["hidden_scope"] == "fold").all():
        if fold is None and query_id is not None:
            fold = m.loc[m["query_id"] == str(query_id), "fold"].iloc[0]
        return set(m.loc[m["fold"] == fold, "truth_connectivity_key"]) if fold is not None else set(m["truth_connectivity_key"])
    return set(m.loc[m["query_id"] == str(query_id), "truth_connectivity_key"])


def class2_reference_mask(reference_connectivity, hidden):
    """Boolean ALLOWED mask over the reference library: every spectrum of a hidden connectivity is removed."""
    return ~pd.Series(reference_connectivity).astype(str).isin(set(hidden)).to_numpy()


def class2_candidate_view(unified, hidden):
    """The candidate universe as a Class-2 query sees it: hidden truths STAY, stripped of TRAIN provenance."""
    v = unified.copy()
    h = v["connectivity_key"].isin(set(hidden))
    v.loc[h, "in_train"] = False
    v.loc[h, "has_reference_spectrum"] = False
    # list-valued columns are rebuilt whole (no .loc assignment of list objects)
    v["candidate_sources"] = [sorted(set(s) - {"TRAIN"}) if hh else s for s, hh in zip(v["candidate_sources"], h)]
    v["source_ids"] = [sorted(i for i in s if not str(i).startswith("TRAIN:")) if hh else s for s, hh in zip(v["source_ids"], h)]
    if "open_representative_smiles" in v:
        v.loc[h, "representative_smiles"] = v.loc[h, "open_representative_smiles"]
    return v


def assert_class2_split(manifest, view, reference_connectivity, allowed_mask, feature_names=()):
    """(1) every truth is in the candidate view; (2) no allowed reference belongs to a hidden truth;
    (3) hidden truths carry no TRAIN / reference-availability trace; (4) no shortcut feature."""
    hidden = set(manifest["truth_connectivity_key"])
    keys = set(view["connectivity_key"])
    missing = hidden - keys
    assert not missing, f"{len(missing)} Class-2 truths missing from the candidate universe"
    rc = pd.Series(reference_connectivity).astype(str).to_numpy()
    leaked = set(rc[np.asarray(allowed_mask, bool)]) & hidden
    assert not leaked, f"{len(leaked)} hidden truths still have accessible reference spectra"
    hv = view[view["connectivity_key"].isin(hidden)]
    assert not hv["in_train"].any() and not hv["has_reference_spectrum"].any(), "hidden truths still flagged in_train / has_reference_spectrum"
    assert not hv["candidate_sources"].map(lambda s: "TRAIN" in s).any(), "hidden truths still list TRAIN as a source"
    bad = [f for f in feature_names if f in FORBIDDEN_SHORTCUT_FEATURES]
    assert not bad, f"reference-availability shortcut features present: {bad}"
    return True
