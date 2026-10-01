import pandas as pd

from casmi.ranking.fold_stability import decision_table, keep_reject


def _per_fold(deltas, base=0.30):
    rows = []
    for f, d in enumerate(deltas):
        rows += [{"variant": "mass_only", "fold": f, "mrr_at_25": base}, {"variant": "new", "fold": f, "mrr_at_25": base + d}]
    return pd.DataFrame(rows)


def test_keep_requires_fold_stability():
    assert keep_reject(_per_fold([0.02, 0.01, 0.03, 0.02, 0.01]), "new", "mass_only")[0] == "KEEP"
    assert keep_reject(_per_fold([0.02, 0.01, 0.03, 0.02, -0.01]), "new", "mass_only")[0] == "KEEP"         # all folds but one
    assert keep_reject(_per_fold([0.10, -0.01, -0.01, 0.01, -0.01]), "new", "mass_only")[0] == "NEEDS_MORE_EVIDENCE"
    assert keep_reject(_per_fold([-0.02, 0.01, -0.03, 0.0, -0.01]), "new", "mass_only")[0] == "REJECT"
    assert keep_reject(_per_fold([0.05, 0.05]), "new", "mass_only")[0] == "NEEDS_MORE_EVIDENCE"            # too few folds


def test_missing_variant_and_decision_table():
    d, reason, _ = keep_reject(_per_fold([0.1] * 5), "absent", "mass_only")
    assert d == "NEEDS_MORE_EVIDENCE" and "no per-fold rows" in reason
    t = decision_table(_per_fold([0.02] * 5), "mass_only").set_index("variant")
    assert t.loc["mass_only", "decision"] == "BASELINE" and t.loc["new", "decision"] == "KEEP" and t.loc["new", "n_positive"] == 5
