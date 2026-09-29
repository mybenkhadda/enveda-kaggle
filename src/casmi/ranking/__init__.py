"""Candidate RANKING using spectral evidence -- the question changes from notebook 03's "is the
correct molecule in the candidate pool?" to "can MS/MS spectral evidence rank it near the top?"

    casmi.ranking.aggregation   generic multi-value reducers (max/mean/median/top-k mean),
                                shared by reference-spectrum and multi-query-spectrum aggregation
    casmi.ranking.features      pair-level (query x candidate) feature table construction
    casmi.ranking.evaluation    rank-from-score, conditional vs. end-to-end MRR@25/Hit@K

Classical baselines only (mass-only, binned cosine, modified cosine, peak overlap, neutral
loss) -- learning-to-rank is notebook 05's job, not implemented here.
"""
