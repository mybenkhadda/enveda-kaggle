"""Ranking metrics for candidate-structure retrieval: Recall@k and MRR@k."""
import numpy as np


def recall_at_k(ranked_candidates, true_labels, k):
    """ranked_candidates: list of ranked candidate-key sequences (best first).
    true_labels: matching list of the correct key for each query.
    Returns the fraction of queries whose true label appears within the top k candidates."""
    hits = 0
    for cands, true in zip(ranked_candidates, true_labels):
        if true in cands[:k]:
            hits += 1
    return hits / len(true_labels) if true_labels else np.nan


def mrr_at_k(ranked_candidates, true_labels, k):
    """Mean reciprocal rank, considering only the top k candidates (0 if the true label is
    absent from the top k, matching the competition-style truncated MRR)."""
    total = 0.0
    for cands, true in zip(ranked_candidates, true_labels):
        rank = None
        for i, c in enumerate(cands[:k]):
            if c == true:
                rank = i + 1
                break
        total += (1.0 / rank) if rank else 0.0
    return total / len(true_labels) if true_labels else np.nan


def evaluate_retrieval(ranked_candidates, true_labels, ks=(1, 5, 10, 25)):
    out = {}
    for k in ks:
        out[f"recall_at_{k}"] = recall_at_k(ranked_candidates, true_labels, k)
    out[f"mrr_at_{max(ks)}"] = mrr_at_k(ranked_candidates, true_labels, max(ks))
    out["avg_n_candidates"] = float(np.mean([len(c) for c in ranked_candidates])) if ranked_candidates else np.nan
    return out
