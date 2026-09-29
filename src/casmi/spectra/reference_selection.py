"""Deterministic reference-compatibility ordering and the EXACT lazy reference walk.

Rule this module exists to enforce: no production code path here may truncate a candidate's
reference list before that reference's eligibility (for every protocol) is known. A prior
version's fixed-size pre-eligibility pool (an earlier "walk cap" / "reference-index cap"
constant) pre-truncated the list BEFORE checking eligibility --
functionally correct only by accident (verified after the fact to never have mattered in this
particular dataset), not by construction. This module makes the walk provably capable of
reaching the end of an arbitrarily long reference list; a reference is skipped only because every
protocol that could still use it already has its 5 accepted references, never because of a fixed
position cutoff.
"""
from dataclasses import dataclass, field

PROTOCOLS = ("standard", "cross_library", "mirror_aware", "near_dup_strict")
MAX_REFS_PER_PROTOCOL = 5
MIRROR_TIERS = ("T1", "T2")


def compat_sort_key(ref_meta, query_meta):
    """`(same_adduct desc, same_ion_mode desc, ce_comparable desc, |ce_diff| asc [incomparable
    sorts last], same_instrument desc, reference_spectrum_id asc)`. CE is "comparable" only when
    both sides have a known, EQUAL unit -- two collision energies in different (or unknown)
    units are never subtracted, and rank behind any comparable pair. `ref_meta`/`query_meta`:
    dict-likes with `adduct`, `ion_mode`, `ce`, `ce_unit`, `instrument`
    (`ref_meta` additionally needs `spectrum_id`, the final deterministic tie-break)."""
    same_adduct = ref_meta.get("adduct") == query_meta.get("adduct")
    same_ion = ref_meta.get("ion_mode") == query_meta.get("ion_mode")
    ce_comparable = (
        ref_meta.get("ce_unit") is not None and query_meta.get("ce_unit") is not None
        and ref_meta["ce_unit"] == query_meta["ce_unit"]
        and ref_meta.get("ce") is not None and query_meta.get("ce") is not None
    )
    ce_diff = abs(ref_meta["ce"] - query_meta["ce"]) if ce_comparable else float("inf")
    same_instrument = ref_meta.get("instrument") == query_meta.get("instrument")
    return (
        0 if same_adduct else 1,
        0 if same_ion else 1,
        0 if ce_comparable else 1,
        ce_diff,
        0 if same_instrument else 1,
        ref_meta["spectrum_id"],
    )


def is_eligible(protocol, ref_meta, query_meta, tier):
    """Pure boolean filter, no similarity computation -- `tier` (one of
    `casmi.spectra.deduplication.TIERS`) must already be known."""
    same_source = ref_meta.get("source") == query_meta.get("source")
    if protocol == "standard":
        return True
    if protocol == "cross_library":
        return not same_source
    if protocol == "mirror_aware":
        return not same_source and tier not in MIRROR_TIERS
    if protocol == "near_dup_strict":
        return not same_source and tier not in MIRROR_TIERS and tier != "T3"
    raise ValueError(f"unknown protocol: {protocol!r}")


@dataclass
class WalkResult:
    accepted: dict = field(default_factory=dict)      # protocol -> [reference_spectrum_id, ...] (compat_rank order, len<=max_refs)
    walk_depth: dict = field(default_factory=dict)     # protocol -> compat_rank (1-based) of the 5th accepted ref, or None
    reached5: dict = field(default_factory=dict)       # protocol -> bool
    n_references_total: int = 0
    n_references_walked: int = 0
    exhausted: bool = False
    rows: list = field(default_factory=list)           # one dict per WALKED reference (for the QCR table)

    def has_at_least(self, protocol, n):
        """Exact regardless of early stopping: whenever the walk stops, EVERY protocol has
        either reached `MAX_REFS_PER_PROTOCOL` accepted references (so `>=n` for any `n<=5` is
        trivially true) OR the full list was exhausted (so the accepted count for that protocol
        IS its true total, not a lower bound) -- see the module docstring's stopping-rule
        argument. Never call this on a protocol that neither reached 5 nor saw `exhausted=True`;
        that state cannot occur under `walk_references`'s stopping rule."""
        return len(self.accepted.get(protocol, ())) >= n


def walk_references(ranked_reference_ids, query_meta, ref_meta_lookup, classify_fn,
                     protocols=PROTOCOLS, max_refs=MAX_REFS_PER_PROTOCOL):
    """`ranked_reference_ids`: the FULL candidate reference list (self-excluded), already sorted
    by `compat_sort_key` for THIS query -- no pre-truncation. `ref_meta_lookup[rid]` gives that
    reference's metadata dict. `classify_fn(rid) -> (tier, cosine, modified_cosine,
    peak_overlap_frac, neutral_loss_cosine)` is called AT MOST ONCE per walked reference here --
    give it a memoizing/caching wrapper (`casmi.spectra.similarity_cache.SimilarityCache`) if it
    isn't already one, so a reference walked for one query doesn't get re-classified from
    scratch if walked again elsewhere.

    Stops as soon as EVERY protocol in `protocols` has accepted `max_refs` references, or the
    reference list is exhausted -- never a fixed position cutoff. Because of this exact stopping
    rule, whenever the walk ends, every protocol has either reached `max_refs` or been fully
    exhausted (see `WalkResult.has_at_least`)."""
    accepted = {p: [] for p in protocols}
    walk_depth = {p: None for p in protocols}
    rows = []
    n_walked = 0

    for rank, rid in enumerate(ranked_reference_ids, start=1):
        if all(len(accepted[p]) >= max_refs for p in protocols):
            break
        n_walked += 1
        ref_meta = ref_meta_lookup[rid]
        tier, cosine, modified_cosine, peak_overlap_frac, neutral_loss_cosine = classify_fn(rid)
        row = {
            "reference_spectrum_id": rid, "compat_rank": rank, "tier": tier,
            "cosine": cosine, "modified_cosine": modified_cosine,
            "peak_overlap_frac": peak_overlap_frac, "neutral_loss_cosine": neutral_loss_cosine,
        }
        for p in protocols:
            eligible = is_eligible(p, ref_meta, query_meta, tier)
            accept = eligible and len(accepted[p]) < max_refs
            row[f"eligible_{p}"] = eligible
            row[f"accepted_{p}"] = accept
            if accept:
                accepted[p].append(rid)
                if len(accepted[p]) == max_refs and walk_depth[p] is None:
                    walk_depth[p] = rank
        rows.append(row)

    exhausted = n_walked == len(ranked_reference_ids)
    reached5 = {p: len(accepted[p]) >= max_refs for p in protocols}
    return WalkResult(accepted=accepted, walk_depth=walk_depth, reached5=reached5,
                       n_references_total=len(ranked_reference_ids), n_references_walked=n_walked,
                       exhausted=exhausted, rows=rows)
