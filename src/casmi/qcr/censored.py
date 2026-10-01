"""Censored eligible-reference-count representation.

The lazy walk (`casmi.spectra.reference_selection.walk_references`) stops each protocol once it
has accepted `max_refs` (5) references. Whenever a protocol's accepted count reaches that cap,
its TRUE total eligible-reference count is only known to be `>= max_refs` (a lower bound) UNLESS
the walk went on to exhaust the entire reference list anyway (for some OTHER protocol's sake) --
in which case every reference was in fact checked for this protocol's eligibility too, and the
true total is recoverable by summing `eligible_<protocol>` across every walked row, even past
the point this protocol itself stopped accepting. Treating a capped-but-unexhausted count as if
it were exact silently biases any reference-richness analysis (C6/C8) toward under-counting
well-referenced candidates -- this module makes the distinction explicit and impossible to
accidentally erase.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CensoredCount:
    lower_bound: int             # accepted count under the walk's stopping rule (always known)
    is_exact: bool                # True iff the true total eligible count is actually known
    exact_value: Optional[int]    # the true total, only when is_exact (may exceed lower_bound)

    def __repr__(self):
        return f"{self.exact_value}" if self.is_exact else f">={self.lower_bound}"


def censored_count_from_walk_result(wr, protocol, max_refs=5):
    """Build a `CensoredCount` for one `(WalkResult, protocol)` pair.

    Case 1 -- `accepted < max_refs`: this can only happen if the walk exhausted the full
    reference list (the walk's overall stop condition requires EVERY protocol to reach
    `max_refs`; if this protocol didn't, the walk kept going until the list ran out), and since
    this protocol was never capped, `accepted` IS its true total. Exact.

    Case 2 -- `accepted == max_refs` and the walk exhausted the full list (for some other
    protocol's sake): every reference was still checked for THIS protocol's eligibility even
    after it hit its cap (`row["eligible_<protocol>"]` is recorded regardless of `accepted_p`),
    so the true total is recoverable by summing that flag across every walked row. Exact
    (possibly `> max_refs`).

    Case 3 -- `accepted == max_refs` and the walk did NOT exhaust the list: the walk may have
    stopped (for this protocol) exactly because it reached quota, with the rest of the
    reference list never examined. Genuinely censored: `>= max_refs`."""
    n_accepted = len(wr.accepted[protocol])
    if n_accepted < max_refs:
        return CensoredCount(lower_bound=n_accepted, is_exact=True, exact_value=n_accepted)
    if wr.exhausted:
        n_eligible_total = sum(1 for r in wr.rows if r.get(f"eligible_{protocol}", False))
        return CensoredCount(lower_bound=n_accepted, is_exact=True, exact_value=n_eligible_total)
    return CensoredCount(lower_bound=n_accepted, is_exact=False, exact_value=None)


def censored_count_summary(counts):
    """`counts`: iterable of `CensoredCount`. Returns `{"n", "share_exact_le5",
    "share_exact_gt5", "share_censored_ge5"}` -- three mutually exclusive, exhaustive buckets;
    NEVER conflates a censored `>=5` with an exact `<=5` count."""
    counts = list(counts)
    n = len(counts)
    if n == 0:
        return {"n": 0, "share_exact_le5": float("nan"), "share_exact_gt5": float("nan"), "share_censored_ge5": float("nan")}
    n_exact_le5 = sum(1 for c in counts if c.is_exact and c.exact_value <= 5)
    n_exact_gt5 = sum(1 for c in counts if c.is_exact and c.exact_value > 5)
    n_censored_ge5 = sum(1 for c in counts if not c.is_exact)
    return {"n": n, "share_exact_le5": n_exact_le5 / n, "share_exact_gt5": n_exact_gt5 / n, "share_censored_ge5": n_censored_ge5 / n}
