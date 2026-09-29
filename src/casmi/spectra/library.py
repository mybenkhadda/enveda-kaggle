"""Reference-spectrum pool construction for the two ranking-evaluation regimes:

    KNOWN-SPECTRUM       -- same-connectivity reference spectra may exist; the query spectrum
                             itself is always excluded (never compared against itself).
    UNSEEN-CONNECTIVITY  -- reference spectra are restricted to OTHER CV folds; the query's own
                             true connectivity therefore has ZERO same-connectivity reference
                             spectra, by construction (every spectrum of a connectivity_key
                             lives in one fold, per `casmi.validation.folds`). Intentional, not
                             a bug -- see `assert_no_connectivity_leakage`.
"""


def build_reference_index(train_metadata, connectivity_col="connectivity_key", id_col="train_spectrum_id"):
    """`{connectivity_key: [spectrum_id, ...]}` -- the raw, unrestricted index every regime
    filters down from."""
    return train_metadata.groupby(connectivity_col)[id_col].apply(list).to_dict()


def known_spectrum_reference_ids(reference_index, connectivity_key, exclude_spectrum_id):
    """Reference spectrum ids for `connectivity_key`, excluding `exclude_spectrum_id` (the
    query's own row) -- the only leakage guard this regime needs."""
    return [i for i in reference_index.get(connectivity_key, []) if i != exclude_spectrum_id]


def unseen_connectivity_reference_ids(reference_index, connectivity_key, query_fold, connectivity_to_fold):
    """Reference spectrum ids for `connectivity_key` whose fold differs from `query_fold`.
    Empty whenever `connectivity_key` IS in `query_fold` -- always true for a query's own
    correct candidate under connectivity-grouped CV."""
    if connectivity_to_fold.get(connectivity_key) == query_fold:
        return []
    return reference_index.get(connectivity_key, [])


def assert_no_connectivity_leakage(validation_connectivities, reference_connectivities):
    """Raises `AssertionError` (never continues silently) if any connectivity appears in both
    the validation set and the reference/search-library set under the unseen-connectivity
    regime."""
    overlap = set(validation_connectivities) & set(reference_connectivities)
    assert not overlap, f"{len(overlap)} connectivities leak between validation and reference sets: {sorted(overlap)[:5]}"
