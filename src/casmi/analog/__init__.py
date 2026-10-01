"""Analog-propagation channel (v2 Phase 5).

    retrieval     spectrally similar LIBRARY spectra for a query (precursor shifts allowed): binned-cosine +
                  neutral-loss prefilter over the whole library (sparse matrix product; optional CUDA), then
                  the EXISTING modified cosine (`casmi.spectra.similarity`) on the shortlist.
    propagation   structural evidence propagated from analog molecules to mass-retrieved candidates
                  (Tanimoto to analogs, weighted by spectral similarity).
    features      the candidate-level `analog_*` feature table + its stable schema.

Spectrum preprocessing is NOT re-implemented: library peaks are the frozen bundle's similarity
representation (remove_invalid_peaks + deterministic top-100), and raw queries go through the same
`casmi.spectra.preprocessing` functions.
"""
