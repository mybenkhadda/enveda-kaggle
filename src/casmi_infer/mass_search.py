"""Candidate retrieval over the exported structure library -- the same arithmetic as training's
`MassIndex._window` + `CandidateGenerator.generate_dataframe` + `dedupe_pool_to_connectivity`:

    tol_da = |neutral_mass| * ppm * 1e-6
    window = searchsorted(masses, nm - tol_da, 'left') : searchsorted(masses, nm + tol_da, 'right')
    mass_error_ppm = 1e6 * (candidate_mass - nm) / nm        abs_mass_error_ppm = |.|
    rows are MASS VARIANTS; dedupe to one row per connectivity keeping the smallest abs ppm.

Fallback sequence (only when the previous level returned zero connectivities): the primary
per-adduct window (training's contract tolerance), then each configured wider ppm window, then the
nearest-N connectivities by |mass difference| (ties: connectivity_key ASC). The level used is
returned so every fallback is logged.
"""
import numpy as np


class MassSearch:
    def __init__(self, masses, conn_idx, conn_keys):
        """`masses`: float64 sorted ASC (bundle/structures_mass.npy); `conn_idx`: connectivity index
        per mass row; `conn_keys`: connectivity_key per connectivity index (for deterministic ties)."""
        self.masses = np.asarray(masses, dtype=np.float64)
        if np.any(np.diff(self.masses) < 0):
            raise ValueError("structure masses must be sorted ascending")
        self.conn_idx = np.asarray(conn_idx, dtype=np.int64)
        self.conn_keys = np.asarray(conn_keys)

    def window(self, neutral_mass, ppm):
        tol = abs(neutral_mass) * ppm * 1e-6
        lo = int(np.searchsorted(self.masses, neutral_mass - tol, side="left"))
        hi = int(np.searchsorted(self.masses, neutral_mass + tol, side="right"))
        return lo, hi

    def _dedupe(self, rows, neutral_mass):
        """(conn_idx, abs_ppm) one per connectivity, min abs ppm, ordered by abs ppm then key."""
        if len(rows) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        cm = self.masses[rows]
        abs_ppm = np.abs(1e6 * (cm - neutral_mass) / neutral_mass)
        ci = self.conn_idx[rows]
        order = np.lexsort((ci, abs_ppm))
        ci, abs_ppm = ci[order], abs_ppm[order]
        _, first = np.unique(ci, return_index=True)
        ci, abs_ppm = ci[first], abs_ppm[first]
        # conn_idx is the position in the connectivity_key-SORTED table, so ordering by conn_idx IS
        # ordering by connectivity_key (asserted at bundle load by `validation.check_bundle_invariants`)
        o = np.lexsort((ci, abs_ppm))
        return ci[o], abs_ppm[o]

    def search(self, neutral_mass, primary_ppm, fallback_ppm=(100.0, 200.0), nearest_n=25):
        """Returns `(conn_idx, abs_ppm, level)`; level in {"primary", "ppm_<x>", "nearest_<n>", "none"}."""
        if neutral_mass is None or not np.isfinite(neutral_mass) or neutral_mass <= 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0), "none"
        for level, ppm in [("primary", primary_ppm)] + [(f"ppm_{p:g}", p) for p in fallback_ppm]:
            lo, hi = self.window(neutral_mass, ppm)
            ci, ap = self._dedupe(np.arange(lo, hi), neutral_mass)
            if len(ci):
                return ci, ap, level
        return (*self.nearest(neutral_mass, nearest_n), f"nearest_{nearest_n}")

    def nearest(self, neutral_mass, n):
        """n closest CONNECTIVITIES by |mass difference| (a symmetric expanding window around the
        insertion point, wide enough to guarantee n distinct connectivities when they exist)."""
        pos = int(np.searchsorted(self.masses, neutral_mass))
        span = max(4 * n, 64)
        while True:
            lo, hi = max(0, pos - span), min(len(self.masses), pos + span)
            ci, ap = self._dedupe(np.arange(lo, hi), neutral_mass)
            if len(ci) >= n or (lo == 0 and hi == len(self.masses)):
                return ci[:n], ap[:n]
            span *= 2
