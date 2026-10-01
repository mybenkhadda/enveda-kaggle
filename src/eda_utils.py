"""Shared, reusable utilities for the CASMI 2026 EDA/validation notebook (01) and any notebook
that comes after it: peak parsing, adduct mass arithmetic, the competition connectivity key
(tautomer-canonical InChIKey14), the MRR@25 metric, and submission-format validation.

Kept separate from the rest of `src/` because it encodes the *competition's own* definitions
(the tautomer-canonical key, MRR@25) rather than this project's baseline-specific choices
(e.g. `src/chemistry.py`'s `connectivity_key`, which is stereo-stripped-SMILES-based, not
InChIKey-based, and is a different, earlier convention used by the baseline notebooks).
"""
import re

import numpy as np

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- Peak parsing -------------------------------------------------------------------------


def parse_peaks(mz_array, intensity_array):
    """Normalize one spectrum's peaks to (mz, intensity) float64 numpy arrays.

    Handles the parallel-list-column storage used by this dataset (`ms2_mzs` /
    `ms2_normalized_intensities`, or generic-arrow lists). Returns two empty arrays for a
    missing/empty spectrum rather than raising, since a handful of empty spectra are expected.
    """
    if mz_array is None or intensity_array is None:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    mz = np.asarray(mz_array, dtype=np.float64)
    inten = np.asarray(intensity_array, dtype=np.float64)
    assert mz.shape == inten.shape, f"mz/intensity length mismatch: {mz.shape} vs {inten.shape}"
    return mz, inten


def spectral_entropy(intensity_array):
    """Shannon entropy of an intensity array, normalized to a probability distribution.

    Returns 0.0 for an empty or all-zero spectrum (a single peak or no signal carries no
    information), never NaN, so it can be aggregated directly.
    """
    inten = np.asarray(intensity_array, dtype=np.float64)
    inten = inten[inten > 0]
    if inten.size == 0:
        return 0.0
    p = inten / inten.sum()
    return float(-(p * np.log(p)).sum())


def _median_mz_decimal_places(mz, max_places=6):
    """Median number of meaningful (non-trailing-zero) decimal digits across an m/z array --
    a cheap proxy for acquisition resolution/rounding convention (e.g. 4 decimals for a
    high-res Orbitrap/timsTOF export vs 1 decimal for a low-res or heavily-rounded source)."""
    def decimals(x):
        s = f"{x:.{max_places}f}".rstrip("0")
        return len(s.split(".")[1]) if "." in s else 0

    return float(np.median([decimals(x) for x in mz]))


def compute_peak_stats(mz, intensity, precursor_mz, above_precursor_da=1.0, precursor_ppm_tol=10.0):
    """Per-spectrum peak-level quality/shape statistics (Section 6 of the EDA notebook).

    `mz`/`intensity` are the arrays from `parse_peaks`; `precursor_mz` may be None/NaN (stats
    that need it are then NaN/False rather than raising).
    """
    n_peaks = len(mz)
    stats = {
        "n_peaks": n_peaks,
        "sorted_ok": bool(np.all(np.diff(mz) >= 0)) if n_peaks > 1 else True,
        "n_duplicate_mz": int(n_peaks - len(np.unique(mz))) if n_peaks else 0,
        "min_mz": float(mz.min()) if n_peaks else np.nan,
        "max_mz": float(mz.max()) if n_peaks else np.nan,
        "max_intensity": float(intensity.max()) if n_peaks else np.nan,
        "entropy": spectral_entropy(intensity),
        "mz_decimal_places_median": _median_mz_decimal_places(mz) if n_peaks else np.nan,
    }

    if n_peaks == 0:
        stats["intensity_scale"] = "empty"
    else:
        m = stats["max_intensity"]
        if np.isclose(m, 1.0, atol=1e-6):
            stats["intensity_scale"] = "normalized_1"
        elif np.isclose(m, 100.0, atol=1e-3):
            stats["intensity_scale"] = "normalized_100"
        elif np.isclose(m, 999.0, atol=1.0) or np.isclose(m, 1000.0, atol=1.0):
            stats["intensity_scale"] = "normalized_999_or_1000"
        else:
            stats["intensity_scale"] = "raw_or_other"

    if n_peaks and precursor_mz is not None and not (isinstance(precursor_mz, float) and np.isnan(precursor_mz)):
        above = mz > (precursor_mz + above_precursor_da)
        stats["frac_peaks_above_precursor"] = float(above.mean())
        total_inten = float(intensity.sum())
        stats["frac_intensity_above_precursor"] = float(intensity[above].sum() / total_inten) if total_inten > 0 else 0.0
        ppm_diff = np.abs(mz - precursor_mz) / precursor_mz * 1e6
        stats["precursor_peak_present"] = bool((ppm_diff <= precursor_ppm_tol).any())
        base_idx = int(np.argmax(intensity))
        stats["base_peak_mz_ratio"] = float(mz[base_idx] / precursor_mz)
    else:
        stats["frac_peaks_above_precursor"] = np.nan
        stats["frac_intensity_above_precursor"] = np.nan
        stats["precursor_peak_present"] = False
        stats["base_peak_mz_ratio"] = np.nan

    return stats


# --- Adduct mass arithmetic ----------------------------------------------------------------

# Monoisotopic atomic masses (Da) for the elements that appear in this dataset's adduct
# fragments (verified against real `adduct` values in data/train.parquet).
ATOMIC_MASSES = {
    "H": 1.0078250319, "D": 2.0141017780, "Li": 7.0160034, "C": 12.0,
    "N": 14.0030740052, "O": 15.9949146221, "F": 18.9984032, "Na": 22.98976928,
    "Mg": 23.9850417, "P": 30.97376151, "S": 31.97207069, "Cl": 34.96885268,
    "K": 38.9637069, "Ca": 39.9625912, "Fe": 55.9349393, "Br": 78.9183371,
    "I": 126.904473,
}
ELECTRON_MASS = 0.00054858

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_ADDUCT_RE = re.compile(r"^\[(\d*)M((?:[+-][A-Za-z0-9]+)*)\](\d*)([+-])$")
_ADDUCT_TERM_RE = re.compile(r"([+-])(\d*)([A-Za-z]+\d*(?:[A-Za-z]+\d*)*)")


def formula_mass(formula):
    """Monoisotopic mass of a simple formula fragment (e.g. 'CH2O2' -> 46.0055 Da).

    Returns None if any element token is unrecognized, so callers can distinguish "zero mass"
    from "couldn't parse" rather than silently mis-scoring an adduct.
    """
    if not formula:
        return 0.0
    mass = 0.0
    consumed = 0
    for sym, count in _FORMULA_TOKEN_RE.findall(formula):
        if not sym:
            continue
        if sym not in ATOMIC_MASSES:
            return None
        n = int(count) if count else 1
        mass += ATOMIC_MASSES[sym] * n
        consumed += len(sym) + len(count)
    if consumed != len(formula):
        return None
    return mass


def parse_adduct(adduct_str):
    """Parse an adduct string, e.g. '[2M+Na]+' or '[M+CH2O2-H]-', into the three quantities
    needed to relate observed precursor m/z to neutral monoisotopic mass:

        ion_mz * |charge| == multiplier * neutral_mass + mass_shift_da

    Returns (multiplier, mass_shift_da, charge) where `charge` is *signed* (positive for a
    cation, negative for an anion) and `mass_shift_da` already folds in the electron-mass
    correction for ion formation. Returns None if the string doesn't match the recognized
    `[nM+-frag...]z+/-` shape, or references an element this module doesn't know the mass of
    (callers should log these rather than guess).
    """
    if not adduct_str or not isinstance(adduct_str, str):
        return None
    m = _ADDUCT_RE.match(adduct_str.strip())
    if not m:
        return None
    n_str, ops_str, z_str, polarity = m.groups()
    multiplier = int(n_str) if n_str else 1
    z = int(z_str) if z_str else 1

    delta = 0.0
    pos = 0
    for tm in _ADDUCT_TERM_RE.finditer(ops_str):
        if tm.start() != pos:
            return None
        pos = tm.end()
        sign, count_str, frag = tm.groups()
        fm = formula_mass(frag)
        if fm is None:
            return None
        count = int(count_str) if count_str else 1
        delta += (1 if sign == "+" else -1) * count * fm
    if pos != len(ops_str):
        return None

    # Forming a +z ion removes z electrons from the neutral assembly; a -z ion adds z.
    electron_term = -z * ELECTRON_MASS if polarity == "+" else z * ELECTRON_MASS
    mass_shift_da = delta + electron_term
    charge = z if polarity == "+" else -z
    return multiplier, mass_shift_da, charge


def expected_precursor_mz(neutral_mass, adduct_str):
    """Expected precursor m/z for a neutral monoisotopic mass under a given adduct, or None
    if the adduct doesn't parse."""
    parsed = parse_adduct(adduct_str)
    if parsed is None or neutral_mass is None:
        return None
    multiplier, shift, charge = parsed
    return (multiplier * neutral_mass + shift) / abs(charge)


def neutral_mass_from_adduct(precursor_mz, adduct_str):
    """Inverse of `expected_precursor_mz`: neutral monoisotopic mass implied by an observed
    precursor m/z and adduct, or None if the adduct doesn't parse."""
    parsed = parse_adduct(adduct_str)
    if parsed is None or precursor_mz is None:
        return None
    multiplier, shift, charge = parsed
    return (precursor_mz * abs(charge) - shift) / multiplier


# --- Competition connectivity key (tautomer-canonical InChIKey14) --------------------------


def build_tautomer_enumerator(max_tautomers=100, max_transforms=100):
    """A `TautomerEnumerator` with bounded search effort.

    We cap the tautomer/transform search space instead of using a wall-clock timeout: a
    per-task signal-based timeout isn't reliably available across a Windows multiprocessing
    pool, whereas this cap bounds worst-case work per molecule directly inside RDKit and is
    portable. Molecules that would have needed more are logged by `conn_key` as `hit_cap=True`
    and fall back to the plain (non-tautomer-canonical) key.
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is required for build_tautomer_enumerator")
    params = rdMolStandardize.CleanupParameters()
    params.maxTautomers = max_tautomers
    params.maxTransforms = max_transforms
    return rdMolStandardize.TautomerEnumerator(params)


def plain_inchikey14(mol):
    """First 14 chars of the standard InChIKey (connectivity block) for an as-is RDKit mol."""
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not ik:
        return None
    return ik[:14]


def conn_key(smiles, enumerator=None):
    """The competition's connectivity key: RDKit tautomer-canonicalize, then take the first
    14 characters (the connectivity block, stereo-ignored) of the resulting InChIKey.

    Returns a dict: {conn_key, plain_key, hit_cap, parse_ok, error}. `hit_cap=True` means the
    tautomer search likely didn't explore the full space for this molecule (its result is the
    best canonical tautomer found within the cap, not a verified global optimum) -- these are
    rare and should be logged, not silently trusted as exact.
    """
    out = {"conn_key": None, "plain_key": None, "hit_cap": False, "parse_ok": False, "error": None}
    if not RDKIT_AVAILABLE:
        out["error"] = "rdkit_not_available"
        return out
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        out["error"] = "parse_failed"
        return out
    out["parse_ok"] = True
    out["plain_key"] = plain_inchikey14(mol)
    if enumerator is None:
        enumerator = build_tautomer_enumerator()
    try:
        n_tautomers = len(enumerator.Enumerate(mol))
        cap_hit = n_tautomers >= enumerator.GetMaxTautomers()
        canon = enumerator.Canonicalize(mol)
        key = plain_inchikey14(canon)
        out["conn_key"] = key if key is not None else out["plain_key"]
        out["hit_cap"] = bool(cap_hit)
    except Exception as exc:
        out["error"] = f"tautomer_canonicalize_failed: {exc}"
        out["conn_key"] = out["plain_key"]
    return out


def _has_stereo(mol):
    """True if `mol` carries any *assigned or possible* stereocenter or double-bond
    stereo. Deliberately not `isomericSmiles=True vs False` string comparison -- that flag
    also toggles isotope display, which would misclassify isotope-labeled-but-achiral
    molecules (e.g. `[13CH4]`) as stereo."""
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    if Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False):
        return True
    return any(b.GetStereo() != Chem.BondStereo.STEREONONE for b in mol.GetBonds())


# --- Full per-structure descriptor record (Section 3 of the EDA notebook) -----------------

STRUCTURE_RECORD_FIELDS = [
    "smiles", "parse_ok", "error", "canonical_smiles", "n_components", "formal_charge",
    "is_charged", "has_stereo", "has_isotopes", "molecular_formula", "exact_mass",
    "heavy_atom_count", "element_set", "num_rings", "num_aromatic_rings", "inchikey14_plain",
    "conn_key", "conn_key_hit_cap", "murcko_scaffold", "generic_scaffold",
]


def compute_structure_record(smiles, enumerator=None):
    """One row of Section-3 descriptors for a single SMILES string. Never raises: a parse
    failure (or any unexpected RDKit error) produces a record with `parse_ok=False` and the
    exception message in `error`, with every other field `None`, so a `Parallel(...)` sweep
    over ~277k unique SMILES can't be taken down by one bad string."""
    rec = {k: None for k in STRUCTURE_RECORD_FIELDS}
    rec["smiles"] = smiles
    rec["parse_ok"] = False
    try:
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            rec["error"] = "parse_failed"
            return rec
        rec["parse_ok"] = True

        rec["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)
        rec["n_components"] = len(Chem.GetMolFrags(mol))
        formal_charge = Chem.GetFormalCharge(mol)
        rec["formal_charge"] = formal_charge
        rec["is_charged"] = formal_charge != 0
        rec["has_stereo"] = _has_stereo(mol)
        rec["has_isotopes"] = any(a.GetIsotope() != 0 for a in mol.GetAtoms())
        rec["molecular_formula"] = rdMolDescriptors.CalcMolFormula(mol)
        rec["exact_mass"] = rdMolDescriptors.CalcExactMolWt(mol)
        rec["heavy_atom_count"] = mol.GetNumHeavyAtoms()
        rec["element_set"] = ",".join(sorted({a.GetSymbol() for a in mol.GetAtoms()}))
        ri = mol.GetRingInfo()
        rec["num_rings"] = ri.NumRings()
        rec["num_aromatic_rings"] = rdMolDescriptors.CalcNumAromaticRings(mol)

        rec["inchikey14_plain"] = plain_inchikey14(mol)
        ck = conn_key(smiles, enumerator)
        rec["conn_key"] = ck["conn_key"]
        rec["conn_key_hit_cap"] = ck["hit_cap"]
        if ck["error"]:
            rec["error"] = ck["error"]

        try:
            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            rec["murcko_scaffold"] = Chem.MolToSmiles(scaffold_mol, canonical=True)
            generic = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
            rec["generic_scaffold"] = Chem.MolToSmiles(generic, canonical=True)
        except Exception as exc:
            rec["error"] = (rec["error"] + "; " if rec["error"] else "") + f"scaffold_failed: {exc}"
    except Exception as exc:
        rec["error"] = f"unexpected_error: {exc}"
    return rec


# --- Metric: MRR@25 -------------------------------------------------------------------------


def mrr_at_25(pred_lists, true_keys, k=25):
    """Mean reciprocal rank @k, matching the CASMI 2026 scoring rule.

    `pred_lists`: list of ranked prediction sequences (each entry is a connectivity key, i.e.
    already mapped from SMILES via `conn_key` -- this function does not call RDKit).
    `true_keys`: the correct connectivity key for each query, same length/order as pred_lists.

    Within each prediction list, duplicate keys are deduplicated *keeping the first (best-
    ranked) occurrence* before truncating to `k` and taking the reciprocal rank of the true
    key (0.0 if absent). This mirrors "up to 25 ranked SMILES" scoring: repeating a guess does
    not let it occupy multiple rank slots.
    """
    assert len(pred_lists) == len(true_keys), "pred_lists and true_keys must be the same length"
    if not pred_lists:
        return float("nan")
    total = 0.0
    for preds, true_key in zip(pred_lists, true_keys):
        seen = set()
        deduped = []
        for p in preds:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
            if len(deduped) >= k:
                break
        try:
            rank = deduped.index(true_key) + 1
            total += 1.0 / rank
        except ValueError:
            pass
    return total / len(true_keys)


# --- Submission validation ------------------------------------------------------------------


def validate_submission(df, molecule_id_col="molecule_id", smiles_col="smiles", max_preds=25, sep=";"):
    """Validate a submission dataframe against the competition's format rules.

    Checks: exactly one row per `molecule_id_col` value, at most `max_preds` semicolon-
    separated SMILES per row, and every SMILES parses with RDKit. Returns a dict summary plus
    a `per_row_issues` list of (molecule_id, issue) tuples (empty if the submission is clean);
    never raises, so it can be used as a pass/fail report rather than a hard assertion.
    """
    issues = []
    n_rows = len(df)
    n_unique_ids = df[molecule_id_col].nunique()
    if n_unique_ids != n_rows:
        dupes = df[molecule_id_col][df[molecule_id_col].duplicated()].unique().tolist()
        issues.append((None, f"duplicate molecule_id rows: {dupes[:20]}"))

    max_preds_seen = 0
    n_unparseable = 0
    n_unparseable_rows = 0
    for mol_id, smiles_field in zip(df[molecule_id_col], df[smiles_col]):
        preds = [s for s in str(smiles_field).split(sep) if s]
        max_preds_seen = max(max_preds_seen, len(preds))
        if len(preds) > max_preds:
            issues.append((mol_id, f"{len(preds)} predictions > max {max_preds}"))
        row_bad = 0
        if RDKIT_AVAILABLE:
            for s in preds:
                if Chem.MolFromSmiles(s) is None:
                    row_bad += 1
        if row_bad:
            n_unparseable += row_bad
            n_unparseable_rows += 1
            issues.append((mol_id, f"{row_bad} unparseable SMILES"))

    return {
        "n_rows": n_rows,
        "n_unique_molecule_ids": n_unique_ids,
        "rows_match_unique_ids": n_unique_ids == n_rows,
        "max_preds_seen": max_preds_seen,
        "max_preds_allowed": max_preds,
        "n_unparseable_smiles": n_unparseable,
        "n_rows_with_unparseable_smiles": n_unparseable_rows,
        "n_issues": len(issues),
        "per_row_issues": issues,
        "is_valid": len(issues) == 0,
    }
