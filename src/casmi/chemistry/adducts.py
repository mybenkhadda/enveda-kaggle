"""Adduct parsing and the mass arithmetic that relates an observed precursor m/z to a
neutral monoisotopic mass.

The grammar (`[nM+-frag...]z+/-`) is generic rather than a fixed lookup table: checked
against every one of the 121 distinct adduct strings in the real training data, it parses
all but a small number referencing elements this module doesn't carry a mass for (which are
reported as `supported=False`, never silently guessed).
"""
import re

import pandas as pd

# Monoisotopic atomic masses (Da) for elements that appear in this dataset's adduct fragments.
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


def _formula_mass(formula):
    """Monoisotopic mass of a simple formula fragment (e.g. 'CH2O2' -> 46.0055 Da), or None
    if any element token is unrecognized."""
    if not formula:
        return 0.0
    mass, consumed = 0.0, 0
    for sym, count in _FORMULA_TOKEN_RE.findall(formula):
        if not sym:
            continue
        if sym not in ATOMIC_MASSES:
            return None
        n = int(count) if count else 1
        mass += ATOMIC_MASSES[sym] * n
        consumed += len(sym) + len(count)
    return mass if consumed == len(formula) else None


def parse_adduct(adduct):
    """Parse an adduct string into structured info. Never guesses: an adduct that doesn't
    match the `[nM+-frag...]z+/-` shape, or references an unrecognized element, comes back
    with `supported=False` and every other field `None`.

    Returns:
        {
            "adduct": the input string,
            "supported": bool,
            "molecule_multiplier": int ("n" in "nM"),
            "charge": int, signed (positive for a cation, negative for an anion),
            "mass_shift": float, already including the electron-mass correction for ion
                formation, such that: ion_mz * |charge| == molecule_multiplier * neutral_mass
                + mass_shift.
        }
    """
    out = {"adduct": adduct, "supported": False, "molecule_multiplier": None, "charge": None, "mass_shift": None}
    if not adduct or not isinstance(adduct, str):
        return out
    m = _ADDUCT_RE.match(adduct.strip())
    if not m:
        return out
    n_str, ops_str, z_str, polarity = m.groups()
    multiplier = int(n_str) if n_str else 1
    z = int(z_str) if z_str else 1

    delta, pos = 0.0, 0
    for tm in _ADDUCT_TERM_RE.finditer(ops_str):
        if tm.start() != pos:
            return out
        pos = tm.end()
        sign, count_str, frag = tm.groups()
        fm = _formula_mass(frag)
        if fm is None:
            return out
        count = int(count_str) if count_str else 1
        delta += (1 if sign == "+" else -1) * count * fm
    if pos != len(ops_str):
        return out

    electron_term = -z * ELECTRON_MASS if polarity == "+" else z * ELECTRON_MASS
    out.update(
        supported=True,
        molecule_multiplier=multiplier,
        charge=z if polarity == "+" else -z,
        mass_shift=delta + electron_term,
    )
    return out


def precursor_from_neutral_mass(neutral_mass, adduct):
    """Expected precursor m/z for a neutral monoisotopic mass under `adduct`, or None if the
    adduct is unsupported or `neutral_mass` is missing."""
    parsed = parse_adduct(adduct)
    if not parsed["supported"] or neutral_mass is None:
        return None
    return (parsed["molecule_multiplier"] * neutral_mass + parsed["mass_shift"]) / abs(parsed["charge"])


def neutral_mass_from_precursor(precursor_mz, adduct):
    """Inverse of `precursor_from_neutral_mass`: the neutral monoisotopic mass implied by an
    observed precursor m/z under `adduct`, or None if the adduct is unsupported."""
    parsed = parse_adduct(adduct)
    if not parsed["supported"] or precursor_mz is None:
        return None
    return (precursor_mz * abs(parsed["charge"]) - parsed["mass_shift"]) / parsed["molecule_multiplier"]


def build_adduct_reference(adduct_series):
    """One row per unique adduct string in `adduct_series`, with its parsed info and
    frequency -- the `adduct_reference.parquet` artifact. Unsupported adducts are kept (not
    dropped) so their frequency is visible for deciding whether they're worth adding support
    for."""
    counts = pd.Series(adduct_series).value_counts(dropna=False)
    rows = []
    for adduct, n in counts.items():
        parsed = parse_adduct(adduct)
        parsed["n_occurrences"] = int(n)
        rows.append(parsed)
    return pd.DataFrame(rows).sort_values("n_occurrences", ascending=False).reset_index(drop=True)
