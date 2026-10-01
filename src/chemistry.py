"""RDKit-backed molecular structure, descriptor, fingerprint, and adduct-mass utilities."""
import re
from collections import Counter

import numpy as np

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# --- Monoisotopic atomic masses (Da) and the electron mass, used for adduct arithmetic ---
ATOMIC_MASSES = {
    "H": 1.0078250319, "D": 2.0141017780, "C": 12.0, "N": 14.0030740052,
    "O": 15.9949146221, "F": 18.9984032, "Na": 22.98976928, "Mg": 23.9850417,
    "P": 30.97376151, "S": 31.97207069, "Cl": 34.96885268, "K": 38.9637069,
    "Ca": 39.9625912, "Fe": 55.9349393, "Br": 78.9183371, "I": 126.904473,
}
ELECTRON_MASS = 0.00054858

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula):
    """Monoisotopic mass of a simple chemical formula fragment, e.g. 'CH2O2' -> 46.0055.

    Returns None if any element token is unrecognized (never silently guesses).
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


_ADDUCT_RE = re.compile(r"^\[(\d*)M((?:[+-][A-Za-z0-9]+)*)\](\d*)([+-])$")
_ADDUCT_TERM_RE = re.compile(r"([+-])(\d*)([A-Za-z]+\d*(?:[A-Za-z]+\d*)*)")


def parse_adduct(adduct):
    """Parse an adduct string like '[2M+Na]+' or '[M+CH2O2-H]-'.

    Returns dict(n=multimer count, terms=[(sign, count, fragment), ...], z=charge magnitude,
    polarity='+'/'-') or None if the string doesn't match a recognized adduct shape.
    """
    if not adduct or not isinstance(adduct, str):
        return None
    m = _ADDUCT_RE.match(adduct.strip())
    if not m:
        return None
    n_str, ops_str, z_str, polarity = m.groups()
    n = int(n_str) if n_str else 1
    z = int(z_str) if z_str else 1
    terms = []
    pos = 0
    for tm in _ADDUCT_TERM_RE.finditer(ops_str):
        if tm.start() != pos:
            return None
        pos = tm.end()
        sign, count_str, frag = tm.groups()
        count = int(count_str) if count_str else 1
        terms.append((1 if sign == "+" else -1, count, frag))
    if pos != len(ops_str):
        return None
    return {"n": n, "terms": terms, "z": z, "polarity": polarity}


def neutral_mass_from_precursor(precursor_mz, adduct):
    """Estimate neutral monoisotopic mass M from an observed precursor m/z and adduct string.

    Returns None for adducts that don't parse or contain unrecognized fragments, rather than
    guessing. Accounts for electron mass so the estimate is consistent to ~sub-ppm for a
    correctly-parsed adduct (before instrument measurement error).
    """
    parsed = parse_adduct(adduct)
    if parsed is None:
        return None
    delta = 0.0
    for sign, count, frag in parsed["terms"]:
        fm = formula_mass(frag)
        if fm is None:
            return None
        delta += sign * count * fm
    z = parsed["z"]
    ion_mass = precursor_mz * z
    # Forming a +z ion removes z electrons from the neutral assembly; forming a -z ion adds z.
    electron_term = -z * ELECTRON_MASS if parsed["polarity"] == "+" else z * ELECTRON_MASS
    neutral_multimer_mass = ion_mass - delta - electron_term
    return neutral_multimer_mass / parsed["n"]


# --- Molecule descriptors ---------------------------------------------------------------

_HALOGENS = {"F", "Cl", "Br", "I"}


def mol_from_smiles(smiles):
    if not RDKIT_AVAILABLE or not smiles or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    return Chem.MolFromSmiles(str(smiles))


def connectivity_smiles(smiles):
    """Canonical SMILES with stereochemistry stripped -- a connectivity-only key."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)


def murcko_scaffold_smiles(smiles):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return None


def compute_molecule_descriptors(smiles):
    """One dict of structural descriptors for a SMILES string, or all-None on parse failure."""
    keys = [
        "canonical_smiles", "connectivity_key", "molecular_formula", "exact_mass",
        "molecular_weight", "num_atoms", "num_heavy_atoms", "num_c", "num_n", "num_o",
        "num_s", "num_p", "num_halogen", "num_rings", "num_aromatic_rings",
        "fraction_aromatic_atoms", "num_rotatable_bonds", "h_bond_donors",
        "h_bond_acceptors", "formal_charge", "tpsa", "logp", "murcko_scaffold",
    ]
    mol = mol_from_smiles(smiles)
    if mol is None:
        return {k: None for k in keys}

    mol_h = Chem.AddHs(mol)
    atoms = mol.GetAtoms()
    elem_counts = Counter(a.GetSymbol() for a in atoms)
    n_heavy = mol.GetNumHeavyAtoms()
    n_aromatic_atoms = sum(1 for a in atoms if a.GetIsAromatic())
    ri = mol.GetRingInfo()

    try:
        scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), canonical=True)
    except Exception:
        scaffold = None

    return {
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "connectivity_key": Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        "exact_mass": rdMolDescriptors.CalcExactMolWt(mol),
        "molecular_weight": Descriptors.MolWt(mol),
        "num_atoms": mol_h.GetNumAtoms(),
        "num_heavy_atoms": n_heavy,
        "num_c": elem_counts.get("C", 0),
        "num_n": elem_counts.get("N", 0),
        "num_o": elem_counts.get("O", 0),
        "num_s": elem_counts.get("S", 0),
        "num_p": elem_counts.get("P", 0),
        "num_halogen": sum(elem_counts.get(h, 0) for h in _HALOGENS),
        "num_rings": ri.NumRings(),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "fraction_aromatic_atoms": (n_aromatic_atoms / n_heavy) if n_heavy else 0.0,
        "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "h_bond_donors": rdMolDescriptors.CalcNumHBD(mol),
        "h_bond_acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "logp": Descriptors.MolLogP(mol),
        "murcko_scaffold": scaffold,
    }


# --- Fingerprints ------------------------------------------------------------------------

_MORGAN_GENERATORS = {}


def _get_morgan_generator(radius, fp_size):
    key = (radius, fp_size)
    if key not in _MORGAN_GENERATORS:
        _MORGAN_GENERATORS[key] = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    return _MORGAN_GENERATORS[key]


def morgan_fp(smiles_or_mol, radius=2, fp_size=2048):
    mol = smiles_or_mol if (RDKIT_AVAILABLE and hasattr(smiles_or_mol, "GetAtoms")) else mol_from_smiles(smiles_or_mol)
    if mol is None:
        return None
    return _get_morgan_generator(radius, fp_size).GetFingerprint(mol)


def bulk_tanimoto(query_fp, fp_list):
    """Tanimoto similarity of one fingerprint against a list of fingerprints (skips Nones)."""
    valid_idx = [i for i, fp in enumerate(fp_list) if fp is not None]
    sims = np.zeros(len(fp_list), dtype=float)
    if not valid_idx or query_fp is None:
        return sims
    valid_fps = [fp_list[i] for i in valid_idx]
    valid_sims = DataStructs.BulkTanimotoSimilarity(query_fp, valid_fps)
    for i, s in zip(valid_idx, valid_sims):
        sims[i] = s
    return sims
