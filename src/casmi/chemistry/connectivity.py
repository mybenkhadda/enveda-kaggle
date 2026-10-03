"""Molecular identity: canonical SMILES, InChIKey(14), and the competition's own connectivity
key (RDKit tautomer-canonical InChIKey14).

The dataset ships its own `inchikey14` column; this module never overwrites it. Three
identities stay explicitly separate everywhere in this package:

    provided inchikey14        -- whatever the raw file says (data.metadata)
    computed inchikey14        -- `inchikey14_from_smiles`, the as-is (non-tautomer) key
    competition connectivity   -- `competition_connectivity_key`, tautomer-canonical

Every function here handles an invalid SMILES by returning `None`, never by raising --
callers processing hundreds of thousands of rows should not have to wrap every call in
try/except.
"""
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

_ENUMERATOR_CACHE = {}


def _mol_from_smiles(smiles):
    if not smiles or not isinstance(smiles, str):
        return None
    return Chem.MolFromSmiles(smiles)


def canonicalize_smiles(smiles):
    """RDKit canonical SMILES (isomeric, i.e. stereo/isotopes preserved), or None if
    unparseable."""
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def inchikey_from_smiles(smiles):
    """Full standard InChIKey, or None if unparseable / InChI generation fails."""
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return ik or None


def inchikey14_from_smiles(smiles):
    """First 14 characters (the connectivity block) of the as-is InChIKey -- NOT tautomer-
    canonicalized. See `competition_connectivity_key` for the competition's actual key."""
    ik = inchikey_from_smiles(smiles)
    return ik[:14] if ik else None


def get_tautomer_enumerator(max_tautomers=None, max_transforms=None):
    """A cached `TautomerEnumerator` for a given (max_tautomers, max_transforms) cap.

    A cap (rather than a wall-clock timeout) bounds worst-case per-molecule search effort:
    signal-based timeouts aren't reliably portable across a multiprocessing pool (notably on
    Windows), whereas RDKit's own `maxTautomers`/`maxTransforms` bound the work directly.
    `None` for either means "use RDKit's own default" (closest to how the competition itself
    presumably computes this key).
    """
    key = (max_tautomers, max_transforms)
    if key not in _ENUMERATOR_CACHE:
        params = rdMolStandardize.CleanupParameters()
        if max_tautomers is not None:
            params.maxTautomers = max_tautomers
        if max_transforms is not None:
            params.maxTransforms = max_transforms
        _ENUMERATOR_CACHE[key] = rdMolStandardize.TautomerEnumerator(params)
    return _ENUMERATOR_CACHE[key]


def competition_connectivity_key_detailed(smiles, enumerator=None, max_tautomers=None, max_transforms=None, compute_hit_cap=True):
    """Full detail behind `competition_connectivity_key`: also reports whether the tautomer
    search likely hit its cap (`hit_cap=True` means the result is the best tautomer found
    within the cap, not a verified global optimum -- rare, but worth logging in bulk runs
    rather than silently trusting).

    `compute_hit_cap=False` skips ONLY the diagnostic `Enumerate` call used to count tautomers
    (`hit_cap` is then None). The key itself always comes from the same
    `enumerator.Canonicalize(mol)` -> `MolToInchiKey` call; default True = the original behaviour.

    Returns dict: {conn_key, plain_inchikey14, hit_cap, parse_ok, error}.
    """
    out = {"conn_key": None, "plain_inchikey14": None, "hit_cap": False if compute_hit_cap else None, "parse_ok": False, "error": None}
    mol = _mol_from_smiles(smiles)
    if mol is None:
        out["error"] = "parse_failed"
        return out
    out["parse_ok"] = True
    out["plain_inchikey14"] = inchikey14_from_smiles(smiles)

    enumerator = enumerator or get_tautomer_enumerator(max_tautomers, max_transforms)
    try:
        hit_cap = None
        if compute_hit_cap:
            n_tautomers = len(enumerator.Enumerate(mol))
            hit_cap = bool(n_tautomers >= enumerator.GetMaxTautomers())
        canon = enumerator.Canonicalize(mol)
        ik = Chem.MolToInchiKey(canon)
        out["conn_key"] = ik[:14] if ik else out["plain_inchikey14"]
        out["hit_cap"] = hit_cap
    except Exception as exc:
        out["error"] = f"tautomer_canonicalize_failed: {exc}"
        out["conn_key"] = out["plain_inchikey14"]
    return out


def competition_connectivity_key(smiles, enumerator=None, max_tautomers=None, max_transforms=None):
    """The competition's connectivity key: RDKit tautomer-canonicalize, then the first 14
    characters (connectivity block, stereo-ignored) of the resulting InChIKey. Returns None
    for an unparseable SMILES."""
    return competition_connectivity_key_detailed(smiles, enumerator, max_tautomers, max_transforms)["conn_key"]
