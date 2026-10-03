"""Per-molecule structural descriptors. Pure functions: no file I/O, no RDKit logging setup
side effects beyond what `casmi.chemistry.connectivity` already configures.
"""
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

_HALOGENS = {"F", "Cl", "Br", "I"}
_HETEROATOMS = {"N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se", "As"}

DESCRIPTOR_FIELDS = [
    "exact_mass", "molecular_weight", "num_atoms", "num_heavy_atoms", "num_c", "num_n",
    "num_o", "num_s", "num_p", "num_halogen", "num_heteroatoms", "num_rings",
    "num_aromatic_rings", "num_rotatable_bonds", "formal_charge", "hbd", "hba", "tpsa",
    "logp", "murcko_scaffold", "scaffold_smiles",
]


IDENTITY_DESCRIPTOR_FIELDS = ["exact_mass", "molecular_weight", "formal_charge"]


def _identity_values(mol):
    """The three descriptors the candidate universe consumes. Shared by `compute_molecular_descriptors`
    and `compute_identity_descriptors`, so both return bit-identical values (same RDKit calls on the
    same `MolFromSmiles(smiles)` object)."""
    return {"exact_mass": rdMolDescriptors.CalcExactMolWt(mol), "molecular_weight": Descriptors.MolWt(mol),
            "formal_charge": Chem.GetFormalCharge(mol)}


def compute_identity_descriptors(smiles):
    """`exact_mass`, `molecular_weight`, `formal_charge` only (all None on a parse failure, never raises).
    Used by the candidate-universe `universe_minimal` standardization profile, which skips the scaffold /
    TPSA / logP / ring / element-count descriptors the universe never reads."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return {k: None for k in IDENTITY_DESCRIPTOR_FIELDS}
    return _identity_values(mol)


def compute_molecular_descriptors(smiles):
    """One dict of structural descriptors for a SMILES string, with every field `None` on a
    parse failure (never raises).

    `murcko_scaffold`: the Bemis-Murcko scaffold SMILES, real atom/bond types preserved.
    `scaffold_smiles`: the *generic* scaffold (all atoms->C, all bonds->single) -- a coarser
    grouping used for the harder scaffold-based validation fold (see `casmi.data.aggregation`).
    Both are `""` (not None) for an acyclic molecule, which has an empty scaffold by
    definition -- distinguished from a parse failure, which is `None`.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return {k: None for k in DESCRIPTOR_FIELDS}

    atoms = mol.GetAtoms()
    elem_counts = Counter(a.GetSymbol() for a in atoms)
    n_heavy = mol.GetNumHeavyAtoms()
    ri = mol.GetRingInfo()

    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        murcko_scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True)
        scaffold_smiles = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold_mol), canonical=True)
    except Exception:
        murcko_scaffold, scaffold_smiles = None, None

    ident = _identity_values(mol)
    return {
        "exact_mass": ident["exact_mass"],
        "molecular_weight": ident["molecular_weight"],
        "num_atoms": Chem.AddHs(mol).GetNumAtoms(),
        "num_heavy_atoms": n_heavy,
        "num_c": elem_counts.get("C", 0),
        "num_n": elem_counts.get("N", 0),
        "num_o": elem_counts.get("O", 0),
        "num_s": elem_counts.get("S", 0),
        "num_p": elem_counts.get("P", 0),
        "num_halogen": sum(elem_counts.get(h, 0) for h in _HALOGENS),
        "num_heteroatoms": sum(c for e, c in elem_counts.items() if e in _HETEROATOMS),
        "num_rings": ri.NumRings(),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "formal_charge": ident["formal_charge"],
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "logp": Descriptors.MolLogP(mol),
        "murcko_scaffold": murcko_scaffold,
        "scaffold_smiles": scaffold_smiles,
    }
