"""External-source structure CLEANUP policies (charge rescue + salt / fragment-parent rescue).

Two named policies, never mixed silently:

    current_policy   no cleanup: the raw source SMILES goes straight to the training canonicalizer and the
                     configured filters (neutral_only / single_component_only) drop charged / multi-component records.
                     This is what built the current TRAIN u COCONUT universe -- it is the default and changes nothing.
    rescue_policy    before canonicalization, for EXTERNAL records only:
                       1. parse the source structure (RDKit, same toolkit as the project canonicalizer);
                       2. multi-component -> parent fragment via RDKit MolStandardize `LargestFragmentChooser`
                          (preferOrganic=True: an organic parent beats a larger inorganic counter-ion). The project has
                          no salt-stripping standardizer of its own, so this RDKit chooser is used and documented;
                       3. charged -> RDKit MolStandardize `Uncharger` (neutralizes where chemically possible);
                       4. still charged after (3) -> RETAINED as a permanently charged candidate (never discarded by the
                          cleanup), flagged `permanently_charged=True`; the effective filters drop `neutral_only`.
                     Provenance is preserved: `original_source_smiles` (as supplied), `original_component_count`,
                     `selected_parent_smiles`, `salt_stripped`, `charge_cleanup_status`, source / source_id untouched.

TRAIN chemistry is NEVER touched by this module (it only receives external source records).
The policy is part of the Stage-A build identity ONLY when it is not `current_policy`, so the existing
COCONUT build keeps its build id and is never invalidated by this module's existence.
"""
import json
from dataclasses import asdict, dataclass, replace

import pandas as pd

CLEANUP_POLICY_VERSION = "casmi-cleanup-1"
CLEANUP_COLUMNS = ["original_source_smiles", "original_component_count", "selected_parent_smiles", "salt_stripped", "uncharged",
                   "formal_charge_before_cleanup", "permanently_charged", "charge_cleanup_status"]
CHARGE_STATUSES = ("not_applied", "neutral", "neutralized", "permanent_charge_retained", "permanent_charge_dropped", "parse_failed")


@dataclass(frozen=True)
class CleanupPolicy:
    name: str = "current_policy"
    uncharge: bool = False
    fragment_parent: bool = False
    retain_permanent_charge: bool = False
    prefer_organic_fragment: bool = True
    version: str = CLEANUP_POLICY_VERSION

    @property
    def active(self):
        return bool(self.uncharge or self.fragment_parent or self.retain_permanent_charge)


POLICIES = {
    "current_policy": CleanupPolicy("current_policy"),
    "rescue_policy": CleanupPolicy("rescue_policy", uncharge=True, fragment_parent=True, retain_permanent_charge=True),
}


def get_policy(policy="current_policy"):
    """A `CleanupPolicy` from a name (`current_policy` / `rescue_policy`), a dict, or a policy."""
    if isinstance(policy, CleanupPolicy):
        return policy
    if isinstance(policy, dict):
        return CleanupPolicy(**{k: v for k, v in policy.items() if k in CleanupPolicy.__dataclass_fields__})
    if policy is None:
        return POLICIES["current_policy"]
    if policy not in POLICIES:
        raise ValueError(f"unknown cleanup policy {policy!r}; expected one of {sorted(POLICIES)}")
    return POLICIES[policy]


def is_default_policy(policy):
    return not get_policy(policy).active


def policy_identity(policy):
    """Deterministic identity dict (part of Stage-A / Stage-B identity when the policy is active)."""
    return asdict(get_policy(policy))


def effective_filters(filters, policy):
    """Filters actually applied after cleanup: a policy that RETAINS permanent charges must not let `neutral_only`
    discard them again. Everything else (mass range, organic, single component) is unchanged."""
    p = get_policy(policy)
    return replace(filters, neutral_only=False) if p.retain_permanent_charge and filters.neutral_only else filters


# ---------------------------------------------------------------------------------------------
# per-structure cleanup
# ---------------------------------------------------------------------------------------------

def _empty_row(smiles):
    return {"original_source_smiles": smiles, "cleaned_smiles": None, "parse_ok": False, "original_component_count": None,
            "selected_parent_smiles": None, "salt_stripped": False, "uncharged": False, "formal_charge_before_cleanup": None,
            "formal_charge_after_cleanup": None, "permanently_charged": False, "charge_cleanup_status": "parse_failed"}


def cleanup_structure(smiles, policy="rescue_policy"):
    """Apply `policy` to ONE source SMILES. Returns a dict with `cleaned_smiles` (the structure handed to the
    competition canonicalizer) and the provenance fields. Never raises; a parse failure keeps `cleaned_smiles=None`
    (the record is then rejected downstream exactly like before)."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    p = get_policy(policy)
    row = _empty_row(smiles)
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
    if mol is None:
        return row
    row["parse_ok"] = True
    frags = Chem.GetMolFrags(mol)
    row["original_component_count"] = int(len(frags))
    row["formal_charge_before_cleanup"] = int(Chem.GetFormalCharge(mol))
    work = mol
    if p.fragment_parent and len(frags) > 1:
        chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=bool(p.prefer_organic_fragment))
        work = chooser.choose(mol)
        row["salt_stripped"] = True
        row["selected_parent_smiles"] = Chem.MolToSmiles(work)
    charged_atoms = any(a.GetFormalCharge() != 0 for a in work.GetAtoms())
    if p.uncharge and charged_atoms:
        before = Chem.MolToSmiles(work)
        work = rdMolStandardize.Uncharger().uncharge(work)
        row["uncharged"] = Chem.MolToSmiles(work) != before
    after = int(Chem.GetFormalCharge(work))
    row["formal_charge_after_cleanup"] = after
    if not p.active:
        row["charge_cleanup_status"] = "not_applied"
    elif after == 0:
        row["charge_cleanup_status"] = "neutralized" if row["formal_charge_before_cleanup"] != 0 or row["uncharged"] else "neutral"
    else:
        row["permanently_charged"] = True
        row["charge_cleanup_status"] = "permanent_charge_retained" if p.retain_permanent_charge else "permanent_charge_dropped"
    row["cleaned_smiles"] = Chem.MolToSmiles(work) if p.active else smiles
    return row


def apply_cleanup_to_records(records, policy="rescue_policy"):
    """Canonical source records (source, source_id, raw_smiles, ...) -> the same records with `raw_smiles` REPLACED by
    the cleaned structure and the provenance columns `CLEANUP_COLUMNS` added (`original_source_smiles` = the SMILES as
    supplied). Under `current_policy` the records are returned unchanged (no columns added)."""
    p = get_policy(policy)
    if not p.active:
        return records
    out = records.copy()
    uniq = pd.unique(pd.Series(out["raw_smiles"], dtype="object").dropna())
    table = {s: cleanup_structure(s, p) for s in uniq}           # one RDKit pass per unique raw SMILES
    rows = [table.get(s) if isinstance(s, str) else None for s in out["raw_smiles"]]
    for c in CLEANUP_COLUMNS:
        out[c] = [r[c] if r else None for r in rows]
    out["original_source_smiles"] = out["raw_smiles"].where(out["raw_smiles"].notna(), None)
    # parse failures keep their raw SMILES so the standardizer rejects them with the usual reason
    out["raw_smiles"] = [(r["cleaned_smiles"] if r and r["cleaned_smiles"] else s) for s, r in zip(out["raw_smiles"], rows)]
    return out


def cleanup_summary(cleaned_records):
    """Counts of salt stripping / neutralization / permanent charges in a cleaned chunk (for logs and reports)."""
    if "charge_cleanup_status" not in cleaned_records:
        return {"policy_active": False}
    from casmi.utils.booleans import as_bool
    r = cleaned_records
    return {"policy_active": True, "n_records": int(len(r)), "n_salt_stripped": int(as_bool(r["salt_stripped"]).sum()),
            "charge_cleanup_status": json.loads(pd.Series(r["charge_cleanup_status"]).value_counts(dropna=False).to_json())}
