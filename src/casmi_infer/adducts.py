"""Adduct arithmetic for inference. The bundle's `adducts.json` is SERIALIZED from
`casmi.chemistry.adducts.parse_adduct` (the training parser) -- never hand-typed -- and inference
applies the stored (multiplier, charge, mass_shift) with the same formula as
`neutral_mass_from_precursor`. `export_adduct_rules` / `AdductRules` round-trip is parity-tested.
"""
import hashlib
import inspect
import json

from casmi.chemistry import adducts as _train_adducts
from casmi.chemistry.adducts import ELECTRON_MASS, parse_adduct


def parser_version():
    """Content hash of the training adduct parser module (grammar + atomic masses + electron mass)."""
    return hashlib.sha256(inspect.getsource(_train_adducts).encode("utf-8")).hexdigest()[:16]


def export_adduct_rules(adduct_strings, observed_in_test=()):
    observed = set(observed_in_test)
    rules = {}
    for a in sorted(set(map(str, adduct_strings)) | observed):
        p = parse_adduct(a)
        rules[a] = {"supported": bool(p["supported"]), "molecule_multiplier": p["molecule_multiplier"], "charge": p["charge"],
                    "mass_shift": p["mass_shift"], "observed_in_test": a in observed}
    return {"parser_version": parser_version(), "electron_mass": ELECTRON_MASS,
            "electron_handling": "mass_shift already includes -z*m_e for cations / +z*m_e for anions (parse_adduct)",
            "formula": "neutral_mass = (precursor_mz * |charge| - mass_shift) / molecule_multiplier",
            "rules": rules}


class AdductRules:
    def __init__(self, payload):
        self.payload = payload
        self.rules = payload["rules"]

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def supported(self, adduct):
        r = self.rules.get(str(adduct))
        return bool(r and r["supported"])

    def neutral_mass(self, precursor_mz, adduct):
        """None for an adduct the bundle does not know or cannot parse (never guessed here)."""
        r = self.rules.get(str(adduct))
        if not r or not r["supported"] or precursor_mz is None:
            return None
        return (float(precursor_mz) * abs(r["charge"]) - r["mass_shift"]) / r["molecule_multiplier"]

    @staticmethod
    def polarity_default(ionization_mode):
        """Explicit, LOGGED last resort for an unsupported adduct: the polarity's protonation adduct."""
        return "[M-H]-" if str(ionization_mode).lower().startswith("neg") else "[M+H]+"
