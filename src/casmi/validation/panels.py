"""Validation PANELS (v2). Four populations, never merged silently; every query row carries its panel.

    ALL_DEV      every grouped-validation dev query (formerly `dev_grouped`). Reporting panel.
    TIMSTOF_DEV  dev queries with instrument_type in the configured values (formerly, WRONGLY, `host_like`). An
                 INSTRUMENT diagnostic: it is not a natural-product panel and NEVER governs a decision.
    HOST         file-backed host panel: the ~250 enveda-np-examples molecules (local
                 data/processed/host_holdout/host_holdout_spectra.parquet, 1,184 spectra / 250 connectivities). Absent file ->
                 NOT_AVAILABLE / HOST_PANEL_SOURCE_MISSING. Never fabricated, never derived from another column.
    NP_PANEL     natural-product-like dev truths by a STRUCTURE-ONLY NP-likeness score >= a USER threshold. Never defined by
                 COCONUT / PubChem membership, reachability, rank, score or any outcome (that would be circular). The method,
                 its version, the threshold and their provenance are recorded. Threshold unset -> NOT_AVAILABLE.

Decision panel: HOST when AVAILABLE, else NP_PANEL when AVAILABLE, else none (decision rules on the NP panel are then
NOT_EVALUATED). TIMSTOF_DEV / ALL_DEV can never be the decision panel.

NP-likeness: the repository has NO NP-likeness implementation (universe `natural_product_score` is NaN by design). The
interface offers `rdkit_contrib_np_score` (Ertl NP-likeness model shipped in RDKit's Contrib/NP_Score, if the installed
RDKit includes Contrib -- checked at run time, NOT_AVAILABLE otherwise) and `file` (a user-supplied per-connectivity score
table). Scores are computed from TRAIN structure-table SMILES (structure only).
"""
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PANEL_PROTOCOL_VERSION = "casmi-panels-2"
ALL_DEV, TIMSTOF_DEV, HOST, NP_PANEL = "ALL_DEV", "TIMSTOF_DEV", "HOST", "NP_PANEL"
PANELS = (ALL_DEV, TIMSTOF_DEV, HOST, NP_PANEL)
DECISION_ELIGIBLE = (HOST, NP_PANEL)
DEPRECATED_ALIASES = {"host_like": TIMSTOF_DEV, "dev_grouped": ALL_DEV}
AVAILABLE, NOT_AVAILABLE = "AVAILABLE", "NOT_AVAILABLE"
HOST_PANEL_SOURCE_MISSING = "HOST_PANEL_SOURCE_MISSING"
NP_THRESHOLD_NOT_SET, NP_METHOD_UNAVAILABLE, NP_SCORES_MISSING = "NP_THRESHOLD_NOT_SET", "NP_METHOD_UNAVAILABLE", "NP_SCORES_MISSING"
NP_METHODS = ("rdkit_contrib_np_score", "file")
QUERY_COLUMNS = ["panel", "population_source", "query_id", "connectivity_key", "fold", "neutral_mass", "truth_exact_mass", "adduct",
                 "ionization_mode", "instrument_type"]


def canonical_panel_name(name):
    """Map deprecated panel names (host_like -> TIMSTOF_DEV, dev_grouped -> ALL_DEV) with a DeprecationWarning."""
    if name in DEPRECATED_ALIASES:
        warnings.warn(f"panel name {name!r} is deprecated; use {DEPRECATED_ALIASES[name]!r} (host_like was the timsTOF instrument subset, "
                      "never a natural-product panel)", DeprecationWarning, stacklevel=2)
        return DEPRECATED_ALIASES[name]
    return name


@dataclass
class PanelConfig:
    timstof_column: str = "instrument_type"
    timstof_values: list = field(default_factory=lambda: ["timsTOF"])
    host_path: str | None = "data/processed/host_holdout/host_holdout_spectra.parquet"
    host_source_label: str = "enveda-np-examples"
    host_expected_n_molecules: int | None = 250
    np_method: str = "rdkit_contrib_np_score"
    np_threshold: float | None = None                       # USER decision, declared before results; None -> NP_PANEL NOT_AVAILABLE
    np_threshold_provenance: str | None = None              # who / what set the threshold (recorded in the panel manifest)
    np_score_file: str | None = None                        # method 'file': parquet / csv with connectivity_key + np_score_column
    np_score_column: str = "np_score"
    np_score_cache: str | None = "data/processed/panels/np_scores.parquet"
    decision_panel_order: list = field(default_factory=lambda: [HOST, NP_PANEL])

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        out = {}
        tt = d.get(TIMSTOF_DEV) or d.get("host_like") or {}
        if "host_like" in d:
            warnings.warn("config panel 'host_like' is deprecated -> TIMSTOF_DEV (instrument subset, never the NP panel)", DeprecationWarning, stacklevel=2)
        if tt:
            out.update(timstof_column=tt.get("column", "instrument_type"), timstof_values=list(tt.get("values") or ["timsTOF"]))
        h = d.get(HOST) or {}
        for k_cfg, k in (("path", "host_path"), ("source_label", "host_source_label"), ("expected_n_molecules", "host_expected_n_molecules")):
            if k_cfg in h:
                out[k] = h[k_cfg]
        n = d.get(NP_PANEL) or {}
        for k_cfg, k in (("method", "np_method"), ("threshold", "np_threshold"), ("threshold_provenance", "np_threshold_provenance"),
                         ("score_file", "np_score_file"), ("score_column", "np_score_column"), ("score_cache", "np_score_cache")):
            if k_cfg in n:
                out[k] = n[k_cfg]
        if "decision_panel_order" in d:
            out["decision_panel_order"] = list(d["decision_panel_order"])
        cfg = cls(**out)
        cfg.validate()
        return cfg

    def validate(self):
        bad = [p for p in self.decision_panel_order if p not in DECISION_ELIGIBLE]
        if bad:
            raise ValueError(f"decision_panel_order may contain only {DECISION_ELIGIBLE}; {bad} can never govern a decision")
        if self.np_method not in NP_METHODS:
            raise ValueError(f"np_method must be one of {NP_METHODS}, got {self.np_method!r}")
        return self

    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# NP-likeness (structure only)
# ---------------------------------------------------------------------------------------------

def _rdkit_np_model():
    """(scorer module, model, version dict) from RDKit Contrib/NP_Score, or (None, None, reason)."""
    import importlib.util
    import os
    try:
        import rdkit
        from rdkit.Chem import RDConfig
    except ImportError as e:
        return None, None, {"error": f"rdkit not importable: {e}"}
    path = os.path.join(RDConfig.RDContribDir, "NP_Score", "npscorer.py")
    if not os.path.isfile(path):
        return None, None, {"error": f"RDKit Contrib NP_Score not installed ({path} missing)", "rdkit": rdkit.__version__}
    spec = importlib.util.spec_from_file_location("casmi_npscorer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.readNPModel()
    model_file = os.path.join(RDConfig.RDContribDir, "NP_Score", "publicnp.model.gz")
    sha = hashlib.sha256(Path(model_file).read_bytes()).hexdigest()[:16] if os.path.isfile(model_file) else None
    return mod, model, {"rdkit": rdkit.__version__, "npscorer": path, "model_sha256": sha, "reference": "Ertl et al. 2008 NP-likeness"}


def np_method_status(method):
    if method == "file":
        return True, {"method": "file"}
    mod, _, info = _rdkit_np_model()
    return mod is not None, {"method": method, **info}


def np_likeness_scores(smiles_by_key, method="rdkit_contrib_np_score", score_file=None, score_column="np_score"):
    """One row per connectivity: connectivity_key, np_score (NaN on failure), np_error, method. `smiles_by_key`: Series
    connectivity_key -> SMILES (TRAIN structure table; structure only). Returns (scores, method_info)."""
    keys = pd.Index(smiles_by_key.index.astype(str), name="connectivity_key")
    if method == "file":
        if not score_file or not Path(score_file).is_file():
            raise FileNotFoundError(f"NP score file missing: {score_file}")
        f = pd.read_parquet(score_file) if str(score_file).endswith(".parquet") else pd.read_csv(score_file)
        s = f.drop_duplicates("connectivity_key").set_index(f["connectivity_key"].astype(str))[score_column].astype(float)
        out = pd.DataFrame({"connectivity_key": keys, "np_score": s.reindex(keys).to_numpy(), "np_error": None, "method": "file"})
        info = {"method": "file", "score_file": Path(score_file).name, "score_column": score_column,
                "score_file_sha256": hashlib.sha256(Path(score_file).read_bytes()).hexdigest()[:16]}
        return out, info
    from rdkit import Chem
    mod, model, info = _rdkit_np_model()
    if mod is None:
        raise RuntimeError(f"{NP_METHOD_UNAVAILABLE}: {info.get('error')}")
    scores, errors = [], []
    for smi in smiles_by_key.to_numpy():
        mol = Chem.MolFromSmiles(str(smi)) if isinstance(smi, str) and smi else None
        if mol is None:
            scores.append(np.nan)
            errors.append("parse_failed")
            continue
        try:
            scores.append(float(mod.scoreMol(mol, model)))
            errors.append(None)
        except Exception as e:      # recorded per molecule, never silently dropped
            scores.append(np.nan)
            errors.append(f"{type(e).__name__}: {e}")
    return pd.DataFrame({"connectivity_key": keys, "np_score": scores, "np_error": errors, "method": method}), {"method": method, **info}


# ---------------------------------------------------------------------------------------------
# panel construction
# ---------------------------------------------------------------------------------------------

def _frame(df, panel, src):
    out = pd.DataFrame({"panel": panel, "population_source": src, "query_id": df["query_id"].astype(str).to_numpy(),
                        "connectivity_key": df["true_connectivity_key"].astype(str).to_numpy()})
    for c, srcc in (("fold", "fold"), ("neutral_mass", "neutral_mass"), ("truth_exact_mass", "exact_mass_true"), ("adduct", "adduct"),
                    ("ionization_mode", "ionization_mode"), ("instrument_type", "instrument_type")):
        out[c] = df[srcc].to_numpy() if srcc in df.columns else None
    if "truth_exact_mass" in out and out["truth_exact_mass"].isna().all() and "exact_mass" in df.columns:
        out["truth_exact_mass"] = df["exact_mass"].to_numpy()
    for c in ("neutral_mass", "truth_exact_mass"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out[QUERY_COLUMNS]


def file_identity(path):
    p = Path(path) if path else None
    if p is None or not p.is_file():
        return None
    return {"name": p.name, "size_bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}


def build_panels(dev, cfg, host_frame=None, host_identity=None, np_scores=None, np_info=None):
    """Query-level table of the four panels + per-panel status. `host_frame`: the HOST table (query_id,
    true_connectivity_key, neutral_mass, ...) or None when the file is missing. `np_scores`: connectivity_key -> np_score
    (structure-only) or None. Returns (queries, status)."""
    d = dev.copy()
    d["query_id"] = d["query_id"].astype(str)
    parts, status = [_frame(d, ALL_DEV, "dev_queries (connectivity-grouped folds)")], {}
    status[ALL_DEV] = {"status": AVAILABLE, "role": "reporting", "reason": ""}
    if cfg.timstof_column in d.columns:
        sub = d[d[cfg.timstof_column].astype(str).isin([str(v) for v in cfg.timstof_values])]
        parts.append(_frame(sub, TIMSTOF_DEV, f"dev_queries[{cfg.timstof_column} in {list(cfg.timstof_values)}]"))
        status[TIMSTOF_DEV] = {"status": AVAILABLE if len(sub) else NOT_AVAILABLE, "role": "instrument diagnostic (never a decision panel)",
                               "reason": "" if len(sub) else f"no dev query with {cfg.timstof_column} in {cfg.timstof_values}"}
    else:
        status[TIMSTOF_DEV] = {"status": NOT_AVAILABLE, "role": "instrument diagnostic (never a decision panel)",
                               "reason": f"dev_queries has no {cfg.timstof_column!r} column"}
    if host_frame is None:
        status[HOST] = {"status": NOT_AVAILABLE, "role": "decision (primary)", "reason": HOST_PANEL_SOURCE_MISSING,
                        "expected_path": cfg.host_path, "source_label": cfg.host_source_label}
    else:
        missing = {"query_id", "true_connectivity_key"} - set(host_frame.columns)
        if missing:
            raise KeyError(f"HOST table lacks {sorted(missing)}")
        hf = _frame(host_frame, HOST, f"file:{Path(cfg.host_path).name if cfg.host_path else '?'} ({cfg.host_source_label})")
        parts.append(hf)
        n_mol = int(hf["connectivity_key"].nunique())
        warn = (f"HOST has {n_mol} molecules, expected {cfg.host_expected_n_molecules}"
                if cfg.host_expected_n_molecules and n_mol != cfg.host_expected_n_molecules else "")
        status[HOST] = {"status": AVAILABLE if len(hf) else NOT_AVAILABLE, "role": "decision (primary)", "reason": "" if len(hf) else "empty HOST table",
                        "source_label": cfg.host_source_label, "source_identity": host_identity, "warning": warn}
    if cfg.np_threshold is None:
        status[NP_PANEL] = {"status": NOT_AVAILABLE, "role": "decision (fallback)", "reason": NP_THRESHOLD_NOT_SET, "method": cfg.np_method}
    elif np_scores is None:
        status[NP_PANEL] = {"status": NOT_AVAILABLE, "role": "decision (fallback)", "reason": NP_SCORES_MISSING, "method": cfg.np_method}
    else:
        s = (np_scores.assign(connectivity_key=np_scores["connectivity_key"].astype(str)).drop_duplicates("connectivity_key")
             .set_index("connectivity_key")["np_score"].astype(float))
        k = d["true_connectivity_key"].astype(str)
        sub = d[(k.map(s) >= float(cfg.np_threshold)).fillna(False).to_numpy(bool)]
        parts.append(_frame(sub, NP_PANEL, f"dev_queries[np_score({cfg.np_method}) >= {cfg.np_threshold}]"))
        status[NP_PANEL] = {"status": AVAILABLE if len(sub) else NOT_AVAILABLE, "role": "decision (fallback)",
                            "reason": "" if len(sub) else "no dev truth reaches the NP threshold", "method": cfg.np_method,
                            "method_info": np_info, "threshold": cfg.np_threshold, "threshold_provenance": cfg.np_threshold_provenance,
                            "n_scored": int(s.notna().sum()), "n_score_missing": int(k.map(s).isna().sum())}
    q = pd.concat(parts, ignore_index=True)
    for name, st in status.items():
        g = q[q["panel"] == name]
        st.update(n_queries=int(len(g)), n_molecules=int(g["connectivity_key"].nunique()))
    return q, status


def select_decision_panel(status, order=(HOST, NP_PANEL)):
    """First AVAILABLE panel of `order` (HOST, then NP_PANEL). TIMSTOF_DEV / ALL_DEV are refused outright."""
    bad = [p for p in order if p not in DECISION_ELIGIBLE]
    if bad:
        raise ValueError(f"{bad} can never govern the decision (only {DECISION_ELIGIBLE})")
    tried = []
    for p in order:
        st = (status or {}).get(p) or {}
        if st.get("status") == AVAILABLE and st.get("n_molecules", 0) > 0:
            return {"decision_panel": p, "reason": f"{p} available ({st.get('n_molecules')} molecules)", "tried": tried}
        tried.append({"panel": p, "status": st.get("status", NOT_AVAILABLE), "reason": st.get("reason", "not built")})
    return {"decision_panel": None, "reason": "no decision panel available: " + "; ".join(f"{t['panel']}={t['reason']}" for t in tried),
            "tried": tried}


def load_np_scores_cache(cache_path, method):
    """(scores, info) from the 11b cache (parquet + .json sidecar) when it was computed with `method`, else (None, None)."""
    p = Path(cache_path) if cache_path else None
    if p is None or not p.is_file() or not p.with_suffix(".json").is_file():
        return None, None
    info = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
    if info.get("method") != method:
        return None, None
    return pd.read_parquet(p), info


def ensure_np_scores(cfg, drive_root, smiles_by_key):
    """Notebook 11b only: compute + cache the structure-only NP scores when a threshold is declared and no cache matches
    the method. Returns (scores, info) or (None, None) when no threshold is set (NP_PANEL stays NOT_AVAILABLE)."""
    if cfg.np_threshold is None:
        return None, None
    cache = Path(drive_root) / cfg.np_score_cache if cfg.np_score_cache else None
    scores, info = load_np_scores_cache(cache, cfg.np_method)
    if scores is not None:
        return scores, info
    sf = Path(drive_root) / cfg.np_score_file if cfg.np_score_file else None
    scores, info = np_likeness_scores(smiles_by_key, cfg.np_method, sf, cfg.np_score_column)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(cache, index=False)
        cache.with_suffix(".json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return scores, info


def panels_for_run(cfg_dict, drive_root, dev):
    """The ONE panel construction used by notebooks 11b and 13 (identical inputs -> identical panel manifest identity).
    HOST from its file (missing -> HOST_PANEL_SOURCE_MISSING); NP scores only from the 11b cache."""
    pcfg = PanelConfig.from_dict(cfg_dict.get("panels"))
    host_path = Path(drive_root) / pcfg.host_path if pcfg.host_path else None
    host = None
    if host_path is not None and host_path.is_file():
        host = pd.read_parquet(host_path) if host_path.suffix == ".parquet" else pd.read_csv(host_path, dtype={"query_id": str})
    scores, info = (load_np_scores_cache(Path(drive_root) / pcfg.np_score_cache, pcfg.np_method)
                    if pcfg.np_threshold is not None and pcfg.np_score_cache else (None, None))
    q, status = build_panels(dev, pcfg, host, file_identity(host_path), scores, info)
    decision = select_decision_panel(status, pcfg.decision_panel_order)
    return {"config": pcfg, "queries": q, "status": status, "decision": decision, "manifest": panel_manifest(status, pcfg, decision),
            "host_frame": host, "host_path": host_path, "np_scores": scores}


def panel_manifest(status, cfg, decision):
    """Identity-bound panel manifest (what each panel IS, from which source, with which method / threshold)."""
    body = {"panel_protocol_version": PANEL_PROTOCOL_VERSION, "config": asdict(cfg), "config_identity": cfg.identity(),
            "panels": {p: {k: v for k, v in s.items() if k not in ("n_queries", "n_molecules")} | {"n_queries": s.get("n_queries"),
                                                                                                   "n_molecules": s.get("n_molecules")}
                       for p, s in status.items()},
            "decision_panel": decision.get("decision_panel"), "decision_panel_reason": decision.get("reason"),
            "deprecated_aliases": DEPRECATED_ALIASES}
    ident = {p: {k: s.get(k) for k in ("status", "reason", "n_queries", "n_molecules", "source_identity", "threshold", "method")}
             for p, s in status.items()}
    body["panel_manifest_identity"] = hashlib.sha256(json.dumps({"cfg": cfg.identity(), "panels": ident, "decision": decision.get("decision_panel")},
                                                                sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]
    return body
