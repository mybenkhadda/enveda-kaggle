"""Configuration-driven readers for LOCAL external structure sources (COCONUT first; LOTUS / NPAtlas /
PubChem later). Nothing here downloads anything or touches the network: the user supplies a file and
a JSON config that NAMES its columns -- the schema is never guessed.

Accepted formats: csv, tsv, parquet, smi (SMILES [whitespace] ID per line; optional header), sdf / sdf.gz
(RDKit ForwardSDMolSupplier; SMILES from a named property or written from the molblock).

Every reader yields chunks in the canonical record schema:
    source, source_id, raw_smiles, name, source_formula, source_metadata (JSON string of the configured
    metadata columns)
A record without a usable SMILES is still yielded (raw_smiles None) so the standardizer REJECTS it
explicitly instead of the reader dropping it silently.
"""
import gzip
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

CANONICAL_SOURCES = {"TRAIN": "TRAIN", "COCONUT": "COCONUT", "LOTUS": "LOTUS", "NPATLAS": "NPATLAS", "PUBCHEM": "PUBCHEM"}
RECORD_COLUMNS = ["source", "source_id", "raw_smiles", "name", "source_formula", "source_metadata"]
FORMATS = ("csv", "tsv", "parquet", "smi", "sdf")


class SourceConfigError(ValueError):
    pass


class SourceNotSupplied(FileNotFoundError):
    """The local source file is not present -- obtain it MANUALLY and point the config at it."""


@dataclass
class SourceConfig:
    source: str = "COCONUT"
    path: str = ""
    format: str = "csv"
    id_col: str | None = None             # tabular / smi; for SDF: the id PROPERTY name (None -> record index)
    smiles_col: str | None = None         # tabular; for SDF: SMILES property (None -> MolToSmiles of the molblock)
    name_col: str | None = None
    formula_col: str | None = None
    metadata_cols: list = field(default_factory=list)
    delimiter: str | None = None          # csv/tsv override
    smi_has_header: bool = False
    chunksize: int = 50_000
    max_records: int | None = None        # smoke runs only
    notes: str = ""

    def validate(self):
        if self.source.upper() not in CANONICAL_SOURCES:
            raise SourceConfigError(f"unknown source {self.source!r}; expected one of {sorted(CANONICAL_SOURCES)}")
        fmt = self.format.lower().replace(".gz", "")
        if fmt not in FORMATS:
            raise SourceConfigError(f"format {self.format!r} not in {FORMATS}")
        if fmt in ("csv", "tsv", "parquet") and not self.smiles_col:
            raise SourceConfigError("tabular sources need `smiles_col` (the schema is configured, not guessed)")
        if fmt in ("csv", "tsv", "parquet") and not self.id_col:
            raise SourceConfigError("tabular sources need `id_col` (source ids must be preserved)")
        return self

    def canonical_source(self):
        return CANONICAL_SOURCES[self.source.upper()]


def load_source_config(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}      # "_comment" keys allowed in the template
    return SourceConfig(**raw).validate()


def file_fingerprint(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _meta_json(df, cols):
    if not cols:
        return pd.Series([None] * len(df), index=df.index)
    return df[cols].apply(lambda r: json.dumps({k: (None if pd.isna(v) else v) for k, v in r.items()}, default=str), axis=1)


def _tabular_chunk(df, cfg):
    need = [c for c in [cfg.id_col, cfg.smiles_col, cfg.name_col, cfg.formula_col, *cfg.metadata_cols] if c]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SourceConfigError(f"configured columns {missing} not in the file; available: {list(df.columns)}")
    out = pd.DataFrame({"source": cfg.canonical_source(), "source_id": df[cfg.id_col].astype(str).to_numpy(),
                        "raw_smiles": df[cfg.smiles_col].where(df[cfg.smiles_col].notna(), None).to_numpy(),
                        "name": df[cfg.name_col].to_numpy() if cfg.name_col else None,
                        "source_formula": df[cfg.formula_col].to_numpy() if cfg.formula_col else None})
    out["source_metadata"] = _meta_json(df, cfg.metadata_cols).to_numpy()
    return out[RECORD_COLUMNS]


def _iter_tabular(cfg):
    fmt = cfg.format.lower()
    path = Path(cfg.path)
    if fmt == "parquet":
        import pyarrow.parquet as pq
        cols = list(dict.fromkeys(c for c in [cfg.id_col, cfg.smiles_col, cfg.name_col, cfg.formula_col, *cfg.metadata_cols] if c))
        pf = pq.ParquetFile(path)
        missing = [c for c in cols if c not in pf.schema.names]
        if missing:
            raise SourceConfigError(f"configured columns {missing} not in the file; available: {pf.schema.names}")
        for b in pf.iter_batches(batch_size=cfg.chunksize, columns=cols):
            yield _tabular_chunk(b.to_pandas(), cfg)
        return
    sep = cfg.delimiter or ("\t" if fmt == "tsv" else ",")
    for df in pd.read_csv(path, sep=sep, chunksize=cfg.chunksize, dtype=str, keep_default_na=True, low_memory=False):
        yield _tabular_chunk(df, cfg)


def _iter_smi(cfg):
    rows, n = [], 0
    opener = gzip.open if str(cfg.path).endswith(".gz") else open
    with opener(cfg.path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and cfg.smi_has_header:
                continue
            parts = line.strip().split(None, 1)
            if not parts:
                continue
            rows.append({"source": cfg.canonical_source(), "source_id": parts[1].strip() if len(parts) > 1 else f"smi_line_{i}",
                         "raw_smiles": parts[0], "name": None, "source_formula": None, "source_metadata": None})
            n += 1
            if len(rows) >= cfg.chunksize:
                yield pd.DataFrame(rows, columns=RECORD_COLUMNS)
                rows = []
    if rows:
        yield pd.DataFrame(rows, columns=RECORD_COLUMNS)


def _iter_sdf(cfg):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    fh = gzip.open(cfg.path, "rb") if str(cfg.path).endswith(".gz") else open(cfg.path, "rb")
    rows = []
    with fh:
        for i, mol in enumerate(Chem.ForwardSDMolSupplier(fh, sanitize=True, removeHs=True)):
            if mol is None:        # unparseable record: yielded with raw_smiles None -> explicit reject downstream
                rows.append({"source": cfg.canonical_source(), "source_id": f"sdf_record_{i}", "raw_smiles": None, "name": None,
                             "source_formula": None, "source_metadata": json.dumps({"sdf_parse": "failed"})})
            else:
                prop = lambda k: mol.GetProp(k) if k and mol.HasProp(k) else None
                smi = prop(cfg.smiles_col) if cfg.smiles_col else Chem.MolToSmiles(mol, isomericSmiles=True)
                rows.append({"source": cfg.canonical_source(), "source_id": str(prop(cfg.id_col) or f"sdf_record_{i}"), "raw_smiles": smi,
                             "name": prop(cfg.name_col), "source_formula": prop(cfg.formula_col),
                             "source_metadata": json.dumps({k: prop(k) for k in cfg.metadata_cols}) if cfg.metadata_cols else None})
            if len(rows) >= cfg.chunksize:
                yield pd.DataFrame(rows, columns=RECORD_COLUMNS)
                rows = []
    if rows:
        yield pd.DataFrame(rows, columns=RECORD_COLUMNS)


def iter_source_records(cfg):
    """Chunks of canonical records from the LOCAL file named by `cfg` (raises SourceNotSupplied if absent)."""
    cfg.validate()
    if not cfg.path or not Path(cfg.path).exists():
        raise SourceNotSupplied(f"{cfg.source}: local file {cfg.path!r} not found. Obtain it MANUALLY (this code never downloads) "
                                f"and set `path` in the source config.")
    fmt = cfg.format.lower().replace(".gz", "")
    it = {"csv": _iter_tabular, "tsv": _iter_tabular, "parquet": _iter_tabular, "smi": _iter_smi, "sdf": _iter_sdf}[fmt](cfg)
    seen = 0
    for chunk in it:
        if cfg.max_records is not None:
            chunk = chunk.iloc[:max(cfg.max_records - seen, 0)]
        if len(chunk):
            seen += len(chunk)
            yield chunk.reset_index(drop=True)
        if cfg.max_records is not None and seen >= cfg.max_records:
            return


def config_record(cfg):
    return {k: v for k, v in asdict(cfg).items()}
