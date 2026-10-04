"""PubChem KEY-MEMBERSHIP index from a LOCAL, user-supplied `CID -> InChIKey` file (no download, no network).

Purpose: estimate how many held-out truths PubChem COULD reach before any PubChem structure is standardized.
The user obtains the file manually (e.g. the PubChem `CID-InChI-Key` export) and places it under
`MyDrive/EnvedaCASMI/data/external/pubchem/`; `configs/v6/pubchem_membership_source.json` names its columns.

    connectivity_key := first InChIKey block (14 characters) of PubChem's stored InChIKey

IMPORTANT -- this is PUBCHEM_KEY_MEMBERSHIP_ESTIMATE, not canonical parity: PubChem computes its InChIKey from its own
stored structure (its own tautomer / charge / salt representation); the competition key is the RDKit
tautomer-canonical InChIKey14 of OUR canonicalizer. The two first blocks agree for most structures but are NOT
guaranteed to: membership here is an estimate, labelled as such everywhere.

Index layout (`out_dir/`): shards `shards/prefix=<P>.parquet` (P = first `shard_prefix_len` key characters), each SORTED by
`connectivity_key` with columns `connectivity_key`, `n_records` (PubChem rows with that first block), `min_cid`;
`membership_manifest.json` with counts, the source file name + byte size and the config (metadata identity, no
content hash). Building is chunked (never the whole file in RAM) and resumable through per-chunk spool markers.
"""
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MEMBERSHIP_INDEX_VERSION = "casmi-pubchem-membership-1"
MEMBERSHIP_LABEL = "PUBCHEM_KEY_MEMBERSHIP_ESTIMATE"
INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
FORBIDDEN_URL_PREFIXES = ("http://", "https://", "ftp://", "s3://", "gs://")


class MembershipConfigError(ValueError):
    pass


@dataclass
class PubChemMembershipConfig:
    path: str = ""
    format: str = "tsv"                 # tsv | csv
    compression: str | None = "infer"   # infer | gzip | None
    cid_col: str = "cid"
    inchikey_col: str = "inchikey"
    delimiter: str | None = None        # overrides the format default
    header: bool = False                # PubChem's CID-InChI-Key export has NO header row
    column_names: list = field(default_factory=lambda: ["cid", "inchi", "inchikey"])   # used when header=False
    chunksize: int = 2_000_000
    shard_prefix_len: int = 1
    notes: str = ""

    def validate(self):
        if any(str(self.path).lower().startswith(p) for p in FORBIDDEN_URL_PREFIXES):
            raise MembershipConfigError("PubChem membership reads LOCAL files only -- download the file manually; URLs are refused")
        if self.format not in ("tsv", "csv"):
            raise MembershipConfigError(f"format must be tsv or csv, got {self.format!r}")
        if self.compression not in ("infer", "gzip", None):
            raise MembershipConfigError("compression must be infer, gzip or null")
        if not self.header and (self.cid_col not in self.column_names or self.inchikey_col not in self.column_names):
            raise MembershipConfigError(f"header=False: cid_col / inchikey_col must be among column_names {self.column_names}")
        if not 1 <= int(self.shard_prefix_len) <= 2:
            raise MembershipConfigError("shard_prefix_len must be 1 or 2")
        return self

    def identity(self):
        p = Path(self.path)
        d = {k: v for k, v in asdict(self).items() if k not in ("path", "notes", "chunksize")}
        d.update(source_file_name=p.name, source_size_bytes=p.stat().st_size if p.is_file() else None)
        return d


def load_membership_config(json_path, path=None):
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    cfg = PubChemMembershipConfig(**{k: v for k, v in raw.items() if k in PubChemMembershipConfig.__dataclass_fields__})
    if path is not None:
        cfg.path = str(path)
    return cfg.validate()


def connectivity_from_inchikey(inchikeys):
    """First InChIKey block for well-formed keys (27 chars, 14-10-1 upper case), <NA> otherwise."""
    from casmi.utils.booleans import as_bool
    s = pd.Series(inchikeys, dtype="object").astype("string").str.strip()
    ok = as_bool(s.str.fullmatch(INCHIKEY_RE.pattern), missing=False)
    return s.str.slice(0, 14).where(ok.to_numpy(), pd.NA)


def iter_membership_chunks(cfg):
    """Chunks of (cid, inchikey) as strings from the local file. Never reads the whole file."""
    cfg.validate()
    p = Path(cfg.path)
    if not p.is_file():
        raise FileNotFoundError(f"PubChem membership file not found: {p} (obtain it manually; this code never downloads)")
    sep = cfg.delimiter or ("\t" if cfg.format == "tsv" else ",")
    kw = dict(sep=sep, dtype=str, chunksize=int(cfg.chunksize), compression=cfg.compression, keep_default_na=False,
              usecols=[cfg.cid_col, cfg.inchikey_col], quoting=3 if cfg.format == "tsv" else 0)
    if not cfg.header:
        kw.update(header=None, names=list(cfg.column_names))
    for df in pd.read_csv(p, **kw):
        yield df.rename(columns={cfg.cid_col: "cid", cfg.inchikey_col: "inchikey"})[["cid", "inchikey"]]


def _cid_int(cid):
    return pd.to_numeric(pd.Series(cid), errors="coerce").astype("Int64")


def aggregate_chunk(df, shard_prefix_len=1):
    """(cid, inchikey) chunk -> one row per connectivity_key with n_records, min_cid, shard; plus the invalid count."""
    keys = connectivity_from_inchikey(df["inchikey"])
    ok = keys.notna()
    d = pd.DataFrame({"connectivity_key": keys[ok].astype(str).to_numpy(), "cid": _cid_int(df.loc[ok.to_numpy(), "cid"]).to_numpy()})
    g = d.groupby("connectivity_key", sort=True).agg(n_records=("cid", "size"), min_cid=("cid", "min")).reset_index()
    g["shard"] = g["connectivity_key"].str.slice(0, int(shard_prefix_len))
    return g, int((~ok).sum())


def build_membership_index(cfg, out_dir, work_dir=None, log=print):
    """Chunked, resumable build. Spool parts (+ per-chunk done markers) go to `work_dir` (default `out_dir/_spool`; a local
    scratch dir is recommended); final sorted shards + manifest go to `out_dir`. Returns the manifest."""
    cfg.validate()
    out_dir = Path(out_dir)
    spool = Path(work_dir) if work_dir else out_dir / "_spool"
    ident = cfg.identity()
    ident_tag = hashlib_tag(ident)
    spool = spool / ident_tag
    spool.mkdir(parents=True, exist_ok=True)
    n_rows = n_invalid = 0
    for i, chunk in enumerate(iter_membership_chunks(cfg)):
        marker = spool / "_done" / f"chunk-{i:05d}.json"
        if marker.is_file():
            rec = json.loads(marker.read_text(encoding="utf-8"))
            n_rows += rec["n_rows"]
            n_invalid += rec["n_invalid"]
            continue
        g, bad = aggregate_chunk(chunk, cfg.shard_prefix_len)
        for shard, part in g.groupby("shard", sort=True):
            f = spool / f"prefix={shard}" / f"part-{i:05d}.parquet"
            f.parent.mkdir(parents=True, exist_ok=True)
            part.drop(columns="shard").to_parquet(f, index=False)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({"chunk": i, "n_rows": int(len(chunk)), "n_invalid": bad}), encoding="utf-8")   # after the parts
        n_rows += len(chunk)
        n_invalid += bad
        if log:
            log(f"[pubchem-membership] chunk {i}: {len(chunk):,} rows | invalid keys {bad:,} | total rows {n_rows:,}")
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    n_unique, shard_counts = 0, {}
    for sd in sorted(p for p in spool.glob("prefix=*") if p.is_dir()):
        parts = [pd.read_parquet(f) for f in sorted(sd.glob("part-*.parquet"))]
        if not parts:
            continue
        a = pd.concat(parts, ignore_index=True)
        a = (a.groupby("connectivity_key", sort=True).agg(n_records=("n_records", "sum"), min_cid=("min_cid", "min")).reset_index())
        tmp = shards_dir / f"{sd.name}.parquet.tmp"
        a.to_parquet(tmp, index=False)
        tmp.replace(shards_dir / f"{sd.name}.parquet")
        n_unique += len(a)
        shard_counts[sd.name.split("=", 1)[1]] = int(len(a))
    manifest = {"membership_index_version": MEMBERSHIP_INDEX_VERSION, "label": MEMBERSHIP_LABEL, "identity": ident,
                "identity_tag": ident_tag, "n_input_rows": int(n_rows), "n_invalid_inchikeys": int(n_invalid),
                "n_unique_connectivity_keys": int(n_unique), "shards": shard_counts,
                "caveat": "first InChIKey block of PubChem's stored structure -- not guaranteed equal to the competition RDKit "
                          "tautomer-canonical key (estimate only)",
                "built_at": datetime.now(timezone.utc).isoformat()}
    (out_dir / "membership_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def hashlib_tag(obj):
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


class PubChemMembershipIndex:
    """Batch membership over the sharded index; only the shards touched by a query batch are loaded (cached)."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        mp = self.out_dir / "membership_manifest.json"
        if not mp.is_file():
            raise FileNotFoundError(f"no PubChem membership index at {self.out_dir} -- run scripts/build_pubchem_membership_index.py")
        self.manifest = json.loads(mp.read_text(encoding="utf-8"))
        self.prefix_len = int(self.manifest["identity"].get("shard_prefix_len", 1))
        self._cache = {}

    @staticmethod
    def available(out_dir):
        return out_dir is not None and (Path(out_dir) / "membership_manifest.json").is_file()

    def _shard(self, prefix):
        if prefix not in self._cache:
            f = self.out_dir / "shards" / f"prefix={prefix}.parquet"
            if f.is_file():
                t = pd.read_parquet(f, columns=["connectivity_key", "n_records", "min_cid"])
                self._cache[prefix] = (t["connectivity_key"].astype(str).to_numpy(), t["n_records"].to_numpy(), t["min_cid"].to_numpy())
            else:
                self._cache[prefix] = (np.zeros(0, dtype=object), np.zeros(0, np.int64), np.zeros(0, object))
        return self._cache[prefix]

    def lookup(self, keys):
        """DataFrame(connectivity_key, in_pubchem, pubchem_n_records, pubchem_min_cid), same order as `keys`."""
        k = pd.Series(keys, dtype="object").astype(str).to_numpy()
        found = np.zeros(len(k), bool)
        nrec = np.zeros(len(k), np.int64)
        mcid = np.full(len(k), None, dtype=object)
        prefixes = np.array([s[: self.prefix_len] for s in k], dtype=object)
        for pre in pd.unique(prefixes):
            sel = np.flatnonzero(prefixes == pre)
            sk, sn, sc = self._shard(pre)
            if not len(sk):
                continue
            pos = np.clip(np.searchsorted(sk, k[sel]), 0, len(sk) - 1)
            hit = sk[pos] == k[sel]
            found[sel[hit]] = True
            nrec[sel[hit]] = sn[pos[hit]]
            mcid[sel[hit]] = sc[pos[hit]]
        return pd.DataFrame({"connectivity_key": k, "in_pubchem": found, "pubchem_n_records": nrec, "pubchem_min_cid": mcid})

    def contains(self, keys):
        return self.lookup(keys)["in_pubchem"].to_numpy(bool)


def membership_summary(index_or_none, truth_keys=None):
    """JSON-able summary for notebook 13 (`pubchem_membership_summary.json`). With no index: status NOT_AVAILABLE."""
    if index_or_none is None:
        return {"status": "NOT_AVAILABLE", "label": MEMBERSHIP_LABEL,
                "reason": "no local PubChem CID->InChIKey membership index (build it with scripts/build_pubchem_membership_index.py)"}
    out = {"status": "AVAILABLE", "label": MEMBERSHIP_LABEL, **{k: index_or_none.manifest.get(k) for k in (
        "membership_index_version", "n_input_rows", "n_invalid_inchikeys", "n_unique_connectivity_keys", "caveat", "built_at")},
           "source_file_name": index_or_none.manifest["identity"].get("source_file_name"),
           "source_size_bytes": index_or_none.manifest["identity"].get("source_size_bytes")}
    if truth_keys is not None:
        hit = index_or_none.contains(truth_keys)
        out.update(n_truth_keys=int(len(hit)), n_truth_in_pubchem=int(hit.sum()), truth_in_pubchem_rate=float(hit.mean()) if len(hit) else None)
    return out
