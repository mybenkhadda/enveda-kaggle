"""USER-RUN: build the PubChem KEY-MEMBERSHIP index (PUBCHEM_KEY_MEMBERSHIP_ESTIMATE) from a LOCAL CID -> InChIKey file.

No download, no network, no API: the file must already be on Drive. Expected location (any name works via --input):

    MyDrive/EnvedaCASMI/data/external/pubchem/CID-InChI-Key.gz

Streaming gzip, chunked (never the whole file in RAM), deduplicated per connectivity key (n_records = CID count,
min_cid kept), resumable (per-chunk spool markers). Spool on local scratch (/content/enveda_work, ENVEDA_SCRATCH_ROOT),
final shards + membership_manifest.json on Drive. Progress with rows/s and an ETA (ESTIMATE_ONLY, against
--expected-rows). Defaults resolve from ENVEDA_DRIVE_ROOT, so on Colab after notebook 10 simply:

    python scripts/build_pubchem_membership_index.py

Exit codes: 0 built, 2 PUBCHEM_MEMBERSHIP_SOURCE_MISSING (Rule A stays NOT_EVALUATED), 4 source identity invalid.
After a successful build: add PUBCHEM to validation_v2.external_sources and re-run notebook 11b (the regime identity
changes -> regimes are re-signed), then notebook 13.
"""
import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_MISSING = "PUBCHEM_MEMBERSHIP_SOURCE_MISSING"
SOURCE_INVALID = "PUBCHEM_MEMBERSHIP_SOURCE_INVALID"
DEFAULT_REL_INPUT = "data/external/pubchem/CID-InChI-Key.gz"
DEFAULT_REL_OUT = "candidates/pubchem_membership"


def _drive_root():
    return Path(os.environ.get("ENVEDA_DRIVE_ROOT", "/content/drive/MyDrive/EnvedaCASMI"))


def _scratch_root():
    return Path(os.environ.get("ENVEDA_SCRATCH_ROOT", "/content/enveda_work"))


def validate_source(path):
    """Local-file identity checks (no parsing of the whole file): exists, non-empty, gzip magic for .gz, first line has
    >= 2 tab-separated fields with an InChIKey-shaped last field. Returns (ok, info)."""
    p = Path(path)
    if str(path).lower().startswith(("http://", "https://", "ftp://")):
        return False, {"status": SOURCE_INVALID, "reason": "URLs are refused: download the file manually to Drive"}
    if not p.is_file():
        return False, {"status": SOURCE_MISSING, "expected_path": str(p),
                       "reason": f"upload the PubChem CID->InChIKey file to {p} (manual); Rule A stays NOT_EVALUATED"}
    size = p.stat().st_size
    info = {"status": "OK", "path": str(p), "name": p.name, "size_bytes": size,
            "modified_at": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()}
    if size == 0:
        return False, {**info, "status": SOURCE_INVALID, "reason": "empty file"}
    with open(p, "rb") as f:
        magic = f.read(2)
    if p.suffix == ".gz" and magic != b"\x1f\x8b":
        return False, {**info, "status": SOURCE_INVALID, "reason": ".gz file without gzip magic bytes"}
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(p, "rt", encoding="utf-8", errors="replace") as f:
        first = f.readline().rstrip("\n")
    fields = first.split("\t")
    info["first_line_fields"] = len(fields)
    last = fields[-1] if fields else ""
    if "\t" not in first:      # csv / header layouts: column checks happen in the library against the source config
        info["first_line_check"] = "skipped (not the headerless TSV layout)"
    elif len(fields) < 2 or not (len(last) == 27 and last[14] == "-" and last[25] == "-"):
        return False, {**info, "status": SOURCE_INVALID, "reason": f"first line does not look like CID<TAB>...<TAB>InChIKey: {first[:120]!r}"}
    return True, info


class Progress:
    """Wraps the library's per-chunk log: rows/s and ETA (ESTIMATE_ONLY) against an expected total row count."""

    def __init__(self, expected_rows):
        self.t0, self.expected = time.time(), expected_rows
        self.rows = 0

    def __call__(self, msg):
        import re
        m = re.search(r"total rows ([\d,]+)", msg)
        if m:
            self.rows = int(m.group(1).replace(",", ""))
        dt = max(time.time() - self.t0, 1e-9)
        rate = self.rows / dt
        eta = ((self.expected - self.rows) / rate) if (self.expected and rate > 0 and self.rows < self.expected) else None
        eta_s = f" | ETA ~{eta / 60:.1f} min (ESTIMATE_ONLY vs {self.expected:,} expected rows)" if eta is not None else ""
        print(f"{msg} | {rate:,.0f} rows/s this session{eta_s}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=None, help=f"local CID -> InChIKey file (default <drive>/{DEFAULT_REL_INPUT})")
    ap.add_argument("--out-dir", default=None, help=f"final shard directory on Drive (default <drive>/{DEFAULT_REL_OUT})")
    ap.add_argument("--work-dir", default=None, help="spool directory (default <scratch>/pubchem_membership_spool)")
    ap.add_argument("--config", default=str(REPO / "configs" / "v6" / "pubchem_membership_source.json"))
    ap.add_argument("--chunksize", type=int, default=None)
    ap.add_argument("--expected-rows", type=int, default=None, help="for the ETA only (ESTIMATE_ONLY); unknown -> no ETA")
    a = ap.parse_args(argv)
    src = Path(a.input) if a.input else _drive_root() / DEFAULT_REL_INPUT
    out = Path(a.out_dir) if a.out_dir else _drive_root() / DEFAULT_REL_OUT
    work = Path(a.work_dir) if a.work_dir else _scratch_root() / "pubchem_membership_spool"
    ok, info = validate_source(src)
    if not ok:
        print(json.dumps(info, indent=2))
        return 2 if info["status"] == SOURCE_MISSING else 4
    sys.path.insert(0, str(REPO / "src"))
    from casmi.candidates.pubchem_membership import build_membership_index, load_membership_config
    cfg = load_membership_config(a.config, path=str(src))
    if a.chunksize:
        cfg.chunksize = a.chunksize
    print(json.dumps({"source": info, "out_dir": str(out), "spool": str(work), "chunksize": cfg.chunksize}, indent=2))
    man = build_membership_index(cfg, out, work, log=Progress(a.expected_rows))
    status = {"status": "BUILT", "source": info, **{k: man[k] for k in ("label", "n_input_rows", "n_invalid_inchikeys",
                                                                          "n_unique_connectivity_keys", "identity_tag", "caveat")},
              "next": "add PUBCHEM to validation_v2.external_sources, re-run notebook 11b (re-sign regimes), then notebook 13"}
    (out / "build_status.json").write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    print(json.dumps(status, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
