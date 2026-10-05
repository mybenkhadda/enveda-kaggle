"""Legacy universe ADOPTION: verify a universe finalized before manifest v3 from its persisted artifacts, and record
the result in a SIDECAR manifest. Nothing is rebuilt: Stage A / B / C outputs are read-only here.

Why: the TRAIN ∪ COCONUT universe on Drive was finalized by the pre-v3 code (commit 85d43d9). Its
`universe_manifest.json` has no `external_source_present` / `sources`, so `universe_source_status` says 'unknown' and
the old protocol check reported "COCONUT has no candidates", even though the buckets carry COCONUT provenance.

Two facts are kept SEPARATE and never merged:

    contents_verified                  every invariant below holds on the persisted artifacts (counts, offsets, keys,
                                       mass / formula index sizes, provenance consistency, expected counts)
    reproducible_build_id_available    False for a legacy universe: there is no Stage-B parent identity, so it can NOT
                                       be reproduced from an identity; the adopted build id is a CONTENT digest only

Invariants (any failure -> VERIFICATION_FAILED, adoption refused):
    manifest present                   universe_manifest.json readable
    n_candidates consistent            manifest == bucket_offsets.json == len(candidate_keys.npy) == sum of bucket rows
    offsets contiguous                 bucket b starts where the previous bucket ends (key-sorted bucket order)
    keys strictly sorted               candidate_keys.npy strictly increasing (global candidate ids are well defined)
    bucket keys == candidate_keys      every bucket's connectivity_key column equals its slice of candidate_keys.npy
    mass / formula index sizes         index/ and formula_index/ meta n_candidates == n_candidates
    provenance consistent              in_train == ('TRAIN' in candidate_sources) for every row
    external source present            >= 1 candidate with a non-TRAIN source (else LEGACY_TRAIN_ONLY, refused)
    expected counts (optional)         TRAIN / COCONUT / overlap / union counts given by the caller match exactly

The sidecar `universe_manifest_v3_adopted.json` sits next to the legacy manifest, which is NEVER rewritten. Buckets are
never rewritten. A sidecar with a different adopted build id is refused unless `force=True`; even then the old sidecar
is archived (renamed with a timestamp), never deleted.

Legacy Stage-A discovery (audit only, read-only): the legacy code wrote `universe_work/_done/<SRC>-chunk-NNNNN.json`
markers and outputs under `universe_work/{standardized,rejected,filtered_out}/<SRC>/chunk-NNNNN.parquet`. The modern
`stage_a.active_stage_a_builds` reads only `stage_a/ACTIVE/*.json` pointers, which is why notebook 13 showed ACTIVE={}
and 0 lost records. `discover_stage_a_builds` returns BOTH, normalized.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

LEGACY_ADOPTION_VERSION = "casmi-legacy-adoption-1"
ADOPTED_MANIFEST_NAME = "universe_manifest_v3_adopted.json"
LEGACY_EXTERNAL_VERIFIED = "LEGACY_EXTERNAL_VERIFIED"
LEGACY_TRAIN_ONLY = "LEGACY_TRAIN_ONLY"
MODERN_MANIFEST = "MODERN_MANIFEST"
VERIFICATION_FAILED = "VERIFICATION_FAILED"
UNIVERSE_ABSENT = "UNIVERSE_ABSENT"
LEGACY_MARKER_RE = re.compile(r"^(?P<source>[A-Za-z0-9_]+)-chunk-(?P<chunk>\d{5})\.json$")
LIMITATIONS = [
    "legacy universe: no Stage-B parent identity, so the universe cannot be REPRODUCED from a build identity",
    "adopted_build_id is a CONTENT digest of candidate_keys.npy, bucket_offsets.json and the legacy manifest, not a build recipe id",
    "filter / prefilter / cleanup policies are those of the legacy code (commit 85d43d9 era); they are not re-derivable from this manifest",
    "valid for READ-ONLY coverage measurement and for regime construction; a rebuild (notebook 12) is required before identity-dependent reuse",
]


class LegacyAdoptionRefused(RuntimeError):
    """An adoption invariant failed, the universe is TRAIN-only, or a different adopted sidecar already exists."""


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _file_identity(path):
    p = Path(path)
    if not p.is_file():
        return None
    return {"name": p.name, "size_bytes": p.stat().st_size, "sha256": _sha256_file(p)}


def _write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
    return path


def _as_set(s):
    if isinstance(s, (list, tuple, set, np.ndarray)):
        return {str(x) for x in s}
    return set()


def is_modern_manifest(manifest):
    """Manifest v3 written by `finalize_universe` on this code line (source counts + Stage-B parent)."""
    m = manifest or {}
    return int(m.get("manifest_version") or 1) >= 3 and "parent_stage_b_build_id" in m and "external_source_present" in m


# ---------------------------------------------------------------------------------------------
# provenance scan (pyarrow, one bucket at a time)
# ---------------------------------------------------------------------------------------------

def scan_bucket_provenance(universe_root, candidate_keys=None):
    """One pass over every bucket listed in bucket_offsets.json. Returns counts + per-bucket invariant details."""
    import pyarrow.parquet as pq
    root = Path(universe_root)
    off = _read_json(root / "bucket_offsets.json") or {}
    offsets = off.get("offsets") or {}
    per_source, overlap = {}, {}
    c = {"n_rows": 0, "train": 0, "coconut": 0, "pubchem": 0, "external_any": 0, "external_only": 0, "train_and_external": 0,
         "train_and_coconut": 0, "provenance_contradictions": 0}
    problems, expected_start = [], 0
    for b, start in sorted(offsets.items(), key=lambda kv: int(kv[1])):
        p = root / "buckets" / f"bucket={b}.parquet"
        if not p.is_file():
            problems.append(f"bucket {b}: file missing")
            continue
        if int(start) != expected_start:
            problems.append(f"bucket {b}: offset {start} != expected {expected_start}")
        t = pq.read_table(p, columns=["connectivity_key", "candidate_sources", "in_train"])
        keys = [str(k) for k in t.column("connectivity_key").to_pylist()]
        srcs = [_as_set(s) for s in t.column("candidate_sources").to_pylist()]
        in_train = [bool(x) for x in t.column("in_train").to_pylist()]
        if candidate_keys is not None:
            sl = np.asarray(candidate_keys[int(start):int(start) + len(keys)])
            if len(sl) != len(keys) or any(a.decode("ascii") != k for a, k in zip(sl, keys, strict=False)):
                problems.append(f"bucket {b}: connectivity keys differ from candidate_keys.npy[{start}:{int(start) + len(keys)}]")
        for s, tr in zip(srcs, in_train, strict=True):
            ext = s - {"TRAIN"}
            has_t = "TRAIN" in s
            c["provenance_contradictions"] += int(tr != has_t)
            c["train"] += int(has_t)
            c["coconut"] += int("COCONUT" in s)
            c["pubchem"] += int("PUBCHEM" in s)
            c["external_any"] += int(bool(ext))
            c["external_only"] += int(bool(ext) and not has_t)
            c["train_and_external"] += int(bool(ext) and has_t)
            c["train_and_coconut"] += int(has_t and "COCONUT" in s)
            for x in s:
                per_source[x] = per_source.get(x, 0) + 1
            k = "+".join(sorted(s)) or "none"
            overlap[k] = overlap.get(k, 0) + 1
        c["n_rows"] += len(keys)
        expected_start = int(start) + len(keys)
    return {"counts": c, "per_source": dict(sorted(per_source.items())), "overlap_counts": dict(sorted(overlap.items())),
            "n_buckets": len(offsets), "bucket_problems": problems, "offsets_n_candidates": off.get("n_candidates")}


# ---------------------------------------------------------------------------------------------
# verification + adoption
# ---------------------------------------------------------------------------------------------

def verify_legacy_universe(universe_root, expected_counts=None, work_root=None):
    """Verify a finalized universe from its persisted artifacts. `expected_counts` (optional): any of
    {'n_candidates', 'train', 'coconut', 'train_and_coconut'} that must match EXACTLY. Returns a report with `status`
    (LEGACY_EXTERNAL_VERIFIED | LEGACY_TRAIN_ONLY | MODERN_MANIFEST | VERIFICATION_FAILED | UNIVERSE_ABSENT)."""
    root = Path(universe_root)
    man_path = root / "universe_manifest.json"
    if not man_path.is_file():
        return {"status": UNIVERSE_ABSENT, "universe_root": str(root), "checks": [], "contents_verified": False}
    man = _read_json(man_path)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})

    check("manifest readable", man is not None, man_path.name)
    keys_path = root / "candidate_keys.npy"
    keys = np.load(keys_path, mmap_mode="r") if keys_path.is_file() else None
    check("candidate_keys.npy present", keys is not None)
    scan = scan_bucket_provenance(root, keys)
    cnt = scan["counts"]
    n_man = (man or {}).get("n_candidates")
    n_keys = int(len(keys)) if keys is not None else None
    check("n_candidates: manifest == offsets == keys == bucket rows",
          n_man is not None and n_man == scan["offsets_n_candidates"] == n_keys == cnt["n_rows"],
          f"manifest {n_man}, offsets {scan['offsets_n_candidates']}, keys {n_keys}, bucket rows {cnt['n_rows']}")
    check("bucket offsets contiguous and bucket keys == candidate_keys slices", not scan["bucket_problems"], "; ".join(scan["bucket_problems"][:5]))
    sorted_ok = bool(keys is not None and (len(keys) < 2 or bool(np.all(keys[:-1] < keys[1:]))))
    check("candidate_keys strictly sorted", sorted_ok)
    mi = _read_json(root / "index" / "mass_index.meta.json") or {}
    fi = _read_json(root / "formula_index" / "formula_index.meta.json") or {}
    check("mass index n_candidates == n_candidates", mi.get("n_candidates") == cnt["n_rows"], f"mass index {mi.get('n_candidates')}")
    check("formula index n_candidates == n_candidates", fi.get("n_candidates") == cnt["n_rows"], f"formula index {fi.get('n_candidates')}")
    check("provenance consistent (in_train == TRAIN in candidate_sources)", cnt["provenance_contradictions"] == 0,
          f"{cnt['provenance_contradictions']} contradicting rows")
    observed = {"n_candidates": cnt["n_rows"], "train": cnt["train"], "coconut": cnt["coconut"], "train_and_coconut": cnt["train_and_coconut"]}
    for k, v in (expected_counts or {}).items():
        check(f"expected count {k}", observed.get(k) == int(v), f"observed {observed.get(k)} expected {v}")
    failed = [c for c in checks if not c["passed"]]
    if failed:
        status = VERIFICATION_FAILED
    elif is_modern_manifest(man):
        status = MODERN_MANIFEST
    elif cnt["external_any"] == 0:
        status = LEGACY_TRAIN_ONLY
    else:
        status = LEGACY_EXTERNAL_VERIFIED
    legacy_meta = root / "metadata.json"
    return {"status": status, "universe_root": str(root), "contents_verified": not failed, "checks": checks,
            "counts": observed | {k: cnt[k] for k in ("pubchem", "external_any", "external_only", "train_and_external")},
            "per_source": scan["per_source"], "overlap_counts": scan["overlap_counts"], "n_buckets": scan["n_buckets"],
            "mass_index_count": mi.get("n_rows"), "mass_index_n_candidates": mi.get("n_candidates"),
            "formula_index_count": fi.get("n_indexed"), "formula_index_n_candidates": fi.get("n_candidates"),
            "legacy_manifest": man, "legacy_manifest_identity": _file_identity(man_path),
            "legacy_metadata_identity": _file_identity(legacy_meta), "legacy_config_identity": _legacy_config_identity(legacy_meta),
            "candidate_keys_identity": _file_identity(keys_path), "bucket_offsets_identity": _file_identity(root / "bucket_offsets.json"),
            "legacy_stage_a": summarize_stage_a_discovery(discover_legacy_stage_a(work_root)) if work_root else None,
            "verified_at": datetime.now(timezone.utc).isoformat()}


def _legacy_config_identity(metadata_path):
    """Config identity recorded by the legacy notebook 12 `write_metadata` (config.universe + input paths), if present."""
    m = _read_json(metadata_path) if metadata_path and Path(metadata_path).is_file() else None
    if not m:
        return None
    cfg = m.get("config") or {}
    return {"config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16],
            "created_by_notebook": m.get("created_by_notebook"), "git_commit": m.get("git_commit"), "created_at": m.get("created_at"),
            "input_paths": m.get("input_paths") or m.get("input_fingerprints")}


def adopted_build_id(verification):
    blob = json.dumps({"adoption_version": LEGACY_ADOPTION_VERSION, "counts": verification["counts"],
                       "candidate_keys": (verification.get("candidate_keys_identity") or {}).get("sha256"),
                       "bucket_offsets": (verification.get("bucket_offsets_identity") or {}).get("sha256"),
                       "legacy_manifest": (verification.get("legacy_manifest_identity") or {}).get("sha256")}, sort_keys=True)
    return "adopted-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_adopted_manifest(verification):
    """Sidecar content (manifest v3 vocabulary, explicitly marked adopted)."""
    if verification["status"] != LEGACY_EXTERNAL_VERIFIED:
        raise LegacyAdoptionRefused(f"adoption requires {LEGACY_EXTERNAL_VERIFIED}, got {verification['status']}: "
                                    + "; ".join(f"{c['check']} ({c['detail']})" for c in verification["checks"] if not c["passed"]))
    c, ps = verification["counts"], verification["per_source"]
    bid = adopted_build_id(verification)
    sources = {s: {"present": n > 0, "n_candidates": int(n)} for s, n in ps.items()}
    return {"manifest_version": 3, "adopted_from_legacy": True, "adoption_policy_version": LEGACY_ADOPTION_VERSION,
            "adopted_build_id": bid, "build_id": bid, "parent_stage_b_build_id": None,
            "contents_verified": True, "reproducible_build_id_available": False,
            "n_candidates": c["n_candidates"], "n_buckets": verification["n_buckets"], "source_list": sorted(ps), "sources": sources,
            "overlap_counts": verification["overlap_counts"], "external_source_present": c["external_any"] > 0,
            "n_train_candidates": c["train"], "n_coconut_candidates": c["coconut"], "n_train_and_coconut": c["train_and_coconut"],
            "n_external_any": c["external_any"], "n_external_only": c["external_only"], "n_train_and_external": c["train_and_external"],
            "mass_index_count": verification["mass_index_count"], "formula_index_count": verification["formula_index_count"],
            "legacy_source_identity": {"universe_manifest": verification["legacy_manifest_identity"],
                                       "legacy_metadata": verification["legacy_metadata_identity"],
                                       "legacy_config": verification["legacy_config_identity"],
                                       "candidate_keys": verification["candidate_keys_identity"],
                                       "bucket_offsets": verification["bucket_offsets_identity"],
                                       "legacy_stage_a": verification.get("legacy_stage_a")},
            "verification_checks": verification["checks"], "limitations": LIMITATIONS,
            "adopted_at": datetime.now(timezone.utc).isoformat()}


def adopt_legacy_universe(universe_root, expected_counts=None, work_root=None, write=True, force=False):
    """Verify, then write the sidecar (idempotent: an identical adopted build id is left untouched). Returns
    {'status', 'adopted_manifest', 'path', 'written', 'verification'}. Raises LegacyAdoptionRefused."""
    root = Path(universe_root)
    v = verify_legacy_universe(root, expected_counts=expected_counts, work_root=work_root)
    if v["status"] == MODERN_MANIFEST:
        return {"status": MODERN_MANIFEST, "adopted_manifest": None, "path": None, "written": False, "verification": v}
    adopted = build_adopted_manifest(v)
    path = root / ADOPTED_MANIFEST_NAME
    old = _read_json(path) if path.is_file() else None
    if old and old.get("adopted_build_id") == adopted["adopted_build_id"]:
        return {"status": v["status"], "adopted_manifest": old, "path": str(path), "written": False, "verification": v}
    if old and not force:
        raise LegacyAdoptionRefused(f"{path} exists with adopted_build_id {old.get('adopted_build_id')} != {adopted['adopted_build_id']}: "
                                    "the universe contents changed since adoption. Inspect it; re-adopt with force=True (the old sidecar is archived)")
    if write:
        if old:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path.rename(path.with_name(f"{path.stem}.superseded-{stamp}.json"))
        _write_json_atomic(path, adopted)
    return {"status": v["status"], "adopted_manifest": adopted, "path": str(path), "written": bool(write), "verification": v}


def read_adopted_manifest(universe_root):
    return _read_json(Path(universe_root) / ADOPTED_MANIFEST_NAME)


def effective_universe_identity(universe_root, check_contents=True):
    """The ONE universe identity downstream code binds to:

        kind = modern            manifest v3 written by finalize_universe (build_id = Stage-C id, reproducible)
        kind = legacy_adopted    legacy manifest + a sidecar whose recorded legacy manifest / candidate-key digests still
                                 match the files on disk (build_id = adopted content digest)
        kind = legacy_unadopted  legacy manifest, no sidecar (coverage is read-only; run the adoption script)
        kind = contradiction     sidecar present but the files changed since adoption, or manifest / sidecar disagree
        kind = absent            no finalized universe
    `external_sources` lists the non-TRAIN sources with >= 1 candidate (from the manifest / sidecar, never guessed)."""
    root = Path(universe_root)
    man = _read_json(root / "universe_manifest.json")
    if man is None:
        return {"kind": "absent", "build_id": None, "external_sources": [], "universe_status": "absent", "problems": []}
    if is_modern_manifest(man):
        ext = sorted(s for s, v in (man.get("sources") or {}).items() if s != "TRAIN" and (v or {}).get("present"))
        return {"kind": "modern", "build_id": man.get("build_id"), "external_sources": ext, "n_candidates": man.get("n_candidates"),
                "universe_status": "external" if man.get("external_source_present") else "train_only", "sources": man.get("sources"),
                "contents_verified": None, "reproducible_build_id_available": True, "problems": []}
    side = read_adopted_manifest(root)
    if side is None:
        return {"kind": "legacy_unadopted", "build_id": None, "external_sources": [], "n_candidates": man.get("n_candidates"),
                "universe_status": "unknown", "contents_verified": False, "reproducible_build_id_available": False,
                "problems": ["legacy manifest without adoption sidecar: run scripts/adopt_legacy_universe_manifest.py (or notebook 11a)"]}
    problems = []
    lsi = side.get("legacy_source_identity") or {}
    if check_contents:
        for name, fname in (("universe_manifest", "universe_manifest.json"), ("candidate_keys", "candidate_keys.npy"),
                            ("bucket_offsets", "bucket_offsets.json")):
            rec = lsi.get(name) or {}
            cur = _file_identity(root / fname)
            if not cur or cur.get("sha256") != rec.get("sha256"):
                problems.append(f"{fname} changed since adoption (recorded {str(rec.get('sha256'))[:12]}, now {str((cur or {}).get('sha256'))[:12]})")
    if side.get("n_candidates") != man.get("n_candidates"):
        problems.append(f"sidecar n_candidates {side.get('n_candidates')} != legacy manifest {man.get('n_candidates')}")
    ext = sorted(s for s, v in (side.get("sources") or {}).items() if s != "TRAIN" and (v or {}).get("present"))
    if bool(ext) != bool(side.get("external_source_present")):
        problems.append("sidecar external_source_present contradicts its per-source counts")
    return {"kind": "contradiction" if problems else "legacy_adopted", "build_id": side.get("adopted_build_id"), "external_sources": ext,
            "n_candidates": side.get("n_candidates"), "universe_status": ("external" if ext else "train_only") if not problems else "contradiction",
            "sources": side.get("sources"), "contents_verified": not problems and bool(side.get("contents_verified")),
            "reproducible_build_id_available": False, "problems": problems, "limitations": side.get("limitations")}


# ---------------------------------------------------------------------------------------------
# legacy Stage-A discovery (audit only)
# ---------------------------------------------------------------------------------------------

def discover_legacy_stage_a(work_root):
    """Legacy `universe_work/_done/<SRC>-chunk-NNNNN.json` markers -> one normalized record per source:
    source, status, legacy=True, chunk_ids, n_chunks, n_input / n_prefiltered / n_rejected / n_filtered / n_kept, namespace,
    paths, missing_outputs, inferred_identity. `status`: complete_contiguous (chunk ids 0..n-1, every output present) |
    gaps (non-contiguous chunk ids) | outputs_missing. The total chunk count of the source file is NOT known here."""
    ws = Path(work_root) if work_root else None
    done = ws / "_done" if ws else None
    if done is None or not done.is_dir():
        return {}
    by_src = {}
    for f in sorted(done.glob("*-chunk-*.json")):
        m = LEGACY_MARKER_RE.match(f.name)
        if not m:
            continue
        rec = _read_json(f) or {}
        by_src.setdefault(m.group("source"), []).append((int(m.group("chunk")), rec, f.name))
    out = {}
    for src, items in sorted(by_src.items()):
        items.sort(key=lambda t: t[0])
        ids = [i for i, _, _ in items]
        tot = {k: int(sum(int(r.get(k) or 0) for _, r, _ in items)) for k in ("n_input", "n_prefiltered", "n_rejected", "n_filtered", "n_kept")}
        paths = {k: str(ws / k / src) for k in ("standardized", "rejected", "filtered_out")}
        missing = [f"{k}/{src}/chunk-{i:05d}.parquet" for i in ids for k in ("standardized", "rejected", "filtered_out")
                   if not (ws / k / src / f"chunk-{i:05d}.parquet").is_file()]
        contiguous = ids == list(range(len(ids)))
        status = "outputs_missing" if missing else ("complete_contiguous" if contiguous else "gaps")
        ident = hashlib.sha256(json.dumps({"source": src, "chunks": ids, "totals": tot, "markers": [n for _, _, n in items]},
                                          sort_keys=True).encode("utf-8")).hexdigest()[:16]
        out[src] = {"source": src, "status": status, "legacy": True, "build_id": f"legacy-{ident}", "chunk_ids": ids, "n_chunks": len(ids),
                    **tot, "namespace": str(ws), "paths": paths, "missing_outputs": missing[:20], "n_missing_outputs": len(missing),
                    "inferred_identity": {"kind": "INFERRED_FROM_MARKERS", "digest": ident,
                                          "note": "not a reproducible Stage-A build identity (legacy markers carry no source/config identity)"},
                    "first_finished_at": min((r.get("finished_at") for _, r, _ in items if r.get("finished_at")), default=None),
                    "last_finished_at": max((r.get("finished_at") for _, r, _ in items if r.get("finished_at")), default=None)}
    return out


def discover_stage_a_builds(work_root):
    """Modern ACTIVE pointers (stage_a/ACTIVE/*.json) + legacy markers, normalized: {SRC: [records]} with `legacy` flag."""
    from casmi.candidates.stage_a import active_stage_a_builds
    out = {}
    for src, v in active_stage_a_builds(work_root).items():
        man = v.get("manifest") or {}
        out.setdefault(src, []).append({"source": src, "status": v["status"], "legacy": False, "build_id": v.get("build_id"),
                                        "namespace": v.get("namespace"), "n_chunks": man.get("n_chunks"), "manifest": man or None})
    for src, rec in discover_legacy_stage_a(work_root).items():
        out.setdefault(src, []).append(rec)
    return out


def summarize_stage_a_discovery(legacy_records):
    return {s: {k: r.get(k) for k in ("status", "build_id", "n_chunks", "n_input", "n_prefiltered", "n_rejected", "n_filtered", "n_kept",
                                       "n_missing_outputs")} for s, r in (legacy_records or {}).items()}


def select_stage_a_namespaces(discovered, universe_identity, sources=None):
    """Stage-A namespaces whose dropped records belong to the FINALIZED universe in use:
    a legacy / adopted universe -> legacy records (status complete_contiguous or gaps, reported); a modern universe ->
    modern complete builds. Returns (namespaces, selection rows) -- the selection is always reported, never silent."""
    legacy_universe = universe_identity.get("kind") in ("legacy_adopted", "legacy_unadopted")
    rows, ns = [], []
    for src, recs in sorted(discovered.items()):
        if sources is not None and src not in sources:
            continue
        for r in recs:
            use = (r["legacy"] and legacy_universe and r["status"] in ("complete_contiguous", "gaps")) or \
                  (not r["legacy"] and not legacy_universe and r["status"] == "complete")
            rows.append({"source": src, "legacy": r["legacy"], "status": r["status"], "build_id": r.get("build_id"),
                         "namespace": r.get("namespace"), "selected": bool(use)})
            if use and r.get("namespace") and r["namespace"] not in ns:
                ns.append(r["namespace"])
    return ns, rows
