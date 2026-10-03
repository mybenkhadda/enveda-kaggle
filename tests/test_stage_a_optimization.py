"""Stage-A optimization: scientific parity with the legacy standardizer + resume / cache identity + Stage-B/C
invalidation + I/O helpers. RDKit-dependent tests are skipped when RDKit is absent; the rest use a deterministic fake
worker (`row_fn`) so the orchestration is tested without chemistry."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from casmi.candidates.filters import UniverseFilterConfig, apply_filters
from casmi.candidates.sources import SourceConfigError, estimate_record_count, iter_source_records, load_source_config
from casmi.candidates.stage_a import (GPU_STAGE_A_POLICY, IDENTITY_COLUMNS, UNIVERSE_STRUCTURE_FIELDS, StageAIncompleteError, StageAPerformance,
                                      StageAProgress, StandardizationCache, StructureEngine, SourceSpec, build_id_of, identity_mismatches,
                                      resolve_stage_a_inputs, run_stage_a, stage_a_identity, standardize_records_fast)
from casmi.candidates.universe import (StageBIncompleteError, UniverseBuildConfig, _write_bucketed, finalize_universe, formula_prefilter,
                                       merge_bucket, run_stage_b, universe_source_status)
from casmi.workspace.parallel import resolve_n_jobs
from casmi.workspace.staging import StagingError, files_intact, persist_files, stage_external_file

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "configs" / "v6" / "coconut_source.json"


# ---------------------------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------------------------

def fake_row(smiles):
    """Deterministic stand-in for the RDKit worker (universe_minimal fields)."""
    if smiles.startswith("bad"):
        row = {k: None for k in UNIVERSE_STRUCTURE_FIELDS}
        row.update(smiles=smiles, parse_ok=False, error="parse_failed")
        return row
    key = hashlib.sha256(smiles.encode("utf-8")).hexdigest().upper()[:14]
    return {"smiles": smiles, "parse_ok": True, "error": None, "canonical_smiles": smiles, "plain_inchikey14": key, "connectivity_key": key,
            "tautomer_hit_cap": False, "exact_mass": 300.0 + len(smiles), "molecular_weight": 300.5 + len(smiles), "formal_charge": 0,
            "molecular_formula": "C10H12O2"}


class CountingRow:
    def __init__(self):
        self.calls = 0

    def __call__(self, smiles):
        self.calls += 1
        return fake_row(smiles)


def write_coconut_csv(path, rows):
    pd.DataFrame(rows, columns=["identifier", "canonical_smiles", "name", "molecular_formula", "unused_wide_column"]).to_csv(path, index=False)
    return path


ROWS = [("CNP0000001", "CCOCCO", "a", "C10H12O2", "x"), ("CNP0000002", "CCOCCO", "dup smiles, other id", "C10H12O2", "x"),
        ("CNP0000003", "CCCCOO", "b", "C10H12O2", "x"), ("CNP0000004", "bad_smiles", "c", "C10H12O2", "x"),
        ("CNP0000005", "CCNCCO", "d", "C1H1", "x"),                       # prefiltered (mass out of range)
        ("CNP0000006", "OCCOCC", "e", None, "x")]


def source_cfg(path):
    cfg = load_source_config(TEMPLATE)
    cfg.path = str(path)
    return cfg


def ucfg(chunk=2):
    return UniverseBuildConfig(chunk_records=chunk, n_jobs=1, filters=UniverseFilterConfig())


def perf(**kw):
    return StageAPerformance(**{"cache_enabled": False, "telemetry": False, "progress_every_records": 0, **kw})


# ---------------------------------------------------------------------------------------------------------------
# build identity
# ---------------------------------------------------------------------------------------------------------------

def test_build_identity_tracks_everything_that_changes_outputs(tmp_path):
    src = write_coconut_csv(tmp_path / "coconut.csv", ROWS)
    base = build_id_of(stage_a_identity(source_cfg(src), src, ucfg(100), perf()))
    assert base == build_id_of(stage_a_identity(source_cfg(src), src, ucfg(100), perf()))
    assert base != build_id_of(stage_a_identity(source_cfg(src), src, ucfg(50), perf()))                                   # chunk size
    assert base != build_id_of(stage_a_identity(source_cfg(src), src, UniverseBuildConfig(chunk_records=100, filters=UniverseFilterConfig(min_exact_mass=100)), perf()))
    assert base != build_id_of(stage_a_identity(source_cfg(src), src, ucfg(100), perf(standardization_profile="full")))   # profile
    assert base != build_id_of(stage_a_identity(source_cfg(src), src, ucfg(100), perf(compute_tautomer_hit_cap=False)))  # diagnostic column
    c = source_cfg(src)
    c.formula_col = None
    assert base != build_id_of(stage_a_identity(c, src, ucfg(100), perf()))                                                # column mapping
    # performance-only knobs never change the identity
    assert base == build_id_of(stage_a_identity(source_cfg(src), src, ucfg(100), perf(standardize_batch_records=7, cache_enabled=True)))
    u = ucfg(100)
    u.n_jobs = 8
    assert base == build_id_of(stage_a_identity(source_cfg(src), src, u, perf()))
    write_coconut_csv(src, ROWS + [("CNP9", "CCCC", "z", "C4H10", "x")])                                                  # source file changed
    assert base != build_id_of(stage_a_identity(source_cfg(src), src, ucfg(100), perf()))


def test_canonicalizer_version_is_part_of_the_identity(tmp_path, monkeypatch):
    import casmi.candidates.stage_a as sa
    src = write_coconut_csv(tmp_path / "coconut.csv", ROWS)
    a = build_id_of(stage_a_identity(source_cfg(src), src, ucfg(), perf()))
    monkeypatch.setattr(sa, "STANDARDIZATION_CONTRACT_VERSION", "casmi-std-contract-999")
    assert a != build_id_of(stage_a_identity(source_cfg(src), src, ucfg(), perf()))


# ---------------------------------------------------------------------------------------------------------------
# staging / persistence
# ---------------------------------------------------------------------------------------------------------------

def test_stage_external_file_copies_once_and_recopies_on_size_change(tmp_path):
    src = tmp_path / "drive" / "coconut.csv"
    src.parent.mkdir()
    src.write_text("a,b\n1,2\n")
    s1 = stage_external_file(src, tmp_path / "scratch", "COCONUT", min_free_gb=0, log=None)
    assert s1.action == "copied" and Path(s1.local_path).read_text() == "a,b\n1,2\n"
    assert stage_external_file(src, tmp_path / "scratch", "COCONUT", min_free_gb=0, log=None).action == "reused"
    src.write_text("a,b\n1,2\n3,4\n")
    s3 = stage_external_file(src, tmp_path / "scratch", "COCONUT", min_free_gb=0, log=None)
    assert s3.action == "copied" and Path(s3.local_path).stat().st_size == src.stat().st_size
    assert src.read_text() == "a,b\n1,2\n3,4\n"                                    # the original is never modified
    with pytest.raises(StagingError):
        stage_external_file(tmp_path / "missing.csv", tmp_path / "scratch", log=None)


def test_persist_files_verifies_size(tmp_path):
    (tmp_path / "local").mkdir()
    a = tmp_path / "local" / "a.parquet"
    a.write_bytes(b"12345")
    out = persist_files([(a, tmp_path / "drive" / "x" / "a.parquet")])
    assert out == {str(tmp_path / "drive" / "x" / "a.parquet"): 5}
    assert files_intact(out) and not files_intact({str(tmp_path / "drive" / "x" / "a.parquet"): 6})


# ---------------------------------------------------------------------------------------------------------------
# Stage A orchestration (fake worker)
# ---------------------------------------------------------------------------------------------------------------

def test_run_stage_a_accounting_dedup_and_provenance(tmp_path):
    src = write_coconut_csv(tmp_path / "coconut.csv", ROWS)
    row = CountingRow()
    res = run_stage_a(source_cfg(src), tmp_path / "work", tmp_path / "scratch", ucfg(100), perf(), log=None, row_fn=row)
    rec = res.summary.iloc[0]
    assert rec["n_input"] == 6 and rec["n_prefiltered"] == 1 and rec["n_rejected"] == 1
    assert rec["n_unique_raw_smiles"] == 4 and rec["n_duplicate_raw_smiles_saved"] == 1 and rec["n_standardizer_calls"] == 4 == row.calls
    std = pd.read_parquet(Path(res.namespace) / "standardized" / "COCONUT" / "chunk-00000.parquet")
    assert set(std["source_id"]) == {"CNP0000001", "CNP0000002", "CNP0000003", "CNP0000006"}   # both ids of the duplicated SMILES kept
    names = dict(zip(std["source_id"], std["name"], strict=True))
    assert names["CNP0000002"] == "dup smiles, other id"
    assert (std["connectivity_key"].str[:1] == std["bucket"]).all()
    assert res.manifest["complete"] and res.manifest["n_input"] == 6


def test_marker_written_only_after_persist(tmp_path, monkeypatch):
    import casmi.workspace.staging as staging
    src = write_coconut_csv(tmp_path / "coconut.csv", ROWS)

    def boom(pairs, log=None):
        raise OSError("drive unavailable")
    monkeypatch.setattr(staging, "persist_files", boom)
    with pytest.raises(OSError):
        run_stage_a(source_cfg(src), tmp_path / "work", tmp_path / "scratch", ucfg(100), perf(), log=None, row_fn=fake_row)
    assert not list((tmp_path / "work").rglob("_done/*/chunk-*.json"))


def test_resume_only_for_identical_build(tmp_path):
    src = write_coconut_csv(tmp_path / "coconut.csv", ROWS)
    r1 = CountingRow()
    a = run_stage_a(source_cfg(src), tmp_path / "work", tmp_path / "scratch", ucfg(2), perf(), log=None, row_fn=r1)
    assert a.manifest["n_chunks"] == 3 and [c["row_start"] for c in a.manifest["chunks"]] == [0, 2, 4]
    r2 = CountingRow()
    b = run_stage_a(source_cfg(src), tmp_path / "work", tmp_path / "scratch", ucfg(2), perf(), log=None, row_fn=r2)
    assert b.build_id == a.build_id and r2.calls == 0                              # every chunk resumed
    r3 = CountingRow()
    c = run_stage_a(source_cfg(src), tmp_path / "work", tmp_path / "scratch", ucfg(3), perf(), log=None, row_fn=r3)
    assert c.build_id != a.build_id and r3.calls > 0                               # new chunk size -> new build, recomputed
    (Path(a.namespace) / "standardized" / "COCONUT" / "chunk-00001.parquet").write_bytes(b"truncated")
    r4 = CountingRow()
    run_stage_a(source_cfg(src), tmp_path / "work", tmp_path / "scratch", ucfg(2), perf(), log=None, row_fn=r4)
    assert r4.calls > 0                                                            # damaged output -> marker not trusted


def test_resolve_inputs_requires_complete_build_for_the_current_file(tmp_path):
    src = write_coconut_csv(tmp_path / "coconut.csv", ROWS)
    run_stage_a(source_cfg(src), tmp_path / "work", tmp_path / "scratch", ucfg(100), perf(), log=None, row_fn=fake_row)
    spec = SourceSpec("COCONUT", str(TEMPLATE), str(src), str(src), True, source_cfg(src))
    inputs = resolve_stage_a_inputs(tmp_path / "work", [spec], ucfg(100), perf())
    assert inputs["sources"]["COCONUT"]["status"] == "complete" and len(inputs["std_files"]) == 1
    missing = SourceSpec("PUBCHEM", "t", str(tmp_path / "nope.csv"), str(tmp_path / "nope.csv"), False, None)
    assert resolve_stage_a_inputs(tmp_path / "work", [spec, missing], ucfg(100), perf())["sources"]["PUBCHEM"]["status"] == "not_provided"
    write_coconut_csv(src, ROWS + [("CNP9", "CCCC", "z", "C4H10", "x")])          # the Drive file changed after Stage A
    with pytest.raises(StageAIncompleteError):
        resolve_stage_a_inputs(tmp_path / "work", [spec], ucfg(100), perf())


def test_cache_avoids_recomputation_and_is_contract_scoped(tmp_path):
    cache = StandardizationCache(tmp_path / "c.sqlite", "contract-A", log=None)
    row = CountingRow()
    eng = StructureEngine("universe_minimal", 1, 2, cache=cache, row_fn=row)
    t1 = eng.structure_table(["CCO", "CCN", "CCC"])
    t2 = eng.structure_table(["CCC", "CCO", "OCC"])
    assert row.calls == 4 and eng.last["n_cache_hits"] == 2
    assert t2.set_index("smiles").loc["CCO"].equals(t1.set_index("smiles").loc["CCO"])
    other = StandardizationCache(tmp_path / "c.sqlite", "contract-B", log=None)
    assert other.get_many(["CCO"]) == {}
    cache.close(); other.close()


def test_standardize_fast_output_order_and_batching_are_deterministic():
    recs = pd.DataFrame({"source": "COCONUT", "source_id": [f"id{i}" for i in range(9)],
                         "raw_smiles": ["CCO", "CCN", "CCO", None, "bad1", "CCC", "CCN", "OCC", ""]})
    out = []
    for batch in (1, 2, 50):
        std, rej, stats = standardize_records_fast(recs, StructureEngine("universe_minimal", 1, batch, row_fn=fake_row))
        out.append((std, rej))
        assert stats["n_unique_raw_smiles"] == 5 and stats["n_blank_smiles"] == 2 and len(std) + len(rej) == 9
    for std, rej in out[1:]:
        assert identity_mismatches(out[0][0], std).empty and identity_mismatches(out[0][1], rej, ["source_id", "failure_reason"]).empty


# ---------------------------------------------------------------------------------------------------------------
# scientific parity (RDKit)
# ---------------------------------------------------------------------------------------------------------------

PARITY_SMILES = ["Oc1ccccn1", "O=c1cccc[nH]1",                    # tautomers
                 "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O",            # stereo
                 "C[N+](C)(C)CC(=O)O", "CC(=O)[O-].[Na+]",         # charged / salt (multi-component)
                 "[13CH3]C(=O)O", "[2H]OC(=O)c1ccccc1",            # isotopes
                 "Clc1ccc(Br)cc1I", "OP(=O)(O)OCC(O)CO", "CS(=O)(=O)Nc1ccccc1", "C[Si](C)(C)OC(=O)CCCCCCCCC",
                 "OB(O)c1ccc(cc1)C(=O)O", "C[Se]CC(N)C(=O)O", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                 "not_a_smiles", "C1CC", "Oc1ccccn1"]                # invalid x2 + duplicate raw SMILES (other id)


def _parity_records():
    return pd.DataFrame({"source": "COCONUT", "source_id": [f"p{i}" for i in range(len(PARITY_SMILES))], "raw_smiles": PARITY_SMILES,
                         "name": [f"n{i}" for i in range(len(PARITY_SMILES))], "source_formula": None, "source_metadata": None})


@pytest.mark.parametrize("profile", ["full", "universe_minimal"])
def test_optimized_paths_are_identical_to_legacy(profile):
    pytest.importorskip("rdkit")
    from casmi.candidates.standardize import standardize_records
    recs = _parity_records()
    std_l, rej_l = standardize_records(recs)
    std_f, rej_f, stats = standardize_records_fast(recs, StructureEngine(profile, 1, 3))
    cols = IDENTITY_COLUMNS + ["tautomer_hit_cap"]
    assert identity_mismatches(std_l, std_f, cols).empty
    assert identity_mismatches(rej_l, rej_f, ["source", "source_id", "raw_smiles", "failure_reason"]).empty
    if profile == "full":
        assert identity_mismatches(std_l, std_f, list(std_l.columns)).empty         # every legacy column, incl. EDA descriptors
    kl, fl = apply_filters(std_l, UniverseFilterConfig())
    kf, ff = apply_filters(std_f, UniverseFilterConfig())
    assert kl["source_id"].tolist() == kf["source_id"].tolist() and fl["filter_reason"].tolist() == ff["filter_reason"].tolist()
    assert stats["n_duplicate_raw_smiles_saved"] == 1


def test_skipping_the_hit_cap_diagnostic_never_changes_the_key():
    pytest.importorskip("rdkit")
    recs = _parity_records()
    a, _, _ = standardize_records_fast(recs, StructureEngine("universe_minimal", 1, 4, compute_hit_cap=True))
    b, _, _ = standardize_records_fast(recs, StructureEngine("universe_minimal", 1, 4, compute_hit_cap=False))
    assert identity_mismatches(a, b, IDENTITY_COLUMNS).empty and b["tautomer_hit_cap"].isna().all()


def test_identity_descriptors_match_full_descriptors():
    pytest.importorskip("rdkit")
    from casmi.chemistry.descriptors import compute_identity_descriptors, compute_molecular_descriptors
    for s in PARITY_SMILES:
        full, ident = compute_molecular_descriptors(s), compute_identity_descriptors(s)
        assert {k: full[k] for k in ident} == ident


# ---------------------------------------------------------------------------------------------------------------
# prefilter / writer / reader
# ---------------------------------------------------------------------------------------------------------------

def test_formula_prefilter_cache_is_equivalent():
    recs = pd.DataFrame({"source": "COCONUT", "source_id": list("abcdef"), "raw_smiles": "C",
                         "source_formula": ["C10H12O2", "CH4", "NaCl", None, "C10H12O2", "C200H402"]})
    cfg = UniverseBuildConfig()
    k0, d0 = formula_prefilter(recs, cfg)
    cache = {}
    k1, d1 = formula_prefilter(recs, cfg, cache=cache)
    k2, d2 = formula_prefilter(recs, cfg, cache=cache)                             # second call served from the cache
    for k, d in ((k1, d1), (k2, d2)):
        assert k.equals(k0) and d.equals(d0)
    assert set(d0["source_id"]) == {"b", "c", "f"} and "d" in set(k0["source_id"])  # missing formula passes to the real standardizer


def _legacy_write_bucketed(df, path, prefix_len):
    """The previous implementation (reference for the writer regression test)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from casmi.candidates.universe import _normalize_dtypes, bucket_of
    df = df.assign(bucket=bucket_of(df["connectivity_key"], prefix_len).to_numpy()).sort_values(["bucket", "connectivity_key"], kind="mergesort")
    df = _normalize_dtypes(df)
    table = pa.Table.from_pandas(df, preserve_index=False)
    with pq.ParquetWriter(path, table.schema) as w:
        for b in df["bucket"].unique():
            w.write_table(pa.Table.from_pandas(df[df["bucket"] == b], schema=table.schema, preserve_index=False))


def test_write_bucketed_matches_previous_writer(tmp_path):
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    keys = ["CZZZZZZZZZZZZZ", "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "ABBBBBBBBBBBBB", "CAAAAAAAAAAAAA"]
    df = pd.DataFrame({"source": "COCONUT", "source_id": list("12345"), "connectivity_key": keys, "raw_smiles": list("abcde"),
                       "neutral_monoisotopic_mass": [300.0, 310.0, 320.0, 330.0, 340.0], "formal_charge": [0, 0, 0, 0, 0],
                       "is_charged": [False] * 5, "name": [None, "x", None, "y", None]})
    _write_bucketed(df, tmp_path / "new.parquet", 1)
    _legacy_write_bucketed(df, tmp_path / "old.parquet", 1)
    new, old = pq.ParquetFile(tmp_path / "new.parquet"), pq.ParquetFile(tmp_path / "old.parquet")
    assert new.schema_arrow.equals(old.schema_arrow) and new.metadata.num_row_groups == old.metadata.num_row_groups == 3
    assert new.read().equals(old.read())
    for i in range(new.metadata.num_row_groups):
        assert len(set(new.read_row_group(i).column("bucket").to_pylist())) == 1
    got = ds.dataset(str(tmp_path / "new.parquet")).to_table(filter=ds.field("bucket") == "C").column("connectivity_key").to_pylist()
    assert got == sorted(k for k in keys if k[0] == "C")


def test_csv_reader_reads_only_configured_columns_without_losing_rows(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text('identifier,canonical_smiles,name,molecular_formula,huge\n'
                 '12345678901234567890,"CC(=O)O","acetic acid, glacial",C2H4O2,zzz\n'
                 'CNP2,C1=CC=CC=C1,,C6H6,\n'
                 'CNP3,,"émodine ✓",,\n'
                 'CNP4,CCO,ethanol,,"a ""quoted"" value"\n', encoding="utf-8")
    cfg = source_cfg(p)
    cfg.chunksize = 2
    chunks = list(iter_source_records(cfg))
    recs = pd.concat(chunks, ignore_index=True)
    assert [len(c) for c in chunks] == [2, 2] and len(recs) == 4
    assert recs["source_id"].tolist() == ["12345678901234567890", "CNP2", "CNP3", "CNP4"]       # large ids kept as text
    assert recs.loc[0, "name"] == "acetic acid, glacial" and recs.loc[2, "name"] == "émodine ✓"
    assert pd.isna(recs.loc[1, "name"]) and pd.isna(recs.loc[2, "raw_smiles"]) and pd.isna(recs.loc[3, "source_formula"])
    bad = source_cfg(p)
    bad.formula_col = "not_a_column"
    with pytest.raises(SourceConfigError):
        list(iter_source_records(bad))
    assert estimate_record_count(p) == 4


# ---------------------------------------------------------------------------------------------------------------
# Stage B / C invalidation
# ---------------------------------------------------------------------------------------------------------------

def _mv():
    return pd.DataFrame({"mass_variant_id": ["mv1"], "connectivity_key": ["AAAAAAAAAAAAAA"], "exact_mass": [300.1],
                         "representative_smiles": ["CCO_train"], "molecular_formula": ["C10H12O2"], "n_train_spectra": [3]})


def _ext_chunk(path, key="ACCCCCCCCCCCCC"):
    d = pd.DataFrame({"source": ["COCONUT"], "source_id": ["c1"], "raw_smiles": ["x"], "normalized_smiles": ["x"], "representative_smiles": ["x"],
                      "connectivity_key": [key], "molecular_formula": ["C10H12O2"], "neutral_monoisotopic_mass": [320.0], "molecular_weight": [320.5],
                      "formal_charge": [0], "n_fragments": [1], "name": [None]})
    _write_bucketed(d, path, 1)


def test_train_only_bucket_is_rebuilt_when_an_external_source_appears(tmp_path):
    std_root, out = tmp_path / "std", tmp_path / "u"
    r1 = merge_bucket("A", std_root, _mv(), out, None, UniverseBuildConfig())
    assert r1["n_external_records"] == 0
    _ext_chunk(std_root / "COCONUT" / "chunk-00000.parquet")
    r2 = merge_bucket("A", std_root, _mv(), out, None, UniverseBuildConfig())
    assert r2["stage_b_id"] != r1["stage_b_id"] and r2["n_external_records"] == 1 and r2["n_connectivities"] == 2


def test_stage_c_refuses_foreign_buckets_and_records_sources(tmp_path):
    out = tmp_path / "u"
    train_only = run_stage_b({"sources": {}, "std_files": []}, _mv(), None, out, UniverseBuildConfig(), log=None)
    man = finalize_universe(out, train_only["buckets"], stage_b_id=train_only["stage_b_id"])
    assert man["external_source_present"] is False and man["sources"]["TRAIN"]["n_candidates"] == 1
    assert universe_source_status(out)["status"] == "train_only"
    f = tmp_path / "a" / "standardized" / "COCONUT" / "chunk-00000.parquet"
    _ext_chunk(f)
    inputs = {"sources": {"COCONUT": {"status": "complete", "build_id": "x", "std_files": {str(f): f.stat().st_size}}}, "std_files": [str(f)]}
    with_ext = run_stage_b(inputs, _mv(), None, out, UniverseBuildConfig(), log=None)
    assert with_ext["stage_b_id"] != train_only["stage_b_id"]
    with pytest.raises(StageBIncompleteError):
        finalize_universe(out, with_ext["buckets"] + ["Z"], stage_b_id=with_ext["stage_b_id"])
    man = finalize_universe(out, with_ext["buckets"], stage_b_id=with_ext["stage_b_id"], build_info={"stage_a": {"COCONUT": "x"}})
    assert man["external_source_present"] and man["sources"]["COCONUT"]["n_candidates"] == 1 and man["n_external_only"] == 1
    assert man["build_id"] == with_ext["stage_b_id"] and universe_source_status(out)["status"] == "external"
    assert json.loads((out / "universe_manifest.json").read_text())["n_train_candidates"] == 1


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------

def test_resolve_n_jobs():
    assert resolve_n_jobs(3, cpu_count=2) == 3                                   # explicit override wins
    assert resolve_n_jobs("auto", cpu_count=2, ram_gb=64) == 2
    assert resolve_n_jobs(-1, cpu_count=16, ram_gb=64) == 16
    assert resolve_n_jobs("auto", cpu_count=16, ram_gb=4, ram_per_worker_gb=1.0) == 4    # RAM cap
    assert resolve_n_jobs(None, cpu_count=1, ram_gb=0.1) == 1


def test_performance_config_and_progress_without_total():
    p = StageAPerformance.from_dict({"standardization_profile": "full", "canonicalization_cache": {"enabled": False},
                                     "progress": {"every_records": 10, "every_seconds": 5}})
    assert p.standardization_profile == "full" and not p.cache_enabled and p.progress_every_records == 10
    with pytest.raises(ValueError):
        StageAPerformance(standardization_profile="fast_but_different")
    logs = []
    pr = StageAProgress("COCONUT", est_total_rows=None, every_records=2, log=logs.append)
    pr.start_chunk(0, 10)
    pr.start_structures(5)
    pr.advance(3)
    pr.end_chunk({"chunk": 0, "n_input": 10, "n_prefiltered": 0, "n_unique_raw_smiles": 5, "n_standardizer_calls": 5,
                  "n_duplicate_raw_smiles_saved": 0, "n_kept": 5})
    assert logs and "rows read 10" in logs[-1]


def test_gpu_policy_is_explicit():
    assert "CPU-bound" in GPU_STAGE_A_POLICY and "not used" in GPU_STAGE_A_POLICY
