from casmi.spectra.similarity_cache import SimilarityCache


def _value(n=0):
    return ("T4", 0.1 + n, 0.2, 0.3, 0.4)


def test_get_or_compute_calls_compute_fn_only_on_miss():
    cache = SimilarityCache(config_hash="cfg1")
    calls = []

    def compute():
        calls.append(1)
        return _value()

    v1 = cache.get_or_compute("host", "q1", "r1", compute)
    v2 = cache.get_or_compute("host", "q1", "r1", compute)  # same key -- must be a hit
    assert v1 == v2
    assert len(calls) == 1
    assert cache.n_computed == 1
    assert cache.n_hits == 1


def test_different_query_dataset_is_a_different_key():
    # HOST and DEV spectrum ids are not guaranteed disjoint -- query_dataset must be part of the key.
    cache = SimilarityCache(config_hash="cfg1")
    cache.get_or_compute("host", "q1", "r1", lambda: _value(1))
    cache.get_or_compute("dev", "q1", "r1", lambda: _value(2))
    assert cache.n_computed == 2
    assert len(cache) == 2


def test_config_hash_is_part_of_the_key():
    cache_a = SimilarityCache(config_hash="cfg_a")
    cache_b = SimilarityCache(config_hash="cfg_b")
    cache_a.get_or_compute("host", "q1", "r1", lambda: _value(1))
    cache_b.get_or_compute("host", "q1", "r1", lambda: _value(2))
    assert cache_a.n_computed == 1
    assert cache_b.n_computed == 1  # not a hit against cache_a's data -- different instance anyway,
    # but the real point is proven by save/load below with a shared file and mismatched config_hash


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "sim_cache.parquet"
    cache = SimilarityCache(config_hash="cfg1")
    cache.get_or_compute("host", "q1", "r1", lambda: _value(1))
    cache.get_or_compute("host", "q1", "r2", lambda: _value(2))
    cache.save(path)

    reloaded = SimilarityCache(config_hash="cfg1")
    n_loaded = reloaded.load(path)
    assert n_loaded == 2
    calls = []
    v = reloaded.get_or_compute("host", "q1", "r1", lambda: (calls.append(1), _value(99))[1])
    assert v == _value(1)
    assert len(calls) == 0  # loaded from disk, never recomputed -- the warm-rerun contract


def test_load_ignores_rows_with_a_different_config_hash(tmp_path):
    path = tmp_path / "sim_cache.parquet"
    old_cache = SimilarityCache(config_hash="cfg_old")
    old_cache.get_or_compute("host", "q1", "r1", lambda: _value(1))
    old_cache.save(path)

    new_cache = SimilarityCache(config_hash="cfg_new")
    n_loaded = new_cache.load(path)
    assert n_loaded == 0
    assert len(new_cache) == 0


def test_load_missing_file_returns_zero_and_never_raises(tmp_path):
    cache = SimilarityCache(config_hash="cfg1")
    assert cache.load(tmp_path / "does_not_exist.parquet") == 0
    assert len(cache) == 0


def test_contains_reflects_full_key():
    cache = SimilarityCache(config_hash="cfg1")
    cache.get_or_compute("host", "q1", "r1", lambda: _value())
    assert ("host", "q1", "r1") in cache
    assert ("dev", "q1", "r1") not in cache
