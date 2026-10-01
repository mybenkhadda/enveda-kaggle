import pytest

from casmi.validation.decision_log import load_decision_log, log_decision, post_host_change_count


def test_log_decision_appends_entry(tmp_path):
    path = tmp_path / "decision_log.jsonl"
    entry = log_decision("capped reference list before filtering", "40-minute timeout", host_visible=True,
                          category="bug_fix", log_path=path)
    assert entry["category"] == "bug_fix"
    assert entry["host_visible"] is True
    loaded = load_decision_log(path)
    assert len(loaded) == 1
    assert loaded[0]["decision"] == entry["decision"]


def test_log_decision_rejects_unknown_category(tmp_path):
    path = tmp_path / "decision_log.jsonl"
    with pytest.raises(ValueError):
        log_decision("x", "y", host_visible=False, category="not_a_real_category", log_path=path)


def test_load_decision_log_missing_file_returns_empty(tmp_path):
    assert load_decision_log(tmp_path / "nonexistent.jsonl") == []


def test_multiple_entries_appended_in_order(tmp_path):
    path = tmp_path / "decision_log.jsonl"
    log_decision("first", "e1", host_visible=False, category="pre_registered_choice", log_path=path)
    log_decision("second", "e2", host_visible=True, category="bug_fix", log_path=path)
    entries = load_decision_log(path)
    assert [e["decision"] for e in entries] == ["first", "second"]


def test_post_host_change_count(tmp_path):
    path = tmp_path / "decision_log.jsonl"
    log_decision("a", "e", host_visible=True, category="pre_registered_choice", log_path=path)
    log_decision("b", "e", host_visible=True, category="post_host_change", log_path=path)
    log_decision("c", "e", host_visible=True, category="post_host_change", log_path=path)
    assert post_host_change_count(path) == 2


def test_post_host_change_count_zero_for_clean_log(tmp_path):
    path = tmp_path / "decision_log.jsonl"
    log_decision("a", "e", host_visible=True, category="bug_fix", log_path=path)
    assert post_host_change_count(path) == 0
