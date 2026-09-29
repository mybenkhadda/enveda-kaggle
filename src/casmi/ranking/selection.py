"""v4b pre-registered, DEV-only model selection + HOST isolation guards.

Order enforced in code (spec section 54):

    1. `write_preregistration`  -- the rule is written (and hashed) BEFORE any metric is computed
    2. DEV-only experiments; `select_from_dev_only` picks the headline model from a table that is
       REFUSED if it carries any HOST-derived column
    3. `lock_dev_selection`     -- persists the selection + all DEV-chosen hyperparameters (alphas)
    4. `require_dev_selection_lock` -- every HOST evaluation call takes the lock object; it cannot
       be obtained before step 3, and it re-verifies the pre-registration hash.

HOST can confirm, quantify degradation and reveal domain shift. It cannot choose.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CANDIDATE_MODELS = ("V1", "V2", "V3", "V4", "V3_RD", "V1_1k", "V1_3k", "V1_10k")

# 2 = source + structure held out, 1 = source held out, 0 = protocol-matched only
HELD_OUT_VALIDITY = {"V1": 0, "V2": 1, "V3": 2, "V4": 2, "V3_RD": 2, "V1_1k": 0, "V1_3k": 0, "V1_10k": 0}
# lower = simpler (fewer features / less training data / no augmentation)
SIMPLICITY_RANK = {"V1": 0, "V1_1k": 0, "V2": 1, "V3": 2, "V1_3k": 3, "V3_RD": 4, "V4": 5, "V1_10k": 6}

PREREGISTERED_RULE = {
    "rule_id": "v4b-modeA-selection-1",
    "primary_metric": "DEV connectivity-safe OOF MRR@25 (all DEV queries in the denominator; zero-candidate queries RR=0)",
    "candidate_models": list(CANDIDATE_MODELS),
    "near_tie_margin": 0.005,
    "near_tie_preference_order": [
        "higher DEV OOF k=1 stress MRR@25",
        "higher held-out validity (source+structure > source > protocol-matched)",
        "simpler model (SIMPLICITY_RANK ascending)",
        "model_id ascending (final deterministic tie-break)",
    ],
    "held_out_validity": HELD_OUT_VALIDITY,
    "simplicity_rank": SIMPLICITY_RANK,
    "alpha_rule": "B4/B5 alpha = argmax DEV MRR@25 over the predefined grid, ties -> smaller alpha",
    "host_role": "confirmation only: HOST metrics may not enter any selection/tuning function; a worse HOST result is reported as a domain-shift warning, never a model swap",
    "near_dup_strict_role": "sensitivity only; never used for selection",
    "domain_shift_warning": "raise a DOMAIN-SHIFT WARNING (no model swap) if the selected model's HOST MRR@25 < its DEV OOF MRR@25 - 0.05",
    "domain_shift_margin": 0.05,
    "missing_model_policy": "a candidate model that was not trained (e.g. scale set not built) is excluded from selection and reported as NOT_RUN",
}

ALLOWED_SELECTION_COLUMNS = ("model_id", "dev_oof_mrr", "dev_k1_mrr", "held_out_validity", "simplicity_rank", "status")

# Which training restrictions each model applies (cumulative). `HELD_OUT_VALIDITY` above is the
# NOMINAL level; the level a model may actually CLAIM is `effective_held_out_validity`, which only
# counts a restriction that removed >= 1 training query (v5 fix: in v4b the source filter removed
# zero DEV queries, so V2 was a no-op duplicate of V1 and must not rank as "more held out").
MODEL_TRAINING_FILTERS = {"V1": (), "V1_1k": (), "V1_3k": (), "V1_10k": (), "V2": ("source",),
                          "V3": ("source", "structure"), "V4": ("source", "structure"), "V3_RD": ("source", "structure")}
# model -> the model it is identical to when every one of its filters is a no-op (same features/training recipe)
NO_OP_PARENT = {"V2": "V1", "V3": "V1"}


def filter_audit(filter_name, before_ids, after_ids):
    """Persistable record of one training restriction: `n_queries_before_filter`,
    `n_queries_after_filter`, `n_queries_removed` (and whether it was a no-op)."""
    before, after = set(before_ids), set(after_ids)
    if not after <= before:
        raise ValueError(f"filter {filter_name!r} added queries; a training restriction may only remove")
    removed = len(before) - len(after)
    return {"filter": filter_name, "n_queries_before_filter": len(before), "n_queries_after_filter": len(after),
            "n_queries_removed": removed, "is_no_op": removed == 0}


def effective_held_out_validity(model_id, filter_audits):
    """Number of this model's training restrictions that actually removed >= 1 query.
    `filter_audits`: `{filter_name: filter_audit(...)}` (per model, or shared across models --
    for cumulative filters pass the audit of each STEP, e.g. source: DEV -> V2, structure: V2 -> V3).
    A restriction without an audit counts as 0 -- a claim of held-out-ness needs evidence."""
    filters = MODEL_TRAINING_FILTERS.get(model_id, ())
    return int(sum(1 for f in filters if filter_audits.get(f, {}).get("n_queries_removed", 0) >= 1))


def no_op_duplicate_annotations(filter_audits):
    """`{model_id: "no-op duplicate of <parent>"}` for every model whose restrictions ALL removed
    zero queries (it trains on exactly its parent's data). Historical model IDs are kept; the
    scientific baseline is reported under the parent's name."""
    out = {}
    for model_id, parent in NO_OP_PARENT.items():
        filters = MODEL_TRAINING_FILTERS[model_id]
        if filters and all(filter_audits.get(f, {}).get("n_queries_removed", 0) == 0 for f in filters):
            out[model_id] = f"no-op duplicate of {parent}: every training restriction removed 0 queries"
    return out


def apply_effective_validity(table, filter_audits, col="held_out_validity"):
    """Replace a selection table's nominal `held_out_validity` by the effective one. Returns
    `(table, corrections)` where `corrections` lists every model whose nominal claim was lowered."""
    t = table.copy()
    eff = t["model_id"].map(lambda m: effective_held_out_validity(m, filter_audits)).astype(int)
    corrections = [{"model_id": m, "nominal": int(n), "effective": int(e)}
                   for m, n, e in zip(t["model_id"], t[col], eff) if int(n) != int(e)]
    t[col] = eff
    return t, corrections


class HostLeakError(RuntimeError):
    """A selection/tuning function was handed HOST-derived information."""


def guard_no_host(obj, where):
    """Raise `HostLeakError` if a DataFrame column / dict key names HOST in any form."""
    names = list(obj.columns) if isinstance(obj, pd.DataFrame) else list(obj.keys()) if isinstance(obj, dict) else []
    leaked = [n for n in names if "host" in str(n).lower()]
    if leaked:
        raise HostLeakError(f"{where}: HOST-derived fields are not allowed as selection inputs: {leaked}")


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_preregistration(path, rule=None, extra=None):
    """Write the rule before any metric exists. Refuses to overwrite a DIFFERENT existing rule
    (re-writing the identical rule is a no-op), so the rule cannot drift after results appear."""
    rule = dict(rule or PREREGISTERED_RULE)
    if extra:
        rule["extra"] = extra
    digest = _hash(rule)
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("rule_sha256") != digest:
            raise RuntimeError(f"{path} already holds a DIFFERENT pre-registered rule "
                               f"({existing.get('rule_sha256')} vs {digest}); the rule may not change once written")
        return existing
    record = {"rule": rule, "rule_sha256": digest, "written_at": datetime.now(timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


def verify_preregistration(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if _hash(record["rule"]) != record["rule_sha256"]:
        raise RuntimeError(f"{path}: pre-registered rule was edited after it was written")
    return record


def select_from_dev_only(dev_table, rule=None, filter_audits=None):
    """`dev_table`: one row per candidate model with ONLY `ALLOWED_SELECTION_COLUMNS`
    (`status` == "OK" rows are eligible). Returns `(selected_model_id, ranked_table)`.
    `filter_audits` (v5): when given, `held_out_validity` is replaced by the EFFECTIVE level
    (a restriction that removed zero queries earns nothing) before the rule is applied.

    Rule: best = max dev_oof_mrr; contenders = eligible models within `near_tie_margin` of best;
    contenders ordered by dev_k1_mrr DESC, held_out_validity DESC, simplicity_rank ASC,
    model_id ASC. Non-contenders follow, ordered by dev_oof_mrr DESC."""
    rule = rule or PREREGISTERED_RULE
    guard_no_host(dev_table, "select_from_dev_only")
    extra = sorted(set(dev_table.columns) - set(ALLOWED_SELECTION_COLUMNS))
    if extra:
        raise HostLeakError(f"select_from_dev_only: unexpected input columns {extra} (only {ALLOWED_SELECTION_COLUMNS} allowed)")
    t = dev_table.copy()
    if filter_audits is not None:
        t, _ = apply_effective_validity(t, filter_audits)
    if "status" not in t.columns:
        t["status"] = "OK"
    ok = t[t["status"] == "OK"].copy()
    if ok.empty:
        raise RuntimeError("no eligible candidate model")
    best = ok["dev_oof_mrr"].max()
    ok["within_margin"] = ok["dev_oof_mrr"] >= best - rule["near_tie_margin"]
    contenders = ok[ok["within_margin"]].sort_values(
        ["dev_k1_mrr", "held_out_validity", "simplicity_rank", "model_id"], ascending=[False, False, True, True], kind="mergesort")
    others = ok[~ok["within_margin"]].sort_values(["dev_oof_mrr", "model_id"], ascending=[False, True], kind="mergesort")
    ranked = pd.concat([contenders, others], ignore_index=True)
    ranked["selection_order"] = range(1, len(ranked) + 1)
    return str(ranked.iloc[0]["model_id"]), ranked


def lock_dev_selection(path, selected_model_id, ranked_table, alphas, preregistration_path, extra=None):
    """Persist every DEV-made choice. Refuses to overwrite an existing lock with a different
    selection (a re-run that reproduces the same choice is fine)."""
    prereg = verify_preregistration(preregistration_path)
    guard_no_host(ranked_table, "lock_dev_selection")
    guard_no_host(alphas, "lock_dev_selection(alphas)")
    record = {"selected_model_id": selected_model_id, "alphas": alphas,
              "ranked_table": ranked_table.to_dict("records"), "preregistration_sha256": prereg["rule_sha256"],
              "locked_at": datetime.now(timezone.utc).isoformat(), **(extra or {})}
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("selected_model_id") != selected_model_id or old.get("alphas") != alphas:
            raise RuntimeError(f"{path} already locks a DIFFERENT DEV selection ({old.get('selected_model_id')}, {old.get('alphas')}); "
                               "delete it deliberately (and log a post_host_change decision) if this is intended")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


class DevSelectionLock:
    """Token proving the DEV selection was persisted. HOST evaluation helpers require one."""

    def __init__(self, record, path):
        self.record = record
        self.path = Path(path)
        self.selected_model_id = record["selected_model_id"]
        self.alphas = record["alphas"]


def require_dev_selection_lock(lock_path, preregistration_path):
    lock_path = Path(lock_path)
    if not lock_path.exists():
        raise HostLeakError("HOST evaluation requested before the DEV selection was locked")
    record = json.loads(lock_path.read_text(encoding="utf-8"))
    prereg = verify_preregistration(preregistration_path)
    if record.get("preregistration_sha256") != prereg["rule_sha256"]:
        raise HostLeakError("DEV selection lock was made under a different pre-registered rule")
    return DevSelectionLock(record, lock_path)
