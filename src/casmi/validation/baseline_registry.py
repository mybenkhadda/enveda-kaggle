"""A versioned, append-only registry of benchmark metrics -- so a later notebook's "we beat the
baseline" claim always has a real, previously-recorded number to point at, and an accidental
re-run can never silently overwrite it with a different result.
"""
import json
from datetime import datetime, timezone
from pathlib import Path


def _load(registry_path):
    registry_path = Path(registry_path)
    if not registry_path.exists():
        return {}
    with open(registry_path, encoding="utf-8") as f:
        return json.load(f)


def register_baseline(name, metrics, registry_path, source_notebook=None, overwrite=False):
    """Add (or, with `overwrite=True`, explicitly replace) one named entry in the registry at
    `registry_path`. Refuses (raises `ValueError`) to silently overwrite an EXISTING entry with
    DIFFERENT metric values unless `overwrite=True` -- re-registering the identical result is a
    harmless no-op, but a changed result under the same name is exactly the silent-overwrite
    this module exists to prevent."""
    registry = _load(registry_path)
    entry = {"metrics": metrics, "source_notebook": source_notebook, "recorded_at": datetime.now(timezone.utc).isoformat()}

    if name in registry and not overwrite:
        existing_metrics = registry[name]["metrics"]
        if existing_metrics != metrics:
            raise ValueError(
                f"baseline {name!r} already registered with different metrics "
                f"(existing={existing_metrics}, new={metrics}). Pass overwrite=True if this is intentional."
            )
        return registry  # identical re-registration -- no-op, nothing written

    registry[name] = entry
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, default=str)
    return registry


def load_baseline(name, registry_path):
    """The registered `metrics` dict for `name`, or `None` if never registered."""
    registry = _load(registry_path)
    entry = registry.get(name)
    return entry["metrics"] if entry else None


class InvalidBaselineError(ValueError):
    """Raised by `load_promoted_baseline` when the requested (name, regime) is marked
    non-promoted (`use_for_model_selection: False`, or a `status` other than one of
    `PROMOTED_STATUSES`) and the caller didn't pass `allow_invalid=True`."""


PROMOTED_STATUSES = {"VALID_INTERNAL_BASELINE", "VALID", "PROMOTED"}


def load_promoted_baseline(name, registry_path, regime=None, allow_invalid=False):
    """Like `load_baseline`, but refuses (raises `InvalidBaselineError`) to hand back a regime
    an audit has marked non-promoted -- so a later notebook can never SILENTLY re-use an
    invalidated number (e.g. Mode-B's fold/missingness-shortcut MRR) just because it's sitting
    in the registry under a plausible-looking key. A metrics dict that carries no `status`/
    `use_for_model_selection` field at all (the older, flat convention -- e.g. notebook 04's
    entry) is treated as promoted by default, since it predates this convention and was never
    flagged invalid.

    `regime`: drill into `metrics[regime]` first (e.g. `regime="mode_b"` for a
    `{"mode_a": {...}, "mode_b": {...}}`-shaped entry) before checking status. Leave `None` for
    a flat entry, or to check the top-level entry's own `status`/`use_for_model_selection`
    fields directly.
    """
    metrics = load_baseline(name, registry_path)
    if metrics is None:
        return None
    scoped = metrics[regime] if regime is not None else metrics
    if not isinstance(scoped, dict):
        return scoped

    status = scoped.get("status")
    use_for_model_selection = scoped.get("use_for_model_selection", True)
    is_promoted = use_for_model_selection and (status is None or status in PROMOTED_STATUSES)

    if not is_promoted and not allow_invalid:
        raise InvalidBaselineError(
            f"baseline {name!r}{f'[{regime!r}]' if regime else ''} is NOT promoted "
            f"(status={status!r}, use_for_model_selection={use_for_model_selection!r}) -- "
            f"refusing to return it for model selection / progress reporting. "
            f"Pass allow_invalid=True if you explicitly need the historical value anyway."
        )
    return scoped


def list_baselines(registry_path):
    return list(_load(registry_path).keys())
