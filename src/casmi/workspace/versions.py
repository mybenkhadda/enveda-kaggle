"""Every version a notebook is checked against, in one place.

A v2 notebook declares the NOTEBOOK_API_VERSION it was written for (`bootstrap(..., notebook_api=...)`), and the
generated bootstrap cell carries BOOTSTRAP_CELL_VERSION in its marker line. Preflight compares both with the
imported source and refuses to continue on any mismatch -- the failure mode it prevents is a notebook from one
commit executed against `src/` from another (e.g. a newer notebook against a stale /content/Enveda clone).

Bump rules (same commit as the change, then re-stamp notebooks with scripts/stamp_notebooks.py):
  CONFIG_SCHEMA_VERSION         configs/casmi_v2_colab.yaml layout changed           (casmi.workspace.config)
  PATHS_API_VERSION             a V2Paths field added / renamed / removed            (casmi.workspace.config)
  ARTIFACT_REGISTRY_API_VERSION an ArtifactRegistry name added / renamed / moved     (casmi.workspace.artifact_registry)
  BOOTSTRAP_CELL_VERSION        the generated bootstrap cell text changed            (casmi.workspace.notebook_cells)
  NOTEBOOK_API_VERSION          any of the above (notebooks must be updated together)
"""
from casmi.workspace.config import CONFIG_SCHEMA_VERSION, PATHS_API_VERSION

ARTIFACT_REGISTRY_API_VERSION = "casmi-v2-artifacts-2"
BOOTSTRAP_CELL_VERSION = "casmi-v2-bootstrap-2"
NOTEBOOK_API_VERSION = "casmi-v2-notebooks-4"

VERSIONS = {
    "config_schema": CONFIG_SCHEMA_VERSION,
    "paths_api": PATHS_API_VERSION,
    "artifact_registry_api": ARTIFACT_REGISTRY_API_VERSION,
    "bootstrap_cell": BOOTSTRAP_CELL_VERSION,
    "notebook_api": NOTEBOOK_API_VERSION,
}
