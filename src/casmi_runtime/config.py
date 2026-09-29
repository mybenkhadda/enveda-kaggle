"""Runtime configuration shared by the Colab and Kaggle notebooks."""
from dataclasses import asdict, dataclass, field
from pathlib import Path

RUNTIMES = ("colab", "kaggle", "local")
DEVICE_PREFERENCES = ("gpu", "cpu")

# the frozen identity every production run must see (values come from the locked v6.3 selection; CONFIG_HASH is
# NOT hard-coded -- pass `expected_config_hash` to pin a specific export)
FROZEN_MODEL_ID = "V1_TL_1K_TESTSIM_STRICT"
FROZEN_AGGREGATOR = "MOST_CONFIDENT_SPECTRUM"
FROZEN_BUNDLE_VERSION = "v2-A7"


@dataclass
class RuntimeConfig:
    """RUNTIME: colab | kaggle | local.
    DEVICE_PREFERENCE='gpu': use a GPU backend only if it is implemented in the frozen bundle AND passes the self-test.
    REQUIRE_GPU=False: a validated CPU backend is acceptable (production). True is for benchmarking only -- the run
    stops when no validated GPU backend exists."""
    runtime: str
    input_root: Path
    work_dir: Path
    device_preference: str = "gpu"
    require_gpu: bool = False
    expected_model_id: str = FROZEN_MODEL_ID
    expected_aggregator: str = FROZEN_AGGREGATOR
    expected_bundle_version: str = FROZEN_BUNDLE_VERSION
    expected_config_hash: str = None
    time_budget_s: float = 8 * 3600
    emergency_cap_enabled: bool = False        # never degrade inference silently
    emergency_cap_n: int = 300
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.input_root, self.work_dir = Path(self.input_root), Path(self.work_dir)
        if self.runtime not in RUNTIMES:
            raise ValueError(f"runtime must be one of {RUNTIMES}, got {self.runtime!r}")
        if self.device_preference not in DEVICE_PREFERENCES:
            raise ValueError(f"device_preference must be one of {DEVICE_PREFERENCES}, got {self.device_preference!r}")
        if self.require_gpu and self.device_preference != "gpu":
            raise ValueError("require_gpu=True needs device_preference='gpu'")
        if self.work_dir.resolve().is_relative_to(self.input_root.resolve()):
            raise ValueError("work_dir must lie OUTSIDE input_root (a previous submission.csv would be discovered as an input)")

    def as_dict(self):
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
