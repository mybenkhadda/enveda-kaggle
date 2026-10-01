"""A reusable timing context manager for pipeline stages."""
import time
from contextlib import contextmanager

from casmi.utils.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def timer(label, n_items=None, logger_=None):
    """Time a block of code and report elapsed seconds (and rows/sec if `n_items` is given).

    Usage:
        with timer("Build structure table", n_items=len(unique_smiles)):
            ...
    """
    log = logger_ or logger
    t0 = time.perf_counter()
    log.info("start: %s", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        if n_items:
            rate = n_items / elapsed if elapsed > 0 else float("inf")
            log.info("done: %s (%.2fs, %d items, %.1f items/s)", label, elapsed, n_items, rate)
        else:
            log.info("done: %s (%.2fs)", label, elapsed)
