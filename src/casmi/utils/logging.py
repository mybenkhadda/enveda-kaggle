"""Project-wide logging setup.

Production modules under `casmi/` should log through `get_logger(__name__)` rather than
scattering ad-hoc `print()` calls; notebooks may still `print()` concise summaries -- that's
orchestration output, not library behavior.
"""
import logging
import sys
from pathlib import Path

_CONFIGURED_LOGGERS = set()
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name, level=logging.INFO, log_file=None):
    """Return a configured logger. Idempotent: calling this again for the same `name` does
    not add duplicate handlers (a common cause of doubled log lines in notebooks that re-run
    a cell defining a module-level logger)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if name not in _CONFIGURED_LOGGERS:
        formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

        if log_file is not None:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False
        _CONFIGURED_LOGGERS.add(name)

    return logger
