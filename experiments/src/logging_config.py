"""Centralized logging configuration for z-gap experiments.

Existing scripts (run_all.py, run_strategy_*.py) use bare ``print()`` calls;
those are preserved to avoid changing user-visible output. New code should
prefer ``get_logger()`` so long multi-model runs produce greppable,
level-tagged output that survives redirection.

Usage:
    from src.logging_config import configure_logging, get_logger

    configure_logging()              # call once at entry point
    logger = get_logger(__name__)
    logger.info("Running %s on %d stimuli", model.name, len(texts))

Level is taken from ``Z_GAP_LOG_LEVEL`` env var if set, else the ``level``
argument (default INFO).
"""

from __future__ import annotations

import logging
import os
import sys


_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_PLAIN_FORMAT = "%(message)s"


def configure_logging(
    level: str | int = "INFO",
    *,
    with_timestamp: bool = True,
    stream=None,
) -> None:
    """Configure the root logger. Idempotent; later calls override format.

    Args:
        level: ``logging`` level name or int. Overridden by ``Z_GAP_LOG_LEVEL``
            env var if present.
        with_timestamp: include asctime/levelname/name prefix. Set ``False`` to
            match legacy ``print()`` output exactly.
        stream: file-like; defaults to ``sys.stderr`` so stdout stays clean
            for any tool that captures it.
    """
    env_level = os.environ.get("Z_GAP_LOG_LEVEL")
    if env_level:
        level = env_level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    fmt = _DEFAULT_FORMAT if with_timestamp else _PLAIN_FORMAT
    logging.basicConfig(
        level=level,
        format=fmt,
        stream=stream or sys.stderr,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a child logger; configure_logging() should have been called once."""
    return logging.getLogger(name)
