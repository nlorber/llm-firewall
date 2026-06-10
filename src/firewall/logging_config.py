# src/firewall/logging_config.py
from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Attach a stdout handler to the root logger for CLI entrypoints.

    The library logs through module loggers (``logging.getLogger(__name__)``). With no
    handler attached, the root logger defaults to WARNING and silently drops every INFO
    line — which is why ``firewall-evaluate`` and ``firewall-robustness`` printed nothing.
    Call this once at the start of a console entrypoint so results reach the terminal.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
