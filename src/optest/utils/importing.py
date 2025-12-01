"""Utility helpers for dynamic imports."""
from __future__ import annotations

import importlib
from typing import Any


def import_string(path: str) -> Any:
    """Return the attribute at the given dotted path.

    Supports ``module:attr`` or ``module.attr`` syntax.
    """

    if not path:
        raise ValueError("Empty import path provided")
    module_name, sep, attr = path.partition(":")
    if not sep:
        module_name, sep, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid import path '{path}'")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover - simple attribute error
        raise AttributeError(f"Module '{module_name}' has no attribute '{attr}'") from exc
