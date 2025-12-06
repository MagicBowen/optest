"""optest package initialization."""
from __future__ import annotations

import importlib
import os

from .version import __version__

__all__ = [
    "__version__",
    "bootstrap",
]

_BOOTSTRAPPED = False


def bootstrap() -> None:
    """Initialize optest (idempotent)."""

    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _load_plugins()
    _BOOTSTRAPPED = True


def _load_plugins() -> None:
    plugin_env = os.environ.get("OPTEST_PLUGINS")
    if not plugin_env:
        return
    for item in plugin_env.split(","):
        module_name = item.strip()
        if not module_name:
            continue
        module = importlib.import_module(module_name)
        register = getattr(module, "register", None)
        if callable(register):
            register()
