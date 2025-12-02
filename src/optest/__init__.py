"""optest package initialization."""
from __future__ import annotations

import importlib
import os

from .backends import AscendBackendDriver, backend_manager, register_stub_backends
from .registry import load_builtins

__all__ = [
    "__version__",
    "bootstrap",
]

__version__ = "0.1.0"

_BOOTSTRAPPED = False


def bootstrap() -> None:
    """Register built-in operators and stub backends (idempotent)."""

    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    load_builtins()
    _load_plugins()
    register_stub_backends()
    backend_manager.register(
        AscendBackendDriver(
            chips=(
                "ascend",
                "ascend910",
                "ascend910a",
                "ascend910b",
                "ascend310",
                "ascend310b",
            )
        )
    )
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
