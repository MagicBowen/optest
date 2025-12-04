"""Helpers for loading user-provided generator/assertion functions."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import Callable


def load_from_source(source: Path, func_name: str) -> Callable:
    """Load a callable named ``func_name`` from a Python file at ``source``."""

    path = source.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Custom source file not found: {path}")
    module_name = f"optest_custom_{path.stem}_{hash(str(path)) & 0xFFFF:x}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert isinstance(loader, importlib.machinery.SourceFileLoader)  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    loader.exec_module(module)
    if not hasattr(module, func_name):
        raise AttributeError(f"Function '{func_name}' not found in {path}")
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"Attribute '{func_name}' in {path} is not callable")
    return func  # type: ignore[return-value]
