"""Operator catalog providing descriptors and lookup."""
from __future__ import annotations

from typing import Dict, Iterable

from optest.core import OperatorDescriptor
from optest.operators import builtin_operators

_catalog: Dict[str, OperatorDescriptor] = {}


def load_builtins() -> None:
    """Populate the catalog with built-in operators."""

    _catalog.clear()
    for cls in builtin_operators.BUILTIN_OPERATOR_CLASSES:
        desc = cls.descriptor()
        _catalog[desc.name] = desc


def get(name: str) -> OperatorDescriptor:
    return _catalog[name]


def descriptors() -> Iterable[OperatorDescriptor]:
    return tuple(_catalog.values())


def names() -> Iterable[str]:
    return tuple(_catalog.keys())
