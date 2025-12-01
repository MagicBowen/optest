"""Operator registry implementation."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Iterator

from op_tester.core import OperatorDescriptor


class OperatorRegistry:
    """Stores operator descriptors and exposes lookup utilities."""

    def __init__(self) -> None:
        self._descriptors: Dict[str, OperatorDescriptor] = {}

    def register(self, descriptor: OperatorDescriptor) -> OperatorDescriptor:
        if descriptor.name in self._descriptors:
            raise ValueError(f"Operator '{descriptor.name}' already registered")
        self._descriptors[descriptor.name] = descriptor
        return descriptor

    def update_or_register(self, descriptor: OperatorDescriptor) -> OperatorDescriptor:
        self._descriptors[descriptor.name] = descriptor
        return descriptor

    def get(self, name: str) -> OperatorDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as exc:  # pragma: no cover - simple error path
            raise KeyError(f"Operator '{name}' is not registered") from exc

    def __contains__(self, name: str) -> bool:
        return name in self._descriptors

    def __iter__(self) -> Iterator[OperatorDescriptor]:
        return iter(self._descriptors.values())

    def names(self) -> Iterable[str]:
        return tuple(self._descriptors.keys())


registry = OperatorRegistry()


def register_factory(factory: Callable[[], OperatorDescriptor]) -> Callable[[], OperatorDescriptor]:
    """Decorator registering a descriptor returned by the decorated factory."""

    descriptor = factory()
    registry.register(descriptor)
    return factory


def register_descriptor(descriptor: OperatorDescriptor) -> OperatorDescriptor:
    return registry.register(descriptor)


def clear_registry() -> None:
    registry._descriptors.clear()


def load_builtins() -> None:
    from . import builtins  # noqa: WPS433

    for descriptor in builtins.BUILTIN_DESCRIPTORS:
        registry.update_or_register(descriptor)
