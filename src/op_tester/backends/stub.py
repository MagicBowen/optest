"""Stub backend driver used for development and CI."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from op_tester.core import resolve_reference
from op_tester.core.models import TestCase

from .base import BackendDriver, backend_manager


class StubBackendDriver(BackendDriver):
    """A backend that simply echoes inputs as outputs for scaffolding."""

    def __init__(self, kind: str, name: str = "stub", chips: Sequence[str] | None = None) -> None:
        self.kind = kind
        self.name = name
        self.chips = tuple(chips or ())

    def run(self, case: TestCase, inputs: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        reference_fn = resolve_reference(case.descriptor, case.reference_override)
        outputs = reference_fn(inputs, case.attributes)
        return tuple(np.array(tensor, copy=True) for tensor in outputs)


def register_stub_backends() -> None:
    """Register default stub drivers for both GPU and NPU kinds."""

    backend_manager.register(StubBackendDriver(kind="gpu"))
    backend_manager.register(StubBackendDriver(kind="npu"))
