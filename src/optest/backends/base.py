"""Backend driver abstractions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from optest.core import BackendTarget, TestCase


class BackendDriver:
    """Base interface for backend drivers."""

    name: str = ""
    kind: str = "gpu"
    chips: Sequence[str] = ()

    def supports(self, case: TestCase) -> bool:
        return case.backend.kind == self.kind and self._chip_supported(case.backend.chip)

    def _chip_supported(self, chip: Optional[str]) -> bool:
        if chip is None or not self.chips:
            return True
        return chip in self.chips

    def run(self, case: TestCase, inputs: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        raise NotImplementedError


class BackendManager:
    """Registry for backend drivers keyed by kind/chip."""

    def __init__(self) -> None:
        self._drivers: Dict[str, BackendDriver] = {}

    def register(self, driver: BackendDriver) -> None:
        key = self._key(driver.kind, driver.name)
        if key in self._drivers:
            raise ValueError(f"Backend '{driver.kind}:{driver.name}' already registered")
        self._drivers[key] = driver

    def get_driver(self, kind: str, chip: Optional[str] = None) -> BackendDriver:
        candidates = [drv for drv in self._drivers.values() if drv.kind == kind and drv._chip_supported(chip)]
        if not candidates:
            raise KeyError(f"No backend registered for kind={kind!r} chip={chip!r}")
        if chip:
            for candidate in candidates:
                if chip in candidate.chips:
                    return candidate
        return candidates[0]

    def drivers(self) -> Iterable[BackendDriver]:
        return tuple(self._drivers.values())

    def _key(self, kind: str, name: str) -> str:
        return f"{kind}:{name}"


backend_manager = BackendManager()
