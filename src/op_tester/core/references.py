"""Reference implementation loading utilities."""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from .models import OperatorDescriptor
from op_tester.utils import import_string

ReferenceCallable = Callable[[Sequence[np.ndarray], Mapping[str, object]], Sequence[np.ndarray]]


def resolve_reference(descriptor: OperatorDescriptor, override: Optional[str] = None) -> ReferenceCallable:
    path = override or descriptor.default_reference
    if not path:
        raise ValueError(f"No reference implementation defined for operator '{descriptor.name}'")
    func = import_string(path)
    if not callable(func):  # pragma: no cover - defensive
        raise TypeError(f"Reference target '{path}' is not callable")
    return func  # type: ignore[return-value]
