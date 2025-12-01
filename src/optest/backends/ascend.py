"""Ascend NPU backend which shells out to user-provided run scripts."""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from optest.core import TestCase

from .base import BackendDriver


@dataclass
class TensorFileSpec:
    tensor: str
    path: Path
    dtype: Optional[str] = None


@dataclass
class AscendCommandConfig:
    workdir: Path
    command: List[str]
    inputs: List[TensorFileSpec]
    outputs: List[TensorFileSpec]
    env: Dict[str, str]


class AscendBackendDriver(BackendDriver):
    """Backend that runs Ascend operator binaries via user scripts."""

    name = "ascend-command"

    def __init__(self, *, chips: Sequence[str]) -> None:
        self.chips = tuple(chips)

    def supports(self, case: TestCase) -> bool:
        return super().supports(case) and _extract_config(case) is not None

    def run(self, case: TestCase, inputs: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        config = _extract_config(case)
        if config is None:
            raise RuntimeError(
                "Ascend backend selected but no backend_config.ascend command was provided"
            )
        self._write_inputs(case, inputs, config)
        self._run_command(config)
        outputs = self._read_outputs(case, config)
        return outputs

    def _write_inputs(
        self, case: TestCase, inputs: Sequence[np.ndarray], config: AscendCommandConfig
    ) -> None:
        name_to_tensor = _map_inputs(case, inputs)
        for spec in config.inputs:
            tensor = name_to_tensor.get(spec.tensor)
            if tensor is None:
                raise RuntimeError(f"Ascend backend input '{spec.tensor}' not found")
            dtype = np.dtype(spec.dtype or tensor.dtype)
            array = np.asarray(tensor, dtype=dtype)
            target_path = config.workdir / spec.path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            array.tofile(target_path)

    def _run_command(self, config: AscendCommandConfig) -> None:
        env = os.environ.copy()
        env.update(config.env)
        process = subprocess.run(
            config.command,
            cwd=str(config.workdir),
            env=env,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "Ascend command failed (exit code %s): %s"
                % (process.returncode, process.stderr.strip())
            )

    def _read_outputs(self, case: TestCase, config: AscendCommandConfig) -> Sequence[np.ndarray]:
        outputs: list[np.ndarray] = []
        for spec in config.outputs:
            target_path = config.workdir / spec.path
            if not target_path.exists():
                raise FileNotFoundError(
                    f"Ascend backend expected output file missing: {target_path}"
                )
            dtype = np.dtype(spec.dtype or case.dtype_spec[-1])
            data = np.fromfile(target_path, dtype=dtype)
            shape = case.shapes.get(spec.tensor)
            array = data.reshape(shape) if shape else data
            outputs.append(array)
        return tuple(outputs)


def _map_inputs(case: TestCase, inputs: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    mapping: Dict[str, np.ndarray] = {}
    for index, tensor in enumerate(inputs):
        mapping[f"input{index}"] = tensor
    return mapping


def _extract_config(case: TestCase) -> Optional[AscendCommandConfig]:
    backend_cfg = case.attributes.get("backend_config") or {}
    ascend_cfg = backend_cfg.get("ascend")
    if not ascend_cfg:
        return None
    workdir = Path(ascend_cfg.get("workdir", ".")).expanduser()
    command = _normalize_command(ascend_cfg.get("command"))
    inputs = [
        TensorFileSpec(
            tensor=spec["tensor"],
            path=Path(spec["path"]),
            dtype=spec.get("dtype"),
        )
        for spec in ascend_cfg.get("inputs", [])
    ]
    outputs = [
        TensorFileSpec(
            tensor=spec.get("tensor", f"output{idx}"),
            path=Path(spec["path"]),
            dtype=spec.get("dtype"),
        )
        for idx, spec in enumerate(ascend_cfg.get("outputs", []))
    ]
    if not inputs or not outputs:
        raise ValueError("Ascend backend requires both 'inputs' and 'outputs' mappings")
    env = {str(k): str(v) for k, v in (ascend_cfg.get("env") or {}).items()}
    return AscendCommandConfig(
        workdir=workdir,
        command=command,
        inputs=inputs,
        outputs=outputs,
        env=env,
    )


def _normalize_command(value) -> List[str]:
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(part) for part in value]
    raise ValueError("Ascend backend requires 'command' as string or list")
