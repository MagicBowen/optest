"""Ascend NPU backend which shells out to user-provided run scripts."""
from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from optest.core import TestCase

from .base import BackendDriver


@dataclass
class TensorFileSpec:
    tensor: str
    path: Path
    dtype: Optional[str] = None


@dataclass
class CommandSpec:
    argv: List[str]


@dataclass
class AscendCommandConfig:
    workdir: Path
    command: List[str]
    build_commands: List[CommandSpec]
    pre_run_commands: List[CommandSpec]
    post_run_commands: List[CommandSpec]
    inputs: List[TensorFileSpec]
    outputs: List[TensorFileSpec]
    golden: List[TensorFileSpec]
    env: Dict[str, str]


class AscendBackendDriver(BackendDriver):
    """Backend that runs Ascend operator binaries via user scripts."""

    name = "ascend-command"
    kind = "npu"

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
        self._run_commands(config.build_commands, config)
        self._write_inputs(case, inputs, config)
        self._run_commands(config.pre_run_commands, config)
        self._run_single_command(config, config.command)
        outputs = self._read_outputs(case, config)
        self._run_commands(config.post_run_commands, config)
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
            alias = _legacy_input_alias(spec.tensor, spec.path)
            if alias:
                alias_path = config.workdir / alias
                if alias_path != target_path:
                    alias_path.parent.mkdir(parents=True, exist_ok=True)
                    array.tofile(alias_path)

    def _run_commands(self, commands: Sequence[CommandSpec], config: AscendCommandConfig) -> None:
        for command in commands:
            self._run_single_command(config, command.argv)

    def _run_single_command(self, config: AscendCommandConfig, argv: Sequence[str]) -> None:
        env = os.environ.copy()
        env.update(config.env)
        process = subprocess.run(
            list(argv),
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
            target_path = _resolve_output_path(config, spec)
            dtype = np.dtype(spec.dtype or case.dtype_spec[-1])
            data = np.fromfile(target_path, dtype=dtype)
            shape = case.shapes.get(spec.tensor)
            array = data.reshape(shape) if shape else data
            outputs.append(array)
        return tuple(outputs)


def _map_inputs(case: TestCase, inputs: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    names = _input_tensor_names(case, len(inputs))
    if len(names) != len(inputs):
        raise ValueError(
            f"Ascend backend expected {len(names)} inputs based on shapes but received {len(inputs)} tensors"
        )
    return {name: tensor for name, tensor in zip(names, inputs)}


def _extract_config(case: TestCase) -> Optional[AscendCommandConfig]:
    backend_cfg = case.attributes.get("backend_config") or {}
    ascend_cfg = backend_cfg.get("ascend")
    if not ascend_cfg:
        return None
    workdir = Path(ascend_cfg.get("workdir", ".")).expanduser()
    command = _normalize_command(ascend_cfg.get("command"))
    build_commands = _normalize_command_list(ascend_cfg.get("build"))
    pre_run_commands = _normalize_command_list(ascend_cfg.get("pre_run"))
    post_run_commands = _normalize_command_list(ascend_cfg.get("post_run"))
    inputs = _normalize_inputs(ascend_cfg.get("inputs"), case)
    outputs = _normalize_outputs(ascend_cfg.get("outputs"), case)
    golden = [
        TensorFileSpec(
            tensor=spec.get("tensor", f"output{idx}"),
            path=Path(spec["path"]),
            dtype=spec.get("dtype"),
        )
        for idx, spec in enumerate(_normalize_golden(ascend_cfg.get("golden")))
    ]
    env = {str(k): str(v) for k, v in (ascend_cfg.get("env") or {}).items()}
    return AscendCommandConfig(
        workdir=workdir,
        command=command,
        build_commands=build_commands,
        pre_run_commands=pre_run_commands,
        post_run_commands=post_run_commands,
        inputs=inputs,
        outputs=outputs,
        golden=golden,
        env=env,
    )


def _normalize_command(value) -> List[str]:
    if value is None:
        raise ValueError("Ascend backend requires 'command' as string, list, or mapping")
    if isinstance(value, (str, os.PathLike)):
        return shlex.split(str(value))
    if isinstance(value, list):
        return [str(part) for part in value]
    if isinstance(value, dict):
        executable = value.get("binary") or value.get("executable")
        if not executable:
            raise ValueError("Ascend backend 'command' mapping requires 'binary' or 'executable'")
        args = value.get("args", [])
        if isinstance(args, (str, os.PathLike)):
            args_list = [str(args)]
        elif isinstance(args, list):
            args_list = [str(part) for part in args]
        else:
            raise ValueError("Ascend backend 'args' must be a list or string")
        return [str(executable)] + args_list
    raise ValueError("Ascend backend requires 'command' as string, list, or mapping")


def _normalize_command_list(raw) -> List[CommandSpec]:
    commands: List[CommandSpec] = []
    if not raw:
        return commands
    if isinstance(raw, (str, os.PathLike, list, dict)):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError("Ascend backend commands must be a list")
    for entry in raw:
        argv = _normalize_command(entry)
        commands.append(CommandSpec(argv=argv))
    return commands


def _normalize_inputs(raw, case: TestCase) -> List[TensorFileSpec]:
    if raw is None:
        raw = _default_inputs(case)
    if isinstance(raw, dict):
        raw = [
            {"tensor": name, "path": value} if not isinstance(value, dict) else {"tensor": name, **value}
            for name, value in raw.items()
        ]
    return [
        TensorFileSpec(
            tensor=spec["tensor"],
            path=Path(spec["path"]),
            dtype=spec.get("dtype"),
        )
        for spec in raw
    ]


def _normalize_outputs(raw, case: TestCase) -> List[TensorFileSpec]:
    if raw is None:
        raw = _default_outputs(case)
    if isinstance(raw, dict):
        raw = [
            {"tensor": name, "path": value} if not isinstance(value, dict) else {"tensor": name, **value}
            for name, value in raw.items()
        ]
    return [
        TensorFileSpec(
            tensor=spec.get("tensor", f"output{idx}"),
            path=Path(spec["path"]),
            dtype=spec.get("dtype"),
        )
        for idx, spec in enumerate(raw)
    ]


def _normalize_golden(raw) -> List[dict]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [{"tensor": name, "path": value} if not isinstance(value, dict) else {"tensor": name, **value} for name, value in raw.items()]
    if not isinstance(raw, list):
        raise ValueError("Ascend backend 'golden' must be a list or mapping")
    return raw


def load_golden_tensors(case: TestCase) -> Optional[Sequence[np.ndarray]]:
    """Load golden tensors from backend_config.ascend.golden if provided."""

    config = _extract_config(case)
    if config is None or not config.golden:
        return None
    golden: list[np.ndarray] = []
    for spec in config.golden:
        target_path = _resolve_output_path(config, spec, allow_missing=True)
        if not target_path:
            return None
        dtype = np.dtype(spec.dtype or case.dtype_spec[-1])
        data = np.fromfile(target_path, dtype=dtype)
        shape = case.shapes.get(spec.tensor)
        array = data.reshape(shape) if shape else data
        golden.append(array)
    return tuple(golden)


def dump_golden_tensors(
    case: TestCase, outputs: Sequence[np.ndarray], *, overwrite: bool = False
) -> None:
    """Persist expected outputs to golden files if configured."""

    config = _extract_config(case)
    if config is None or not config.golden:
        return
    if len(config.golden) != len(outputs):
        raise ValueError("Golden spec count does not match outputs")
    for spec, array in zip(config.golden, outputs):
        target_path = config.workdir / spec.path
        if target_path.exists() and not overwrite:
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dtype = np.dtype(spec.dtype or array.dtype)
        np.asarray(array, dtype=dtype).tofile(target_path)


def _legacy_input_alias(tensor: str, default_path: Path) -> Optional[Path]:
    if tensor == "input0":
        return default_path.parent / "input_x.bin"
    if tensor == "input1":
        return default_path.parent / "input_y.bin"
    return None


def _legacy_output_alias(spec: TensorFileSpec) -> Optional[Path]:
    if spec.tensor == "output0":
        return spec.path.parent / "output_z.bin"
    return None


def _resolve_output_path(
    config: AscendCommandConfig, spec: TensorFileSpec, *, allow_missing: bool = False
) -> Optional[Path]:
    target_path = config.workdir / spec.path
    if target_path.exists():
        return target_path
    alias = _legacy_output_alias(spec)
    if alias:
        alias_path = config.workdir / alias
        if alias_path.exists():
            return alias_path
    if allow_missing:
        return None
    raise FileNotFoundError(f"Ascend backend expected output file missing: {target_path}")


def _default_inputs(case: TestCase) -> List[dict]:
    specs: List[dict] = []
    for name in _input_tensor_names(case, case.descriptor.num_inputs):
        idx = int(name.replace("input", "")) if name.startswith("input") and name[5:].isdigit() else None
        dtype = None
        if idx is not None and idx < len(case.dtype_spec):
            dtype = case.dtype_spec[idx]
        specs.append({"tensor": name, "path": f"input/{name}.bin", "dtype": dtype})
    return specs


def _default_outputs(case: TestCase) -> List[dict]:
    specs: List[dict] = []
    for name in _output_tensor_names(case, case.descriptor.num_outputs):
        specs.append({"tensor": name, "path": f"output/{name}.bin"})
    return specs


def _input_tensor_names(case: TestCase, fallback: int) -> List[str]:
    names = _tensor_names_from_shapes(case.shapes, prefix="input")
    if names:
        return names
    return [f"input{idx}" for idx in range(fallback)]


def _output_tensor_names(case: TestCase, fallback: int) -> List[str]:
    names = _tensor_names_from_shapes(case.shapes, prefix="output")
    if names:
        return names
    return [f"output{idx}" for idx in range(fallback)]


def _tensor_names_from_shapes(shapes: Mapping[str, object], prefix: str) -> List[str]:
    candidates = [name for name in shapes.keys() if str(name).startswith(prefix)]
    def _suffix_key(name: str) -> tuple[int, str]:
        suffix = name[len(prefix):]
        return (int(suffix), "") if suffix.isdigit() else (1_000_000, suffix)
    return sorted(candidates, key=_suffix_key)
