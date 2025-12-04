# optest config redesign requirements

This document defines the new optest plan format, command-line usage, and validation rules so multiple cases of the same operator share common configuration without duplication.

## Goals
- Reduce duplication across cases by promoting shared defaults to the plan level.
- Make data generation, golden generation, and execution templating explicit and extensible.
- Enable backend-aware selection, skips, and expected failures.
- Provide strong validation (schema/versioning, shape/dtype/IO consistency).
- Allow targeted execution via tags, priorities, and case filters.

## Execution plan (prioritized with dependencies)
1) Define data models and loader/validator for the new plan schema (operator/backends/cases, IO defaults, overrides, output dtype handling).  
2) Implement command templating plus backend runner (cann/cuda shells) and IO pipeline (generate inputs, ensure dirs, invoke command, read outputs).  
3) Implement generator/assertion resolution (builtin and custom via `source`+function name) and comparison logic (rtol/atol/metric).  
4) Wire CLI and selection (backend/chip, cases/tags/priority filters, cache/list modes) to the new plan loader.  
5) Update examples and docs to the new format; add unit tests for loader, templating, selection, and custom extensions.  
6) Execute examples as verification (golden path and failure cases).

## Plan YAML schema
Top-level fields:
- `operator` (required): logical operator name.
- `description` (optional): human-readable summary.
- `inputs` (required): default list of input file paths (relative to backend workdir unless absolute). Cases can override when layout or count differ.
- `outputs` (required): default list of output file paths emitted by the tested operator (relative to backend workdir unless absolute). The framework loads these files to compare against golden outputs. Cases can override when layout or count differ.
- `generator` (optional): default input data generator. If omitted, defaults to `builtin.random`.
- `assertion` (optional): default golden generator/comparator. If omitted, defaults to `builtin.identity` (compare backend output to provided golden).
- `backends` (required): list of backend definitions (only `cann` and `cuda` supported; new backends added by optest, not by users).
- `cases` (required): list of case groups describing dtype/shape combinations and overrides.
- `cache` (optional): `reuse` (default) or `regen` to control regeneration of inputs/goldens.
- `tags` (optional): list of strings used for selection.
- `priority` (optional): integer; lower is higher priority.

### File conventions
- Binaries are little-endian, row-major, tightly packed; no headers. Multi-output ops require one file per output.
- Paths are resolved relative to the backend `workdir`; if not set, relative to the plan file directory.
- optest must ensure parent directories for inputs/outputs exist before running commands.
- Default `inputs`/`outputs` live at the plan top level to avoid duplication; cases may override `inputs`/`outputs` when they need different file layouts or counts.

## Generators
Structure:
```
generator:
  name: builtin.random | builtin.uniform | builtin.ones | custom_fn_name
  source: ./custom_gen.py # required for custom; python file containing the function
  seed: 123            # optional, deterministic when set
  params: {}           # generator-specific parameters (e.g., low/high for uniform)
  per_input:           # optional per-input override by index
    "0": {name: builtin.random, seed: 1}
    "1": {name: builtin.ones}
  constants:           # optional constants/attributes passed to runner/assertion
    axis: 1
```
- Cases may override `generator` partially or entirely.
- Custom generators: set `name` to the Python function name and `source` to the file path; optest imports and calls it with the agreed prototype.

## Assertions (golden generation and comparison)
Structure:
```
assertion:
  name: builtin.elementwise_add | builtin.identity | custom_fn_name
  source: ./custom_assert.py   # required for custom; python file containing the function
  rtol: 1e-4          # optional
  atol: 1e-6          # optional
  metric: max_abs | mean_abs | cosine | custom
  output_dtypes: [float32]  # preferred place to declare output dtypes; defaults to case dtypes if omitted
  params: {}          # assertion-specific parameters
```
- Cases may override `assertion`.
- Output data types should be declared under `assertion.output_dtypes` (plan-level default), with optional per-case overrides when outputs differ from inputs (e.g., accumulation or casting ops).
- Golden data may be produced by running a reference implementation (builtin) or user-provided custom assertion.

## Custom generators and assertions (simple Python functions)
Users extend by pointing to a Python file and a function name; no packaging required. optest loads the file, finds the function, and executes it with the fixed prototypes below.

Custom generator prototype (writes input binaries):
```python
def my_generator(*, input_paths, shapes, dtypes, params, seed, constants, rng):
    """
    input_paths: list[str] paths to write; len matches plan inputs (or case override)
    shapes: list[list[int]] per input shape for the current shape entry
    dtypes: list[str] per input dtype for the current case
    params: dict merged from plan/case generator.params
    seed: int|None; already used to seed rng
    constants: dict from generator.constants (optional)
    rng: numpy.random.Generator seeded by optest (use instead of global)
    Write little-endian, row-major, headerless binaries to input_paths.
    """
```

Custom assertion prototype (generates goldens and compares):
```python
from typing import NamedTuple, Dict, Any, List

class AssertionResult(NamedTuple):
    ok: bool
    details: str = ""          # human-readable message
    metrics: Dict[str, Any] = {}  # optional numeric metrics

def my_assertion(*, input_paths, output_paths, shapes, dtypes, output_dtypes,
                 params, rtol, atol, metric):
    """
    input_paths: list[str] input binaries
    output_paths: list[str] operator-produced output binaries to validate
    shapes: dict with "inputs": [...], "outputs": [...] for current shape entry
    dtypes: list[str] per input dtype
    output_dtypes: list[str] per output dtype (from assertion.output_dtypes or inferred)
    params: dict merged from plan/case assertion.params
    rtol/atol: float tolerances
    metric: selected metric name
    Must return AssertionResult. Should read operator outputs, compute golden outputs
    (inline or via reference code), compare with tolerances/metric, and fill details/metrics.
    """
```

Usage in plan:
```yml
generator:
  name: my_generator
  source: ./custom_gen.py
assertion:
  name: my_assertion
  source: ./custom_assert.py
  output_dtypes: [float32]
```

## Backends
Allowed `type`: `cann` | `cuda`.
Structure:
```
- type: cann
  chip: Ascend910b              # required per backend entry
  workdir: /ascend/ops/add      # optional; defaults to plan directory
  env: {ASCEND_VISIBLE_DEVICES: "0"}   # optional per-backend env
  timeout: 300                  # seconds; optional
  retries: 1                    # optional
  prepare: [["./setup.sh"]]     # optional pre-commands run in workdir
  cleanup: [["./cleanup.sh"]]   # optional post-commands
  command: ["./run.sh", "-r", "cpu", "-v", "{chip}", "-d", "{dtype}", "-s", "{shape}", "-i", "{input0}", "-o", "{output0}"]
  only_cases: []                # optional allowlist of case names
  skip_cases: []                # optional skip list
  xfail_cases: []               # optional expected-fail list
```
Templating tokens available in `command`/`prepare`/`cleanup`:
- `{chip}`, `{backend}`.
- `{case}` (case name).
- `{dtype}` (for single dtype) or `{dtypes}` (comma-joined list).
- `{shape}` (flattened "NxM" form) or `{shapes}` (json-like string).
- `{input0}`, `{input1}`, ..., `{inputs}` (comma-joined paths).
- `{output0}`, `{output1}`, ..., `{outputs}` (comma-joined paths).
- `{workdir}` (resolved per backend).
Tokens are replaced literally; values must be shell-escaped by optest when invoking via a shell.

## Cases
Structure:
```
- name: float32_big
  dtypes: [float32, float32]      # per-input dtypes; outputs inferred unless overridden in assertion
  shapes:
    - inputs: [[1, 1024], [1, 1024]]
      outputs: [[1, 1024]]
    - inputs: [[1, 2056], [1, 2056]]
      outputs: [[1, 2056]]
  inputs: ["data/input0.bin", "data/input1.bin"]   # optional override
  outputs: ["output/output0.bin"]                  # optional override
  generator: {seed: 42}           # optional override
  assertion: {rtol: 1e-5}         # optional override
  backends:
    only: ["cann"]                # optional; mutually exclusive with skip
    skip: []                      # optional
    xfail: ["cuda"]               # optional expected fail list
  tags: ["smoke", "wide"]         # optional
  priority: 10                    # optional; lower runs first
```
- Each case is applied across all shapes listed.
- Validation: number of input shapes must match `inputs` length; output shapes must match `outputs` length; dtype count must match input count.

## Validation rules
- `inputs` and `outputs` lengths must match all shape entries.
- Dtype list length must equal number of inputs.
- Backend `type` must be one of the allowed types.
- Commands must reference only known tokens; unknown tokens fail validation.
- When `output_dtypes` provided, its length must match outputs.
- No case may be in both `only` and `skip`; no backend may be both in `skip` and `xfail`.
- Custom `generator`/`assertion` entries must provide `source` and `name` must resolve to a function in that file.

## CLI usage
Core:
```
optest run --plan vector_add.yaml --backend cann --chip Ascend910b
```
Selection and control:
```
optest run --plan vector_add.yaml --backend cann --chip Ascend910b \
  --cases float32_big,int32_tail \
  --tags smoke,wide --skip-tags slow \
  --priority-max 20 \
  --cache reuse|regen \
  --report terminal|json --report-path ./report.json \
  --list            # dry-run: list matched cases without executing
```
- `--cases` filters by case name (comma-separated, supports globs).
- `--tags` / `--skip-tags` filter by tags.
- `--priority-max` limits to cases with priority <= value.
- `--backend` and `--chip` must match a backend entry; case-level allow/skip/xfail is enforced.
- `--cache` overrides plan-level cache policy.
- `--report` selects terminal (default) or json; report format is CLI-only (not in YAML).
- Terminal output is colored by default; pass `--no-color` to disable.

## Example plan: vector_add.yaml
```yml
operator: vector_add
description: z = x + y; x, y, z are one-dimensional vectors
inputs: ["data/input0.bin", "data/input1.bin"]
outputs: ["output/output0.bin"]
generator:
  name: builtin.random
  seed: 123
assertion:
  name: builtin.elementwise_add
  rtol: 1e-5
  atol: 1e-7
backends:
  - type: cann
    chip: ascend910b
    workdir: /ascend/ops/add
    env: {ASCEND_VISIBLE_DEVICES: "0"}
    command: ["./run.sh", "-r", "cpu", "-v", "{chip}", "-d", "{dtypes}", "-s", "{shapes}", "-i", "{inputs}", "-o", "{outputs}"]
    timeout: 300
    retries: 1
  - type: cuda
    chip: H100
    workdir: /h100/ops/add
    command: ["./build/add", "-d", "{dtypes}", "-s", "{shapes}", "-i", "{inputs}", "-o", "{outputs}"]
cases:
  - name: float32_big
    dtypes: [float32, float32]
    shapes:
      - {inputs: [[1, 1024], [1, 1024]], outputs: [[1, 1024]]}
      - {inputs: [[1, 2056], [1, 2056]], outputs: [[1, 2056]]}
    tags: ["smoke"]
    priority: 5
  - name: int32_tail
    dtypes: [int32, int32]
    shapes:
      - {inputs: [[1, 1000], [1, 1000]], outputs: [[1, 1000]]}
    generator: {name: builtin.uniform, params: {low: -10, high: 10}}
    assertion: {rtol: 0, atol: 0}
    backends:
      xfail: ["cuda"]
    tags: ["wide"]
    priority: 20
```

## Execution example
Run cann smoke tests only:
```
optest run --plan vector_add.yaml --backend cann --chip Ascend910b --tags smoke
```
