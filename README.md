# optest

optest is a Python CLI for validating AI operators across GPU/NPU targets using a declarative YAML plan (new format only). Plans describe inputs/outputs, generators, assertions, backends/commands, and test cases; optest renders templates, generates data, runs your command, and checks results.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .   # or: python -m build && pip install dist/optest-*.whl
```

## Plan at a glance
```yaml
operator: vector_add
description: z = x + y
inputs: ["data/input0.bin", "data/input1.bin"]
outputs: ["output/output0.bin"]
generator: {name: builtin.random, seed: 123}
assertion: {name: builtin.elementwise_add, rtol: 1e-5, atol: 1e-7}
backends:
  - type: cuda
    chip: local
    workdir: .
    command: ["./run.sh", "-d", "{dtypes}", "-s", "{shapes}", "-i", "{inputs}", "-o", "{outputs}"]
cases:
  - name: float32_basic
    dtypes: [float32, float32]
    shapes:
      - {inputs: [[1, 4], [1, 4]], outputs: [[1, 4]]}
    tags: ["smoke"]
```
Key points:
- Defaults live at the plan level (`inputs`, `outputs`, `generator`, `assertion`) and can be overridden per case.
- Command templating tokens include `{chip}`, `{backend}`, `{case}`, `{dtype}`/`{dtypes}`, `{shape}`/`{shapes}`, `{input0}`/`{inputs}`, `{output0}`/`{outputs}`, `{workdir}`.
- Custom generators/assertions: set `name` to your function and `source` to the Python file; optest loads and calls it with the agreed prototypes.

## CLI
```bash
optest run --plan ./plan.yaml --backend cuda --chip local \
  --cases float32_basic --tags smoke --skip-tags slow \
  --priority-max 10 --cache reuse|regen \
  --report terminal|json [--report-path path] [--list] [--no-color]
```
- `--backend/--chip` select a backend entry.
- `--cases` (globs), `--tags`, `--skip-tags`, `--priority-max` filter what runs.
- `--cache` overrides plan cache (`reuse`/`regen`).
- `--report` chooses terminal (default) or JSON; report format is CLI-only.
- Colors are on by default; use `--no-color` to disable.

## Examples
- `examples/vector_add_v2/`: built-in generator/assertion.
- `examples/ascend_operator/`: Ascend-style binary driven by the new plan (build C++ binary first).
- `examples/custom_op_with_plugins/`: custom generator + assertion via `source`+function names.
- `vector_add_v2/` includes an intentionally failing case tagged `fail` (select with `--tags fail` if desired).

Run from each example directory so relative paths resolve:
```bash
cd examples/vector_add_v2
optest run --plan ./plan.yaml --backend cuda --chip local --tags smoke
```
