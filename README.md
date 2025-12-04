# optest

optest is a Python CLI for validating AI operators across GPU/NPU targets using a declarative YAML plan. A plan describes inputs/outputs, generators, assertions, backends/commands, and test cases. optest renders templates, generates data, runs your command, and checks results with colored terminal output or JSON.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .   # or: python -m build && pip install dist/optest-*.whl
```

## Plan format (single source of truth)
Top-level fields:
```yaml
operator: vector_add                  # required
description: z = x + y                # optional
inputs: ["data/input0.bin", "data/input1.bin"]   # required; defaults for all cases
outputs: ["output/output0.bin"]                   # required; defaults for all cases
generator:                            # optional; defaults to builtin.random
  name: builtin.random
  seed: 123
  params: {low: -1, high: 1}
  constants: {value: null, scale: 1.0, shift: 0.0}  # built-ins honor these
  per_input:
    "0": {name: builtin.uniform, params: {low: 0, high: 1}}
assertion:                            # optional; defaults to builtin.identity
  name: builtin.elementwise_add
  rtol: 1e-5
  atol: 1e-7
  metric: max_abs                     # max_abs or mean_abs supported
  output_dtypes: [float32]
backends:                             # required; supports type: cuda | cann
  - type: cuda
    chip: local
    workdir: .
    env: {CUDA_VISIBLE_DEVICES: "0"}
    timeout: 300
    retries: 1
    prepare: [["echo", "prep"]]
    cleanup: [["echo", "done"]]
    command: ["./run.sh", "-d", "{dtypes}", "-s", "{shapes}", "-i", "{inputs}", "-o", "{outputs}"]
    only_cases: []                    # optional allowlist
    skip_cases: []                    # optional skip list
    xfail_cases: []                   # optional expected-fail list
cases:                                # required
  - name: float32_basic
    dtypes: [float32, float32]        # len matches inputs
    shapes:                           # each entry defines a shape combo
      - {inputs: [[1, 4], [1, 4]], outputs: [[1, 4]]}
    generator: {seed: 42}             # optional override
    assertion: {rtol: 1e-5}           # optional override
    inputs: ["data/input0.bin", "data/input1.bin"]   # optional override
    outputs: ["output/output0.bin"]                  # optional override
    backends: {only: [], skip: [], xfail: []}        # optional per-case filter
    tags: ["smoke"]
    priority: 10
cache: reuse                          # reuse | regen (optional)
tags: ["plan-tag"]                    # optional
priority: 1                           # optional; default priority for cases
```

Templating tokens (available in `command`/`prepare`/`cleanup`): `{chip}`, `{backend}`, `{case}`, `{dtype}`, `{dtypes}`, `{shape}`, `{shapes}`, `{input0}`/`{inputs}`, `{output0}`/`{outputs}`, `{workdir}`. Values are formatted then shell-escaped.

Built-in generators: `builtin.random`, `builtin.uniform`, `builtin.ones` (honor `constants` value/scale/shift). Custom generators/assertions: set `name` to your function and `source` to a Python file; optest loads and calls:
- Generator function signature: `fn(input_paths, shapes, dtypes, params, seed, constants, rng)`
- Assertion function signature: `fn(input_paths, output_paths, shapes, dtypes, output_dtypes, params, rtol, atol, metric)` returning `AssertionResult(ok, details, metrics)` or `(ok, details)`.

## CLI usage
```bash
optest run --plan ./plan.yaml --backend cuda --chip local \
  --cases float32_basic --tags smoke --skip-tags slow \
  --priority-max 10 --cache reuse|regen \
  --report terminal|json [--report-path path] [--list] [--no-color]
```
- `--backend/--chip` select a backend entry; must match plan backends.
- `--cases` (globs), `--tags`, `--skip-tags`, `--priority-max` filter what runs.
- `--cache` overrides plan cache.
- `--report` chooses terminal (default) or JSON; `--report-path` writes JSON to a file.
- `--list` prints matched cases without running.
- Colors are on by default; use `--no-color` to disable.

## Execution output formats
- Terminal: colored, aligned lines per case: `PASS/FAIL/XFAIL/XPASS/ERROR <case@backend:chip/shape_idx>` with indented details/metrics, plus a summary `total=<n> passed=<p> xfail=<xf> failed=<f>`.
- JSON: inline-schema validated object with `summary` (total/failures) and `cases` (id, status, details, metrics, xfail).

## User extension points
- **Generators** (Python function): point to a file + function; no packaging needed.
  ```yaml
  generator:
    name: my_generator
    source: ./custom_gen.py
  ```
  ```python
  # custom_gen.py
  import numpy as np
  def my_generator(*, input_paths, shapes, dtypes, params, seed, constants, rng):
      for path, shape, dtype in zip(input_paths, shapes["inputs"], dtypes):
          data = rng.standard_normal(size=shape).astype(dtype)
          np.full(shape, params.get("fill", 0), dtype=dtype)  # use params/constants if desired
          np.asarray(data).tofile(path)
  ```
- **Assertions** (Python function): compute goldens and compare outputs.
  ```yaml
  assertion:
    name: my_assertion
    source: ./custom_assert.py
    rtol: 1e-4
    atol: 1e-5
  ```
  ```python
  # custom_assert.py
  import numpy as np
  from optest.plan.models import AssertionResult
  def my_assertion(*, input_paths, output_paths, shapes, dtypes, output_dtypes, params, rtol, atol, metric):
      x = np.fromfile(input_paths[0], dtype=dtypes[0]).reshape(shapes["inputs"][0])
      out = np.fromfile(output_paths[0], dtype=output_dtypes[0]).reshape(shapes["outputs"][0])
      golden = np.square(x) + 1
      if np.allclose(out, golden, rtol=rtol or 0, atol=atol or 0):
          return AssertionResult(ok=True, details="")
      diff = float(np.max(np.abs(out - golden)))
      return AssertionResult(ok=False, details=f"max_abs={diff}", metrics={"max_abs": diff})
  ```
- **Operators** (plugins): register new descriptors before running.
  ```python
  # my_plugin.py
  from optest.core import OperatorDescriptor, Tolerance
  from optest.operators import catalog
  def register():
      desc = OperatorDescriptor(
          name="my_op",
          category="custom",
          num_inputs=1,
          dtype_variants=(("float32",),),
          default_reference="my_plugin:my_reference",
          default_tolerance=Tolerance(absolute=1e-5, relative=1e-5),
      )
      catalog._catalog[desc.name] = desc  # or provide a helper to register cleanly
  def my_reference(inputs, attrs):
      (x,) = inputs
      return (x + 1,)
  ```
  Run with `OPTEST_PLUGINS=my_plugin optest run --plan ...`.
- **Backends**: implement and register a `BackendDriver` to target new hardware; or rely on command backends in the plan. Example skeleton:
  ```python
  from optest.backends import BackendDriver, backend_manager
  class MyBackend(BackendDriver):
      name = "my-backend"
      kind = "gpu"
      chips = ("x1",)
      def run(self, case, inputs):
          # launch your kernel; return numpy outputs
          return inputs
  backend_manager.register(MyBackend())
  ```

## How to adapt your code to optest
1) **Build your operator** (binary or script) so it accepts CLI args for dtype/shape/IO paths.
2) **Author a plan** next to the binary:
   ```yaml
   operator: my_add
   inputs: ["input/in0.bin", "input/in1.bin"]
   outputs: ["output/out0.bin"]
   backends:
     - type: cuda
       chip: local
       workdir: .
       command: ["./build/my_add", "--dtype", "{dtypes}", "--shape", "{shapes}", "--in0", "{input0}", "--in1", "{input1}", "--out0", "{output0}"]
   cases:
     - name: f32
       dtypes: [float32, float32]
       shapes: [{inputs: [[1, 1024], [1, 1024]], outputs: [[1, 1024]]}]
       tags: ["smoke"]
   ```
3) **Run optest** from the plan directory:
   ```bash
   optest run --plan ./plan.yaml --backend cuda --chip local --tags smoke
   ```
4) **Read results**:
   - Terminal: aligned colored statuses, details/metrics; summary at end.
   - JSON: `--report json --report-path report.json` produces a schema-validated report.
5) **Iterate**: update your binary or plan (add cases/tags, custom generator/assertion) and rerun.

Everything you need is in this README; no external examples required.
