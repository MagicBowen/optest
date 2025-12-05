# optest

`optest` is a Python CLI that validates operators via a declarative YAML plan. It renders template tokens, generates inputs, runs your binary or script, and checks outputs using built-in or custom assertions. Reports are printed in color or emitted as JSON.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .   # or: python -m build && pip install dist/optest-*.whl
```

## Quick start
```bash
optest run --plan examples/vector_add/plan.yaml --backend cuda --chip local
```
`vector_add` uses `builtin.ones` to generate inputs, runs a tiny Python adder, and validates with `builtin.elementwise_add`.

## C++ matmul example (wrap your own entry)
1) Build the runner:
   ```bash
   cd examples/matmul_cpp/operator
   bash build.sh  # produces build/matmul_runner
   ```
2) Run the plan (float + int shapes):
   ```bash
   optest run --plan examples/matmul_cpp/plan.yaml --backend cuda --chip local
   ```
3) Failure demos (intentional errors):
   ```bash
   optest run --plan examples/matmul_cpp/plan.yaml --backend cuda --chip local --tags xfail-demo
   ```

Plan excerpt (tokens flow into the C++ runner):
```yaml
operator: matmul_cpp
inputs: ["data/input0.bin", "data/input1.bin"]
outputs: ["out/output0.bin"]
generator: {name: builtin.uniform, seed: 0}
assertion: {name: builtin.matmul}
backends:
  - type: cuda
    chip: local
    workdir: .
    command: ["./operator/build/matmul_runner",
              "--input0", "{input0}", "--input1", "{input1}",
              "--output0", "{output0}", "--dtype", "{dtype}", "--shapes", "{shapes}"]
cases:
  - name: float_small
    dtypes: [float32, float32]
    shapes: [{inputs: [[2,3],[3,4]], outputs: [[2,4]]}]
```

Wrapper snippet (see `examples/matmul_cpp/operator/matmul_runner.cpp`):
```cpp
int main(int argc, char** argv) {
    Options opts = parse_args(argc, argv);        // accepts --dtype/--input0/--input1/--output0/--shapes
    MatmulShape shape = parse_shapes(opts.shapes_json);
    if (opts.dtype == "float32") {
        run_matmul<float>(opts, shape);           // loads inputs, calls matmul_kernel, writes output
    } else if (opts.dtype == "int32") {
        run_matmul<int32_t>(opts, shape);
    } else {
        throw std::runtime_error("unsupported dtype");
    }
}
```
The compute kernel lives in `matmul_kernel.cpp` to keep math separate from optest IO/parsing.

## Plan file reference (paths relative to plan file if not absolute)
- `operator` (required)
- `description` (optional, default `""`)
- `inputs` (required): list of input file paths
- `outputs` (required): list of output file paths
- `generator` (optional, per-case override allowed, default `{name: builtin.random}`):
  - `name` (default `builtin.random`), `source` (Python path), `seed` (int | null), `params` (dict, default `{}`),
    `constants` (dict, default `{}`), `per_input` (dict index->generator, default `{}`)
- `assertion` (optional, per-case override allowed, default `{name: builtin.identity}`):
  - `name`, `source`, `rtol` (default builtin tolerance or `1e-5`), `atol` (default builtin tolerance or `1e-4`),
    `metric` (`max_abs` default), `output_dtypes` (defaults to case dtypes), `params` (dict, default `{}`)
- `backends` (required, non-empty list):
  - `type` (`cuda` | `cann`), `chip` (string), `workdir` (default plan dir),
    `env` (dict, default `{}`, templated), `timeout` (seconds, default `null`),
    `retries` (default `0`), `prepare` (list of commands, default `[]`),
    `cleanup` (list of commands, default `[]`), `command` (required),
    `only_cases`/`skip_cases`/`xfail_cases` (lists, default `[]`)
- `cases` (required, non-empty list):
  - `name`, `dtypes` (match `inputs` length), `shapes` (list of `{inputs, outputs}`),
    optional `generator`, `assertion`, `inputs`, `outputs`, `backends` (`{only, skip, xfail}` default empty),
    `tags` (list, default `[]`), `priority` (int | null, default plan priority)
- `cache` (optional, default `reuse`; `regen` forces new inputs)
- `tags` (optional list)
- `priority` (optional default priority for cases)

Templating tokens (rendered in `command`/`prepare`/`cleanup` and `env`): `{chip}`, `{backend}`, `{case}`, `{dtype}`, `{dtypes}`, `{shape}`,
`{shapes}`, `{input0}`/`{inputs}`, `{output0}`/`{outputs}`, `{workdir}`. Tokens are shell-escaped for argv; env keys/values are formatted without shell escaping.

Built-in generators: `builtin.random`, `builtin.uniform`, `builtin.ones` (support `constants` value/scale/shift).
Built-in assertions: all operators in `optest.operators.builtin_operators` plus `builtin.identity` (output self-check).

## CLI reference
`optest run [OPTIONS]`
- `--plan / --config PATH` (required): plan YAML.
- `--backend STRING`, `--chip STRING`: select a backend entry.
- `--cases STRING`: comma-separated globs.
- `--tags STRING`: comma-separated tags to include.
- `--skip-tags STRING`: comma-separated tags to skip.
- `--priority-max INT`: skip cases above this priority.
- `--cache [reuse|regen]`: override plan cache.
- `--list`: list matched cases without running.
- `--report [terminal|json]` and `--report-path PATH`: output format (default terminal).
- `--no-color`: disable ANSI colors.
- `--verbose`: extra logging (placeholder).
- Exit code: 0 on full success, 1 on failures/errors.

## Extend and adapt
- **Custom generator**: point to a Python file + function. Use `params/constants/seed` to drive behavior.
  ```yaml
  generator:
    name: my_gaussian
    source: ./custom_gen.py
    params: {mean: 0.0, std: 0.1}
    constants: {scale: 2.0, shift: 1.0}
  ```
  ```python
  # custom_gen.py
  import numpy as np

  def my_gaussian(*, input_paths, shapes, dtypes, params, seed, constants, rng):
      mean = float(params.get("mean", 0.0))
      std = float(params.get("std", 1.0))
      scale = float(constants.get("scale", 1.0))
      shift = float(constants.get("shift", 0.0))
      for path, shape, dtype in zip(input_paths, shapes["inputs"], dtypes):
          data = rng.normal(loc=mean, scale=std, size=shape).astype(dtype)
          (data * scale + shift).astype(dtype).tofile(path)
  ```

- **Custom assertion**: compute goldens however you like; return `AssertionResult` or `(ok, details)`.
  ```yaml
  assertion:
    name: my_assert
    source: ./custom_assert.py
    rtol: 1e-4
    atol: 1e-5
  ```
  ```python
  # custom_assert.py
  import numpy as np
  from optest.plan.models import AssertionResult

  def my_assert(*, input_paths, output_paths, shapes, dtypes, output_dtypes, params, rtol, atol, metric):
      x = np.fromfile(input_paths[0], dtype=dtypes[0]).reshape(shapes["inputs"][0])
      y = np.fromfile(output_paths[0], dtype=output_dtypes[0]).reshape(shapes["outputs"][0])
      golden = np.sin(x)
      if np.allclose(y, golden, rtol=rtol or 1e-5, atol=atol or 1e-4):
          return AssertionResult(ok=True, details="")
      diff = float(np.max(np.abs(y - golden)))
      return AssertionResult(ok=False, details=f"max_abs={diff}", metrics={"max_abs": diff})
  ```

- **Native operators**: accept CLI args for dtype/shape/IO paths and call your kernel. Example command in a plan:
  ```yaml
  command: ["./build/my_op", "--input0", "{input0}", "--output0", "{output0}", "--dtype", "{dtype}", "--shape", "{shape}"]
  ```
  Your C/C++/Rust/Go entry point only needs to parse these args, read/write raw binaries, and run the kernel (see `examples/matmul_cpp` for a full C++ reference).

- **Per-backend setup**: use templated env/prepare/cleanup.
  ```yaml
  backends:
    - type: cuda
      chip: local
      env: {CUDA_VISIBLE_DEVICES: "0", RUN_ID: "{case}-{shape}"}
      prepare: [["bash", "scripts/setup.sh", "{backend}", "{chip}"]]
      cleanup: [["bash", "scripts/teardown.sh", "{case}"]]
      command: ["./bin/run_op", "--input0", "{input0}", "--output0", "{output0}", "--dtype", "{dtype}", "--shape", "{shape}"]
  ```

- **Case selection for CI**: tag and filter.
  ```bash
  # run only smoke tests
  optest run --plan ./plan.yaml --backend cuda --chip local --tags smoke
  # skip slow tests
  optest run --plan ./plan.yaml --backend cuda --chip local --skip-tags slow
  # enforce priority ceiling
  optest run --plan ./plan.yaml --backend cuda --chip local --priority-max 5
  ```
