# optest

optest is a Python CLI for validating AI operators across GPU/NPU targets. It generates inputs, runs your operator binary/script, computes NumPy references (or user references), and reports detailed pass/fail results (TTY and JSON).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .   # or: python -m build && pip install dist/optest-*.whl
```

## Core concepts

- **Plan (YAML)**: describes backend, chip, and test cases. Each case sets `op`, `dtypes`, `shapes`, optional `attributes`, and `backend_config`.
- **Backends**: optest runs through a backend driver. The Ascend command backend shells out to your build output in a given `workdir`.
- **Generators/References**: optest can use built-ins, or you can supply dotted-path overrides or register plugins.

### Plan anatomy (common fields)

```yaml
backend: npu            # gpu or npu
chip: ascend            # chip label
report:
  format: terminal      # or json
cases:
  - op: elementwise_add
    dtypes: [float32, float32]
    shapes:
      input0: [2, 3]
      input1: [2, 3]
      output0: [2, 3]
    attributes:
      backend_config:
        ascend:
          workdir: ./operator  # path where command runs, often in the directory of the current operator
          command: ["./build/add_custom", "--dtype", "float32"]
          # optional env, inputs/outputs if you need custom paths
```

optest writes inputs to `input/input{idx}.bin` (and reads outputs from `output/output{idx}.bin`) unless you override `inputs`/`outputs`.

## How to test your own operator (step-by-step)

1) **Build your binary** in your operator folder (example: `operator/build.sh` → `./build/add_custom`).
2) **Write `plan.yaml`** next to your operator folder:
   - Set `backend: npu`, `chip: ascend`.
   - Under the case, set `backend_config.ascend.workdir` to the operator folder (relative is fine, e.g., `./operator`).
   - Set `command` to your binary invocation (array form recommended).
   - Add `inputs`/`outputs` only if your binary expects non-default paths.
3) **Run optest from the directory containing `plan.yaml`**:
   ```bash
   optest run --plan ./plan.yaml --no-color
   ```
4) **Inspect results**: passes show per-case summaries; failures show mismatched counts, max_abs/max_rel, failing indices, and stderr from your command.
5) **Iterate**: fix your operator code, rebuild, rerun optest.

## Example scenarios
### Ascend operator (C++)

Located in `examples/ascend_operator/`.

```bash
cd examples/ascend_operator/operator
bash build.sh
cd ..
optest run --plan ./plan.yaml --backend npu --chip ascend --no-color
```

The plan runs two cases (float32/int32) against `./operator/build/add_custom`.

### Custom operator with plugin (non-builtin op)

Located in `examples/custom_op_with_plugins/`.

```bash
cd examples/custom_op_with_plugins/operator
bash build.sh
cd ..
PYTHONPATH=$PWD OPTEST_PLUGINS=plugin optest run --plan ./plan.yaml --backend npu --chip ascend --no-color
```

The plugin registers `custom_square` and supplies generator/reference defaults. `plan_failure.yaml` uses a buggy runner to demonstrate mismatch reporting.

## Custom generators and references

- Per-case overrides: set `generator:` or `reference:` to a dotted path.
- Plugin registration: set `OPTEST_PLUGINS=your.module` and implement `register()` to add descriptors or defaults (see `examples/custom_op_with_plugins/plugin.py`).

## Reporting

- Terminal (default): streaming status plus failure blocks with shapes, dtypes, tolerances, seeds, stderr.
- JSON: `optest run --plan plan.yaml --report json --report-path ./reports/run.json`.

## Useful commands
、
- Run all tests: `PYTHONPATH=src pytest`
- Build package: `python -m build`
- Quick single op run (no plan): `optest run --op elementwise_add --dtype float32,float32 --shape input0=2x2 --shape input1=2x2`

## Coding & layout tips

- Before using optest，you should compile operator first, usually the operator binary is under `./build/`, you should config the executable binary in plans for optest invoke it. You can also wrapper the operator binary in python or shell (for env config or other preparing action), then config the python/shell to optest.
- Use default IO paths when possible; only override `inputs`/`outputs` if your binary uses custom filenames.
- For Ascend, set env in `backend_config.ascend.env` only if your binary or wrapper script needs it; optest itself doesn’t require RUN_MODE/SOC_VERSION.

## Tutorial: IO paths, built-ins, and plugins

### 1) Specifying input/output paths

- Default: optest writes `input/input{idx}.bin` and reads `output/output{idx}.bin` (also aliases `input_x.bin`/`output_z.bin` for the first tensors).
- Custom paths: map tensor names to files under `backend_config.ascend.inputs`/`outputs`:
  ```yaml
  backend_config:
    ascend:
      workdir: ./operator
      command: ["./build/my_op", "--dtype", "float32"]
      inputs:
        input0: data/in0.bin
        input1: data/in1.bin
      outputs:
        output0: results/out.bin
  ```
  Names must align with your op (`input0`, `input1`, … and `output0`, …). optest generates data, writes to `inputs`, runs your command, then reads `outputs`.

### 2) Using built-in operators (no custom reference needed)

optest ships NumPy references for common ops. Set `op`, `dtypes`, and `shapes`; no `reference` needed. 

Examples: 
- `elementwise_add/sub/mul/div`
- comparisons (`equal/greater/less/less_equal/greater_equal`)
- `vector_dot`, `vector_sum`, `vector_norm`, `matmul`, `gemm`, `conv2d`, `maxpool2d`, `avgpool2d`
- activations (`relu`, `leaky_relu`, `sigmoid`, `tanh`, `softmax`)
- reductions (`reduce_sum`, `reduce_mean`), `broadcast_to`. 

Sample case:

```yaml
- op: reduce_sum
  dtypes: [float32]
  shapes: {input0: [4, 4], output0: [4, 4]}
  attributes:
    backend_config:
      ascend:
        workdir: ./operator
        command: ["./build/reduce_bin", "--dtype", "float32"]
```

### 3) Creating a plugin with custom generator/reference

If you operator is not supported by optest builtins, you can develop a plugin for optest:

1) Create `plugins/my_plugin.py`:

   ```python
   from optest.core import OperatorDescriptor
   from optest.registry import registry
   import numpy as np

   def register():
       registry.register(
           OperatorDescriptor(
               name="my_custom_op",
               category="example",
               num_inputs=1,
               dtype_variants=(("float32",),),
               default_generator="plugins.my_plugin:my_generator",
               default_reference="plugins.my_plugin:my_reference",
           )
       )

   def my_generator(case, rng):
       shape = case.shapes["input0"]
       data = np.arange(int(np.prod(shape)), dtype=case.dtype_spec[0]).reshape(shape)
       return [data], None

   def my_reference(inputs, attrs):
       (x,) = inputs
       return (np.square(x),)
   ```

2) Write `plan.yaml` next to your operator:

   ```yaml
   backend: npu
   chip: ascend
   cases:
     - op: my_custom_op
       dtypes: [float32]
       shapes: {input0: [2, 2], output0: [2, 2]}
       attributes:
         backend_config:
           ascend:
             workdir: ./operator
             command: ["./build/my_custom_bin", "--dtype", "float32"]
   ```

3) Run from the plan directory:

   ```bash
   PYTHONPATH=$PWD OPTEST_PLUGINS=plugins.my_plugin optest run --plan ./plan.yaml --backend npu --chip ascend --no-color
   ```
optest will generate inputs via your generator, run your binary, and compare outputs to your reference.
