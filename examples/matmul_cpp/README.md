# C++ Matmul example

This example shows how to wrap a simple C++ matmul entry point so it can be driven by optest. The binary reads inputs written by optest, respects the dtype/shape provided by the plan templating tokens, and writes the result back to disk.

## Layout
- `operator/matmul_kernel.cpp` and `operator/matmul_kernel.h`: pure compute kernel (`C = A x B`) with explicit instantiations for `float32` and `int32`.
- `operator/matmul_runner.cpp`: optest-facing wrapper that parses CLI args, reads inputs, validates shapes, calls the kernel, and writes the output.
- `operator/CMakeLists.txt`: build rules for the runner.
- `operator/build.sh`: convenience script to configure and build.
- `plan.yaml`: optest plan targeting the runner with multiple shapes and dtypes.

## Build
```bash
cd examples/matmul_cpp/operator
bash build.sh
# binary is at ./build/matmul_runner
```

## Plan walkthrough
- `inputs` / `outputs` are relative to the plan directory.
- `generator`: `builtin.uniform` with a fixed seed to produce deterministic inputs.
- `assertion`: `builtin.matmul` uses the built-in NumPy reference to check results.
- `backends[0].command`: passes optest tokens into the runner:
  - `{input0}`, `{input1}`, `{output0}`: data file paths.
  - `{dtype}`: `float32` or `int32`.
  - `{shapes}`: JSON string of all input/output shapes (parsed by the runner).
- `cases`: two dtype groups (`float_small` for floats, `int_basic` for ints) each with multiple shapes.
- `cache: regen`: inputs are regenerated per shape so a single set of paths can be reused safely.
- Two negative cases are tagged `xfail-demo`:
  - `bad_shape_output`: output shape intentionally wrong.
  - `bad_dtype`: uses `float16`, which the runner rejects.

## How the runner parses shapes
`{shapes}` is the JSON emitted by optest, e.g. `{"inputs": [[2, 3], [3, 4]], "outputs": [[2, 4]]}`. The runner extracts the numbers in order, validates:
- `A` is `m x k`
- `B` is `k x n`
- output is `m x n`

This keeps the example dependency-free (no external JSON library) while still consuming optest’s templated shapes.

## Run with optest
```bash
# From repo root, after building the runner:
optest run --plan examples/matmul_cpp/plan.yaml --backend cuda --chip local

# To run only the failure demos (expect non-zero exit):
optest run --plan examples/matmul_cpp/plan.yaml --backend cuda --chip local --tags xfail-demo

# To exclude failures:
optest run --plan examples/matmul_cpp/plan.yaml --backend cuda --chip local --skip-tags xfail-demo
```

The `cuda` backend here is just the command runner; no CUDA toolchain is required for the example.
