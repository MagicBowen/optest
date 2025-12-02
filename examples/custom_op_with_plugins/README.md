# Custom operator with user generator/reference

This example shows how to test an operator not registered in optest’s built-ins by supplying your own generator and reference implementation, plus a tiny C++ operator that optest drives via the Ascend command backend.

## Files
- `custom_plugins.py`: contains `SequentialGenerator` and `square_reference`.
- `plugin.py`: registers the `custom_square` descriptor via `OPTEST_PLUGINS`.
- `operator/`: C++ program that reads `input/input0.bin`, squares it, and writes `output/output0.bin`.
- `plan.yaml`: declares two cases for `custom_square` with custom generator/reference and commands to run the binary.

## Prereqs
- From the repo root, install optest in editable mode so the examples package is importable:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -e .
  ```

## Build the operator
```bash
cd examples/custom_op_with_plugins/operator
bash build.sh
```
Binary is at `examples/custom_op_with_plugins/operator/build/custom_square`.

## Run
Load the plugin and run optest **from this example directory** so paths line up with the plan:
```bash
cd examples/custom_op_with_plugins
PYTHONPATH=$PWD OPTEST_PLUGINS=plugin optest run --plan ./plan.yaml --backend npu --chip ascend --no-color
```
Here `custom_square` is registered by the plugin and uses the plan’s `generator`/`reference` overrides. The Ascend command backend writes inputs, runs `build/custom_square`, and compares outputs to the reference.

## See a failure report
Run the failure plan to see optest surface a comparison mismatch (intentional bug writes unsquared data):
```bash
cd examples/custom_op_with_plugins
PYTHONPATH=$PWD OPTEST_PLUGINS=plugin optest run --plan ./plan_failure.yaml --backend npu --chip ascend --no-color
```
