# optest examples

This folder contains runnable examples using the new plan format.

- `vector_add/` – minimal vector add using the new plan and built-in generator/assertion.
- `ascend_operator/` – CANN-style add operator build driven by the plan.
- `custom_op_with_plugins/` – custom generator + assertion without plugins.
- `matmul_cpp/` – C++ matmul runner (float/int) showing how to wire a native binary to optest, with demo failure cases tagged `xfail-demo`.

Before running any example, install optest (editable or wheel):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .   # or: python -m build && pip install dist/optest-*.whl
```

Run each example from within its directory (so `plan.yaml` and `workdir` paths resolve), following the per-example README. `vector_add/plan.yaml` also includes an intentionally failing case tagged `fail`; run it explicitly with `--tags fail` if you want to see failure output.
