# optest examples

This folder contains runnable examples that show how to swap legacy test scripts for optest plans.

- `ascend_operator/` – a minimal Ascend-style operator build + optest plans (no legacy test scripts).
- `custom_op_with_plugins/` – shows how to add a user-defined generator/reference for an op not in builtins.

Before running any example, install optest (editable or wheel):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .   # or: python -m build && pip install dist/optest-*.whl
```

Run each example from within its directory (so `plan.yaml` and `workdir` paths resolve), following the per-example README.
