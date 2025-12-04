# Custom generator and assertion (plan)

This example shows how to extend optest with a user-defined generator and assertion using the new plan format. No plugins or packaging needed—point the plan to a Python file and a function name.

## Files
- `plan.yaml`: new-format plan pointing to `custom_impl.py` for generator + assertion.
- `custom_impl.py`: implements `custom_generator` (writes deterministic inputs) and `custom_assertion` (computes golden = x^2 + 1 and compares).
- `run.py`: simple runner that squares the input and adds 1.

## Run
```bash
cd examples/custom_op_with_plugins
optest run --plan ./plan.yaml --backend cuda --chip local --tags custom
```

Use `--report json --report-path report.json` to emit a JSON report.
