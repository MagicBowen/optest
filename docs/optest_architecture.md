# optest Architecture & Extension Guide

## 1. High-Level Software Architecture

```mermaid
graph LR
    A[CLI & Config Layer] -->|RunOptions| B(Plan Loader)
    B -->|TestCases| C[Test Runner]
    C -->|Invokes| D[Generators]
    C -->|Invokes| E[Backends]
    C -->|Compares| F[Comparator]
    C -->|Results| G[Report Manager]
    G --> H[Terminal Output]
    G --> I[JSON Output]
    subgraph Catalog & References
        J[Operator Catalog]
        K[Reference Impl Resolver]
    end
    B --> J
    D --> J
    E --> J
    E --> K
    C --> K
```

- **CLI & Config Layer** – parses flags (`--plan`, `--backend`, `--chip`, `--cases`, `--tags`, `--report`, etc.), loads YAML plans, and builds execution options.
- **Plan Loader** – validates YAML against a JSON Schema, normalizes cases (dtype/shape/IO checks), and prepares backend configs.
- **Operator Catalog** – stores metadata for built-in operators (descriptors + reference implementations) and can be extended via plugins.
- **Generators** – create deterministic NumPy inputs and, optionally, golden outputs. Built-in generators support `constants` (`value`, `scale`, `shift`) and can be replaced by user functions via `source`+`name`.
- **Reference Resolver** – loads NumPy-only reference implementations and built-in operator classes; used when generators don’t emit goldens.
- **Backends** – encapsulate kernel execution (CUDA, CANN, etc.) via command templates and lifecycle hooks.
- **Test Runner** – orchestrates generator → backend → assertion flow per test case, collects metrics, and streams structured results.
- **Outputs** – terminal output (colored, aligned) and JSON output validated against an inline schema.

## 2. Core Design Concepts

### 2.1 Operator Descriptor & Test Case
- `OperatorDescriptor` captures static metadata (name, category, dtype tuples, attribute schema, generator/reference hooks, tolerance defaults).
- `TestCase` binds descriptors to concrete dtype tuples, shapes, backend targets, user attributes, tolerances, and optional overrides.

### 2.2 Generators & Reference Hooks
- Generators must expose a `.generate(case, rng)` method returning `(inputs, expected_outputs|None)`. Built-in generators support `constants` (`value`, `scale`, `shift`).
- Reference implementations live in built-in operator classes (e.g., `optest.operators.builtin_operators.ElementwiseAdd.run`).
- Custom generators or assertions are provided via plan entries pointing to `source`+`name` functions (no registry needed).

### 2.3 Backend Abstraction
- Base class `BackendDriver` defines `kind`, `chips`, and `.run(case, inputs)`.
- `BackendManager` registers drivers and selects the best match based on plan (`--backend`, `--chip`).
- Command backends write inputs, execute a user-provided command, and load outputs; failures surface with stderr/stdout context.

### 2.4 Reporting Pipeline
- Runner prints aligned, colored terminal output (toggle with `--no-color`) and validates JSON reports against an inline schema before writing.

## 3. Extending optest with Custom GPU/NPU Backends

1. **Implement a backend driver**
   ```python
   from optest.backends import BackendDriver, backend_manager

   class MyCudaBackend(BackendDriver):
       kind = "gpu"
       name = "cuda-runtime"
       chips = ("a100", "h100")

       def run(self, case, inputs):
           # 1. Convert numpy inputs to device tensors
           # 2. Launch your CUDA kernel identified by case.descriptor.name
           # 3. Copy outputs back to numpy arrays and return a tuple/list
           return (run_my_kernel(case, inputs),)

   backend_manager.register(MyCudaBackend())
   ```

2. **Expose driver to the CLI**
   - Create a Python module (e.g., `my_company.optest_plugins.backends`) that registers the driver at import time.
   - Set `OPTEST_PLUGINS=my_company.optest_plugins.backends` before invoking `optest`, or import the module in your own CLI wrapper before calling `bootstrap()`.

3. **Handle chip-specific behavior**
   - Declare `chips = ("chipA", "chipB")` and branch inside `.run()` based on `case.backend.chip`. Return helpful errors if unsupported chips are requested.

4. **Tie into operator metadata**
   - If your backend only supports specific dtypes/shapes, extend or override operator descriptors by loading plugins before running tests.

### Tips
- Use case attributes (e.g., strides, paddings) to configure kernel launches.
- For remote hardware, your backend can call RPCs instead of local launches as long as `.run()` returns NumPy arrays.
- Implement additional drivers (e.g., `MyNpuBackend(kind="npu")`) to target NPUs; CLI selection happens via `--backend npu --chip ascend310b`.

### 3.1 Command Backends (CUDA / CANN)
- optest shells out to your binary/script using templated commands; it writes inputs to disk, runs the command, and loads outputs for comparison.
- Plan fields: `workdir`, `env`, `prepare`/`cleanup`, `timeout`, `retries`, and `command` with tokens `{chip}`, `{backend}`, `{case}`, `{dtypes}`, `{shape}`/`{shapes}`, `{inputN}`/`{inputs}`, `{outputN}`/`{outputs}`, `{workdir}`.
- optest ensures parent directories exist and surfaces errors with context (missing files, command failures with stderr/stdout).

## 4. Packaging & Distribution

### 4.1 Building Wheels / Source Distributions
- Ensure dependencies are up-to-date: `python -m pip install --upgrade build`.
- Run `python -m build` from the project root to create wheels and sdists under `dist/`.
- Publish via an internal index or PyPI (e.g., `python -m twine upload dist/*`).

### 4.2 Editable Installs for Contributors
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 4.3 Installing Released Packages
- For end users: `pip install optest` (or your published name/version).
- They’ll receive the `optest` console script plus Python API for embedding (e.g., `from optest.core.runner import TestRunner`).

### 4.4 Delivering with Custom Plugins
- Package plugins in a separate distribution (e.g., `company-optest-plugins`) that depends on `optest`.
- Document environment variables or wrapper scripts that import plugin modules to register custom operators/backends automatically.

## 5. Usage Workflow Overview
1. Install the package (editable or from wheel).
2. Register any custom operators/backends via plugin modules.
3. Author YAML plans in `cases:` format or pass CLI overrides.
4. Run `optest run --plan my_plan.yaml --backend gpu --chip a100 --report json`.
5. Inspect terminal summaries and, if enabled, JSON reports under the provided path.
