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
- Built-in generators live in the plan runner (`builtin.random`, `builtin.uniform`, `builtin.ones`) and honor `constants` (`value`, `scale`, `shift`) plus per-input overrides.
- Custom generators are plain functions referenced via `generator.source` + `generator.name`; the runner invokes them with paths/shapes/dtypes/params and an already-seeded `numpy.random.Generator`.
- Reference implementations live in built-in operator classes (e.g., `optest.operators.builtin_operators.ElementwiseAdd.run`) or custom assertions supplied via plan entries.

### 2.3 Backend Abstraction
- Backends are YAML plan entries (`backends:`) that describe command templates plus env/timeout/retry hooks; allowed `type` values are `cann` and `cuda`.
- Runner resolves tokens (paths, dtypes, shapes, chip/backend) into the command/prepare/cleanup argv, writes inputs, executes the commands, and loads outputs for comparison.
- Selection happens via plan and CLI filters (`--backend`, `--chip`); there is no Python backend registry anymore.

### 2.4 Reporting Pipeline
- Runner prints aligned, colored terminal output (toggle with `--no-color`) and validates JSON reports against an inline schema before writing.

## 3. Backend command templates

- Define backend commands directly in the plan (`backends:`); no Python subclasses or registries are needed.
- A backend entry can call any runner script or binary (local or remote) via `command`, with optional `prepare`/`cleanup`, `env`, `timeout`, and `retries`. Chip- or backend-specific behavior should live in that script, using tokens to parameterize calls.
- Example:
  ```yaml
  backends:
    - type: cuda
      chip: local
      workdir: .
      command: ["python", "run.py", "--input0", "{input0}", "--input1", "{input1}", "--output0", "{output0}", "--dtype", "{dtype}", "--shape", "{shape}"]
      prepare: [["./setup.sh"]]
      cleanup: [["./teardown.sh"]]
  ```
- Remote runners (e.g., `ssh host run_op.sh ...`) work as long as they read the generated input files and write the expected outputs where optest will load them.

### 3.1 Command tokens
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
- They’ll receive the `optest` console script; programmatic use can import `optest.plan.runner.run_plan` directly if needed.

### 4.4 Delivering with Custom Plugins
- Package plugins in a separate distribution (e.g., `company-optest-plugins`) that depends on `optest`.
- Document environment variables or wrapper scripts that import plugin modules to register custom operators/backends automatically.

## 5. Usage Workflow Overview
1. Install the package (editable or from wheel).
2. Provide any custom generators/assertions (via plan `source` files or optional plugins).
3. Author YAML plans in `cases:` format or pass CLI overrides.
4. Run `optest run --plan my_plan.yaml --backend gpu --chip a100 --report json`.
5. Inspect terminal summaries and, if enabled, JSON reports under the provided path.
