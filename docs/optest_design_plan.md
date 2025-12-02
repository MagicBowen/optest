# optest Design & Implementation Plan

## 1. Goals & Scope
- Deliver `optest`, a Python package installable via pip that exposes a CLI for validating AI operators targeting CUDA GPUs or diverse NPUs (chip tag aware).
- Support core operator families (elementwise tensor ops, GEMM, Conv2D, pooling, activations) with extensibility for future ops.
- Automate tensor/input plus golden/reference generation with NumPy by default (no SciPy dependency initially) while allowing users to plug in custom generators/reference functions.
- Provide backend abstraction so real GPU/NPU integrations can be plugged in later; ship a stub backend driver that reuses reference implementations for self-test.
- Produce actionable test reports via terminal output (default, colorized unless `--no-color`) and optional JSON output adhering to a strict schema.

## 2. Requirements Summary
### Functional
1. Discover and run operator tests through CLI (`optest run`).
2. Allow backend selection (`gpu` or `npu`) plus optional `--chip <id>` for hardware specificity.
3. Support dtype matrix including `float32`, `float16`, `int8`, `int16`, `int32` (extensible via registry metadata).
4. Accept concise CLI syntax for shapes (`--shape m=128,n=256,k=64`) and dtype tuples (`--dtype float32,int8`).
5. Default tensor data/reference generation via NumPy-based utilities; support user-defined generators/reference implementations via registry hooks or CLI/config overrides.
6. Execute backend kernels (stub initially), compare outputs vs golden with dtype-aware tolerances, and capture diagnostics for mismatches.
7. Emit terminal summary by default; allow `--report json --report-path path.json` for strict-schema JSON files.
8. Parse YAML plan files describing batches of test cases; CLI overrides take precedence.
9. Provide deterministic runs with configurable RNG seed; expose fail-fast and verbosity controls.

### Non-Functional
- Zero SciPy dependency in early versions (custom Conv/Pool reference code where needed).
- Modular package layout enabling reuse/extension; plugin-friendly registries.
- Clear separation between CLI/config, test orchestration, data generation, backend invocation, and reporting.
- Colorized terminal output with `--no-color` escape hatch; JSON schema versioning for stability.
- Installable package with console script entry point (`optest`).

## 3. Architectural Overview
```
+-------------------+      +-------------------+      +------------------+
| CLI & Config Layer| -->  | Test Orchestration| -->  | Reporters        |
+-------------------+      +-------------------+      +------------------+
           |                        |                            ^
           v                        v                            |
      +-----------+           +-----------+                +-----------+
      | Registry  | <-------> | Generators|                | Backends  |
      +-----------+           +-----------+                +-----------+
```
- CLI aggregates direct flags and YAML plan files, producing execution plans.
- Registry holds operator descriptors, supported dtypes/shapes/backends, default generators/reference implementations, and tolerance defaults.
- Generators provide deterministic input/random data + reference outputs using NumPy utilities or user-supplied callables.
- Backends encapsulate actual operator execution (GPU/NPU). Stub backend delegates to reference implementation for self-testing.
- Test orchestrator expands cases, runs generators/backends/comparators, and streams events to reporters.
- Reporters render terminal text and/or JSON files with strict schema.

## 4. Core Data Model & Components
1. **OperatorDescriptor**: name, category, arity, attribute schema, supported dtype combos, shape validators, backend support flags, default tolerance, generator/ref hooks.
2. **TestCase**: fully resolved operator invocation (operator descriptor + dtype tuple + shapes + backend + chip + tolerance override + generator/ref overrides + misc attrs).
3. **Generator API**: `generate(case: TestCase, rng: Generator) -> (inputs, reference_outputs, aux_metadata)`; base implementations live under `optest/generators/defaults`.
4. **BackendDriver API**: attributes `name`, `kind`, `chips`; methods `supports(op_descriptor, test_case)` and `run(test_case, inputs)` returning backend outputs. Stub driver implements GPU/NPU kinds by calling reference function.
5. **Comparator**: dtype-aware comparison strategies (float: abs+rel tolerance; int: exact/abs tolerance). Returns `ComparisonResult` with pass/fail flag, max errors, mismatch indices, snippet metadata.
6. **TestRunner**: orchestrates case execution, handles seeding, error capture, fail-fast, and passes structured events to reporters.
7. **Reporters**: Terminal reporter (colorized or monochrome) and JSON reporter (strict schema with version + validation). Collect aggregated metrics and failure details.
8. **Config Loader**: YAML parser merging defaults + file + env + CLI; outputs normalized plan describing cases/backends/reporting options.
9. **Plugin Loader**: optional env var (e.g., `OPTEST_PLUGINS`) listing modules to import at startup so they can register operators/generators/backends.

## 5. CLI Design
- Console script `optest` built on `click` (or Typer). Primary command `optest run`.
- Options:
  - `-o/--op NAME` (repeatable); `--all` for every registered op.
  - `--dtype <spec>`: comma-separated tuple per operator input (e.g., `--dtype float32,float32`). Multiple flags allowed for multi-input ops; default derived from registry.
  - `--shape <key=value,...>`: parse into dicts; allow repeating for multi-input shapes (prefix keys like `a:m=128,n=256`).
  - `--backend {gpu,npu}` & `--chip TEXT`.
  - `--config PATH` (YAML) with ability to provide multiple files.
  - `--report {terminal,json}` default terminal; `--report-path PATH` (default `./optest_report_<timestamp>.json` when JSON selected).
  - `--no-color` to disable terminal colors.
  - `--generator dotted.path` / `--reference dotted.path` for overrides.
  - `--tolerance abs=1e-4,rel=1e-5`.
  - `--seed INT`, `--fail-fast`, `--verbose`, `--max-workers` (placeholder, sequential for MVP).
- Flow: parse CLI -> load YAML plan if provided -> merge -> build execution plan -> run -> output.

## 6. YAML Plan Schema
```yaml
version: 1
backend: gpu
chip: a100
seed: 123
report:
  format: json
  path: ./reports/a100_run.json
cases:
  - op: gemm
    dtypes: [float32, float32]
    shape:
      m: 128
      n: 256
      k: 64
    tolerance:
      abs: 1e-4
      rel: 1e-5
  - op: conv2d
    input_shape: [1, 64, 112, 112]
    weight_shape: [64, 64, 3, 3]
    stride: [1, 1]
    padding: same
    dtype: float16
```
- Loader validates version, ensures cases map to registered operators, and merges CLI overrides (e.g., CLI `--chip` overrides file `chip`).

## 7. Operator Registry & Extension Points
- API: `operator_registry.register(OperatorDescriptor)` or decorator for built-ins.
- Built-in descriptors for elementwise ops (`add`, `mul`), `gemm`, `conv2d`, `maxpool2d`, `avgpool2d`, `relu`, `sigmoid`.
- Operator-specific metadata includes parameter schemas (stride/padding, activation alpha, etc.) validated pre-run.
- Generators/reference functions can be overridden via registry or CLI-specified dotted paths (resolved at runtime via importlib).
- Plugin discovery: environment variable `OPTEST_PLUGINS="pkg1.module,pkg2.module"`; CLI loads them before building the registry.

## 8. Backend Strategy
- `BackendManager` keeps dictionary of drivers keyed by `(kind, chip)`.
- Stub backend (`StubBackendDriver`) implements both GPU/NPU modes and uses descriptor reference implementation to simulate execution. Useful for CI/self-test when real hardware is absent.
- Real GPU/NPU integrations will conform to the same interface (exposing chip metadata, dtype support). Documented expectations so users can implement later.

## 9. Reporting & Diagnostics
- Terminal reporter:
  - Shows streaming progress `[3/10] gemm float32 backend=gpu chip=a100 ... PASS (35ms)`.
  - On failure, prints structured blocks with backend/chip, seed, shapes, tolerance, worst mismatch index plus actual/expected values, and aggregates a final failure summary.
  - Honors `--no-color` to disable ANSI codes.
- JSON reporter:
  - Schema v1 with fields: `schema_version`, `generated_at`, `summary` (pass/fail counts, duration), and `cases` array (each includes `id`, `operator`, `backend`, `chip`, `dtypes`, `shapes`, `status`, `duration_ms`, `tolerance`, `metrics`, `failure` details, `inputs_metadata`).
  - Validate schema before writing (using jsonschema module or manual check) to guarantee consistency.

## 10. Implementation Plan
1. **Phase 0 – Project Skeleton & Packaging**
   - Create `pyproject.toml`/`setup.cfg` (or `setup.py`) plus `README`, `LICENSE`, `requirements` lists.
   - Establish package directories: `optest/cli`, `core`, `registry`, `generators`, `backends`, `reporting`, `plans`, `utils`.
   - Wire console entry point `optest` invoking CLI stub.
2. **Phase 1 – Core Models & Registry**
   - Implement dataclasses for descriptors/test cases/tolerances.
   - Build registry module with registration decorators and lookup utilities; add built-in operator descriptors (without implementations yet).
3. **Phase 2 – Generators & Reference Implementations**
   - Implement NumPy-based default generators + reference functions for the required operator set (elementwise, GEMM, Conv2D, pooling, activations).
   - Provide deterministic RNG wrapper, dtype range helpers, and override plumbing.
4. **Phase 3 – Backend Interface & Stub Driver**
   - Define backend base class + manager; implement stub driver that routes to reference implementations.
   - Ensure backend selection respects kind/chip CLI params and surfaces helpful errors.
5. **Phase 4 – Test Runner & Comparator**
   - Implement comparator utilities, structured comparison results, and test runner orchestrating generator/backend/comparator sequence with fail-fast + logging.
6. **Phase 5 – CLI & YAML Loader**
   - Flesh out CLI parsing with concise shape/dtype syntax, plan loading, override precedence, plugin auto-loading, and reporting flags (`--no-color`).
7. **Phase 6 – Reporting**
   - Implement terminal reporter (color + fallback) and JSON reporter (schema v1) along with reporting pipeline integration.
8. **Phase 7 – Packaging & Distribution Enhancements**
   - Document plugin/extension instructions, finalize dependency list (numpy, click/typer, pyyaml, colorama, jsonschema, rich/prettytable optional), and ensure installability.
9. **Phase 8 – Testing & Examples**
   - Add unit tests covering registry, generators, comparator, backend stub, runner, CLI parsing, YAML loader, and reporters.
   - Provide sample YAML plans and example reports in `examples/`.

## 11. Risks & Considerations
- **Backend integration**: Real GPU/NPU adapters must match the defined interface; documentation + stub driver intended to smooth this transition.
- **Custom conv/pool references**: Without SciPy, custom implementations need careful validation to avoid correctness bugs; tests will focus on known patterns.
- **Shape/dtype syntax**: Must ensure parser remains intuitive yet expressive for multi-input ops.
- **Performance**: Large tensors may make NumPy reference runs slow; future optimization could include shape filters or sampling.
- **JSON schema evolution**: Need versioning strategy for backward compatibility; start with `schema_version: 1`.

## 12. Next Steps
- Proceed with project bootstrap (Phase 0) following this plan.
- While implementing, keep documentation updated and add examples demonstrating CLI/YAML usage.
