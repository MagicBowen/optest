# Using optest for Ascend sample operators

This note explains how to replace the per-operator test scripts in `samples/operator/ascendc/0_introduction` with optest.

## Key pieces
- **Convention-over-configuration IO**: if `inputs`/`outputs` are not provided, optest writes tensors to `input/{tensor}.bin` and reads from `output/{tensor}.bin`, using dtypes from the case (and also mirrors to legacy `input_x.bin`/`output_z.bin` names for the samples). Optional `golden` entries let you dump or load reference outputs for debugging.
- **Override paths when needed**: map tensor names to the exact files your binary reads/writes (e.g., `input0: input/input_x.bin`, `output0: output/output_z.bin`) via `backend_config.ascend.inputs`/`outputs`. optest generates and writes the data, runs your binary, and reads back the result from those paths. Provide entries for every input/output tensor (dict or list syntax) if the defaults do not match.
- **Minimal backend config**: you only need `workdir` and `command` (or a binary path). optest handles data generation, file writes, execution, and comparison using built-in references (or a user-specified reference).
- **Env vars optional**: include `env` entries like `RUN_MODE`/`SOC_VERSION` only if your binary requires them at runtime; optest itself does not depend on them once the binary is built.
- **Plan generator**: `tools/gen_ascend_sample_plans.py` can emit YAML plan stubs that point optest at each `run.sh` using the default IO conventions (adjust op names/paths as needed).

## Running a sample with optest
1. Ensure the sample is built or provide a `build` hook:
   ```yaml
   cases:
     - op: ascend_matmul
       shapes:
         input0: [2, 2]
         input1: [2, 2]
         output0: [2, 2]
       attributes:
         backend_config:
           ascend:
             workdir: /path/to/sample
             command: ./ascendc_kernels_bbit   # or ["bash","run.sh"] if that runs the binary
             golden:
               output0: output/golden0.bin
             env:
               ASCEND_INSTALL_PATH: /usr/local/Ascend/ascend-toolkit/latest
   ```
2. Run optest:
   ```bash
   optest run --plan /path/to/plan.yaml --backend npu --chip ascend310b
   ```
3. optest writes inputs, runs the command, reads outputs, and compares to NumPy references (or golden files if configured).

## Onboarding a new sample operator
1. Ensure the operator is registered in optest (use built-ins like `elementwise_add`, `matmul`, etc., or add a descriptor).
2. Create a plan entry pointing to the sample directory and the run binary/script; map inputs/outputs to match the binary.
3. Remove legacy `gen_data.py`/`verify_result.py` pieces—the plan + backend handles data/gen/compare.

## Generating plan stubs
Run the helper to create a draft plan for all `run.sh` entries:
```bash
python3 tools/gen_ascend_sample_plans.py --root tmp/ascend_samples/operator/ascendc/0_introduction --output tmp/ascend_samples_plan.yaml
```
Fill in shapes/dtypes per operator and run with optest.
