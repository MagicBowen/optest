# Ascend backend example with optest

This example ships a tiny add operator plus build scripts and relies on optest for all testing.

## 1) Build the operator
```bash
cd examples/ascend_operator/operator
bash build.sh
```
The binary is produced at `./build/add_custom`.

## 2) Run optest against the binary
Two cases are defined (float32 and int32, different shapes):
```bash
cd ..
optest run --plan ./plan.yaml --backend npu --chip ascend --no-color
```
optest writes inputs to `input/input{0,1}.bin`, runs `./build/add_custom --dtype ...` from the plan, and compares `output/output0.bin` against NumPy reference outputs. If the command fails, optest reports the failure and stderr.

## Notes
- Modify `plan.yaml` to add more cases (shapes/dtypes) or change `workdir` if you copy the operator elsewhere.
- There are no Python/shell test scripts in the operator; testing is fully handled by optest.
