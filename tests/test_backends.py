from pathlib import Path

import numpy as np

from optest.backends.ascend import AscendBackendDriver, _extract_config
from optest.core import BackendTarget, OperatorDescriptor, TestCase, Tolerance


def _make_case(config) -> TestCase:
    descriptor = OperatorDescriptor(
        name="dummy",
        category="test",
        num_inputs=1,
        num_outputs=1,
        dtype_variants=(("float32",),),
    )
    return TestCase(
        descriptor=descriptor,
        dtype_spec=("float32",),
        shapes={"input0": (2,), "output0": (2,)},
        backend=BackendTarget(kind="npu"),
        tolerance=Tolerance(),
        attributes={"backend_config": config},
    )


def test_ascend_backend_defaults_io_paths() -> None:
    config = {"ascend": {"workdir": ".", "command": ["true"]}}
    case = _make_case(config)
    cfg = _extract_config(case)
    assert cfg is not None
    assert cfg.inputs[0].path == Path("input/input0.bin")
    assert cfg.outputs[0].path == Path("output/output0.bin")


def test_ascend_backend_lifecycle_hooks(tmp_path) -> None:
    workdir = tmp_path
    build_script = workdir / "build.py"
    build_script.write_text("open('built.txt','w').write('ok')", encoding="utf-8")
    pre_script = workdir / "pre.py"
    pre_script.write_text("open('pre.txt','w').write('ok')", encoding="utf-8")
    run_script = workdir / "run.py"
    run_script.write_text(
        "import numpy as np;import pathlib;"
        "data=np.fromfile('input/input0.bin',dtype=np.float32)*2;"
        "pathlib.Path('output').mkdir(exist_ok=True);"
        "data.astype(np.float32).tofile('output/output0.bin')",
        encoding="utf-8",
    )
    post_script = workdir / "post.py"
    post_script.write_text("open('post.txt','w').write('ok')", encoding="utf-8")
    for script in (build_script, pre_script, run_script, post_script):
        script.chmod(script.stat().st_mode | 0o111)

    config = {
        "ascend": {
            "workdir": str(workdir),
            "build": ["python3", str(build_script)],
            "pre_run": ["python3", str(pre_script)],
            "command": ["python3", str(run_script)],
            "post_run": ["python3", str(post_script)],
            "inputs": {"input0": "input/input0.bin"},
            "outputs": {"output0": "output/output0.bin"},
        }
    }
    case = _make_case(config)
    driver = AscendBackendDriver(chips=("sim",))
    output = driver.run(case, [np.array([1.0, 2.0], dtype=np.float32)])
    assert output[0].tolist() == [2.0, 4.0]
    assert (workdir / "built.txt").exists()
    assert (workdir / "pre.txt").exists()
    assert (workdir / "post.txt").exists()


def test_ascend_backend_default_paths(tmp_path) -> None:
    workdir = tmp_path
    run_script = workdir / "run.py"
    run_script.write_text(
        "import numpy as np;import pathlib;"
        "pathlib.Path('output').mkdir(parents=True, exist_ok=True);"
        "data=np.fromfile('input/input_x.bin',dtype=np.float32)+5;"
        "data.astype(np.float32).tofile('output/output_z.bin')",
        encoding="utf-8",
    )
    run_script.chmod(run_script.stat().st_mode | 0o111)
    config = {
        "ascend": {
            "workdir": str(workdir),
            "command": ["python3", str(run_script)],
        }
    }
    case = _make_case(config)
    driver = AscendBackendDriver(chips=("sim",))
    output = driver.run(case, [np.array([1.0, 2.0], dtype=np.float32)])
    assert output[0].tolist() == [6.0, 7.0]


def test_load_golden_tensors(tmp_path) -> None:
    workdir = tmp_path
    golden_file = workdir / "golden.bin"
    np.array([3.0, 4.0], dtype=np.float32).tofile(golden_file)
    config = {
        "ascend": {
            "workdir": str(workdir),
            "command": {"binary": "true"},
            "inputs": {"input0": "input/input0.bin"},
            "outputs": {"output0": "output/output0.bin"},
            "golden": {"output0": "golden.bin"},
        }
    }
    case = _make_case(config)
    from optest.backends.ascend import load_golden_tensors

    tensors = load_golden_tensors(case)
    assert tensors is not None
    assert tensors[0].tolist() == [3.0, 4.0]
