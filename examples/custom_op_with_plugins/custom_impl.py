import numpy as np
from pathlib import Path
from optest.plan.models import AssertionResult


def custom_generator(*, input_paths, shapes, dtypes, params, seed, constants, rng):
    """Write deterministic inputs for each input path."""
    for path, shape, dtype in zip(input_paths, shapes["inputs"], dtypes):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = rng.standard_normal(size=shape).astype(dtype)
        np.asarray(data).astype(dtype).tofile(target)


def custom_assertion(*, input_paths, output_paths, shapes, dtypes, output_dtypes, params, rtol, atol, metric):
    """Compute goldens via NumPy and compare to backend output."""
    x = np.fromfile(input_paths[0], dtype=dtypes[0]).reshape(shapes["inputs"][0])
    out = np.fromfile(output_paths[0], dtype=output_dtypes[0]).reshape(shapes["outputs"][0])
    golden = np.square(x) + 1
    if np.allclose(out, golden, rtol=rtol or 0, atol=atol or 0):
        return AssertionResult(ok=True, details="")
    diff = np.max(np.abs(out - golden))
    return AssertionResult(ok=False, details=f"max_abs_diff={diff}", metrics={"max_abs": float(diff)})
