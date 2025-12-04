import argparse
import numpy as np
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input0", required=True)
    parser.add_argument("--input1", required=True)
    parser.add_argument("--output0", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--shape", required=True)
    args = parser.parse_args()

    shape = tuple(int(part) for part in args.shape.replace("X", "x").split("x") if part)
    a = np.fromfile(args.input0, dtype=args.dtype).reshape(shape)
    b = np.fromfile(args.input1, dtype=args.dtype).reshape(shape)
    out = np.add(a, b).astype(args.dtype)
    output_path = Path(args.output0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(output_path)


if __name__ == "__main__":
    main()
