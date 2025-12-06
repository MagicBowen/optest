import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input0", required=True)
    parser.add_argument("--output0", required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--shape", required=True)
    args = parser.parse_args()

    shape = tuple(int(part) for part in args.shape.replace("X", "x").split("x") if part)
    x = np.fromfile(args.input0, dtype=args.dtype).reshape(shape)
    y = np.square(x) + 1
    output_path = Path(args.output0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y.astype(args.dtype).tofile(output_path)


if __name__ == "__main__":
    main()
