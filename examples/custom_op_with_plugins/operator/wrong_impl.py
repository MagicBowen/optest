#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--input", default="input/input0.bin")
    parser.add_argument("--output", default="output/output0.bin")
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    data = np.fromfile(args.input, dtype=dtype)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # Intentional bug: write the raw input instead of squaring it
    data.tofile(args.output)


if __name__ == "__main__":
    main()
