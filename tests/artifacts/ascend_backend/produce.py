#!/usr/bin/env python3
import numpy as np
from pathlib import Path

Path('output').mkdir(exist_ok=True)
input_data = np.fromfile('input/input_x.bin', dtype=np.float32)
(input_data * 2).astype(np.float32).tofile('output/output_z.bin')
