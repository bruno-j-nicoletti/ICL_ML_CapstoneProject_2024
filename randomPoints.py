#! /usr/bin/env python

import numpy as np
import capstone

for funcIndex in range(1, 9) :
    print(f"FUNCTION {funcIndex}")
    X, Y, _ = capstone.loadData(funcIndex)
    N = len(X[0])
    x = np.random.uniform(size = N)
    print(f"{capstone.formatVec(x)}\n")
