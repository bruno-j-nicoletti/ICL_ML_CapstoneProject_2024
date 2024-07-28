#! /usr/bin/env python
import numpy as np
import capstone
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("func", type=int,
                        help="The index of the function to explore.")
    parser.add_argument("-d", "--drop", type=int, default = -1,
                        help="Drop the given dimension.")
    parser.add_argument("-c", "--corners", action='store_true',
                        help="Include the corners of the volume.")
    args = parser.parse_args()

    X, Y, _ = capstone.loadData(args.func)
    if args.drop != -1:
        X = np.delete(X, [args.drop], 1)
    x, closestDistance = capstone.loneliestPoint(X, args.corners)
    print(f"{capstone.formatVec(x)}\n")
    print(f"Closest point is {closestDistance}")
