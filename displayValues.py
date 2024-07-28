#! /usr/bin/env python
import argparse
import sys
import capstone
import numpy as np

def printIt(i, x, y, o) :
    print(f"{i:2}({o:2}) = {y:0.10f} <- {capstone.formatVec(x)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("func", type=int,
                        help="The index of the function to load.")
    parser.add_argument("-d", "--diff", nargs=2, type=int,required=False)
    parser.add_argument("-s", "--submission", action='store_true',
                        help="Display in order of submission.")
    args = parser.parse_args()

    X, Y, order = capstone.loadData(args.func, not args.submission)

    print(f"N values {len(Y)}")
    d1 = 0
    d2 = 1
    if args.diff == None :
        i = 0
        for x, y, o in zip(X, Y, order) :
            printIt(i, x, y, o)
            i+=1
        print()
    else :
        d1 = args.diff[0]
        d2 = args.diff[1]

    x1, y1 = X[d1], Y[d1]
    x2, y2 = X[d2], Y[d2]
    print(f"Diffing...")
    printIt(d1, x1, y1, order[d1])
    printIt(d2, x2, y2, order[d2])
    dX = x1-x2
    dY = y1-y2
    print(f"delta is {np.sqrt(sum(dX * dX))}")
    printIt("D", abs(dX), abs(dY), 0)
    printIt("M", x2 + dX/2, y2 + dY/2, 0)
