#! /usr/bin/env python
import argparse
import numpy as np
import sys
import capstone
import os.path
import typing

def findSavedTurboState(func : int) -> typing.Tuple[int, typing.Optional[capstone.TurboState]] :
    lastI = None
    i = 1
    while True :
        fname = f"Data/turboState_{func}_{i}.json"
        if not os.path.isfile(fname) :
            break
        else :
            lastI = i
            i += 1
    if lastI is not None :
        fname = f"Data/turboState_{func}_{lastI}.json"
        print(f"Using turbo state {fname}")
        with open(fname, 'r') as file:
            data = file.read()
        return (i, capstone.TurboState.fromJSON(data))
    return (i, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("func", type=int,
                        help="The index of the function to explore.")
    parser.add_argument("-d", "--drop", type=int, default = -1,
                        help="Drop the given dimension.")
    parser.add_argument("-l", "--length", type=float, default = 0.2,
                        help="Size of the trust region.")
    parser.add_argument("-s", "--save", action='store_true',
                        help="Save the turbo state for the next iteration.")

    args = parser.parse_args()

    X, Y, originalOrder = capstone.loadData(args.func)
    if args.drop != -1 :
        X =  np.delete(X, [args.drop], 1)

    iteration, prevState = findSavedTurboState(args.func)

    if prevState is not None :
        lastSampleIndex = np.argmax(originalOrder)
        lastY = Y[lastSampleIndex]
        prevState.update(lastY)

    x, turboState  = capstone.turboSearch(X, Y, prevState, args.length)

    print(f"RESULT:\n{capstone.formatVec(x)}")

    if args.save :
        with open(f"Data/turboState_{args.func}_{iteration}.json", "w") as f :
            f.write(turboState.toJSON())
