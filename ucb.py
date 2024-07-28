#! /usr/bin/env python
import argparse
import numpy as np
import sys
import capstone
from sklearn.gaussian_process import GaussianProcessRegressor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("func", type=int,
                        help="The index of the function to explore.")
    parser.add_argument("selection", type=int,
                        help="Which element to centre the search on.")
    parser.add_argument("-s", "--search", type=float, default=-0.2, help="Search size")
    parser.add_argument("-b", "--beta", type=float, default=1.96, help="Beta for the UCB search")
    parser.add_argument("-a", "--alpha", type=float, default=1e-10, help="Beta for the UCB search")
    parser.add_argument("-d", "--drop", type=int, default = -1,
                        help="Drop the given dimension.")
    parser.add_argument("-n", "--nSamples", type=int, default = 10000,
                        help="The total number of samples in the sampling grid.")
    parser.add_argument("--normalise",  action='store_true',
                        help="Normalise Y for the Gaussian Process Regressor.")

    args = parser.parse_args()

    X, Y, _ = capstone.loadData(args.func)
    if args.drop != -1 :
        X =  np.delete(X, [args.drop], 1)

    nDims = len(X[0])
    gpr = GaussianProcessRegressor(copy_X_train=True, random_state=0, normalize_y=args.normalise)
    gpr.fit(X, Y)

    if args.search > 0 :
        centre = X[args.selection]
        print(f"Centering on {args.selection} = {Y[args.selection]}\ncent {capstone.printIt(centre)}")
    else :
        args.search = 0.5
        centre = np.array([.5] * nDims)
        print(f"Searching entire space")

    x, ucb  = capstone.gridSearch(args.nSamples, centre, args.search, gpr, args.beta)
    print(f"RESULT:\n{capstone.formatVec(x)} -> {ucb}")
