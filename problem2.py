#! /usr/bin/env python
import argparse
import numpy as np
import sys
import capstone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

noiseVariance = 0.4

kernel = 1.0 * RBF(1.0, length_scale_bounds="fixed") + WhiteKernel(noise_level=noiseVariance)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=int,
                        help="Which element to centre the search on.")
    parser.add_argument("-s", "--search", type=float, default=0.2, help="Search size")
    parser.add_argument("-b", "--beta", type=float, default=1.96, help="Beta for the UCB search")
    parser.add_argument("-a", "--alpha", type=float, default=1e-10, help="Alpha for the UCB search")
    parser.add_argument("-n", "--nSamples", type=int, default = 100000,
                        help="The total number of samples in the sampling grid.")
    parser.add_argument("--normalise",  action='store_true',
                        help="Normalise Y for the Gaussian Process Regressor.")

    args = parser.parse_args()

    X, Y, _ = capstone.loadData(2)

    gpr = GaussianProcessRegressor(kernel, copy_X_train=True, random_state=0, normalize_y=args.normalise)
    gpr.fit(X, Y)

    centre = X[args.selection]
    print(f"Centering on {args.selection} = {Y[args.selection]}\ncent {capstone.printIt(centre)}")

    x, ucb  = capstone.gridSearch(args.nSamples, centre, args.search, gpr, args.beta)
    print(f"RESULT:\n{capstone.formatVec(x)} -> {ucb}")
