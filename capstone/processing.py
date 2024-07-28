import numpy as np
import typing
import itertools
from scipy.spatial import distance

def printIt(p) -> str :
    p = np.array(p)
    value = np.array2string(p, precision=6, separator='-', floatmode='fixed', formatter={'float': '{:0.6f}'.format})
    return value[1:-1]


from sklearn.gaussian_process import GaussianProcessRegressor

def makeGridIntersections(nValuesWanted: int,
                          centre : np.array,
                          delta : float) -> np.array :
    nDims = len(centre)
    nValuesPerAxis = float(nValuesWanted) ** float(1/nDims)
    N = int(nValuesPerAxis)
    grid = []
    for d in range(nDims) :
        left = max(0.0, centre[d]-delta)
        right =  min(1.0, centre[d]+delta)
        x = np.linspace(left, right, N)
        grid.append(x)
    return grid

def makeGrid(nValuesWanted: int,
             centre : np.array,
             delta : float,
             prune: bool = False) -> np.array :
    nDims = len(centre)
    nValuesPerAxis = float(nValuesWanted) ** float(1/nDims)
    N = int(nValuesPerAxis)

    grid = []
    strides = []
    intersections = []
    nValues = 1
    mins = []
    maxes = []
    for d in range(nDims) :
        strides.append(nValues)
        left = max(0.0, centre[d]-delta)
        right =  min(1.0, centre[d]+delta)
        mins.append(max(0, left))
        maxes.append(min(1.0, right))
        x = np.linspace(left, right, N)
        grid.append(x)
        nValues *= len(x)

#    print(f"Mins {printIt(mins)}")
#    print(f"Maxs {printIt(maxes)}")

    for i in range(nValues) :
        outOfRange = False
        intersection = np.zeros(nDims)
        for d, stride in zip(range(nDims), strides) :
            intersection[d] = grid[d][(i//stride) %N]
            if prune :
                if intersection[d] >= 1.0 or intersection[d] < 0.0 :
                    outOfRange = True
                    break
        if not outOfRange :
            intersections.append(intersection)
    return np.array(intersections)


def regressorUCB(intersections :  np.array,
                 gpr : GaussianProcessRegressor,
                 beta: float = 1.96) -> np.array :
    mean, std = gpr.predict(intersections, return_std = True)
    print("std:", np.min(std), np.max(std))
    print("mean:", np.min(mean), np.max(mean))
    ucb = mean + beta * std
    return ucb

def gridSearch(nValuesWanted: int,
               centre : np.array,
               delta : float,
               gpr : GaussianProcessRegressor,
               beta: float = 1.96) -> np.array :
    intersections = makeGrid(nValuesWanted, centre, delta, prune=True)
    ucb = regressorUCB(intersections, gpr, beta)
    i = np.argmax(ucb)
    return (intersections[i], ucb[i])


def sortValues(X : np.array, Y : np.array) :
    sortedIndex = np.flip(np.argsort(Y))
    return (X[sortedIndex], Y[sortedIndex])
