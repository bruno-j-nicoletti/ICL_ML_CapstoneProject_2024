from os import path
import typing
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

dimensions = [2, 2, 3, 4, 4, 5, 6, 8]

def loadData(fIndex : int, sortByY : bool = True) -> typing.Tuple[np.array] :
    X = np.load(f'initial_data/function_{fIndex}/initial_inputs.npy')
    Y = np.load(f'initial_data/function_{fIndex}/initial_outputs.npy')

    # read the data from the drops
    d = pd.read_csv('Data/589_data.csv')

    # get the outputs, no need to parse
    outputs = d[f"f{fIndex}_output"].to_numpy()

    # get the inputs, will be strings
    inputsStrs = d[f"f{fIndex}"]

    # parse the strings into np arrays
    inputs = []
    for iStr in inputsStrs :
        iStr = iStr.replace("[", "").replace("]", "")
        iSplit = iStr.split()
        values = []
        for i in iSplit :
            values.append(float(i))
        assert(len(values) == dimensions[fIndex-1])
        inputs.append(values)

    # add data from drops to the initial data
    X = np.append(X, inputs, axis=0);
    Y = np.append(Y, outputs)

    originalOrder = np.array([x for x in range(len(Y))])
    sortedIndex = np.flip(np.argsort(Y))
    if sortByY :
        X = X[sortedIndex]
        Y = Y[sortedIndex]
        order = originalOrder[sortedIndex]
    else :

        order = [sortedIndex.tolist().index(x) for x in range(len(Y))]
    return (X, Y, order)

def loadIntoRegressor(fIndex: int) -> typing.Tuple[np.array, np.array, GaussianProcessRegressor] :
    X, Y, _ = loadData(fIndex)
    gpr = GaussianProcessRegressor(copy_X_train=True, random_state=10)
    gpr.fit(X, Y)
    return X, Y, gpr
