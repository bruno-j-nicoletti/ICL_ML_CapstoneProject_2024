import numpy as np

def formatVec(p : np.array) -> str :
    value = np.array2string(p, precision=6, separator='-', floatmode='fixed', formatter={'float': '{:0.6f}'.format})
    return value[1:-1]
