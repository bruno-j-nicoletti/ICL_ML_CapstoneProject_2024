import numpy as np
import copy
from scipy.spatial import Delaunay, Voronoi
import itertools
import math

def doSpatial(spatial, X: np.array, corners: bool) :
    """Run the spacial algorithm on the given points, adding in corners or not"""
    nDims = X.shape[1]
    points = np.copy(X)
    if corners :
        cornersPoints = np.array([x for x in itertools.product([0.0, 1.0], repeat=nDims)])
        points  = np.append(points, cornersPoints, axis=0)
    return spatial(points)


def delauny(X: np.array, corners : bool = False) -> Delaunay:
    """Get the delauny simplexification of the given points."""
    return doSpatial(Delaunay, X, corners)

def voronoi(X: np.array, corners : bool = False) -> Voronoi:
    """Get the voronoi points for the given points."""
    return doSpatial(Voronoi, X, corners)

def _dist(p0: np.array, p1: np.array) -> float :
    d = 0.0
    for x0, x1 in zip(p0, p1) :
        delta = x1 - x0
        d += delta * delta
    return math.sqrt(d)

def loneliestPoint(X : np.array, corners : bool = False, clamp: bool = False) -> np.array :
    """Returns the point that is furthest from all the other points
    within the convect hull of the given points()"""


    # This would be easier with a Delauny triangulation if the library
    # returned the circumhyperspheres and their centers, but it doesn't
    # so I have to do this search from the Voronoi points instead
    vor = voronoi(X, corners)

    nDims = len(X[0])
    maxR = 0
    point = None

    # the distance to the closest point for each vor point
    furthestDistance = 0
    furthestPoint = None

    # now find the distance to the closest input point
    for vorPoint in vor.vertices:
        thisD = np.linalg.norm(vorPoint - X[0])
        clampedPoint = copy.copy(vorPoint)
        outOfBounds = False
        for i, v in enumerate(vorPoint) :
            if clamp :
                clampedPoint[i] = min(max(v, 0.0), 1.0)
            else :
                if v < 0.0 or v > 1.0 :
                    outOfBounds = True
                    break
        if outOfBounds :
            continue

        for p in X :
            d = _dist(clampedPoint, p)
            if d < thisD :
                thisD = d
        if thisD > furthestDistance :
            furthestDistance = thisD
            furthestPoint = clampedPoint

    return  furthestPoint, furthestDistance
