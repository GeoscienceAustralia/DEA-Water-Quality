#! /usr/bin/env python3

import numpy as np

def calcUBackward(reflBelowSurf_WL_PIX, g0, g1):
    """
    Calculate u from Eq. 2 in Brando 2012.

    reflBelowSurf_WL_PIX is the [WL, PIX] NumPy array of below surface reflectance.
    g0, g1 are the model constants.

    Returns a NumPy array of uIOPRatio.
    """

    uIOPRatio_WL_PIX = (-g0 + np.sqrt(g0 * g0 + 4 * g1 * reflBelowSurf_WL_PIX)) / (2 * g1)		# Eq. 2
    return uIOPRatio_WL_PIX

# The '__main__' entry point is only used for testing. 
if __name__ == '__main__':
    import sys

    ME = sys.argv[0]
    if len(sys.argv) != 1:
        print("usage:  " + sys.argv[0])
        sys.exit(1)

    # Define inputs:
    reflBelowSurf_WL_PIX = np.array([[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]], dtype=float)
    g0 = .084		# Lee
    g1 = .17		# Lee
    print("Inputs:")
    print("reflBelowSurf_WL_PIX =", reflBelowSurf_WL_PIX)
    print("g0 =", g0)
    print("g1 =", g1)

    uIOPRatio_WL_PIX = calcUBackward(reflBelowSurf_WL_PIX, g0, g1)
    print("Results:")
    print("uIOPRatio_WL_PIX =", uIOPRatio_WL_PIX)

    print(sys.argv[0] + " done")             # DEBUG
