#! /usr/bin/env python3

def calcReflForward(uIOPRatioPredicted_WL_PIX, g0, g1):
    reflBelowSurfPredicted_WL_PIX = (g0 + (g1 * uIOPRatioPredicted_WL_PIX)) * uIOPRatioPredicted_WL_PIX
    return reflBelowSurfPredicted_WL_PIX

# The '__main__' entry point is only used for testing. 
if __name__ == '__main__':
    import sys
    import numpy as np

    ME = sys.argv[0]
    if len(sys.argv) != 1:
        print("usage:  " + sys.argv[0])
        sys.exit(1)

    # Define inputs:
    uIOPRatioPredicted_WL_PIX = np.arange(.1, .9, .1, dtype=float)
    uIOPRatioPredicted_WL_PIX.shape = (len(uIOPRatioPredicted_WL_PIX), 1)		# required by calcReflForward
    g0 = .084		# Lee
    g1 = .17		# Lee
    print("Inputs:")
    print("uIOPRatioPredicted_WL_PIX =", uIOPRatioPredicted_WL_PIX)
    print("g0 =", g0)
    print("g1 =", g1)

    reflBelowSurfPredicted_WL_PIX = calcReflForward(uIOPRatioPredicted_WL_PIX, g0, g1)
    print("\nResults:")
    print("reflBelowSurfPredicted_WL_PIX =", reflBelowSurfPredicted_WL_PIX)

    print(sys.argv[0] + " done")             # DEBUG
    sys.exit(0)
