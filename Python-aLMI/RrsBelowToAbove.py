#! /usr/bin/env python3

def RrsBelowToAbove(reflBelowSurf, coeff0=.52, coeff1=1.7):
    """
    Convert from below-surface rrs to above-surface remote sensing reflectance (Rrs).

    Returns a NumPy array of the same shape as the input
    """

    return coeff0 * reflBelowSurf / (1. - coeff1 * reflBelowSurf)

# The '__main__' entry point is only used for testing. 
if __name__ == '__main__':
    import sys
    import numpy as np

    ME = sys.argv[0]
    if len(sys.argv) != 1:
        print("usage:  " + sys.argv[0])
        sys.exit(1)

    # Define inputs:
    reflBelowSurf = np.array([0., .2, .4, .5, .6], dtype=float)
    print("Inputs:")
    print("reflBelowSurf =", reflBelowSurf)

    refAboveSurf = RrsBelowToAbove(reflBelowSurf)
    print("\nResults:")
    print("refAboveSurf =", refAboveSurf)

    print(sys.argv[0] + " done")             # DEBUG
    sys.exit(0)
