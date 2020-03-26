#! /usr/bin/env python3

def RrsAboveToBelow(reflAboveSurf, coeff0=.52, coeff1=1.7):
    """
    Convert from above-surface remote sensing reflectance (Rrs) to below-surface rrs.

    Returns a NumPy array of the same shape as the input
    """

    return reflAboveSurf / (coeff0 + coeff1 * reflAboveSurf)

# The '__main__' entry point is only used for testing. 
if __name__ == '__main__':
    import sys
    import numpy as np

    ME = sys.argv[0]
    if len(sys.argv) != 1:
        print("usage:  " + sys.argv[0])
        sys.exit(1)

    # Define inputs:
    reflAboveSurf = np.array([0., .2, .4, .5, .6], dtype=float)
    print("Inputs:")
    print("reflAboveSurf =", reflAboveSurf)

    reflBelowSurf = RrsAboveToBelow(reflAboveSurf)
    print("\nResults:")
    print("reflBelowSurf =", reflBelowSurf)

    print(sys.argv[0] + " done")             # DEBUG
    sys.exit(0)
