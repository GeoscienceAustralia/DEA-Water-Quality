#! /usr/bin/env python3

import numpy as np

def calcY(uIOPRatio_WL_PIX, absWater_WL, backscatWater_WL):
    """
    Calculate the (water-only) Y half of the matrix equation for the chunk.
    Note that Y will need to be adjusted if we are not calculating all the components (of the SIOP sets?),
    which must be done inside the siopSet loop, as it depends on absStar and backscatStar of components.

    uIOPRatio_WL_PIX is the IOP "u" in Eq. 2 and 3 of Brando 2012.
    absWater_WL is the absorption coeff for water.
    backscatWater_WL is the backscatter coeff for water.

    Returns the IOP y in Eq. 12 of Brando 2012.
    """

    shape = (len(absWater_WL), 1)		# so they are broadcastable to the shape of uIOPRatio_WL_PIX
    absWater_WL_1 = np.reshape(absWater_WL, shape)			# make it a [nWL, 1] array
    backscatWater_WL_1 = np.reshape(backscatWater_WL, shape)		# make it a [nWL, 1] array

    yIOP_WL_PIX = -absWater_WL_1 * uIOPRatio_WL_PIX + backscatWater_WL_1 * (1 - uIOPRatio_WL_PIX)		# Eq. 12, y = ...
    return yIOP_WL_PIX

# The '__main__' entry point is only used for testing. 
if __name__ == '__main__':
    import sys

    ME = sys.argv[0]
    if len(sys.argv) != 1:
        print("usage:  " + sys.argv[0])
        sys.exit(1)

    print("Inputs:")
    uIOPRatio_WL_PIX = np.linspace(.01, .41, num=40)
    uIOPRatio_WL_PIX.shape = (8, 5)
    absWater_WL = np.linspace(.1, .8, num=8)
    backscatWater_WL = np.linspace(.15, .85, num=8)
    print("uIOPRatio_WL_PIX =", uIOPRatio_WL_PIX)
    print("absWater_WL =", absWater_WL)
    print("backscatWater_WL =", backscatWater_WL)

    yIOP_WL_PIX = calcY(uIOPRatio_WL_PIX, absWater_WL, backscatWater_WL)
    print("Results:")
    print("yIOP_WL_PIX =", yIOP_WL_PIX)

    print(sys.argv[0] + " done")             # DEBUG
