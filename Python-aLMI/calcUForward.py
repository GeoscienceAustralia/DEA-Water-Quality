#! /usr/bin/env python3

import numpy as np

def calcUForward(conc_COMP_PIX, components, a_star, bb_star):
    """
    Use the concentrations from the LMI and convert back to the u ratio.

    conc_COMP_PIX:  a NumPy array of the calculated concentrations of each component
    components:  a list of the component names
    a_star:  a dict of the SIOP set a_star values
    bb_star:  a dict of the SIOP set bb_star values

    Returns a dict of NumPy arrays:  predicted uIOPRatio, total absorption, total backscatter, 
	absorption by component and wavelength, backscatter by component and wavelength
    """

    nWL = len(a_star['WATER'])
    absWater_WL = np.reshape(a_star['WATER'], (nWL, 1))		# so it is broadcastable to the shape of totalAbs_WL_PIX, etc.
    backscatWater_WL = np.reshape(bb_star['WATER'], (nWL, 1))	# so it is broadcastable to the shape of totalAbs_WL_PIX, etc.

    (nCOMP, nPIX) = conc_COMP_PIX.shape

    bShape = (nWL, nCOMP, nPIX)
    abs_WL_COMP_PIX = np.empty(bShape, dtype=float)		# absorption at each WL, for each component
    backscat_WL_COMP_PIX = np.empty(bShape, dtype=float)	# backscatter at each WL, for each component

    for j in range(len(components)):		# index of component in arrays
        COMP = components[j]
        absStar_WL = np.reshape(a_star[COMP], (nWL, 1, 1))
        backscatStar_WL = np.reshape(bb_star[COMP], (nWL, 1, 1))

        conc_PIX = np.reshape(conc_COMP_PIX[j], (1, 1, nPIX))

        abs_WL_COMP_PIX[:,j,:] = np.reshape(conc_PIX * absStar_WL, (nWL, nPIX))
        backscat_WL_COMP_PIX[:,j,:] = np.reshape(conc_PIX * backscatStar_WL, (nWL, nPIX))

    totalAbs_WL_PIX = np.sum(abs_WL_COMP_PIX, axis = 1)		# total absorption (by WL)
    totalBackscat_WL_PIX = np.sum(backscat_WL_COMP_PIX, axis = 1)		# total backscatter (by WL)

    uIOPRatioPredicted_WL_PIX = (totalBackscat_WL_PIX + backscatWater_WL) / (totalBackscat_WL_PIX + totalAbs_WL_PIX + backscatWater_WL + absWater_WL)

    return {'uIOPRatioPredicted':uIOPRatioPredicted_WL_PIX, 'totalAbs':totalAbs_WL_PIX, 'totalBackscat':totalBackscat_WL_PIX, 'abs':abs_WL_COMP_PIX, 'backscat':backscat_WL_COMP_PIX}

# The '__main__' entry point is only used for testing. 
if __name__ == '__main__':
    import sys
    from SIOP_sets_load import SIOP_sets_load
    from configUtils import *

    ME = sys.argv[0]
    if len(sys.argv) != 2:
        print("usage:  " + ME + " CONFIG_FILE")
        sys.exit(1)

    CONFIG_FILE = sys.argv[1]
    configVerbose = False
    configSet = configLoad(CONFIG_FILE, verbose=configVerbose)

    # Define inputs:
    conc_COMP_PIX = np.array([[.4, .7], [.5, .8], [.6, .9]], dtype=float)		# shape is (3, 2)
    components = ['CHL', 'CDOM', 'NAP']
    SIOP_SETS_FILE = getConfigOption(configSet, 'inputParameters', 'SIOP_SETS_FILE', optional=False, verbose=configVerbose)
    siopSets = SIOP_sets_load(SIOP_SETS_FILE)
    siopSet = siopSets[list(siopSets.keys())[0]]		# first SIOP set
    a_star = siopSet['a_star']
    bb_star = siopSet['bb_star']
    print("Inputs:")
    print("conc_COMP_PIX =", conc_COMP_PIX, "shape =", conc_COMP_PIX.shape)
    print("components =", components)
    print("a_star[" + "WATER" + "] =", a_star["WATER"])
    print("bb_star[" + "WATER" + "] =", bb_star["WATER"])
    j = 0		# index of component in array
    for COMP in components:
        print("a_star[" + COMP + "] =", a_star[COMP])
        print("bb_star[" + COMP + "] =", bb_star[COMP])

    result = calcUForward(conc_COMP_PIX, components, a_star, bb_star)
    print("Results:")
    print("uIOPRatioPredicted_WL_PIX =", result['uIOPRatioPredicted'])
    print("totalAbs_WL_PIX =", result['totalAbs'])
    print("totalBackscat_WL_PIX =", result['totalBackscat'])
    print("abs_WL_COMP_PIX =", result['abs'])
    print("backscat_WL_COMP_PIX =", result['backscat'])

    print(sys.argv[0] + " done")             # DEBUG
