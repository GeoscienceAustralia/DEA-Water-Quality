#! /usr/bin/env python3

import numpy as np

def svd_LMI(yIOP_WL, uIOPRatio_WL, a_star, bb_star, components, verbose=False):
    """
    Setup arrays for inversion, and solve.

    yIOP_WL is the data model constant (y in Eq. 11 from Brando 2012).
    uIOPRatio_WL is the NumPy vector by wavelength of the ratio of backscatter to (absorption + backscatter)
    a_star and bb_star are dicts of absorption and backscatter coeffs; keys are component names ("CHL", etc.);
        (_star signifies that it is a concentration specific coeff.; it's a misnomer in the case of "water")
        values are NumPy vectors by wavelength 
    components is a list of component names that are to be solved for (e.g., ['CHL', 'CDOM', 'NAP']).

    Returns a NumPy vector of component concentrations.
    """

    nWL=len(uIOPRatio_WL)		# number of wavelengths in the inversion
    nCOMP=len(components)

    # setup linear system
    aShape = (nWL, nCOMP)
    aMatrix_WL_COMP = np.empty(aShape, dtype=float)	# the data model matrix (A in Eq. 11 from Brando 2012); need float64?
    j = 0
    for COMP in components:
        aMatrix_WL_COMP[:,j] = a_star[COMP] * uIOPRatio_WL - bb_star[COMP] * (1 - uIOPRatio_WL)
        j += 1

    if verbose:
        print("uIOPRatio_WL =", uIOPRatio_WL)
        print("aMatrix_WL_COMP =", aMatrix_WL_COMP)
        print("yIOP_WL =", yIOP_WL)

    # Solve yIOP_WL = aMatrix_WL_COMP * conc_COMP for conc_COMP (y = Ax; Eq. 11 in Brando 2012):
    conc_COMP, residuals, rank, singVals = np.linalg.lstsq(aMatrix_WL_COMP, yIOP_WL, rcond=0.000001)

    if verbose:
        print("conc_COMP =", conc_COMP)
        print("residuals =", residuals)
        print("rank =", rank)
        print("singVals =", singVals)

    return conc_COMP, residuals, rank, singVals, aMatrix_WL_COMP
