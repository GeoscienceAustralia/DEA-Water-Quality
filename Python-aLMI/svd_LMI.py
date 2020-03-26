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

# The '__main__' entry point is only used for testing.
if __name__ == '__main__':
    import sys

    # Setup to test:
    components = ['CHL', 'CDOM', 'NAP']

    a_star = np.array([
  [0.004717857, 0.06823214, 2.6294, 0.02611947],
  [0.0067218, 0.07067308, 0.9552832, 0.01848856],
  [0.014114, 0.04300423, 0.2106099, 0.01101934],
  [0.04311167, 0.01682276, 0.0514635, 0.006799811],
  [0.052834, 0.01216308, 0.02805054, 0.00552704],
  [0.42976, 0.02240692, 0.0005145862, 0.001406568],
  [0.4566833, 0.02727147, 0.0003563196, 0.001239731],
  [2.84149, 0.0008930769, 3.384405e-05, 0.0005541346]])		# from GBR_CLU4_01 in siops_MODIS_all_CLT4.nc

    bb_star = np.array([
  [0.003515716, 0.0007320828, 0, 0.007808884],
  [0.002592526, 0.0006985981, 0, 0.007451713],
  [0.001704467, 0.0006549357, 0, 0.006985981],
  [0.001191986, 0.0006198538, 0, 0.006611774],
  [0.001031243, 0.0006061982, 0, 0.006466113],
  [0.0004402566, 0.0005317714, 0, 0.005672229],
  [0.0004101905, 0.0005260086, 0, 0.005610758],
  [0.0002680388, 0.0004926718, 0, 0.005255166]])		# from GBR_CLU4_01 in siops_MODIS_all_CLT4.nc

    a_star_dict = {}
    a_star_dict['water'] = a_star[:,0]
    a_star_dict['CHL'] = a_star[:,1]
    a_star_dict['CDOM'] = a_star[:,2]
    a_star_dict['NAP'] = a_star[:,3]

    bb_star_dict = {}
    bb_star_dict['water'] = bb_star[:,0]
    bb_star_dict['CHL'] = bb_star[:,1]
    bb_star_dict['CDOM'] = bb_star[:,2]
    bb_star_dict['NAP'] = bb_star[:,3]

    # Use data from A20120403_0410.20130805213444.L2.hdf.lmi.log (FIRST call of svd_LMI):
    yIOP_WL = np.array([0.00276979, 0.00166469, -0.000407164, -0.00419131, -0.00464664, -0.00616998, -0.00587786, -0.00588015])		# was YY
    print("yIOP_WL.shape =", yIOP_WL.shape)

    uIOPRatio_WL = np.array([0.0905963, 0.0996138, 0.133492, 0.121509, 0.105409, 0.0153655, 0.0137566, 0.00216352])		# was r_f
    print("uIOPRatio_WL.shape =", uIOPRatio_WL.shape)

    aMatrixExpected = np.array([
 [  0.00551582,     0.238214,  -0.00473510],
 [  0.00641100,    0.0951594,  -0.00486770],
 [  0.00517319,    0.0281146,  -0.00458242],
 [  0.00149958,   0.00625328,  -0.00498215],
 [ 0.000739798,   0.00295678,  -0.00520193],
 [-0.000179307,  7.90687e-06,  -0.00556346],
 [-0.000143610,  4.90175e-06,  -0.00551652],
 [-0.000489674,  7.32222e-08,  -0.00524260]])		# was AAA
    print("aMatrixExpected.shape =", aMatrixExpected.shape)

    resultExpected = np.array([0.81754931, 0.014034555, 1.0584364])		# was XX

    result, residuals, rank, singVals, aMatrix = svd_LMI(yIOP_WL, uIOPRatio_WL, a_star_dict, bb_star_dict, components, verbose=True)
    print("aMatrix diff =", aMatrix - aMatrixExpected)
    print("result.shape =", result.shape)
    print("result diff =", result - resultExpected)

    print(sys.argv[0] + " done")             # DEBUG

