#! /usr/bin/env python3
# Calculate k490, kd_par, and Secchi Depth

import numpy as np
import math
from calc_mu_d import calc_mu_d
from var_dump import var_dump

def calc_kd_SD(conc_COMP, a_star_WL_COMP, bb_star_WL_COMP, sun_zen_deg, fv, verbose=False):
    """
    conc_COMP:		the calculated concentrations (vector)
    a_star_WL_COMP, bb_star_WL_COMP:	the a_star & bb_star values for each wavelength to be used (usually 420 to 750, by 10nm)
    sun_zen_deg:	the solar zenith angle, in degrees
    fv:			fill value to use - may have invalid concentration(s)

    Returns k490, kd_par, SD
    """
    if np.any(conc_COMP < 0.):		# invalid concentration(s); occurs when aLMI fails
        if verbose:
            print("calc_kd_SD:  WARNING - invalid concentration(s): ", conc_COMP)
        nWLs = a_star_WL_COMP.shape[0]
        return {'kd':np.zeros((nWLs,), dtype=np.float32) + fv, 'kd_par':fv, 'SD':fv}

    if verbose:
        print("calc_kd_SD:  var_dump(conc_COMP):")
        var_dump(conc_COMP, print_values=True, debug=False)
        print("calc_kd_SD:  var_dump(a_star_WL_COMP):")
        var_dump(a_star_WL_COMP, print_values=True, debug=False)
        print("calc_kd_SD:  var_dump(sun_zen_deg):")
        var_dump(sun_zen_deg, print_values=True, debug=False)

    conc_COMP = np.reshape(np.hstack(([1.], conc_COMP)), (len(conc_COMP) + 1, 1))		# include WATER; make it a column vector
    a_tot_WL = np.dot(a_star_WL_COMP, conc_COMP)
    if verbose:
        print("calc_kd_SD:  var_dump(a_tot_WL):")
        var_dump(a_tot_WL, print_values=True, debug=False)
    bb_tot_WL = np.dot(bb_star_WL_COMP, conc_COMP)
    # kd by Lee et al 2005 JGR
    m0 = 1. + 0.005 * sun_zen_deg
    m1 = 4.18
    m2 = 0.52
    m3 = 10.8
    v_WL = m1 * (1. - m2 * np.exp(-m3 * a_tot_WL))
    kd_WL = a_tot_WL * m0 + bb_tot_WL * v_WL
    if verbose:
        print("calc_kd_SD:  var_dump(kd_WL):")
        var_dump(kd_WL, print_values=True, debug=False)

    # retrieve SD according to Brando_2002_11ARSPC
    mu_d = calc_mu_d(sun_zen_deg)
    f_mu_d = 1. / (1. + 2. * mu_d)		# walker 1994

    secchi_WL = f_mu_d / kd_WL

    SD = np.amax(secchi_WL)
    kd_par = np.mean(kd_WL)
    return {'kd':kd_WL, 'kd_par':kd_par, 'SD':SD}

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

    print("Inputs:")
    SIOP_SETS_FILE = getConfigOption(configSet, 'inputParameters', 'SIOP_SETS_FILE', optional=False, verbose=configVerbose)
    siopSets = SIOP_sets_load(SIOP_SETS_FILE)
    siopSetName = list(siopSets.keys())[0]			# first SIOP set
    siopSet = siopSets[siopSetName]
    print("var_dump(siopSet):")
    var_dump(siopSet, print_values=True, debug=False)
    components = siopSet['component']
    a_star = siopSet['a_star']
    bb_star = siopSet['bb_star']
    conc_COMP = np.array([.3, .2, .1])
    sun_zen_deg = 33.

    # Create a matrix of a_star and bb_star values:
    # print "var_dump(a_star[components[0]]):"
    # var_dump(a_star[components[0]], print_values=True, debug=False)
    nWLs = a_star[components[0]].shape[0]
    nCOMPs = len(components)
    print("nWLs =", nWLs, "; nCOMPs =", nCOMPs)
    a_star_WL_COMP = np.empty((nWLs, nCOMPs), dtype=np.float32)
    bb_star_WL_COMP = np.empty((nWLs, nCOMPs), dtype=np.float32)
    for j in range(nCOMPs):
        COMP = components[j]
        a_star_WL_COMP[:,j] = a_star[COMP]
        bb_star_WL_COMP[:,j] = bb_star[COMP]

    fv = -999.
    kdResults = calc_kd_SD(conc_COMP, a_star_WL_COMP, bb_star_WL_COMP, sun_zen_deg, fv, verbose=True)
    print("Results:")
    print("kd =", kdResults['kd'])
    print("kd_par =", kdResults['kd_par'])
    print("SD =", kdResults['SD'])

    print(sys.argv[0] + " done")             # DEBUG
