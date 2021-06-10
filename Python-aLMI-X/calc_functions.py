#! /usr/bin/env python3

import numpy as np
import math
from util_functions import var_dump


def RrsBelowToAbove(reflBelowSurf, coeff0=.52, coeff1=1.7):
    """
    Convert from below-surface rrs to above-surface remote sensing reflectance (Rrs).

    Returns a NumPy array of the same shape as the input
    """
    return coeff0 * reflBelowSurf / (1. - coeff1 * reflBelowSurf)


def RrsAboveToBelow(reflAboveSurf, coeff0=.52, coeff1=1.7):
    """
    Convert from above-surface remote sensing reflectance (Rrs) to below-surface rrs.

    Returns a NumPy array of the same shape as the input
    """
    return reflAboveSurf / (coeff0 + coeff1 * reflAboveSurf)


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


def calcUBackward(reflBelowSurf_WL_PIX, g0, g1):
    """
    Calculate u from Eq. 2 in Brando 2012.

    reflBelowSurf_WL_PIX is the [WL, PIX] NumPy array of below surface reflectance.
    g0, g1 are the model constants.

    Returns a NumPy array of uIOPRatio.
    """
    uIOPRatio_WL_PIX = (-g0 + np.sqrt(g0 * g0 + 4 * g1 * reflBelowSurf_WL_PIX)) / (2 * g1)		# Eq. 2
    return uIOPRatio_WL_PIX


def calcReflForward(uIOPRatioPredicted_WL_PIX, g0, g1):
    reflBelowSurfPredicted_WL_PIX = (g0 + (g1 * uIOPRatioPredicted_WL_PIX)) * uIOPRatioPredicted_WL_PIX
    return reflBelowSurfPredicted_WL_PIX


def calcCost(input_WL_PIX, predicted_WL_PIX, delta_WL_PIX, numWLs, costType):
    if costType == "RMSE":
        err_WL_PIX = delta_WL_PIX
        meanSq_PIX = np.sum(err_WL_PIX * err_WL_PIX, axis=0) / numWLs
        d_r = np.sqrt(meanSq_PIX) / np.mean(input_WL_PIX, axis=0)	# normalise by input (?)
    elif costType == "RMSRE":
        err_WL_PIX = 1. - predicted_WL_PIX / input_WL_PIX
        meanSq_PIX = np.sum(err_WL_PIX * err_WL_PIX, axis=0) / numWLs
        d_r = np.sqrt(meanSq_PIX)
    elif costType == "RMSE_LOG":
        err_WL_PIX = np.log10(input_WL_PIX) - np.log10(predicted_WL_PIX)
        meanSq_PIX = np.sum(err_WL_PIX * err_WL_PIX, axis=0) / numWLs
        d_r = np.sqrt(meanSq_PIX)
    else:
        raise ValueError("invalid costType; must be one of 'RMSE', 'RMSRE', 'RMSE_LOG'")

    return d_r


def calc_mu_d(sun_zen_deg):
    """
    IDL code:
	function compute_mu_d, sun_zen_deg
		 ; and now compute f_mu_d
		 fr_diff=0.0;
		 sun_zen_rad=!dtor*sun_zen_deg
		 n_w=1.333 ; air/water refractive index
		 sun_zen_uw=ASIN(1./n_w*SIN(sun_zen_rad))
		 mu_0=COS(sun_zen_uw)
		 mu_d=0.5*fr_diff+mu_0*(1-fr_diff)
		 return, mu_d
	end

    Returns mu_d
    """
    fr_diff = 0.
    sun_zen_rad = np.radians(sun_zen_deg)
    n_w = 1.333		# air/water refractive index
    sun_zen_uw = np.arcsin(1. / n_w * np.sin(sun_zen_rad))		# radians, in [-pi/2, pi/2] (same as IDL)
    mu_0 = np.cos(sun_zen_uw)
    mu_d = 0.5 * fr_diff + mu_0 * (1. - fr_diff)
    # print "calc_mu_d:  sun_zen_deg = ", sun_zen_deg, "; mu_d = ", mu_d                # DEBUG
    return mu_d


# Calculate k490, kd_par, and Secchi Depth
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
