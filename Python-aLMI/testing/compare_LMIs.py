#! /usr/bin/env python3

import numpy as np
import math
from RrsBelowToAbove import RrsBelowToAbove
from calcUBackward import calcUBackward
from RrsAboveToBelow import RrsAboveToBelow

def _rms(array):
    return math.sqrt(np.mean(np.square(array, dtype=np.float64)))

def compare_LMIs(IDL_file, PY_file, outputType=None, verbose=False):
    """
    Compare the output files of LMI runs using the IDL and the Python codes.
    The variables in IDL LMI files don't have _FillValue attributes.
    Returns:  
    """

    ME = 'compare_LMIs'
    WL = "412"			# WL to compare
    g0Lee = 0.084
    g1Lee = 0.17

    print(ME + ":  compare IDL_file " + IDL_file + " and Python file " + PY_file)
    #== if outputType is None:  outputType = "reflAboveSurf"
    if outputType is None or outputType=="''" or outputType=='""':  outputType = "reflAboveSurf"
    print("outputType =", outputType)

    if outputType.lower() == 'reflAboveSurf'.lower():
        outVarPrefix = 'Rrs_MIM_'
        deltaVarPrefix = 'Delta_Rrs_below_MIM_'
        spectrumIgnoreIDL = -.02
    elif outputType.lower() == 'reflBelowSurf'.lower():
        outVarPrefix = 'Rrs_below_MIM_'
        deltaVarPrefix = 'Delta_Rrs_below_MIM_'
        spectrumIgnoreIDL = RrsAboveToBelow(-.02)
    elif outputType.lower() == 'uIOPRatio'.lower():
        outVarPrefix = 'u_MIM_'
        deltaVarPrefix = 'Delta_u_MIM_'
        spectrumIgnoreIDL = calcUBackward(RrsAboveToBelow(-.02), g0Lee, g1Lee)

    # list of variables to compare:
    var_info = []	# list; each element is a list:  IDL var name, IDL ignore value, Py var name, Py ignore value, Py fill value
    var_info.append(['Chl_MIM', -.2, 'CHL_MIM', -999., -.1])
    var_info.append(['CDOM_MIM', -.2, 'CDOM_MIM', -999., -.1])
    var_info.append(['Nap_MIM', -.2, 'NAP_MIM', -999., -.1])
    var_info.append(['siop_MIM', -2, 'SIOPindex', -999, -1])		# IDL has -1 for SVD failure
    var_info.append(['dR_MIM', 1000., 'cost_Rrs_below', -999., 100.])		# IDL ignore:  also 100?
    # var_info.append(['Rrs_MIM_' + WL, spectrumIgnoreIDL, outVarPrefix + WL, -999., -.01])
    var_info.append([outVarPrefix + WL, spectrumIgnoreIDL, outVarPrefix + WL, -999., -.01])
    # var_info.append(['Rrs_MIM_' + WL, spectrumIgnoreIDL, outVarPrefix + WL, -999., RrsBelowToAbove(-.01)])
    var_info.append(['Delta_Rrs_MIM_' + WL, -.02, deltaVarPrefix + WL, -999., -.01])
    var_info.append(['a_tot_MIM_441', -.2, 'a_tot_MIM_441', -999., -.3])
    var_info.append(['a_phy_MIM_441', -.2, 'a_phy_MIM_441', -999., -.1])
    var_info.append(['a_CDOM_MIM_441', -.2, 'a_CDOM_MIM_441', -999., -.1])
    var_info.append(['a_CDM_MIM_441', -.2, 'a_CDM_MIM_441', -999., -.1])
    var_info.append(['a_P_MIM_441', -.2, 'a_P_MIM_441', -999., -.1])
    var_info.append(['a_NAP_MIM_441', -.2, 'a_NAP_MIM_441', -999., -.1])
    var_info.append(['bb_P_MIM_551', -.2, 'bb_P_MIM_551', -999., -.2])
    var_info.append(['bb_phy_MIM_551', -.2, 'bb_phy_MIM_551', -999., -.1])
    var_info.append(['bb_NAP_MIM_551', -.2, 'bb_NAP_MIM_551', -999., -.1])
    var_info.append(['Kd_par_MIM', -.2, 'Kd_par_MIM', -999., -.1])
    var_info.append(['Kd_490_MIM', -.2, 'Kd_490_MIM', -999., -.1])
    var_info.append(['SD_MIM', -.2, 'SD_MIM', -999., -.1])

    IDL_h = generic_io.open(IDL_file)
    PY_h = generic_io.open(PY_file)

    total_rms_rel_err = 0.

    for v in var_info:
        print("\nv =", v)
        IDL_VAR = v[0]
        IDL_BAD = v[1]		# "bad" values in the IDL array
        PY_VAR = v[2]
        PY_BAD = v[3]		# "bad" values in the PY array
        BOTH_IG = v[4]		# values that should be ignored in IDL and PY array

        try:
            IDL_arr = generic_io.get_variable(IDL_h, IDL_VAR)[:].flat
        except:
            print(IDL_file + " does not have variable " + IDL_VAR + "; skip")
            continue

        try:
            PY_arr = generic_io.get_variable(PY_h, PY_VAR)[:].flat
        except:
            print(PY_file + " does not have variable " + PY_VAR + "; skip")
            continue

        num_elements = len(IDL_arr)
        print("number of elements =", num_elements)

        # IDL_bad = np.nonzero(IDL_arr == IDL_BAD)[0]		# indices of bad values
        # PY_bad = np.nonzero(PY_arr == PY_BAD)[0]		# indices of bad values
        # BOTH_ig = np.nonzero(np.logical_or(PY_arr == BOTH_IG, IDL_arr == BOTH_IG))[0]

        IDL_ok = np.nonzero(np.logical_and(IDL_arr != BOTH_IG, IDL_arr != IDL_BAD))[0]		# indices of good values
        PY_ok = np.nonzero(np.logical_and(PY_arr != BOTH_IG, PY_arr != PY_BAD))[0]		# indices of good values

        # print "IDL_bad:", len(IDL_bad), "IDL_ok:", len(IDL_ok)
        # print " PY_bad:", len(PY_bad), " PY_ok:", len(PY_ok)

        # if len(BOTH_ig) + len(IDL_bad) + len(IDL_ok) != num_elements:
        #     print "	IDL lengths error; ig:", len(BOTH_ig), "bad:", len(IDL_bad), "ok:", len(IDL_ok), "err:", num_elements - (len(BOTH_ig) + len(IDL_bad) + len(IDL_ok))

        # if len(BOTH_ig) + len(PY_bad) + len(PY_ok) != num_elements:
        #     print "	PY lengths error; ig:", len(BOTH_ig), "bad:", len(PY_bad), "ok:", len(PY_ok), "err:", num_elements - (len(BOTH_ig) + len(PY_bad) + len(PY_ok))

        if len(IDL_ok) != len(PY_ok):
            if len(PY_ok) < len(IDL_ok):
                Ok = PY_ok
                print("	ok lengths differ:", len(IDL_ok), len(PY_ok), "using PY_ok")
            else:
                Ok = IDL_ok
                print("	ok lengths differ:", len(IDL_ok), len(PY_ok), "using IDL_ok")
        else:
            print("	ok lengths  agree:", len(IDL_ok), len(PY_ok), "using IDL_ok")
            Ok = IDL_ok
            if np.any(PY_ok != IDL_ok):
                print("	WARNING:  ok indices differ")

        # get stats on the arrays
        IDL_min = np.amin(IDL_arr[Ok])
        IDL_max = np.amax(IDL_arr[Ok])
        IDL_median = np.median(IDL_arr[Ok])
        IDL_std = np.std(IDL_arr[Ok])
        PY_min = np.amin(PY_arr[Ok])
        PY_max = np.amax(PY_arr[Ok])
        PY_median = np.median(PY_arr[Ok])
        PY_std = np.std(PY_arr[Ok])
        print("IDL_min =", IDL_min, "IDL_max =", IDL_max, "IDL_median =", IDL_median, "IDL_std =", IDL_std)
        print(" PY_min =",  PY_min, " PY_max =",  PY_max, " PY_median =",  PY_median, " PY_std =",  PY_std)
        err = PY_arr[Ok] - IDL_arr[Ok]
        rel_err = err / IDL_median
        rms_err = _rms(err)
        rms_rel_err = _rms(rel_err)
        print("RMS error =", rms_err, "RMS relative error =", rms_rel_err)
        total_rms_rel_err += rms_rel_err

    print("Total RMS relative error =", total_rms_rel_err)
    generic_io.close(IDL_h)
    generic_io.close(PY_h)
    return None

# The '__main__' entry point is only used for testing. 
# It requires the generic_io and lmi_io modules, which will be found if PYTHONPATH is set correctly:
#	(tcsh)	setenv PYTHONPATH ${PYTHONPATH}:io_layer
if __name__ == '__main__':
    import sys
    import generic_io

    ME = sys.argv[0]

    if (len(sys.argv) < 3) | (len(sys.argv) > 4):
        print("usage:  " + ME + " IDL_LMI_FILE PYTHON_LMI_FILE [outputType]")
        sys.exit(1)

    IDL_LMI_FILE = sys.argv[1]
    PYTHON_LMI_FILE = sys.argv[2]
    if len(sys.argv) > 3:
        outputType = sys.argv[3]
    else:
        outputType = None

    compare_LMIs(IDL_LMI_FILE, PYTHON_LMI_FILE, outputType=outputType, verbose=True)

    print("\n" + ME + " done")             # DEBUG
