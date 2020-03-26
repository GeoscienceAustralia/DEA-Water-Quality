#! /usr/bin/env python3

import generic_io
import string
import math
from collections import OrderedDict

def SIOP_sets_load(filename, wavelengths=None, tolerance=0., get_attrs=False, verbose=False):
    """
	Load the SIOP sets from a SIOP_sets netCDF file.

	Returns the SIOP sets in an OrderedDict; keys are the SIOP set names, values are the SIOP sets.
	Each SIOP set is a dict; keys are 'component', 'wavelength', 'a_star', 'bb_star'; 
		'component' value is a list of component names (strings), corresponding to the columns of a_star and bb_star;
		'wavelength' value is a NumPy array (float) of wavelengths in nm, corresponding to the rows of a_star and bb_star;
		'a_star' value is a dict; keys are component names (strings); values are NumPy arrays (float) of absorption coefficients;
		'bb_star' value is a dict; keys are component names (strings); values are NumPy arrays (float) of backscatter coefficients.
	If the wavelengths keyword (a sequence) is specified, then select only those wavelengths from each SIOP set.
	If the tolerance keyword is specified, then the wavelengths must agree within that amount.
	If the get_attrs keyword is True, then return a tuple of (SIOPsets, attrs), using variables from the first group.
    """

    print("SIOP_sets_load:  using generic_io from %s" % generic_io.__file__)

    tolerance = float(tolerance)
    result = OrderedDict()
    fid = generic_io.open(filename)
    SIOP_set_names = generic_io.get_group_names(fid)		# a list
    for SIOP_set_name in SIOP_set_names:
        result[SIOP_set_name] = _get_SIOP_set(fid, SIOP_set_name, wavelengths=wavelengths, tolerance=tolerance, verbose=verbose)

    if get_attrs:
        attrs = generic_io.get_attrs(fid)		# a dict of global attrs
        if verbose:  print("DEBUG:  global attrs =", attrs)		# DEBUG
        gid0 = generic_io.get_group(fid, SIOP_set_names[0])	# handle for the first group
        var_names = generic_io.get_variable_names(gid0)
        for var_name in var_names:
            var_attrs = generic_io.get_attrs(gid0, var_name=var_name)	# dict of var attrs
            if verbose:  print("DEBUG:  " + var_name + " attrs =", var_attrs)		# DEBUG
            for key in list(var_attrs.keys()):
                attrs[var_name + ":" + key] = var_attrs[key]	# update with the var attr
        if verbose:  print("DEBUG:  all attrs =", attrs)		# DEBUG

    generic_io.close(fid)

    if get_attrs:
        return (result, attrs)
    else:
        return result

def _get_SIOP_set(fid, SIOP_set_name, wavelengths=None, tolerance=0., verbose=False):
    """
	Get a SIOP set from a SIOP_sets netCDF file.
	If the wavelengths keyword (a sequence) is specified, then select only those wavelengths from the SIOP set.
	If the tolerance keyword is specified, then the wavelengths must agree within that amount (ignored if wavelengths=None).
    """

    gid =  generic_io.get_group(fid, SIOP_set_name)
    componentVar = generic_io.get_variable(gid, "component")
    wavelengthVar = generic_io.get_variable(gid, "wavelength")
    a_starVar = generic_io.get_variable(gid, "a_star")
    bb_starVar = generic_io.get_variable(gid, "bb_star")

    if wavelengths:
        WLs = []			# indices of selected wavelengths
        have_wavelengths = wavelengthVar[:].tolist()	# a list of floats
        if verbose:  print("DEBUG:  select wavelengths from " + str(have_wavelengths))
        for jj in range(len(wavelengths)):
            WL = wavelengths[jj]
            if verbose:  print("DEBUG:  jj =", jj, "; WL =", WL)

            found = []		# indices of values in have_wavelengths that match WL
            for ii in range(len(have_wavelengths)):
                have_wavelength = have_wavelengths[ii]
                if math.fabs(float(have_wavelength) - float(WL)) <= tolerance:
                    found.append(ii)
    
            if len(found) == 0:
                raise ValueError("Could not find a wavelength within " + str(tolerance) + " of " + str(WL) + " in " + str(have_wavelengths))
            elif len(found) == 1:
                found = found[0]
            elif len(found) > 1:		# choose the closest wavelength; if equal deltas, choose the larger wavelength
                # raise ValueError("Found too many wavelengths within " + str(tolerance) + " of " + str(WL) + ":  indices " + str(found) + " in " + str(have_wavelengths))
                ord = []		# list of tuples:  (delta, wavelength, kk)
                for kk in range(len(found)):
                    have_wl = float(have_wavelengths[found[kk]])
                    delta = math.fabs(have_wl - float(WL))
                    ord.append((delta, -have_wl, kk))		# use -have_wl so "larger" wavelength comes earlier in sort
                ord.sort()		# sort by delta, then by wavelength
                found2 = found[ord[0][2]]		# index into have_wavelengths, of "best" match
                if verbose:  print("Found %s wavelengths within %s of %s:  indices %s in %s; choose %s" % (len(found), tolerance, WL, found, have_wavelengths, have_wavelengths[found2]))
                found = found2		# the new "best" index
    
            WLs.append(found)
    else:
        WLs = slice(0, None, None)		# all

    if verbose:  print("DEBUG:  WLs =", WLs, "type =", type(WLs))
    wavelength = wavelengthVar[WLs]		# WLs can be a slice object or a list
    if verbose:  print("DEBUG:  wavelength =", wavelength, "type =", type(wavelength), "shape =", wavelength.shape, "dtype =", wavelength.dtype)	# float32
    component = componentVar[:]
    if verbose:  print("DEBUG:  component =", component, "type =", type(component), "shape =", component.shape, "dtype =", component.dtype)	# object
    nCOMP = component.shape[0]
    a_star = OrderedDict()
    bb_star = OrderedDict()
    for j in range(nCOMP):
        COMP = component[j]
        # print "DEBUG:  COMP =", COMP, "type =", type(COMP)
        a_star[COMP] = a_starVar[WLs, j]
        bb_star[COMP] = bb_starVar[WLs, j]

    return OrderedDict([("component", component), ("wavelength", wavelength), ("a_star", a_star), ("bb_star", bb_star)])

# The '__main__' entry point is only used for testing.
if __name__ == '__main__':
    import sys
    from configUtils import *

    ME = sys.argv[0]
    if len(sys.argv) < 2:
        print("usage:  " + ME + " CONFIG_FILE [GET_ATTRS_FLAG]")
        sys.exit(1)

    if len(sys.argv) >= 3:
        get_attrs = True
    else:
        get_attrs = False

    CONFIG_FILE = sys.argv[1]
    configVerbose = False
    configSet = configLoad(CONFIG_FILE, verbose=configVerbose)

    # Define inputs:
    SIOP_SETS_FILE = getConfigOption(configSet, 'inputParameters', 'SIOP_SETS_FILE', optional=False, verbose=configVerbose)
    SIOP_SETS_10nm_FILE = getConfigOption(configSet, 'optionalParameters', 'SIOP_SETS_10nm_FILE', optional=True, verbose=configVerbose)
    wavelengths = [441.500, 546.5, 407.5]
    tolerance = float(getConfigOption(configSet, 'inputParameters', 'tolerance', optional=False, verbose=configVerbose))

    def _doTest(SIOP_SETS_FILE, wavelengths=None, tolerance=0., get_attrs=False):
        print("_doTest", SIOP_SETS_FILE, "; wavelengths =", wavelengths, "; tolerance =", tolerance, ":")
        result = SIOP_sets_load(SIOP_SETS_FILE, wavelengths=wavelengths, tolerance=tolerance, get_attrs=get_attrs, verbose=False)
        if get_attrs:
            SIOP_sets_dict = result[0]		# a dict
            attrs = result[1]		# a dict
        else:
            SIOP_sets_dict = result			# a dict
        # print "SIOP_sets_dict =", SIOP_sets_dict		# BIG!
        print("SIOP_sets_dict.keys() =", list(SIOP_sets_dict.keys()))
        aKey = list(SIOP_sets_dict.keys())[0]
        aSiopSet = SIOP_sets_dict[aKey]
        # print "aSiopSet =", aSiopSet		# biggish
        print("aSiopSet.keys() =", list(aSiopSet.keys()))		# OK
        for key in list(aSiopSet.keys()):
            print("key =", key, "value =", type(aSiopSet[key]), aSiopSet[key])		# OK

        if get_attrs:
            print("\nattrs =", attrs)

        print()
        return

    _doTest(SIOP_SETS_FILE, wavelengths=wavelengths, tolerance=tolerance, get_attrs=get_attrs)
    if SIOP_SETS_10nm_FILE:
        _doTest(SIOP_SETS_10nm_FILE, wavelengths=wavelengths, tolerance=tolerance, get_attrs=get_attrs)

    print(sys.argv[0] + " done")
