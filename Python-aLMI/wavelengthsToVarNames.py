#! /usr/bin/env python3

import generic_io
import re
import math
import numpy as np

def wavelengthsToVarNames(filename, wavelengths, tolerance=0., var_prefix="Rrs_", var_suffix="", verbose=False):
    """
	Determine the variable names in the input data file that correspond to the given list of wavelengths.
	The variable names are expected to be:  var_prefix + WAVELENGTH + var_suffix; e.g., "Rrs_" + 412 + ""

	filename is the name of the input data file.
	wavelengths is a list of wavelengths (may be int or float, or a str version of int or float).

	If the tolerance keyword is specified, then the wavelengths must agree within that amount.
	The var_prefix and var_suffix keywords can be used to specify the prefix and suffix of the variable names.

	Returns a list of variable names, in the same order as the specified wavelengths.
	Raises an exception if no variable corresponding to a requested wavelength is found.
    """

    fid = generic_io.open(filename)
    var_names = generic_io.get_variable_names(fid)		# a list
    generic_io.close(fid)
    have_wavelengths = _what_wavelengths(var_names, var_prefix, var_suffix)	# a list of strings
    if verbose:  print("have_wavelengths =", have_wavelengths)		# DEBUG
    result = []
    for wavelength in wavelengths:
        found = []
        for have_wavelength in have_wavelengths:
            if math.fabs(float(have_wavelength) - float(wavelength)) <= tolerance:
                found.append(var_prefix + have_wavelength + var_suffix)

        if len(found) == 0:
            raise ValueError("Could not find a variable corresponding to wavelength '" + str(wavelength) + "'; have " + str(have_wavelengths))
        elif len(found) > 1:
            raise ValueError("Found too many variables at tolerance = " + str(tolerance) + ":  " + str(found) + "; \n\trequested " + str(wavelengths) + "; have " + str(have_wavelengths))

        result.extend(found)

    return result

def _what_wavelengths(var_names, var_prefix, var_suffix):
    # Return the list of wavelengths (strings) from the variable names that match the specified prefix and suffix.

    result = []
    pattern = var_prefix + "(.+)" + var_suffix + "$"		# require the match to be at least 1 char
    for var_name in var_names:
        matches = re.match(pattern, var_name)
        if matches:
            result.append(matches.group(1))

    return result

# The '__main__' entry point is only used for testing.
if __name__ == '__main__':
    import sys

    ME = sys.argv[0]
    if len(sys.argv) >= 4:
        filename = sys.argv[1]
        var_prefix = sys.argv[2]
        print(ME + ":  filename = " + filename)
        wavelengths = sys.argv[3:]
    else:
        print("usage:  " + sys.argv[0] + " FILENAME VAR_PREFIX WAVELENGTH ...")
        sys.exit(1)

    print("var_prefix = ", var_prefix)
    print("wavelengths =", wavelengths)

    result = wavelengthsToVarNames(filename, wavelengths, tolerance=15, var_prefix=var_prefix, verbose=True)
    print("result =", result)

    print(sys.argv[0] + " done")
