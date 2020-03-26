#! /usr/bin/env python3

"""
 Code for doing LMI-specific I/O.
 Supported formats:  formats that are supported by generic_io.py (currently netCDF, HDF).
"""

import generic_io
import numpy as np
from collections import OrderedDict

def get_reflectances(handle, var_names, index_expr=None, verbose=False):
    """
    Read the specified reflectances from the input file (specified with a handle).

    var_names is a list of the variable names of the reflectances (order matters).
    index_expr is a Python index expression that specifies the indexing into each array of data.
        We expect index_expr to be a sequence of 2 slices.  E.g.,
        to get line 10, pixels 100 thru 199, specify:  (slice(10,11), slice(100,200))
        to get lines 10 thru 19, pixels 100 thru 199, specify:  (slice(10,20), slice(100,200))

    Returns a 3-D NumPy array; dimensions are [bands, lines, pixels].
    """
    # ME = "get_reflectances"
    ME = __name__
    if verbose:  print(ME + ":  get_dims =", generic_io.get_dims(handle, var_name=var_names[0]))		# DEBUG
    nbands = len(var_names)
    var0 = generic_io.get_variable(handle, var_names[0])        # an object
    if verbose:  print(ME + ":  var0 =", var0)
    # if verbose:  print ME + ":  var0[:] =", var0[:]		# a NumPy array (BIG!)
    (nlinesTotal, npixelsTotal) = var0[:].shape
    if index_expr:
        lineSlice = index_expr[0]
        pixelSlice = index_expr[1]
        nlines = _slice_size(lineSlice, nlinesTotal)
        npixels = _slice_size(pixelSlice, npixelsTotal)
    else:
        nlines = nlinesTotal
        npixels = npixelsTotal
        index_expr = (slice(0, nlines), slice(0, npixels))
    if verbose:  print(ME + ":  index_expr =", index_expr)

    shape = (nbands, nlines, npixels)
    if verbose:  print(ME + ":  shape =", shape)
    result = np.empty(shape, dtype=float)
    result[0] = var0[index_expr]
    j = 1       # already have var0
    for var_name in var_names[1:]:
        var = generic_io.get_variable(handle, var_name)
        result[j] = var[index_expr]
        j += 1

    return result

def _slice_size(Slice, Len):
    """
    Calculate the number of elements that will be indexed by the given slice, from a vector of the given length.
    """
    if Slice:
        (start, stop, step) = Slice.indices(Len)
        return int((stop - start + step - 1) / step)        #== (stop - start + step - 1) / step
    else:
        return Len		# i.e., all elements

def _make_slice(SSS):	      # make a slice object (or None)
    # SSS is a list of 3 strings, corresponding to Start, Stop, Step
    if not SSS:
        return slice(None)		# i.e., all elements

    try:
        start = int(SSS[0])
    except:
        start = None

    try:
        stop = int(SSS[1])
    except:
        stop = None

    try:
        step = int(SSS[2])
    except:
        step = None

    return slice(start, stop, step)

# The '__main__' entry point is only used for testing.
if __name__ == '__main__':
    import sys
    import os.path
    import argparse

    # Create an ArgumentParser:
    parser = argparse.ArgumentParser(description='Perform I/O on an LMI data file')

    # Positional arguments (required): 
    parser.add_argument('data_file', metavar='DATA_FILE', type=str, help='Data file')
    parser.add_argument('var_names', metavar='VAR_NAMES', type=str, nargs="*", default=None, help='Names of variables to retrieve (default:  None)')

    # Optional arguments:
    parser.add_argument('--lines_SSS', metavar='LINES_SSS', type=str, nargs=3, default=None, help='Line slice parameters:  Start Stop Step (default:  None)')
    parser.add_argument('--pixels_SSS', metavar='PIXELS_SSS', type=str, nargs=3, default=None, help='Pixel slice parameters:  Start Stop Step (default:  None)')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbosity (default:  False)')

    cmd_args = parser.parse_args()
    print("lmi_io.py->__main__:  cmd_args =", cmd_args)

    kw_args = vars(cmd_args)            # convert to a dict
    data_file = kw_args.pop('data_file')                    # don't want it twice!
    print("lmi_io.py->__main__:  kw_args =", kw_args)

    # Open file; get names of all variables:
    hh = generic_io.open(data_file)
    all_vars = generic_io.get_variable_names(hh)
    # print "all_vars =", all_vars

    var_names = kw_args.get("var_names")	# an empty list if none specified on command line
    print("var_names =", var_names)

    # If no var_names specified, get all Rrs_*:
    if len(var_names) == 0:
        for var in all_vars:
            if var.startswith("Rrs_"):
                var_names.append(var)
        var_names.sort()
        print("now var_names =", var_names)
    if len(var_names) == 0:
        raise ValueError("no variables to retrieve")

    lines_SSS = kw_args.get("lines_SSS")	# a list, or None if not specified on command line
    print("lines_SSS =", lines_SSS)
    lines_slice = _make_slice(lines_SSS)		# make a slice object (or None)
    print("lines_slice =", lines_slice)

    pixels_SSS = kw_args.get("pixels_SSS")	# a list, or None if not specified on command line
    print("pixels_SSS =", pixels_SSS)
    pixels_slice = _make_slice(pixels_SSS)	# make a slice object (or None)
    print("pixels_slice =", pixels_slice)

    # Get reflectances, with an index_expr:
    print("Get reflectances, with an index_expr:")
    index_expr = (lines_slice, pixels_slice)
    print("index_expr =", index_expr)
    verbose = cmd_args.verbose
    rrsInd = get_reflectances(hh, var_names, index_expr=index_expr, verbose=verbose)
    print("rrsInd.shape =", rrsInd.shape)

    generic_io.close(hh)

    print(sys.argv[0] + " done")             # DEBUG
