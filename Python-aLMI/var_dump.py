#! /usr/bin/env python3

# Print a variable (recursively)
#
# Keywords for use by the initial caller:
# If the VALUES keyword is set, print array values, otherwise just the number of elements.
# If the STEP keyword is set, use that for subsequent indenting, otherwise use the default.
# If the DEBUG keyword is set, print diagnostic info (not indented).
#
# Keywords for use by recursive calls:
# If the INDENT keyword is set, print with that string prepending all output.
# If the PREFIX keyword is set, print with that string (after any indent) prepending all output.
# If the TAGNAME keyword is set, print that for nested structures.

# from var_dump.pro, in SPLIT4

import string
import numpy as np
from collections import OrderedDict

def _output(indent_and_prefix, expr):		# expr may be a string, number, etc. (but not a NumPy array of dtype 'object')
    # print indent_and_prefix, str(expr)
    print(indent_and_prefix, expr)
    return

def _dims_string(n_dims, dims):			# dims is a sequence
    if n_dims <= 0: return ""
    result = "["
    for j in range(n_dims):
        result = result + str(dims[j]).strip()
        if j != (n_dims - 1): result = result + ","
    result = result + "]"
    return result

def var_dump(var, print_values=False, indent="", step="    ", prefix="", tagname=None, debug=False):
    # if debug:  _output(indent + "starting ...", "")
    # _output(indent + "starting ...", "")

    if tagname: tagname_str = "(" + tagname + ") "
    else: tagname_str = ""
    if debug: print("DEBUG:  tagname_str = '" + tagname_str + "'")

    # Determine the type of var:
    vtype = type(var)
    if debug: print("DEBUG:  vtype =", vtype)

    dims_str = ""

    """
    try:
        #IDL n_dims = size(var, /n_dim)		# only if var has a shape (np.array)
        #IDL dims = size(var, /dim)		# only if var has a shape (np.array)
        dims_str = _dims_string(n_dims, dims)
        if debug: print "DEBUG:  n_els =", n_els, ", dims =", dims, ", vtype =", vtype, ", dims_str =", dims_str		# DEBUG
    except:
        dims_str = 'DIMS_STR'
        print "'dims_str =' failed; set dims_str to", dims_str
    """

    if vtype == dict or vtype == OrderedDict:
        keys = list(var.keys())
        if debug: print("DEBUG:  a dict; keys =", keys)
        _output(indent + tagname_str + str(vtype) + dims_str + ":  KEYS = " + ", ".join(keys), "")
        for key in keys:
            value = var[key]
            tag_j = key		# ???
            var_dump(value, print_values=print_values, indent=indent+step, prefix="(" + tag_j + ") ", tagname=tag_j, debug=debug)
    elif vtype == int or vtype == float:
        if print_values:  _output(indent + prefix + str(vtype) + dims_str + ":  ", var)
    elif vtype == str:
        if print_values:  _output(indent + prefix + str(vtype) + dims_str + ":  '" + var + "'", "")
    elif vtype == list:
        # Determine the dimensions of var:
        try:
            n_els = len(var)
            if debug: print("DEBUG:  n_els =", n_els)
            dims_str = _dims_string(1, [n_els])
        except:
            n_els = -1
            print("_dims_string() failed; set n_els to", n_els)
        if print_values:  _output(indent + prefix + str(vtype) + dims_str + ":  ", var)
    elif vtype == tuple:
        # Determine the dimensions of var:
        try:
            n_els = len(var)
            if debug: print("DEBUG:  n_els =", n_els)
            dims_str = _dims_string(1, [n_els])
        except:
            n_els = -1
            print("_dims_string() failed; set n_els to", n_els)
        if print_values:  _output(indent + prefix + str(vtype) + dims_str + ":  ", var)
    elif vtype == np.ndarray:
        if debug: print("DEBUG:  NumPy dtype =", var.dtype)
        # Determine the dimensions of var:
        try:
            n_els = len(var)
            if debug: print("DEBUG:  n_els =", n_els)
            shape = var.shape
            dims_str = _dims_string(len(shape), shape)
        except:
            n_els = -1
            print("_dims_string() failed; set n_els to", n_els)
        if var.dtype == np.object:
            if debug: print("DEBUG:  var.dtype is object")
            vflat = var.flat
            if debug: print("DEBUG:  type(var[0]) =", type(vflat[0]))
            values = repr(vflat[0])
            for j in range(1, n_els):
                values += ", " + repr(vflat[j])
            _output(indent + tagname_str + str(vtype) + dims_str + ":  VALUES = " + values, "")
        else:
            if print_values:  _output(indent + prefix + str(vtype) + dims_str + ":  ", var)
    else:
        if debug: print("DEBUG:  not a known type")
        if print_values:
            _output(indent + prefix + str(vtype) + ":  ", var)

    return

# The '__main__' entry point is only used for testing. 
if __name__ == '__main__':
    import sys

    ME = sys.argv[0]
    if len(sys.argv) != 1:
        print("usage:  " + sys.argv[0])
        sys.exit(1)
 
    d1 = {'a':11, 'b':22., 'c':33}
    d2 = {'A':111, 'B':222, 'C':333, 'D':d1}

    debug = False
    print_values = True

    var_dump(17, print_values=print_values, debug=debug)
    var_dump(17., print_values=print_values, debug=debug)
    var_dump([17], print_values=print_values, debug=debug)
    var_dump((17,), print_values=print_values, debug=debug)
    var_dump(d1, print_values=print_values, debug=debug)
    var_dump(d2, print_values=print_values, debug=debug)
    var_dump(True, print_values=print_values, debug=True)
    vv = np.array([44, 55, 66])
    var_dump(vv, print_values=print_values, debug=True)

    print(sys.argv[0] + " done")

