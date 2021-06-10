#! /usr/bin/env python3

import configparser
from collections import OrderedDict
import ast
import os
import sys

import generic_io
import string
import math
import numpy as np



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
# 
# from var_dump.pro, in SPLIT4
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


### SET UP THE BOTH THE CONFIG ANG ARG PARSERS ###
def configLoad(CONFIG_FILE, verbose=False):
    configSet = configparser.SafeConfigParser(None, OrderedDict, True)
    found = configSet.read(CONFIG_FILE)
    if verbose:  print("configLoad:  found config file:", found)		# (a list!)
    if len(found) == 0:
        raise IOError("configLoad:  could not get config from " + CONFIG_FILE)
    return configSet


# Get value of a config option (should have been called a 'parameter', because they are not always optional).
# If optional is False, then throw an exception if the option (or section) is not present.
# If optional is True, then:
#	if the option (or section) is not present, returns False; if specified with no value, returns None; else, returns the value
def getConfigOption(configSet, section, option, optional=False, verbose=False):
    if configSet.has_section(section):
        if verbose:  print("getConfigOption:  " + "has section '" + section + "'")
        if configSet.has_option(section, option):
            if verbose:  print("getConfigOption:  " + section + " has option '" + option + "'")
            value = configSet.get(section, option)
            return value
        else:
            msg = "getConfigOption:  section '" + section + "' does not have option '" + option + "'"
            if optional:
                if verbose:  print(msg)
                return False
            else:
                raise ValueError(msg)
    else:
        msg = "getConfigOption:  configSet does not have section '" + section + "'"
        if optional:
            if verbose:  print("#### WARNING:  " + msg)
            return False
        else:
            raise ValueError(msg)


