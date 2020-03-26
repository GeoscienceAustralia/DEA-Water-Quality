#! /usr/bin/env python3
"""
This is the main call file for the aLMI system.

Please run:

    aLMI_main.py --help

Spring 2014

NOTES:
- Requires the generic_io and lmi_io modules, which will be found if PYTHONPATH is set correctly:
    (tcsh)  setenv PYTHONPATH ${PYTHONPATH}:io_layer
    (bash) export PYTHONPATH=$PYTHONPATH:io_layer

index suffixes:
	WL	wavelength
	LINE	scan line
	PIX	pixel, within scan line
	EL	element, within a chunk
	COMP	component

"""

import lmi_io
from calcUBackward import calcUBackward
from calcReflForward import calcReflForward
from chunkProcessLMI import chunkProcessLMI
from RrsAboveToBelow import RrsAboveToBelow
from RrsBelowToAbove import RrsBelowToAbove
from SIOP_sets_load import SIOP_sets_load
from var_dump import var_dump
from calc_kd_SD import calc_kd_SD
from configUtils import *

import argparse
from collections import OrderedDict
import configparser     #== to read / create .ini files
import datetime
import math
import numpy as np
import os
import sys

### SET UP THE ARG PARSER ###
#== Returns an object XYZ with fields 'XYZ.ANN_FILE', 'XYZ.CONFIG_FILE', 'XYZ.OUTPUT_FILE', 'XYZ.verbosity'
def argLoad():
    parser = argparse.ArgumentParser() #setting up the arg parser
    parser.add_argument('ANN_FILE', type=str, help='The first compulsory argument is the radiometery data file to be used as the input data.') # setting up the first input which will be the data file.
    parser.add_argument("CONFIG_FILE", help='The second compulsory argument is the configuration file that contains the settings to be used with this run') #setting up the second input which will be the config file
    parser.add_argument('OUTPUT_FILE', help='The third compulsory argument is the name of the output file to be created on this run. The suffix will determine the file type created - e.g. filename.nc will make a netcdf')
    parser.add_argument("-v","--verbosity", action="count",help="used to control the level of verbosity. Low level (default) verbosity is -v, mid level verbosity is -vv (or --verbosity --verbosity), and the highest level of verbosity is -vvv (or --verbosity --verbosity --verbosity). Use the high level with caution as it prints large matrices and data arrays.")
    inputArgs = parser.parse_args()
    return inputArgs

# This function is for reshaping the results data that comes back from the aLMI step, into the chunk shape.    
#== Returns an array with shape 'outShape' with values from 'inArray' copied where indicated by 'ok_index'; values 
#== in the returned array not copied from 'inArray' are set either to 'fillValue' or 0 (if 'fillValue' is None).
#==   inArray: array of OK pixels, may have fewer pixels due to some having been dropped (QA issues)
#==   inShape: shape of the whole array containing both OK and not-OK pixels 
#==   ok_index: indices of the OK pixels in the 2nd / last dimension of the whole array (of shape inShape)
#==   outShape: shape of the new / output array, once the whole array (inShape) has been resized
#== Assumes that ok_index refers to the 2nd dimension of the array at most, or the last dimension.
def reshapeResults(inArray, ok_index, inShape, outShape, fillValue, verbose=False):
    if verbose:  print("reshapeResults:  inArray.shape =", inArray.shape, "; ok_index.shape =", ok_index.shape, "; inShape =", inShape, "; outShape =", outShape)
    dtype = inArray.dtype
    # print "reshapeResults:  dtype is", dtype			# DEBUG
    if fillValue is not None:
        outArray = np.zeros(inShape, dtype=dtype) + fillValue
    else:
        outArray = np.zeros(inShape, dtype=dtype)
    outArray[:,ok_index] = inArray
    outArray.shape = outShape
    return outArray

# Add two arrays, with FV-awareness (if either element is FV, result is FV)
def addWithFV(array1, array2, fv):
    result = array1 + array2
    missing = np.logical_or(array1 == fv, array2 == fv)
    if np.any(missing):
        result[missing] = fv
    return result

#== Creates a variable in an output (.nc, etc) file with given handle
def _init_variable(handle, name, dtype, dims, fv, attrs, verbose=False):
    # There are no CVs in the output file, so all variables should be compressed (will be ignored if file is not NETCDF4)
    varObj = lmi_io.generic_io.create_variable(handle, name, dtype, dims=dims, fill_value=fv, compress=True, verbose=verbose)
    lmi_io.generic_io.set_attrs(handle, attrs, var_name=name, verbose=False)
    return varObj

#== Creates abs or backsc variable in an output (.nc, etc) file with given handle
#== Wrapper function for _init_variable()
# Convenience routine for initialising abs and backscat variables.
# If attrs is not specified, use the default (specify None for no attributes).
def _init_abs_backscat_variable(handle, var_name, dtype, dims, fv, attrs=False, verbose=False):
    if attrs is False:  attrs = OrderedDict([('long_name', var_name), ('units', 'm-1')])		# the default #== unit here is "m minus one"...
    varObj = _init_variable(handle, var_name, dtype, dims, fv, attrs, verbose=verbose)
    return varObj

#== index_expr_out is the slice object ('list' of indices) where the data is written in the variable object
# Routine to write values to the output file:
# args:  handle of output variable (*VarObj), NumPy array (SIOPindexOutput_LINE_PIX, e.g.), index_expr_out
# E.g.:  writeValues(SIOPindexVarObj, SIOPindexOutput_LINE_PIX, index_expr_out)
def writeValues(varObj, varOutput, index_expr_out, verbose=False):
    in_type = varOutput.dtype
    out_type = varObj[:].dtype
    if in_type != out_type:                 #== changes variable's data type to match output type
        tmp = varOutput.astype(out_type)
        if verbose:  was = "; was " + str(in_type)
    else:
        tmp = varOutput
        if verbose:  was = ""
    varObj[index_expr_out] = tmp            #== writes data to variable object on file
    if verbose:  print("Wrote %s%s" % (out_type, was))

# Get the index of a value in a list (the first found!), or return None if not found.
def indexOf(aList, aValue, verbose=False):
    try:
        result = aList.index(aValue)
        if verbose:  print("%s found at index %s in %s" % (aValue, result, aList))
    except:
        result = None
        print("Warning:  %s not found in %s" % (aValue, aList))
    return result

# Main routine starts here:
def aLMI_main():
    ME = os.path.basename(sys.argv[0])
    startTime = datetime.datetime.utcnow().isoformat(' ') + ' UTC'
    # load in the arguments parsed to this run:
    inputArgs = argLoad()
    print(ME + ":  startTime = " + startTime)
    print("Verbosity level = ", inputArgs.verbosity)

    print("using lmi_io from %s" % lmi_io.__file__)					# DEBUG
    print("using lmi_io.generic_io from %s" % lmi_io.generic_io.__file__)		# DEBUG

    ANN_FILE = inputArgs.ANN_FILE # pull out the file name of the datafile.
    CONFIG_FILE = inputArgs.CONFIG_FILE #pull out the config file you want to use.
    OUTPUT_FILE = inputArgs.OUTPUT_FILE # pull out the output file name.
    print('**** INPUT FILES ****')
    print('The input file to be used is:  ' + ANN_FILE)
    print('The config file to be used is:  ' + CONFIG_FILE)
    print('The output file to be created is:  ' + OUTPUT_FILE)
    print('****')

    # Load in the configuration file parsed into this run.
    configVerbose = (inputArgs.verbosity >= 1)
    configSet = configLoad(CONFIG_FILE, verbose=configVerbose)
    
#==
#== Read in parameters from configuration (.ini) file --> check out .ini file.
#==
    
    # Get non-optional parameters from the config file:
    in_var_names = ast.literal_eval(getConfigOption(configSet, 'inputParameters', 'inputSpectrumVarNames', optional=False, verbose=configVerbose))		# strings
    useWavelengths = ast.literal_eval(getConfigOption(configSet, 'inputParameters', 'useWavelengths', optional=False, verbose=configVerbose))			# numerics
    outWavelengthLabels = ast.literal_eval(getConfigOption(configSet, 'outputParameters', 'outWavelengthLabels', optional=False, verbose=configVerbose))	# strings
    tolerance = float(getConfigOption(configSet, 'inputParameters', 'tolerance', optional=False, verbose=configVerbose))
    SIOP_SETS_FILE = getConfigOption(configSet, 'inputParameters', 'SIOP_SETS_FILE', optional=False, verbose=configVerbose)
    g0 = float(getConfigOption(configSet, 'ggParameters', 'g0', optional=False, verbose=configVerbose))
    g1 = float(getConfigOption(configSet, 'ggParameters', 'g1', optional=False, verbose=configVerbose))
    components = ast.literal_eval(getConfigOption(configSet, 'inputParameters', 'components', optional=False, verbose=configVerbose))
    chunkProcessConfig = ast.literal_eval(getConfigOption(configSet, 'inputParameters', 'chunkProcessConfig', optional=False, verbose=configVerbose))
    inputType = getConfigOption(configSet, 'inputParameters', 'inputType', optional=False, verbose=configVerbose)
    outputType = getConfigOption(configSet, 'outputParameters', 'outputType', optional=False, verbose=configVerbose)
    numberOfLinesPerChunk = int(getConfigOption(configSet, 'inputParameters','numberOfLinesPerChunk', optional=False, verbose=configVerbose))

    # Get optional parameters from the config file:
    #== MIM parameters are True/False flags
    copyInputSpectrum_option = getConfigOption(configSet, 'optionalParameters', 'copyInputSpectrum', optional=True, verbose=configVerbose)
    a_wavelength = getConfigOption(configSet, 'optionalParameters', 'a_wavelength', optional=True, verbose=configVerbose)
    bb_wavelength = getConfigOption(configSet, 'optionalParameters', 'bb_wavelength', optional=True, verbose=configVerbose)
    a_phy_MIM_option = getConfigOption(configSet, 'optionalParameters', 'a_phy_MIM', optional=True, verbose=configVerbose)
    a_CDOM_MIM_option = getConfigOption(configSet, 'optionalParameters', 'a_CDOM_MIM', optional=True, verbose=configVerbose)
    a_NAP_MIM_option = getConfigOption(configSet, 'optionalParameters', 'a_NAP_MIM', optional=True, verbose=configVerbose)
    a_P_MIM_option = getConfigOption(configSet, 'optionalParameters', 'a_P_MIM', optional=True, verbose=configVerbose)
    a_CDM_MIM_option = getConfigOption(configSet, 'optionalParameters', 'a_CDM_MIM', optional=True, verbose=configVerbose)
    a_tot_MIM_option = getConfigOption(configSet, 'optionalParameters', 'a_tot_MIM', optional=True, verbose=configVerbose)
    bb_phy_MIM_option = getConfigOption(configSet, 'optionalParameters', 'bb_phy_MIM', optional=True, verbose=configVerbose)
    bb_NAP_MIM_option = getConfigOption(configSet, 'optionalParameters', 'bb_NAP_MIM', optional=True, verbose=configVerbose)
    bb_P_MIM_option = getConfigOption(configSet, 'optionalParameters', 'bb_P_MIM', optional=True, verbose=configVerbose)

    # Kd_par, Kd_490, and SecchiDepth estimates require a 10nm SIOPS file, and a list of which wavelengths to use
    Kd_par_MIM_option = getConfigOption(configSet, 'optionalParameters', 'Kd_par_MIM', optional=True, verbose=configVerbose)
    Kd_490_MIM_option = getConfigOption(configSet, 'optionalParameters', 'Kd_490_MIM', optional=True, verbose=configVerbose)
    SD_MIM_option = getConfigOption(configSet, 'optionalParameters', 'SD_MIM', optional=True, verbose=configVerbose)
    if Kd_par_MIM_option is not False or Kd_490_MIM_option is not False or SD_MIM_option is not False:
        doKdEstimate = True
        SIOP_SETS_10nm_FILE = getConfigOption(configSet, 'optionalParameters', 'SIOP_SETS_10nm_FILE', optional=True, verbose=configVerbose)
        if type(SIOP_SETS_10nm_FILE) != str:
            print("config requires 'SIOP_SETS_10nm_FILE' when any of 'Kd_par_MIM', 'Kd_490_MIM', 'SD_MIM' are specified")
            sys.exit(1)
        Kd_WavelengthsRange_option = getConfigOption(configSet, 'optionalParameters', 'Kd_WavelengthsRange', optional=True, verbose=configVerbose)
        if Kd_WavelengthsRange_option is not False:		# generate list from "start" to "end" (inclusive)
            Kd_WavelengthsRange = ast.literal_eval(Kd_WavelengthsRange_option)			# numerics
            Kd_Wavelengths = list(range(Kd_WavelengthsRange[0], Kd_WavelengthsRange[1] + 1, 10))	# assumes WLs are at 10nm steps
        else:
            print("config requires 'Kd_WavelengthsRange' when any of 'Kd_par_MIM', 'Kd_490_MIM', 'SD_MIM' are specified")
            sys.exit(1)
    else:
        doKdEstimate = False
        SIOP_SETS_10nm_FILE = None			# so it's defined

    file_fmt = getConfigOption(configSet, 'optionalParameters', 'file_fmt', optional=True, verbose=configVerbose)
    if file_fmt is False:  file_fmt = None		# will use default file format from generic_io.open
    if (inputArgs.verbosity >= 1):  print("file_fmt =", file_fmt)

    ### SETUP THE RUN ###

    # Setup for "sensible" printing of NumPy arrays:
    precision=3  #TODO Put this into the config file? #== for var_dump() purposes...?
    np.set_printoptions(linewidth=160, formatter={'all':lambda x: '%15.6f' % x}, edgeitems=1000, precision=precision)

    if inputType.lower() == 'reflAboveSurf'.lower():
        inputIsU = False
    elif inputType.lower() == 'reflBelowSurf'.lower():
        inputIsU = False
    elif inputType.lower() == 'uIOPRatio'.lower():
        inputIsU = True
    else:
        print("inputType = " + inputType + "; must be one of reflAboveSurf|reflBelowSurf|uIopRatio")
        sys.exit(1)
    print("process input as " + inputType)

    if outputType=='""' or outputType=="''": outputType = inputType     #==

    if outputType.lower() == 'reflAboveSurf'.lower():
        outputIsU = False
        outVarPrefix = 'Rrs_MIM_'   #== MIM = Matrix Inversion Method
        almiInputType = 'Rrs_below'     #== aLMI *input* type defined based on outputType... #==> all OK according to JA & HB!
    elif outputType.lower() == 'reflBelowSurf'.lower():
        outputIsU = False
        outVarPrefix = 'Rrs_below_MIM_'
        almiInputType = 'Rrs_below'
    elif outputType.lower() == 'uIOPRatio'.lower():
        outputIsU = True
        outVarPrefix = 'u_MIM_'
        almiInputType = 'u'
    else:
        print("outputType = " + outputType + "; must be one of reflAboveSurf|reflBelowSurf|uIopRatio|''")
        sys.exit(1)
    print("output will be " + outputType + "; output variable prefix will be " + outVarPrefix)

    if inputType.lower() == 'uIOPRatio'.lower() or outputType.lower() == 'uIOPRatio'.lower():
        deltaVarPrefix = 'Delta_u_MIM_'
    else:
        deltaVarPrefix = 'Delta_Rrs_below_MIM_'

    print("model constants (that capture the water's sun angle geometry and scattering phase functions):  g0 = ", g0, " g1 = ", g1)
    print('Constituent concentrations requested: ', components)                                                                                       
    print('cost function type = ', chunkProcessConfig['costType'])
    print('min valid conc = ',  chunkProcessConfig['minValidConc'])
    print('outputFillValue = ',  chunkProcessConfig['outputFillValue'])
    print('costThreshhold (accepted pixels must have a cost < costThreshhold) = ',  chunkProcessConfig['costThreshhold'])

#== ### LOAD DATA FILES ###

    siopSets = SIOP_sets_load(SIOP_SETS_FILE, wavelengths=useWavelengths, tolerance=tolerance)
    print("Using SIOP sets from", SIOP_SETS_FILE)
    print("Using these SIOP wavelengths: ", siopSets[list(siopSets.keys())[0]]['wavelength'])
    if inputArgs.verbosity >= 1:
        var_dump(siopSets, print_values=True, debug=False)

#== Deal with 10nm file / parameters

    if type(SIOP_SETS_10nm_FILE) == str:
        siopSets10nm = SIOP_sets_load(SIOP_SETS_10nm_FILE, wavelengths=Kd_Wavelengths, tolerance=tolerance)
        print("Using 10nm SIOP sets from", SIOP_SETS_10nm_FILE)
        print("Using these SIOP wavelengths: ", siopSets10nm[list(siopSets10nm.keys())[0]]['wavelength'])
        # Check that the 10nm SIOP sets have the same groups as the channel SIOP sets:
        if inputArgs.verbosity >= 1:
            print("Channel SIOP set names: ", list(siopSets.keys()))
            print("10nm    SIOP set names: ", list(siopSets10nm.keys()))
        if list(siopSets.keys()) != list(siopSets10nm.keys()):
            print("Warning:  channel and 10nm SIOP sets have different names")
        if inputArgs.verbosity >= 1:
            var_dump(siopSets10nm, print_values=True, debug=False)
        if Kd_490_MIM_option is not False:
            i490 = indexOf(Kd_Wavelengths, 490, verbose=(inputArgs.verbosity >= 1))		# index in list of 10nm WLs
        else:
            i490 = None
            print("Warning:  490 not found in %s; will not produce Kd_490_MIM" % Kd_Wavelengths)
    else:
        i490 = None

    if type(a_wavelength) == str:
        iAbs = indexOf(outWavelengthLabels, a_wavelength, verbose=(inputArgs.verbosity >= 1))	# index in WLs
    else:
        iAbs = None
    if type(bb_wavelength) == str:
        iBackscat = indexOf(outWavelengthLabels, bb_wavelength, verbose=(inputArgs.verbosity >= 1))	# index in WLs
    else:
        iBackscat = None
    iCHL = indexOf(components, "CHL", verbose=(inputArgs.verbosity >= 1))		# index of CHL in components    #== ...hardcoded??... nCOMPs later on
    iCDOM = indexOf(components, "CDOM", verbose=(inputArgs.verbosity >= 1))		# index of CDOM in components
    iNAP = indexOf(components, "NAP", verbose=(inputArgs.verbosity >= 1))		# index of NAP in components

    print("The input spectrum variable names in the ANN file are", in_var_names)

#==
#== inputSpectrumVarNames must have same count as useWavelengths and outWavelengthLabels
#== outWavelengthLabels values must be within tolerance of useWavelengths
#==

    if len(in_var_names) != len(useWavelengths):
        print("len(useWavelengths) =", len(useWavelengths), ", len(in_var_names) =", len(in_var_names), "; must be equal")
        sys.exit(1)

    if len(outWavelengthLabels) != len(useWavelengths):
        print("len(useWavelengths) =", len(useWavelengths), ", len(outWavelengthLabels) =", len(outWavelengthLabels), "; must be equal")
        sys.exit(1)

    out_var_names = []
    for j in range(len(outWavelengthLabels)):
        if math.fabs(float(outWavelengthLabels[j]) - useWavelengths[j]) > tolerance:
            print("outWavelengthLabel", outWavelengthLabels[j], "must be within", tolerance, "of useWavelength", useWavelengths[j])
            sys.exit(1)
        out_var_names.append(outVarPrefix + outWavelengthLabels[j])

    print("The spectrum variable names in the LMI output file will be", out_var_names)

    ### OPEN INPUT ANN FILE ###
    handle = lmi_io.generic_io.open(ANN_FILE)
    if inputArgs.verbosity >= 1:
        print("The file handle is =", handle)

    createVarVerbose = True     #== hardcoded?...
    nCOMPs = len(components)		# doesn't include WATER
    fv = chunkProcessConfig['outputFillValue']		# used for some output variables
    if inputArgs.verbosity >= 1:
        print("fv =", fv)

    ### EXTRACT CHUNKS ###
    #== The code below assumes that the inputSpectrumVarNames[0] can be found in ANN_FILE...
    inputShape = lmi_io.generic_io.get_dims(handle, var_name=in_var_names[0]) # Get the dimensions of one of the Rrs datasets.
    totalNumberOfLines = inputShape.get(list(inputShape.keys())[0]) # pull out how many lines there are.
    totalNumberOfPixelsPerLine = inputShape.get(list(inputShape.keys())[1]) # pull out how many pixels there are per line.

    fullFileDims = {'numberOfLines':totalNumberOfLines, 'numberOfPixelsPerLine':totalNumberOfPixelsPerLine}

    print("fullFileDims =", fullFileDims, "; lines per chunk =", numberOfLinesPerChunk)

    ### CREATE AN OUTPUT FILE ###
    #== create file, and create a series of variables to be written to an output (e.g. .nc) file
    writeHandle = lmi_io.generic_io.open(OUTPUT_FILE, file_fmt=file_fmt, access_mode='w')

    # copy some variables "as is":
    vars_copy = ["longitude", "latitude", "l2_flags", "nn_flags"] #TODO: move to config file.   #== hardcoded #== add wavelength??
    if copyInputSpectrum_option is not False:
        vars_copy.extend(in_var_names)

    for var_name in vars_copy:
        var_in = lmi_io.generic_io.get_variable(handle, var_name)
        datatype = var_in[:].dtype
        if var_name is "wavelength":    #== ?? var_name: "longitude", "latitude", "l2_flags", "nn_flags" + inputSpectrumVarNames
            fakeDimNum = list(lmi_io.generic_io.get_dims(handle, var_name=var_name).values())[0]
            dims = {'numberOfInputWavelengths':fakeDimNum}
        elif var_name is "msec":        #== ?? var_name: "longitude", "latitude", "l2_flags", "nn_flags" + inputSpectrumVarNames
            dims = {'numberOfLines':totalNumberOfLines}
        else:
            dims = fullFileDims
        attrs_in = lmi_io.generic_io.get_attrs(handle, var_name=var_name)

        # Fix "long_name" attribute for "longitude" and "latitude":     #== ??
        if var_name in ["longitude", "latitude"]:
            attrs_in["long_name"] = var_name + " at control points"

        # Discard slope and intercept for float variables:     #== ??
        if datatype == np.float32 and attrs_in.get("slope") == "1" and attrs_in.get("intercept") == "0":
            del(attrs_in["slope"])
            del(attrs_in["intercept"])
            if inputArgs.verbosity >= 1:
                print("deleted superfluous 'slope' and 'intercept' attributes from '" + var_name + "' variable")

        # Select the fill value to use:     #== hardcoded??...
        if var_name in ["l2_flags", "nn_flags"]:
            fillValue = None
        elif var_name in in_var_names:
            fillValue = -999.			# what the IDL ANN code produces
        else:   #== "longitude", "latitude"
            fillValue = fv
        
        #== creates / initialises the variable in output file:
        var_out = _init_variable(writeHandle, var_name, datatype, dims, fillValue, attrs_in, verbose=createVarVerbose)
        var_out[:] = var_in[:]      #== copy data 'as-is' from input variable to the created output variable

    # Copy the global attributes:   #== some hardcoded?...
    ## TODO: Get (some of?) these from the config file. #==
    global_attrs_in = lmi_io.generic_io.get_attrs(handle)		# an OrderedDict
    global_attrs = OrderedDict()
    global_attrs['ATMCORR_VERSION'] = global_attrs_in.get('ATMCORR VERSION', global_attrs_in.get('ATMCORR_VERSION'))
    global_attrs['ANN_CREATOR'] = lmi_io.generic_io.get_attrs(handle)['CREATOR']
    global_attrs['ANN_CREATION_DATE'] = global_attrs_in.get('CREATION DATE', global_attrs_in.get('CREATION_DATE'))
    global_attrs['LMI_VERSION'] = 'Python aLMI, Version unknown'  #== hardcoded...
    global_attrs['CREATOR'] = os.environ['USER']
    global_attrs['CREATION_DATE'] = startTime
    global_attrs['EMAIL_ADDRESS'] = 'user@org'  #== hardcoded...

    # Add some LMI global attributes:
    global_attrs['SIOP_SETS_FILE'] = SIOP_SETS_FILE
    global_attrs['history'] = startTime + ": " + ME + ' ' + ANN_FILE + ' ' + CONFIG_FILE + ' ' + OUTPUT_FILE + "\n" + global_attrs_in.get('history', '')
    #== configHandle = file(CONFIG_FILE)
    configHandle = open(CONFIG_FILE)
    global_attrs['config'] = configHandle.read()
    configHandle.close()

    # Set the global attributes into the output file:
    lmi_io.generic_io.set_attrs(writeHandle, global_attrs)

    if doKdEstimate:
        sun_zen_var = lmi_io.generic_io.get_variable(handle, 'zen')		# need solar zenith angle for Kd calculations

    ##########
    # Create output file variables, and make a list of the variable objects that we can use to write into later on
    concVarObjList = []
    concUnits = {'CHL':'ug/l', 'CDOM':'m-1', 'NAP':'mg/L'}		# get from config?  #== hardcoded
    for j in range(nCOMPs):
        var_name = components[j] + '_MIM'
        attrs = OrderedDict([('long_name', 'Concentration of ' + components[j] + ', MIM SVDC on ' + almiInputType), ('units', concUnits[components[j]])])
        concVarObjList.append(_init_variable(writeHandle, var_name, np.float32, fullFileDims, fv, attrs, verbose=createVarVerbose))

    attrs = OrderedDict([('long_name', 'SIOP index, MIM SVDC on ' + almiInputType), ('units', 'dimensionless')])
    SIOPindexVarObj = _init_variable(writeHandle, 'SIOPindex', np.int16, fullFileDims, int(fv), attrs, verbose=createVarVerbose)

    costType = chunkProcessConfig['costType']
    if costType == "RMSE":
        costUnits = "dimensionless"
    elif costType == "RMSRE":
        costUnits = "dimensionless"
    elif costType == "RMSE_LOG":
        costUnits = "<same as input spectrum>"
    else:
        raise ValueError("invalid costType; must be one of 'RMSE', 'RMSRE', 'RMSE_LOG'")
    attrs = OrderedDict([('long_name', 'distance on optical closure, MIM SVDC on ' + almiInputType), ('units', costUnits), ('costType', costType)])
    costVarObj = _init_variable(writeHandle, 'cost_' + almiInputType, np.float32, fullFileDims, fv, attrs, verbose=createVarVerbose)

    if type(a_wavelength) == str:
        if (a_phy_MIM_option is not False) and (iCHL is not None):
            a_phy_MIM_VarObj = _init_abs_backscat_variable(writeHandle, "a_phy_MIM_" + a_wavelength, np.float32, fullFileDims, fv, attrs=False, verbose=createVarVerbose)

        if (a_CDOM_MIM_option is not False) and (iCDOM is not None):
            a_CDOM_MIM_VarObj = _init_abs_backscat_variable(writeHandle, "a_CDOM_MIM_" + a_wavelength, np.float32, fullFileDims, fv, attrs=False, verbose=createVarVerbose)

        if (a_NAP_MIM_option is not False) and (iNAP is not None):
            a_NAP_MIM_VarObj = _init_abs_backscat_variable(writeHandle, "a_NAP_MIM_" + a_wavelength, np.float32, fullFileDims, fv, attrs=False, verbose=createVarVerbose)

        if (a_P_MIM_option is not False) and (iCHL is not None) and (iNAP is not None):
            a_P_MIM_VarObj = _init_abs_backscat_variable(writeHandle, "a_P_MIM_" + a_wavelength, np.float32, fullFileDims, fv, attrs=False, verbose=createVarVerbose)

        if (a_CDM_MIM_option is not False) and (iCDOM is not None) and (iNAP is not None):
            a_CDM_MIM_VarObj = _init_abs_backscat_variable(writeHandle, "a_CDM_MIM_" + a_wavelength, np.float32, fullFileDims, fv, attrs=False, verbose=createVarVerbose)

        if a_tot_MIM_option is not False:
            a_tot_MIM_VarObj = _init_abs_backscat_variable(writeHandle, "a_tot_MIM_" + a_wavelength, np.float32, fullFileDims, fv, attrs=False, verbose=createVarVerbose)

    if type(bb_wavelength) == str:
        if (bb_phy_MIM_option is not False) and (iCHL is not None):
            bb_phy_MIM_VarObj = _init_abs_backscat_variable(writeHandle, "bb_phy_MIM_" + bb_wavelength, np.float32, fullFileDims, fv, attrs=False, verbose=createVarVerbose)

        if (bb_NAP_MIM_option is not False) and (iNAP is not None):
            bb_NAP_MIM_VarObj = _init_abs_backscat_variable(writeHandle, "bb_NAP_MIM_" + bb_wavelength, np.float32, fullFileDims, fv, attrs=False, verbose=createVarVerbose)

        if bb_P_MIM_option is not False:
            bb_P_MIM_VarObj = _init_abs_backscat_variable(writeHandle, "bb_P_MIM_" + bb_wavelength, np.float32, fullFileDims, fv, attrs=False, verbose=createVarVerbose)

    if type(SIOP_SETS_10nm_FILE) == str:
        if Kd_par_MIM_option is not False:
            attrs = OrderedDict([('long_name', 'Kd_par, MIM SVDC on ' + almiInputType), ('units', 'm-1')])
            Kd_par_MIM_VarObj = _init_variable(writeHandle, "Kd_par_MIM", np.float32, fullFileDims, fv, attrs, verbose=createVarVerbose)

        if Kd_490_MIM_option is not False:
            attrs = OrderedDict([('long_name', 'Kd_490, MIM SVDC on ' + almiInputType), ('units', 'm-1')])
            Kd_490_MIM_VarObj = _init_variable(writeHandle, "Kd_490_MIM", np.float32, fullFileDims, fv, attrs, verbose=createVarVerbose)

        if SD_MIM_option is not False:
            attrs = OrderedDict([('long_name', 'Secchi Depth, MIM SVDC on ' + almiInputType), ('units', 'm')])
            SD_MIM_VarObj = _init_variable(writeHandle, "SD_MIM", np.float32, fullFileDims, fv, attrs, verbose=createVarVerbose)

    spectrumVarObjList = []
    for var_name in out_var_names:
        spectrumVarObjList.append(_init_variable(writeHandle, var_name, np.float32, fullFileDims, fv, OrderedDict([('long_name', var_name), ('units', 'm-1')]), verbose=createVarVerbose))

    deltaVarObjList = []
    for j in range(len(out_var_names)):
        var_name = deltaVarPrefix + outWavelengthLabels[j]
        deltaVarObjList.append(_init_variable(writeHandle, var_name, np.float32, fullFileDims, fv, OrderedDict([('long_name', var_name), ('units', 'm-1')]), verbose=createVarVerbose))

    # longitude
    # latitude
    # l2_flags
    # nn_flags
    # Chl_MIM
    # CDOM_MIM
    # Nap_MIM
    # siop_MIM      #== SIOPindex...?
    # a_phy_MIM_441
    # a_CDOM_MIM_441
    # a_NAP_MIM_441
    # a_P_MIM_441
    # a_CDM_MIM_441
    # a_tot_MIM_441
    # bb_phy_MIM_551
    # bb_NAP_MIM_551
    # bb_P_MIM_551
    # a_budget_MIM_441      #== ??
    # Rrs_MIM_412
    # Rrs_MIM_441
    # Rrs_MIM_488
    # Rrs_MIM_531
    # Rrs_MIM_551
    # Rrs_MIM_667
    # Rrs_MIM_678
    # Rrs_MIM_748
    # Delta_Rrs_MIM_412
    # Delta_Rrs_MIM_441
    # Delta_Rrs_MIM_488
    # Delta_Rrs_MIM_531
    # Delta_Rrs_MIM_551
    # Delta_Rrs_MIM_667
    # Delta_Rrs_MIM_678
    # Delta_Rrs_MIM_748
    #== and also: Kd_par_MIM, Kd_490_MIM, SD_MIM

    ### START WORKING ###
    Once = True		# for getting just one print of some things in the following loop
    reshapeResultsVerbose = False       #== hardcoded
    startOfChunks = list(range(0, totalNumberOfLines, numberOfLinesPerChunk))
    numberOfChunks = len(startOfChunks)

    varList = sorted(vars())		# will have the names of all variables, including all "output variable objects"
    if inputArgs.verbosity >= 1:
        print("varList has:")
        for varName in varList:
            vtype = eval('type(' + varName + ')')
            print("\t" + varName, "; type is", vtype)
    
    for chunkIndex in range(numberOfChunks):
        sys.stdout.flush()
        startOfChunk = startOfChunks[chunkIndex]
        endOfChunk = min(startOfChunk + numberOfLinesPerChunk, totalNumberOfLines)		# 1 + (index of last line of chunk)
        if inputArgs.verbosity >= 1:
            print('start chunk', chunkIndex + 1, 'of', numberOfChunks, ': ', startOfChunk + 1, ' to end of the chunk: ', endOfChunk)	# "natural" numbering (from 1)
        writeValuesVerbose = Once and (inputArgs.verbosity >= 1)

        # Make an index expression based on the current chunk.
        index_expr = (slice(startOfChunk,endOfChunk), slice(None, None))    #== line indices; all pixels per line

        # extract this slice of refls|uIOPRatio.
        input_WL_LINE_PIX = lmi_io.get_reflectances(handle, in_var_names, index_expr=index_expr, verbose=(inputArgs.verbosity >= 1))

        ### CHECK THE DATA ###
        input_WL_EL = input_WL_LINE_PIX.copy()
        WL_LINE_PIX_shape = (nWLs, nLines, nPixels) = input_WL_LINE_PIX.shape
        LINE_PIX_shape = (nLines, nPixels)
        nELs = nLines * nPixels     #== total nr of (ok and not-ok) pixels
        input_WL_EL.shape = WL_EL_shape = (nWLs, nELs)      #== re-shapes input_WL_EL into a nWLs x nELs array
        ok = np.all(input_WL_EL > 0.000001, axis=0)   
                #== array of booleans, 1 x nELs    #== conc?!...   # locate pixels where all concentrations are valid (TODO:  add 'minInput' config parameter)
        ok_index = np.nonzero(ok)[0]    #== indices of non-zero elements (of length <= nELs)                                                                            

        nValid = len(ok_index)
        if nValid > 0:
            if inputArgs.verbosity >= 1:
                print("number of valid pixels =", nValid)			# number of valid pixels in this chunk
        else:
            if inputArgs.verbosity >= 1:
                print("no valid pixels; skip this chunk")
            continue

        inputValid_WL_EL = input_WL_EL[:,ok]                                                                              
        if Once and (inputArgs.verbosity >= 1):
            print("inputValid_WL_EL.shape =", inputValid_WL_EL.shape)
            print("input_WL_LINE_PIX.shape =", input_WL_LINE_PIX.shape)
            print("input_WL_EL.shape =", input_WL_EL.shape)
        
        #== Transform data into below-surface refl. so that chunkProcessLMI can calculate u-ratio:
        if inputType.lower() == 'reflAboveSurf'.lower():
            reflBelowSurf_WL_EL = RrsAboveToBelow(inputValid_WL_EL)		### CONVERT FROM ABOVE TO BELOW WATER REFL ###
            LMIinput_WL_EL = reflBelowSurf_WL_EL		#== optim		# will calculate uIOPRatio in chunkProcessLMI
        elif inputType.lower() == 'reflBelowSurf'.lower():
            LMIinput_WL_EL = inputValid_WL_EL				# will calculate uIOPRatio in chunkProcessLMI
        elif inputType.lower() == 'uIOPRatio'.lower():  #== optim
            LMIinput_WL_EL = inputValid_WL_EL				# input is uIOPRatio
            # print "not implemented yet"
            # sys.exit(1)

        ### PROCESS DATA ####
        if Once and (inputArgs.verbosity >= 1):
            print('**** Begin processing data ****')

        ### RUN THE LMI ON THIS CHUNK ####
        result = chunkProcessLMI(siopSets, LMIinput_WL_EL, g0, g1, components, chunkProcessConfig, inputIsU=inputIsU, outputIsU=outputIsU, verbose=(Once and (inputArgs.verbosity >= 2)))
        if Once and (inputArgs.verbosity >= 1):
            print("shape of results from chunkProcessLMI are:")
            for key in list(result.keys()):
                print("\t" + key + ".shape: ", result[key].shape)

        #### PRINT RESULTS ####
        # Lots of this will be wrapped up inside a verbosity level option.
        # May be very big; just print once:
        if Once and (inputArgs.verbosity >= 2):
            ndx = np.arange(len(result['cost']))    #== result['cost'] is 1 x nPIX array, i.e. 1 x nValidPix
            print("<ndx>, cost.T, SIOPindex.T, conc.T:")     #== .T = transposed; 'conc' has several "columns" for each COMP
            print(np.vstack((ndx, result['cost'], result['SIOPindex'], result['conc'])).T)  

        #### RESHAPE THE DATA INTO A CHUNK SIZED MATRIX, AND WRITE TO OUTPUT FILE ####

        SIOPindexOutput_LINE_PIX = reshapeResults(result['SIOPindex'], ok_index, (1, nELs), LINE_PIX_shape, -999, verbose=reshapeResultsVerbose)    #== fv instead?...
        writeValues(SIOPindexVarObj, SIOPindexOutput_LINE_PIX, index_expr, verbose=writeValuesVerbose)

        costOutput_LINE_PIX = reshapeResults(result['cost'], ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
        writeValues(costVarObj, costOutput_LINE_PIX, index_expr, verbose=writeValuesVerbose)

        #### THE CONCENTRATION DATA ####
        concOutput_COMP_LINE_PIX = reshapeResults(result['conc'], ok_index, (nCOMPs, nELs), (nCOMPs, nLines, nPixels), fv, verbose=reshapeResultsVerbose)
        for varNum in range(0, len(concVarObjList)):
            writeValues(concVarObjList[varNum], concOutput_COMP_LINE_PIX[varNum,:,:], index_expr, verbose=writeValuesVerbose)


        #### THE SPECTRUM DATA ####
        if outputType.lower() == 'reflAboveSurf'.lower():
            if 'reflBelowSurfPredicted' in list(result.keys()):
                reflBelowSurfPredicted_WL_EL = result['reflBelowSurfPredicted']
            else:
                reflBelowSurfPredicted_WL_EL = calcReflForward(result['uIOPRatioPredicted'], g0, g1)
            ### CONVERT FROM BELOW TO ABOVE WATER REFL ###                                                          
            spectrum_WL_EL = RrsBelowToAbove(reflBelowSurfPredicted_WL_EL)	# note:  converts a few -0.01 to -.005113
        elif outputType.lower() == 'reflBelowSurf'.lower():
            if 'reflBelowSurfPredicted' in list(result.keys()):
                spectrum_WL_EL = result['reflBelowSurfPredicted']
            else:
                spectrum_WL_EL = calcReflForward(result['uIOPRatioPredicted'], g0, g1)
        elif outputType.lower() == 'uIOPRatio'.lower():
            spectrum_WL_EL = result['uIOPRatioPredicted']

        spectrumOutput_WL_LINE_PIX = reshapeResults(spectrum_WL_EL, ok_index, WL_EL_shape, WL_LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
        for varNum in range(0,nWLs):
            writeValues(spectrumVarObjList[varNum], spectrumOutput_WL_LINE_PIX[varNum,:,:], index_expr, verbose=writeValuesVerbose)

        #### THE DELTA DATA ####
        if 'deltaReflBelowSurf' in list(result.keys()):
            delta_WL_EL = result['deltaReflBelowSurf']
        else:
            delta_WL_EL = result['delta_uIOPRatio']
        deltaOutput_WL_LINE_PIX = reshapeResults(delta_WL_EL, ok_index, WL_EL_shape, WL_LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
        for varNum in range(0,nWLs):
            writeValues(deltaVarObjList[varNum], deltaOutput_WL_LINE_PIX[varNum,:,:], index_expr, verbose=writeValuesVerbose)

        #### WRITE OUT THE OTHER RESULTS ####
        if iAbs is not None:
            if 'a_tot_MIM_VarObj' in varList:
                absTotalOutput_WL_LINE_PIX = reshapeResults(result['absTotal'], ok_index, WL_EL_shape, WL_LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                writeValues(a_tot_MIM_VarObj, absTotalOutput_WL_LINE_PIX[iAbs,:,:], index_expr, verbose=writeValuesVerbose)
            abs_WL_COMP_EL = result['abs']
            abs_COMP_EL = abs_WL_COMP_EL[iAbs,:,:]
            if 'a_phy_MIM_VarObj' in varList:
                #== absCHLOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iCHL,:], ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                absCHLOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iCHL,:], ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                writeValues(a_phy_MIM_VarObj, absCHLOutput_LINE_PIX, index_expr, verbose=writeValuesVerbose)
            if 'a_CDOM_MIM_VarObj' in varList:
                #== absCDOMOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iCDOM,:], ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                absCDOMOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iCDOM,:], ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                writeValues(a_CDOM_MIM_VarObj, absCDOMOutput_LINE_PIX, index_expr, verbose=writeValuesVerbose)
            if 'a_NAP_MIM_VarObj' in varList:
                #== absNAPOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iNAP,:], ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                absNAPOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iNAP,:], ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                writeValues(a_NAP_MIM_VarObj, absNAPOutput_LINE_PIX, index_expr, verbose=writeValuesVerbose)
            if 'a_CDM_MIM_VarObj' in varList:
                # absCDMOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iCDOM,:] + abs_COMP_EL[iNAP,:], ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                #== absCDMOutput_LINE_PIX = reshapeResults(addWithFV(abs_COMP_EL[iCDOM,:], abs_COMP_EL[iNAP,:], fv), ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                absCDMOutput_LINE_PIX = reshapeResults(addWithFV(abs_COMP_EL[iCDOM,:], abs_COMP_EL[iNAP,:], fv), ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                writeValues(a_CDM_MIM_VarObj, absCDMOutput_LINE_PIX, index_expr, verbose=writeValuesVerbose)
            if 'a_P_MIM_VarObj' in varList:
                #== absPOutput_LINE_PIX = reshapeResults(addWithFV(abs_COMP_EL[iCHL,:], abs_COMP_EL[iNAP,:], fv), ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                absPOutput_LINE_PIX = reshapeResults(addWithFV(abs_COMP_EL[iCHL,:], abs_COMP_EL[iNAP,:], fv), ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                writeValues(a_P_MIM_VarObj, absPOutput_LINE_PIX, index_expr, verbose=writeValuesVerbose)
        if iBackscat is not None:
            if 'bb_P_MIM_VarObj' in varList:
                backscatTotalOutput_WL_LINE_PIX = reshapeResults(result['backscatTotal'], ok_index, WL_EL_shape, WL_LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                writeValues(bb_P_MIM_VarObj, backscatTotalOutput_WL_LINE_PIX[iBackscat,:,:], index_expr, verbose=writeValuesVerbose)
            backscat_WL_COMP_EL = result['backscat']
            backscat_COMP_EL = backscat_WL_COMP_EL[iBackscat,:,:]
            if 'bb_phy_MIM_VarObj' in varList:
                #== backscatCHLOutput_LINE_PIX = reshapeResults(backscat_COMP_EL[iCHL,:], ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                backscatCHLOutput_LINE_PIX = reshapeResults(backscat_COMP_EL[iCHL,:], ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                writeValues(bb_phy_MIM_VarObj, backscatCHLOutput_LINE_PIX, index_expr, verbose=writeValuesVerbose)
            if 'bb_NAP_MIM_VarObj' in varList:
                #== backscatNAPOutput_LINE_PIX = reshapeResults(backscat_COMP_EL[iNAP,:], ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                backscatNAPOutput_LINE_PIX = reshapeResults(backscat_COMP_EL[iNAP,:], ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                writeValues(bb_NAP_MIM_VarObj, backscatNAPOutput_LINE_PIX, index_expr, verbose=writeValuesVerbose)

        if doKdEstimate:
            sun_zen_deg = sun_zen_var[index_expr]
            sun_zen_deg_ok = sun_zen_deg.flat[ok_index]
            conc_COMP_EL = result['conc']
            SIOPindex_EL = result['SIOPindex']
            siopSetNames = list(siopSets10nm.keys())
            SIOPcomponents = siopSets10nm[siopSetNames[0]]['component']		# includes WATER
            nWLs10nm = len(siopSets10nm[siopSetNames[0]]['wavelength'])

            a_star_WL_COMP = np.empty((nWLs10nm, nCOMPs + 1), dtype=np.float32)
            bb_star_WL_COMP = np.empty((nWLs10nm, nCOMPs + 1), dtype=np.float32)

            if 'Kd_par_MIM_VarObj' in varList:
                Kd_par = np.zeros((nValid,), dtype=np.float32) + fv
            if 'Kd_490_MIM_VarObj' in varList:
                Kd_490 = np.zeros((nValid,), dtype=np.float32) + fv
            if 'SD_MIM_VarObj' in varList:
                SecchiDepth = np.zeros((nValid,), dtype=np.float32) + fv
            for PIX in range(nValid):
                conc_COMP = conc_COMP_EL[:,PIX]			# a vector
                SIOPindex = SIOPindex_EL[PIX]       #== SIOP set selected at this pixel
                a_star_10nm = siopSets10nm[siopSetNames[SIOPindex]]['a_star']		# for this pixel
                bb_star_10nm = siopSets10nm[siopSetNames[SIOPindex]]['bb_star']		# for this pixel
                for j in range(nCOMPs + 1):					# create arrays of a_star & bb_star
                    COMP = SIOPcomponents[j]
                    a_star_WL_COMP[:,j] = a_star_10nm[COMP]
                    bb_star_WL_COMP[:,j] = bb_star_10nm[COMP]
                KdResults = calc_kd_SD(conc_COMP, a_star_WL_COMP, bb_star_WL_COMP, sun_zen_deg_ok[PIX], fv, verbose=False)
                if 'Kd_par_MIM_VarObj' in varList:
                    Kd_par[PIX] = KdResults['kd_par']
                if 'Kd_490_MIM_VarObj' in varList:
                    Kd_490[PIX] = KdResults['kd'][i490]
                if 'SD_MIM_VarObj' in varList:
                    SecchiDepth[PIX] = KdResults['SD']

            if 'Kd_par_MIM_VarObj' in varList:
                #== Kd_par_Output_LINE_PIX = reshapeResults(Kd_par, ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                Kd_par_Output_LINE_PIX = reshapeResults(Kd_par, ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                writeValues(Kd_par_MIM_VarObj, Kd_par_Output_LINE_PIX, index_expr, verbose=writeValuesVerbose)
            if 'Kd_490_MIM_VarObj' in varList:
                #== Kd_490_Output_LINE_PIX = reshapeResults(Kd_490, ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                Kd_490_Output_LINE_PIX = reshapeResults(Kd_490, ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                writeValues(Kd_490_MIM_VarObj, Kd_490_Output_LINE_PIX, index_expr, verbose=writeValuesVerbose)
            if 'SD_MIM_VarObj' in varList:
                #== SD_Output_LINE_PIX = reshapeResults(SecchiDepth, ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                SD_Output_LINE_PIX = reshapeResults(SecchiDepth, ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                writeValues(SD_MIM_VarObj, SD_Output_LINE_PIX, index_expr, verbose=writeValuesVerbose)

        # siopSetNames = siopSets.keys()
        # aSiopSet = siopSets[siopSetNames[0]]
        # absWater_WL = aSiopSet['a_star']['WATER']
        # abs_star_CDOM_WL = aSiopSet['a_star']['CDOM']
        # abs_star_CHL_WL = aSiopSet['a_star']['CHL']
        # backscat_start_CHL_WL = aSiopSet['bb_star']['CHL']
        # backscat_star_CDOM_WL = aSiopSet['bb_star']['CDOM']
        # backscatWater_WL = aSiopSet['bb_star']['WATER']
        # siopWavelength_WL = aSiopSet['wavelength'] 
        # idx441 = abs(siopWavelength_WL-441) == min(abs(siopWavelength_WL-441))
        #
        # backscatter_CHL_441 = backscatWater_WL[idx441] + backscat_start_CHL_WL[idx441] * concOutput_COMP_LINE_PIX[1,:,:]
        #
        # a_phy_MIM_441 - not currently output by the chunkProcessLMI code
        # a_CDOM_MIM_441  - not currently output by the chunkProcessLMI code
        # a_NAP_MIM_441 - not currently output by the chunkProcessLMI code
        # a_P_MIM_441 - not currently output by the chunkProcessLMI code
        # a_CDM_MIM_441 - not currently output by the chunkProcessLMI code
        # a_tot_MIM_441 - produced by the chunkProcessLMI code and returned in the results dict as a component of 'absTotal'
        # bb_phy_MIM_551 - not currently output by the chunkProcessLMI code
        # bb_NAP_MIM_551 - not currently output by the chunkProcessLMI code
        # bb_P_MIM_551 - not currently output by the chunkProcessLMI code
        # a_budget_MIM_441 

        Once = False

    # CLOSE FILES - is this the best place to do this?
    lmi_io.generic_io.close(writeHandle, verbose=True)
    lmi_io.generic_io.close(handle, verbose=True)

    # For diagnostic purposes:
    if inputArgs.verbosity >= 1:
        print("\nPython variables:")
        for key in sorted(locals().keys()):
            vtype = eval('type(' + key + ')')
            if vtype is np.ndarray:
                print("variable '" + key + "' type is", vtype, "; shape is", eval(key + '.shape'), "; OWNDATA =", eval(key + '.flags["OWNDATA"]'))
            else:
                print("variable '" + key + "' type is", vtype)

    print(ME + " done")    

# The '__main__' entry point.
if __name__ == '__main__':
    aLMI_main()
    sys.exit(0)
