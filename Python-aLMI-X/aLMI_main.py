#! /usr/bin/env python3
"""
This is the main call file for the aLMI system -- updated to run with Xarray inputs rather than 
.nc/.hdf files. Returns an output Xarray of results (instead of writing to .nc/.hdf file).

If the input Xarray object happens to contain a 'time' dimension, the code assumes that the 
Xarray has at most one time slice. In other words, this code only processes one single image 
of input RS data at a time.

Please run:

    aLMI_main.py --help

Spring 2014 -- updated May 2021, eric.lehmann@csiro.au

index suffixes:
	WL	wavelength
	LINE	scan line
	PIX	pixel, within scan line
	EL	element, within a chunk
	COMP	component
"""

### TODO: parallelise the chunk processing for-loop
###       alternatively, integrate / demonstrate the use of aLMI with Dask (...and remove chunk-based processing?)

import xarray as xr
import copy

from calc_functions import calcUBackward, calcReflForward, calc_kd_SD, RrsAboveToBelow, RrsBelowToAbove
from util_functions import SIOP_sets_load, var_dump, configLoad, getConfigOption
from chunkProcessLMI import chunkProcessLMI

from collections import OrderedDict
import configparser     #== to read / create .ini files
import datetime
import math
import numpy as np
import os
import sys
import ast


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
    missing = np.logical_or(array1==fv, array2==fv)
    if np.any(missing):
        result[missing] = fv
    return result


# Get the index of a value in a list (the first found!), or return None if not found.
def indexOf(aList, aValue, verbose=False):
    try:
        result = aList.index(aValue)
        if verbose:  print("%s found at index %s in %s" % (aValue, result, aList))
    except:
        result = None
        print("### Warning:  %s not found in %s" % (aValue, aList))
    return result



# Main routine starts here:
def aLMI_main(INPUT_X_ARRAY_DS, CONFIG_FILE):

    # SolarZenithBandName = "oa_solar_zenith"   # name of the solar zenith band in input dataset --> load from CONFIG file instead
    # inputArgs_verbosity = 1       # ... load from CONFIG file instead

    ME = os.path.basename(sys.argv[0])
    startTime = datetime.datetime.utcnow().isoformat(' ') + ' UTC'
    print(ME + ":  startTime = " + startTime)
    print('The config file to be used is:  ' + CONFIG_FILE)
    
    
    ### Load in the configuration file parsed into this run ###
    #== Read in parameters from configuration (.ini) file
    configSet = configLoad(CONFIG_FILE, verbose=False)
    
    # Get non-optional parameters from the config file:
    inputArgs_verbosity = int(getConfigOption(configSet, 'inputParameters','verbosity', optional=False, verbose=False))
    print("Verbosity level = ", inputArgs_verbosity)
    configVerbose = (inputArgs_verbosity>=2)
    
    SolarZenithBandName = getConfigOption(configSet, 'inputParameters', 'SolarZenithBandName', optional=False, verbose=configVerbose)
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
        assert isinstance(SIOP_SETS_10nm_FILE,str), "Config requires 'SIOP_SETS_10nm_FILE' when any of 'Kd_par_MIM', 'Kd_490_MIM', 'SD_MIM' are specified."
        Kd_WavelengthsRange_option = getConfigOption(configSet, 'optionalParameters', 'Kd_WavelengthsRange', optional=True, verbose=configVerbose)
        if Kd_WavelengthsRange_option is not False:		# generate list from "start" to "end" (inclusive)
            Kd_WavelengthsRange = ast.literal_eval(Kd_WavelengthsRange_option)			# numerics
            Kd_Wavelengths = list(range(Kd_WavelengthsRange[0], Kd_WavelengthsRange[1] + 1, 10))	# assumes WLs are at 10nm steps
        else:
            raise AssertionError("Config requires 'Kd_WavelengthsRange' when any of 'Kd_par_MIM', 'Kd_490_MIM', 'SD_MIM' are specified")
    else:
        doKdEstimate = False
        SIOP_SETS_10nm_FILE = None			# so it's defined
    
    
    ### CHECKING INPUT XARRAY DATASET ###
    
    # If 'time' is a coordinate in the dataset, ensure it has only one slice and then remove it:
    if 'time' in INPUT_X_ARRAY_DS.coords.variables:
        assert INPUT_X_ARRAY_DS.coords.dims['time']==1, "The input dataset has more than 1 time slice."
        INPUT_X_ARRAY_DS = INPUT_X_ARRAY_DS.isel(time=0)   # take the first time slice
        INPUT_X_ARRAY_DS = INPUT_X_ARRAY_DS.drop_vars('time').squeeze(drop=True)   # remove time coordinate
    
    inDS_dimnames = list(INPUT_X_ARRAY_DS.dims.keys())   # [kk for kk in INPUT_X_ARRAY_DS.dims.keys()]   # x/y, or lat/lon, etc.
    assert len(inDS_dimnames)==2, f"The input dataset has more than 2 dimensions: {inDS_dimnames}."
    
    # Ensure that all selected aLMI vars are in the input dataset:
    inDS_data_vars = list(INPUT_X_ARRAY_DS.data_vars)   # [vv for vv in INPUT_X_ARRAY_DS.data_vars]
    assert all([vv in inDS_data_vars for vv in in_var_names]), "The input dataset is missing one or more of the selected aLMI bands."
    
    totalNumberOfLines, totalNumberOfPixelsPerLine = INPUT_X_ARRAY_DS[ in_var_names[0] ].shape
    
    # The code below assumes the lat/lon, resp. x/y objects are 1D...
    dimshape1 = INPUT_X_ARRAY_DS[inDS_dimnames[0]].shape
    dimshape2 = INPUT_X_ARRAY_DS[inDS_dimnames[1]].shape
    assert len(dimshape1)==1, f"Input dataset's '{inDS_dimnames[0]}' coord is a multi-dim array (vector expected)."
    assert len(dimshape2)==1, f"Input dataset's '{inDS_dimnames[1]}' coord is a multi-dim array (vector expected)."
    
    # Ensure that the DS dimensions match the DS shape:
    if dimshape1[0]!=totalNumberOfLines or dimshape2[0]!=totalNumberOfPixelsPerLine:
        inDS_dimnames.reverse()   # e.g. ['x','y'] becomes ['y','x']
    tmp = (INPUT_X_ARRAY_DS[inDS_dimnames[0]].shape[0]==totalNumberOfLines and INPUT_X_ARRAY_DS[inDS_dimnames[1]].shape[0]==totalNumberOfPixelsPerLine)
    assert tmp, "Problem with input dataset dimensions."
    
    if doKdEstimate:		# need solar zenith angle for Kd calculations
        assert SolarZenithBandName in inDS_data_vars, f"Cannot find the solar zenith angle variable {SolarZenithBandName} in the input dataset."
    
    
    ### SETUP THE RUN ###

    # Setup for "sensible" printing of NumPy arrays:
    precision = 3  #TODO Put this into the config file? #== for var_dump() purposes...?
    np.set_printoptions(linewidth=160, formatter={'all':lambda x: '%15.6f' % x}, edgeitems=1000, precision=precision)

    if inputType.lower()=='reflAboveSurf'.lower():
        inputIsU = False
    elif inputType.lower()=='reflBelowSurf'.lower():
        inputIsU = False
    elif inputType.lower()=='uIOPRatio'.lower():
        inputIsU = True
    else:
        raise AssertionError("inputType = " + inputType + "; must be one of reflAboveSurf|reflBelowSurf|uIopRatio")
    print("process input as " + inputType)

    if outputType=='""' or outputType=="''": outputType = inputType

    if outputType.lower()=='reflAboveSurf'.lower():
        outputIsU = False
        outVarPrefix = 'Rrs_MIM_'   #== MIM = Matrix Inversion Method
        almiInputType = 'Rrs_below'     #== aLMI *input* type defined based on outputType... #==> all OK according to JA & HB!
    elif outputType.lower()=='reflBelowSurf'.lower():
        outputIsU = False
        outVarPrefix = 'Rrs_below_MIM_'
        almiInputType = 'Rrs_below'
    elif outputType.lower()=='uIOPRatio'.lower():
        outputIsU = True
        outVarPrefix = 'u_MIM_'
        almiInputType = 'u'
    else:
        raise AssertionError("outputType = " + outputType + "; must be one of reflAboveSurf|reflBelowSurf|uIopRatio|''")
    print("output will be " + outputType + "; output variable prefix will be " + outVarPrefix)

    if inputType.lower()=='uIOPRatio'.lower() or outputType.lower()=='uIOPRatio'.lower():
        deltaVarPrefix = 'Delta_u_MIM_'
    else:
        deltaVarPrefix = 'Delta_Rrs_below_MIM_'

    print("Model constants (that capture the water's sun angle geometry and scattering phase functions):  g0 = ", g0, " g1 = ", g1)
    print('Constituent concentrations requested: ', components)                                                                                       
    print('Cost function type = ', chunkProcessConfig['costType'])
    print('Min valid conc = ',  chunkProcessConfig['minValidConc'])
    print('OutputFillValue = ',  chunkProcessConfig['outputFillValue'])
    print('CostThreshhold (accepted pixels must have a cost < costThreshhold) = ',  chunkProcessConfig['costThreshhold'])
    
    
    ### LOAD DATA FILES ###
    
    siopSets = SIOP_sets_load(SIOP_SETS_FILE, wavelengths=useWavelengths, tolerance=tolerance)
    print("Using SIOP sets from", SIOP_SETS_FILE)
    print("Using these SIOP wavelengths: ", siopSets[list(siopSets.keys())[0]]['wavelength'])
    if inputArgs_verbosity>=2:
        var_dump(siopSets, print_values=True, debug=False)

    # Deal with 10nm file and parameters:
    if isinstance(SIOP_SETS_10nm_FILE,str):
        siopSets10nm = SIOP_sets_load(SIOP_SETS_10nm_FILE, wavelengths=Kd_Wavelengths, tolerance=tolerance)
        print("Using 10nm SIOP sets from", SIOP_SETS_10nm_FILE)
        print("Using these SIOP wavelengths: ", siopSets10nm[list(siopSets10nm.keys())[0]]['wavelength'])
        # Check that the 10nm SIOP sets have the same groups as the channel SIOP sets:
        if inputArgs_verbosity>=1:
            print("Channel SIOP set names: ", list(siopSets.keys()))
            print("10nm    SIOP set names: ", list(siopSets10nm.keys()))
        if list(siopSets.keys())!=list(siopSets10nm.keys()):
            print("### Warning:  channel and 10nm SIOP sets have different names.")
        if inputArgs_verbosity>=2:
            var_dump(siopSets10nm, print_values=True, debug=False)
        if Kd_490_MIM_option is not False:
            i490 = indexOf(Kd_Wavelengths, 490, verbose=(inputArgs_verbosity>=1))		# index in list of 10nm WLs
        else:
            i490 = None
            print(f"### Warning:  490 not found in {Kd_Wavelengths}; will not produce Kd_490_MIM.")
    else:
        i490 = None

    if isinstance(a_wavelength,str):
        iAbs = indexOf(outWavelengthLabels, a_wavelength, verbose=(inputArgs_verbosity>=1))	# index in WLs
    else:
        iAbs = None
    if isinstance(bb_wavelength,str):
        iBackscat = indexOf(outWavelengthLabels, bb_wavelength, verbose=(inputArgs_verbosity>=1))	# index in WLs
    else:
        iBackscat = None
    iCHL = indexOf(components, "CHL", verbose=(inputArgs_verbosity>=1))		# index of CHL in components    #== ...hardcoded??... nCOMPs later on
    iCDOM = indexOf(components, "CDOM", verbose=(inputArgs_verbosity>=1))		# index of CDOM in components
    iNAP = indexOf(components, "NAP", verbose=(inputArgs_verbosity>=1))		# index of NAP in components

    print("The input spectrum variable names in the input dataset are", in_var_names)

#==
#== inputSpectrumVarNames must have same count as useWavelengths and outWavelengthLabels
#== outWavelengthLabels values must be within tolerance of useWavelengths
#==
    
    assert len(in_var_names)==len(useWavelengths), f"len(useWavelengths) = {len(useWavelengths)}, len(in_var_names) = {len(in_var_names)}; must be equal"
    
    assert len(outWavelengthLabels)==len(useWavelengths), f"len(useWavelengths) = {len(useWavelengths)}, len(outWavelengthLabels) = {len(outWavelengthLabels)}; must be equal"

    out_var_names = []
    for ii,ll in enumerate(outWavelengthLabels):
        if math.fabs(float(ll) - useWavelengths[ii])>tolerance:
            raise AssertionError(f"outWavelengthLabel {ll} must be within {tolerance} of useWavelength {useWavelengths[ii]}")
        out_var_names.append(outVarPrefix + ll)

    print("The spectrum variable names in the aLMI output dataset will be", out_var_names)

    nCOMPs = len(components)		# doesn't include WATER
    fv = chunkProcessConfig['outputFillValue']		# used for some output variables
    if inputArgs_verbosity>=1:
        print("fv =", fv)
    
    
    ### CREATE AND INITIALISE OUTPUT DS VARIABLES ###
    # TODO: create function to avoid code duplication
    
    OUTPUT_X_ARRAY_DS = xr.Dataset(coords=INPUT_X_ARRAY_DS.coords, attrs=INPUT_X_ARRAY_DS.attrs)   # copy input DS coords (x/y, lat/lon, etc.) and attrs ('crs', 'grid_mapping', etc.)
    inDS_var_attrs = INPUT_X_ARRAY_DS[in_var_names[0]].attrs    # template dict of variables' attributes
    tot_shape = (totalNumberOfLines,totalNumberOfPixelsPerLine)
    
    concUnits = {'CHL':'ug/l', 'CDOM':'m-1', 'NAP':'mg/L'}		# get from config?  #== hardcoded
    concVarNameList = []
    for cc in components:
        var_name = cc + '_MIM'
        concVarNameList.append(var_name)
        tmp = np.zeros(tot_shape, dtype=np.float32) + fv
        OUTPUT_X_ARRAY_DS[var_name] = (inDS_dimnames, tmp)
        tmp = copy.deepcopy(inDS_var_attrs)
        tmp['long_name'] = 'Concentration of ' + cc + ', MIM SVDC on ' + almiInputType
        tmp['units'] = concUnits[cc]
        tmp['nodata'] = fv
        OUTPUT_X_ARRAY_DS[var_name].attrs = tmp
    
    SIOPindexVarName = 'SIOPindex'
    tmp = np.zeros(tot_shape, dtype=np.int16) + int(fv)
    OUTPUT_X_ARRAY_DS[SIOPindexVarName] = (inDS_dimnames, tmp)
    tmp = copy.deepcopy(inDS_var_attrs)
    tmp['long_name'] = 'SIOP index, MIM SVDC on ' + almiInputType
    tmp['units'] = 'dimensionless'
    tmp['nodata'] = int(fv)
    OUTPUT_X_ARRAY_DS[SIOPindexVarName].attrs = tmp
    
    costType = chunkProcessConfig['costType']
    if costType=="RMSE":
        costUnits = "dimensionless"
    elif costType=="RMSRE":
        costUnits = "dimensionless"
    elif costType=="RMSE_LOG":
        costUnits = "<same as input spectrum>"
    else:
        raise ValueError("invalid costType; must be one of 'RMSE', 'RMSRE', 'RMSE_LOG'")
    
    costVarName = 'cost_' + almiInputType
    tmp = np.zeros(tot_shape, dtype=np.float32) + fv
    OUTPUT_X_ARRAY_DS[costVarName] = (inDS_dimnames, tmp)
    tmp = copy.deepcopy(inDS_var_attrs)
    tmp['long_name'] = 'distance on optical closure, MIM SVDC on ' + almiInputType
    tmp['units'] = costUnits
    tmp['costType'] = costType
    tmp['nodata'] = fv
    OUTPUT_X_ARRAY_DS[costVarName].attrs = tmp
    
    if isinstance(a_wavelength,str):
        if a_phy_MIM_option is not False and iCHL is not None:
            a_phy_MIM_VarName = "a_phy_MIM_" + a_wavelength
            tmp = np.zeros(tot_shape, dtype=np.float32) + fv
            OUTPUT_X_ARRAY_DS[a_phy_MIM_VarName] = (inDS_dimnames, tmp)
            tmp = copy.deepcopy(inDS_var_attrs)
            tmp['long_name'] = a_phy_MIM_VarName
            tmp['units'] = 'm-1'
            tmp['nodata'] = fv
            OUTPUT_X_ARRAY_DS[a_phy_MIM_VarName].attrs = tmp
            
        if a_CDOM_MIM_option is not False and iCDOM is not None:
            a_CDOM_MIM_VarName = "a_CDOM_MIM_" + a_wavelength
            tmp = np.zeros(tot_shape, dtype=np.float32) + fv
            OUTPUT_X_ARRAY_DS[a_CDOM_MIM_VarName] = (inDS_dimnames, tmp)
            tmp = copy.deepcopy(inDS_var_attrs)
            tmp['long_name'] = a_CDOM_MIM_VarName
            tmp['units'] = 'm-1'
            tmp['nodata'] = fv
            OUTPUT_X_ARRAY_DS[a_CDOM_MIM_VarName].attrs = tmp
        
        if a_NAP_MIM_option is not False and iNAP is not None:
            a_NAP_MIM_VarName = "a_NAP_MIM_" + a_wavelength
            tmp = np.zeros(tot_shape, dtype=np.float32) + fv
            OUTPUT_X_ARRAY_DS[a_NAP_MIM_VarName] = (inDS_dimnames, tmp)
            tmp = copy.deepcopy(inDS_var_attrs)
            tmp['long_name'] = a_NAP_MIM_VarName
            tmp['units'] = 'm-1'
            tmp['nodata'] = fv
            OUTPUT_X_ARRAY_DS[a_NAP_MIM_VarName].attrs = tmp
        
        if a_P_MIM_option is not False and iCHL is not None and iNAP is not None:
            a_P_MIM_VarName = "a_P_MIM_" + a_wavelength
            tmp = np.zeros(tot_shape, dtype=np.float32) + fv
            OUTPUT_X_ARRAY_DS[a_P_MIM_VarName] = (inDS_dimnames, tmp)
            tmp = copy.deepcopy(inDS_var_attrs)
            tmp['long_name'] = a_P_MIM_VarName
            tmp['units'] = 'm-1'
            tmp['nodata'] = fv
            OUTPUT_X_ARRAY_DS[a_P_MIM_VarName].attrs = tmp
        
        if a_CDM_MIM_option is not False and iCDOM is not None and iNAP is not None:
            a_CDM_MIM_VarName = "a_CDM_MIM_" + a_wavelength
            tmp = np.zeros(tot_shape, dtype=np.float32) + fv
            OUTPUT_X_ARRAY_DS[a_CDM_MIM_VarName] = (inDS_dimnames, tmp)
            tmp = copy.deepcopy(inDS_var_attrs)
            tmp['long_name'] = a_CDM_MIM_VarName
            tmp['units'] = 'm-1'
            tmp['nodata'] = fv
            OUTPUT_X_ARRAY_DS[a_CDM_MIM_VarName].attrs = tmp

        if a_tot_MIM_option is not False:
            a_tot_MIM_VarName = "a_tot_MIM_" + a_wavelength
            tmp = np.zeros(tot_shape, dtype=np.float32) + fv
            OUTPUT_X_ARRAY_DS[a_tot_MIM_VarName] = (inDS_dimnames, tmp)
            tmp = copy.deepcopy(inDS_var_attrs)
            tmp['long_name'] = a_tot_MIM_VarName
            tmp['units'] = 'm-1'
            tmp['nodata'] = fv
            OUTPUT_X_ARRAY_DS[a_tot_MIM_VarName].attrs = tmp

    if isinstance(bb_wavelength,str):
        if bb_phy_MIM_option is not False and iCHL is not None:
            bb_phy_MIM_VarName = "bb_phy_MIM_" + bb_wavelength
            tmp = np.zeros(tot_shape, dtype=np.float32) + fv
            OUTPUT_X_ARRAY_DS[bb_phy_MIM_VarName] = (inDS_dimnames, tmp)
            tmp = copy.deepcopy(inDS_var_attrs)
            tmp['long_name'] = bb_phy_MIM_VarName
            tmp['units'] = 'm-1'
            tmp['nodata'] = fv
            OUTPUT_X_ARRAY_DS[bb_phy_MIM_VarName].attrs = tmp
        
        if bb_NAP_MIM_option is not False and iNAP is not None:
            bb_NAP_MIM_VarName = "bb_NAP_MIM_" + bb_wavelength
            tmp = np.zeros(tot_shape, dtype=np.float32) + fv
            OUTPUT_X_ARRAY_DS[bb_NAP_MIM_VarName] = (inDS_dimnames, tmp)
            tmp = copy.deepcopy(inDS_var_attrs)
            tmp['long_name'] = bb_NAP_MIM_VarName
            tmp['units'] = 'm-1'
            tmp['nodata'] = fv
            OUTPUT_X_ARRAY_DS[bb_NAP_MIM_VarName].attrs = tmp
        
        if bb_P_MIM_option is not False:
            bb_P_MIM_VarName = "bb_P_MIM_" + bb_wavelength
            tmp = np.zeros(tot_shape, dtype=np.float32) + fv
            OUTPUT_X_ARRAY_DS[bb_P_MIM_VarName] = (inDS_dimnames, tmp)
            tmp = copy.deepcopy(inDS_var_attrs)
            tmp['long_name'] = bb_P_MIM_VarName
            tmp['units'] = 'm-1'
            tmp['nodata'] = fv
            OUTPUT_X_ARRAY_DS[bb_P_MIM_VarName].attrs = tmp
    
    if isinstance(SIOP_SETS_10nm_FILE,str):
        if Kd_par_MIM_option is not False:
            Kd_par_MIM_VarName = "Kd_par_MIM"
            tmp = np.zeros(tot_shape, dtype=np.float32) + fv
            OUTPUT_X_ARRAY_DS[Kd_par_MIM_VarName] = (inDS_dimnames, tmp)
            tmp = copy.deepcopy(inDS_var_attrs)
            tmp['long_name'] = 'Kd_par, MIM SVDC on ' + almiInputType
            tmp['units'] = 'm-1'
            tmp['nodata'] = fv
            OUTPUT_X_ARRAY_DS[Kd_par_MIM_VarName].attrs = tmp
    
        if Kd_490_MIM_option is not False:
            Kd_490_MIM_VarName = "Kd_490_MIM"
            tmp = np.zeros(tot_shape, dtype=np.float32) + fv
            OUTPUT_X_ARRAY_DS[Kd_490_MIM_VarName] = (inDS_dimnames, tmp)
            tmp = copy.deepcopy(inDS_var_attrs)
            tmp['long_name'] = 'Kd_490, MIM SVDC on ' + almiInputType
            tmp['units'] = 'm-1'
            tmp['nodata'] = fv
            OUTPUT_X_ARRAY_DS[Kd_490_MIM_VarName].attrs = tmp
    
        if SD_MIM_option is not False:
            SD_MIM_VarName = "SD_MIM"
            tmp = np.zeros(tot_shape, dtype=np.float32) + fv
            OUTPUT_X_ARRAY_DS[SD_MIM_VarName] = (inDS_dimnames, tmp)
            tmp = copy.deepcopy(inDS_var_attrs)
            tmp['long_name'] = 'Secchi Depth, MIM SVDC on ' + almiInputType
            tmp['units'] = 'm'
            tmp['nodata'] = fv
            OUTPUT_X_ARRAY_DS[SD_MIM_VarName].attrs = tmp
    
    spectrumVarNameList = []
    for var_name in out_var_names:
        spectrumVarNameList.append(var_name)
        tmp = np.zeros(tot_shape, dtype=np.float32) + fv
        OUTPUT_X_ARRAY_DS[var_name] = (inDS_dimnames, tmp)
        tmp = copy.deepcopy(inDS_var_attrs)
        tmp['long_name'] = var_name
        tmp['units'] = 'm-1'
        tmp['nodata'] = fv
        OUTPUT_X_ARRAY_DS[var_name].attrs = tmp
    
    deltaVarNameList = []
    for jj in range(len(out_var_names)):
        var_name = deltaVarPrefix + outWavelengthLabels[jj]
        deltaVarNameList.append(var_name)
        tmp = np.zeros(tot_shape, dtype=np.float32) + fv
        OUTPUT_X_ARRAY_DS[var_name] = (inDS_dimnames, tmp)
        tmp = copy.deepcopy(inDS_var_attrs)
        tmp['long_name'] = var_name
        tmp['units'] = 'm-1'
        tmp['nodata'] = fv
        OUTPUT_X_ARRAY_DS[var_name].attrs = tmp
    
    
    ### START PROCESSING ###
    
    Once = True		# for getting just one print of some things in the following loop
    reshapeResultsVerbose = False       #== hardcoded
    startOfChunks = list(range(0, totalNumberOfLines, numberOfLinesPerChunk))
    numberOfChunks = len(startOfChunks)

    varList = sorted(vars())		# will have the names of all variables, including all "output variables"
    if inputArgs_verbosity>=2:
        print("varList has:")
        for varName in varList:
            vtype = eval('type(' + varName + ')')
            print("\t" + varName, "; type is", vtype)
    
    
    ### CHUNK PROCESSING ###
    
    # Processing of each chunk in sequence:
    for chunkIndex in range(numberOfChunks):
        sys.stdout.flush()
        startOfChunk = startOfChunks[chunkIndex]
        endOfChunk = min(startOfChunk + numberOfLinesPerChunk, totalNumberOfLines)		# 1 + (index of last line of chunk)
        if inputArgs_verbosity>=1:
            print('--- Processing chunk', chunkIndex + 1, 'of', numberOfChunks, ': ', startOfChunk + 1, ' to end of the chunk: ', endOfChunk)	# "natural" numbering (from 1)
            
#== input_WL_LINE_PIX is a 3-D NumPy array; dimensions are [bands, lines, pixels], where 'lines' and 'pixels' are defined by 'index_expr'
#== in_var_names is a list of the variable names of the reflectances (order matters)
        
        lineSlice = slice(startOfChunk,endOfChunk)
        pixelSlice = slice(None, None)
        input_WL_LINE_PIX = INPUT_X_ARRAY_DS[in_var_names].to_array().values[:,lineSlice,pixelSlice]   # class = numpy.ndarray [bands,nlines,npix]
        
        ### CHECK THE DATA ###
        input_WL_EL = input_WL_LINE_PIX.copy()
        WL_LINE_PIX_shape = (nWLs, nLines, nPixels) = input_WL_LINE_PIX.shape
        LINE_PIX_shape = (nLines, nPixels)
        nELs = nLines * nPixels     #== total nr of (ok and not-ok) pixels
        input_WL_EL.shape = WL_EL_shape = (nWLs, nELs)      #== re-shapes input_WL_EL into a nWLs x nELs array
        with np.errstate(invalid='ignore'):    # ignore warnings: RuntimeWarning: invalid value encountered in greater
            ok = np.all(input_WL_EL > 0.000001, axis=0)
                #== array of booleans, 1 x nELs    # locate pixels where all concentrations are valid (TODO:  add 'minInput' config parameter)   #== conc?!...   
        ok_index = np.nonzero(ok)[0]    #== indices of non-zero/True elements (of length <= nELs)                                                                            

        nValid = len(ok_index)
        if nValid > 0:
            if inputArgs_verbosity>=1:
                print("Number of valid pixels =", nValid)			# number of valid pixels in this chunk
        else:   #== no OK pixel found -- do nothing (output variables have been pre-filled with FV
            if inputArgs_verbosity>=1:
                print("No valid pixels; skipping this chunk")
            continue

        inputValid_WL_EL = input_WL_EL[:,ok]
        if Once and (inputArgs_verbosity>=1):
            print("inputValid_WL_EL.shape =", inputValid_WL_EL.shape)
            print("input_WL_LINE_PIX.shape =", input_WL_LINE_PIX.shape)
            print("input_WL_EL.shape =", input_WL_EL.shape)
        
        #== Transform data into below-surface refl. so that chunkProcessLMI can calculate u-ratio:
        if inputType.lower()=='reflAboveSurf'.lower():
            reflBelowSurf_WL_EL = RrsAboveToBelow(inputValid_WL_EL)		### CONVERT FROM ABOVE TO BELOW WATER REFL ###
            LMIinput_WL_EL = reflBelowSurf_WL_EL		#== optim		# will calculate uIOPRatio in chunkProcessLMI
        elif inputType.lower()=='reflBelowSurf'.lower():
            LMIinput_WL_EL = inputValid_WL_EL				# will calculate uIOPRatio in chunkProcessLMI
        elif inputType.lower()=='uIOPRatio'.lower():  #== optim
            LMIinput_WL_EL = inputValid_WL_EL				# input is uIOPRatio
        
        ### RUN THE LMI ON THIS CHUNK ####
        # chunkProcessLMI returns:  a dict of results, i.e. dict of numpy arrays
        result = chunkProcessLMI(siopSets, LMIinput_WL_EL, g0, g1, components, chunkProcessConfig, inputIsU=inputIsU, outputIsU=outputIsU, verbose=(Once and (inputArgs_verbosity>=2)))
        if Once and (inputArgs_verbosity>=1):
            print("shape of results from chunkProcessLMI are:")
            for key in list(result.keys()):
                print("\t" + key + ".shape: ", result[key].shape)

        #### PRINT RESULTS ####
        # Lots of this will be wrapped up inside a verbosity level option.
        # May be very big; just print once:
        if Once and (inputArgs_verbosity>=2):
            ndx = np.arange(len(result['cost']))    #== result['cost'] is 1 x nPIX array, i.e. 1 x nValidPix
            print("<ndx>, cost.T, SIOPindex.T, conc.T:")     #== .T = transposed; 'conc' has several "columns" for each COMP
            print(np.vstack((ndx, result['cost'], result['SIOPindex'], result['conc'])).T)  

        #### RESHAPE THE DATA INTO A CHUNK SIZED MATRIX AND SAVE TO DATASET ####
        #== SIOPindexOutput_LINE_PIX = reshapeResults(result['SIOPindex'], ok_index, (1, nELs), LINE_PIX_shape, -999, verbose=reshapeResultsVerbose)    #== int(fv) instead?...
        SIOPindexOutput_LINE_PIX = reshapeResults(result['SIOPindex'], ok_index, (1, nELs), LINE_PIX_shape, int(fv), verbose=reshapeResultsVerbose)    #== int(fv) instead?...
        OUTPUT_X_ARRAY_DS[SIOPindexVarName][lineSlice,pixelSlice] = SIOPindexOutput_LINE_PIX

        costOutput_LINE_PIX = reshapeResults(result['cost'], ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
        OUTPUT_X_ARRAY_DS[costVarName][lineSlice,pixelSlice] = costOutput_LINE_PIX

        #### THE CONCENTRATION DATA ####
        concOutput_COMP_LINE_PIX = reshapeResults(result['conc'], ok_index, (nCOMPs, nELs), (nCOMPs, nLines, nPixels), fv, verbose=reshapeResultsVerbose)
        for ii,vv in enumerate(concVarNameList):
            OUTPUT_X_ARRAY_DS[vv][lineSlice,pixelSlice] = concOutput_COMP_LINE_PIX[ii,:,:]

        #### SPECTRUM DATA ####
        if outputType.lower()=='reflAboveSurf'.lower():
            if 'reflBelowSurfPredicted' in result:
                reflBelowSurfPredicted_WL_EL = result['reflBelowSurfPredicted']
            else:
                reflBelowSurfPredicted_WL_EL = calcReflForward(result['uIOPRatioPredicted'], g0, g1)
            ### CONVERT FROM BELOW TO ABOVE WATER REFL ###                                                          
            spectrum_WL_EL = RrsBelowToAbove(reflBelowSurfPredicted_WL_EL)	# note:  converts a few -0.01 to -.005113
        elif outputType.lower()=='reflBelowSurf'.lower():
            if 'reflBelowSurfPredicted' in result:
                spectrum_WL_EL = result['reflBelowSurfPredicted']
            else:
                spectrum_WL_EL = calcReflForward(result['uIOPRatioPredicted'], g0, g1)
        elif outputType.lower()=='uIOPRatio'.lower():
            spectrum_WL_EL = result['uIOPRatioPredicted']

        spectrumOutput_WL_LINE_PIX = reshapeResults(spectrum_WL_EL, ok_index, WL_EL_shape, WL_LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
        for ii,vv in enumerate(spectrumVarNameList):
            OUTPUT_X_ARRAY_DS[vv][lineSlice,pixelSlice] = spectrumOutput_WL_LINE_PIX[ii,:,:]

        #### DELTA DATA ####
        if 'deltaReflBelowSurf' in result:
            delta_WL_EL = result['deltaReflBelowSurf']
        else:
            delta_WL_EL = result['delta_uIOPRatio']
        deltaOutput_WL_LINE_PIX = reshapeResults(delta_WL_EL, ok_index, WL_EL_shape, WL_LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
        for ii,vv in enumerate(deltaVarNameList):
            OUTPUT_X_ARRAY_DS[vv][lineSlice,pixelSlice] = deltaOutput_WL_LINE_PIX[ii,:,:]

        #### OTHER RESULTS ####
        if iAbs is not None:
            if 'a_tot_MIM_VarName' in varList:
                absTotalOutput_WL_LINE_PIX = reshapeResults(result['absTotal'], ok_index, WL_EL_shape, WL_LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                OUTPUT_X_ARRAY_DS[a_tot_MIM_VarName][lineSlice,pixelSlice] = absTotalOutput_WL_LINE_PIX[iAbs,:,:]
            abs_WL_COMP_EL = result['abs']
            abs_COMP_EL = abs_WL_COMP_EL[iAbs,:,:]
            if 'a_phy_MIM_VarName' in varList:
                #== absCHLOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iCHL,:], ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                absCHLOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iCHL,:], ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                OUTPUT_X_ARRAY_DS[a_phy_MIM_VarName][lineSlice,pixelSlice] = absCHLOutput_LINE_PIX
            if 'a_CDOM_MIM_VarName' in varList:
                #== absCDOMOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iCDOM,:], ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                absCDOMOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iCDOM,:], ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                OUTPUT_X_ARRAY_DS[a_CDOM_MIM_VarName][lineSlice,pixelSlice] = absCDOMOutput_LINE_PIX
            if 'a_NAP_MIM_VarName' in varList:
                #== absNAPOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iNAP,:], ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                absNAPOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iNAP,:], ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                OUTPUT_X_ARRAY_DS[a_NAP_MIM_VarName][lineSlice,pixelSlice] = absNAPOutput_LINE_PIX
            if 'a_CDM_MIM_VarName' in varList:
                # absCDMOutput_LINE_PIX = reshapeResults(abs_COMP_EL[iCDOM,:] + abs_COMP_EL[iNAP,:], ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                #== absCDMOutput_LINE_PIX = reshapeResults(addWithFV(abs_COMP_EL[iCDOM,:], abs_COMP_EL[iNAP,:], fv), ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                absCDMOutput_LINE_PIX = reshapeResults(addWithFV(abs_COMP_EL[iCDOM,:], abs_COMP_EL[iNAP,:], fv), ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                OUTPUT_X_ARRAY_DS[a_CDM_MIM_VarName][lineSlice,pixelSlice] = absCDMOutput_LINE_PIX
            if 'a_P_MIM_VarName' in varList:
                #== absPOutput_LINE_PIX = reshapeResults(addWithFV(abs_COMP_EL[iCHL,:], abs_COMP_EL[iNAP,:], fv), ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                absPOutput_LINE_PIX = reshapeResults(addWithFV(abs_COMP_EL[iCHL,:], abs_COMP_EL[iNAP,:], fv), ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                OUTPUT_X_ARRAY_DS[a_P_MIM_VarName][lineSlice,pixelSlice] = absPOutput_LINE_PIX
        if iBackscat is not None:
            if 'bb_P_MIM_VarName' in varList:
                backscatTotalOutput_WL_LINE_PIX = reshapeResults(result['backscatTotal'], ok_index, WL_EL_shape, WL_LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                OUTPUT_X_ARRAY_DS[bb_P_MIM_VarName][lineSlice,pixelSlice] = backscatTotalOutput_WL_LINE_PIX[iBackscat,:,:]
            backscat_WL_COMP_EL = result['backscat']
            backscat_COMP_EL = backscat_WL_COMP_EL[iBackscat,:,:]
            if 'bb_phy_MIM_VarName' in varList:
                #== backscatCHLOutput_LINE_PIX = reshapeResults(backscat_COMP_EL[iCHL,:], ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                backscatCHLOutput_LINE_PIX = reshapeResults(backscat_COMP_EL[iCHL,:], ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                OUTPUT_X_ARRAY_DS[bb_phy_MIM_VarName][lineSlice,pixelSlice] = backscatCHLOutput_LINE_PIX
            if 'bb_NAP_MIM_VarName' in varList:
                #== backscatNAPOutput_LINE_PIX = reshapeResults(backscat_COMP_EL[iNAP,:], ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                backscatNAPOutput_LINE_PIX = reshapeResults(backscat_COMP_EL[iNAP,:], ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                OUTPUT_X_ARRAY_DS[bb_NAP_MIM_VarName][lineSlice,pixelSlice] = backscatNAPOutput_LINE_PIX

        if doKdEstimate:
            sun_zen_deg = INPUT_X_ARRAY_DS[SolarZenithBandName].values[lineSlice,pixelSlice]   # .to_array() not needed (1-band obj is already DataArray)
            sun_zen_deg_ok = sun_zen_deg.flat[ok_index]
            conc_COMP_EL = result['conc']
            SIOPindex_EL = result['SIOPindex']
            siopSetNames = list(siopSets10nm.keys())
            SIOPcomponents = siopSets10nm[siopSetNames[0]]['component']		# (list) includes WATER
            nWLs10nm = len(siopSets10nm[siopSetNames[0]]['wavelength'])

            a_star_WL_COMP = np.empty((nWLs10nm, nCOMPs + 1), dtype=np.float32)
            bb_star_WL_COMP = np.empty((nWLs10nm, nCOMPs + 1), dtype=np.float32)

            if 'Kd_par_MIM_VarName' in varList:
                Kd_par = np.zeros((nValid,), dtype=np.float32) + fv
            if 'Kd_490_MIM_VarName' in varList:
                Kd_490 = np.zeros((nValid,), dtype=np.float32) + fv
            if 'SD_MIM_VarName' in varList:
                SecchiDepth = np.zeros((nValid,), dtype=np.float32) + fv
            
            for PIX in range(nValid):
                conc_COMP = conc_COMP_EL[:,PIX]			# a vector
                SIOPindex = SIOPindex_EL[PIX]       #== SIOP set selected at this pixel
                a_star_10nm = siopSets10nm[siopSetNames[SIOPindex]]['a_star']		# for this pixel
                bb_star_10nm = siopSets10nm[siopSetNames[SIOPindex]]['bb_star']		# for this pixel
                # for j in range(nCOMPs + 1):					# create arrays of a_star & bb_star
                for ii,cc in enumerate(SIOPcomponents):
                    # COMP = cc   # SIOPcomponents[j]
                    a_star_WL_COMP[:,ii] = a_star_10nm[cc]
                    bb_star_WL_COMP[:,ii] = bb_star_10nm[cc]
                KdResults = calc_kd_SD(conc_COMP, a_star_WL_COMP, bb_star_WL_COMP, sun_zen_deg_ok[PIX], fv, verbose=False)
                if 'Kd_par_MIM_VarName' in varList:
                    Kd_par[PIX] = KdResults['kd_par']
                if 'Kd_490_MIM_VarName' in varList:
                    Kd_490[PIX] = KdResults['kd'][i490]
                if 'SD_MIM_VarName' in varList:
                    SecchiDepth[PIX] = KdResults['SD']

            if 'Kd_par_MIM_VarName' in varList:
                #== Kd_par_Output_LINE_PIX = reshapeResults(Kd_par, ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                Kd_par_Output_LINE_PIX = reshapeResults(Kd_par, ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                OUTPUT_X_ARRAY_DS[Kd_par_MIM_VarName][lineSlice,pixelSlice] = Kd_par_Output_LINE_PIX
            if 'Kd_490_MIM_VarName' in varList:
                #== Kd_490_Output_LINE_PIX = reshapeResults(Kd_490, ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                Kd_490_Output_LINE_PIX = reshapeResults(Kd_490, ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                OUTPUT_X_ARRAY_DS[Kd_490_MIM_VarName][lineSlice,pixelSlice] = Kd_490_Output_LINE_PIX
            if 'SD_MIM_VarName' in varList:
                #== SD_Output_LINE_PIX = reshapeResults(SecchiDepth, ok_index, (nELs,), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                SD_Output_LINE_PIX = reshapeResults(SecchiDepth, ok_index, (1, nELs), LINE_PIX_shape, fv, verbose=reshapeResultsVerbose)
                OUTPUT_X_ARRAY_DS[SD_MIM_VarName][lineSlice,pixelSlice] = SD_Output_LINE_PIX

        Once = False
    #== End: for chunkIndex in range(numberOfChunks)
    
    # CLOSE FILES - is this the best place to do this?

    # For diagnostic purposes:
    if inputArgs_verbosity>=2:
        print("\nPython variables:")
        for key in sorted(locals().keys()):
            vtype = eval('type(' + key + ')')
            if vtype is np.ndarray:
                print("variable '" + key + "' type is", vtype, "; shape is", eval(key + '.shape'), "; OWNDATA =", eval(key + '.flags["OWNDATA"]'))
            else:
                print("variable '" + key + "' type is", vtype)
    
    print(ME + " done")    
    
    # remove potential "spurious" time dimension:
    if 'time' in list(OUTPUT_X_ARRAY_DS.dims):
        OUTPUT_X_ARRAY_DS = OUTPUT_X_ARRAY_DS.isel(time=0).drop_vars('time').squeeze(drop=True)  
    
    return OUTPUT_X_ARRAY_DS
