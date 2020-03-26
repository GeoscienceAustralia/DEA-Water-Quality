#! /usr/bin/env python3
"""

"""

import configparser
from collections import OrderedDict
import ast
import os
import sys

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
            if verbose:  print("WARNING:  " + msg)
            return False
        else:
            raise ValueError(msg)


# The '__main__' entry point.
if __name__ == '__main__':
    ME = sys.argv[0]
    if len(sys.argv) < 2:
        print("usage:  " + ME + " configFile [SECTION OPTION [VERBOSE]]")
        sys.exit(1)

    configFile = sys.argv[1]
    configVerbose = True
    if len(sys.argv) >= 4:
        get_option = True		# get the value of an option; print to stdout
        section = sys.argv[2]
        option = sys.argv[3]
        if len(sys.argv) == 4:  configVerbose = False		# can use to get option value
    else:
        get_option = False		# test many options used by aLMI_main

    if configVerbose:  print(ME + ":  configFile = " + configFile)
    configSet = configLoad(configFile, verbose=configVerbose)

    if get_option:		# just get the value of one option
        value = getConfigOption(configSet, section, option, optional=True, verbose=configVerbose)
        if value is False:		# not found
            sys.exit(1)
        else:
            if value:  print(value)		# if None, print nothing
            sys.exit(0)

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

    print(sys.argv[0] + " done")
