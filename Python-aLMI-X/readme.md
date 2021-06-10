# Adaptive Linear Matrix Inversion in Python &ndash; using Xarray input/output

The code base provided in this repository has been written to be able to run the adaptive linear matrix inversion (aLMI) method on the National Computational Infrastructure (NCI) or Virtual Desktop Infrastructure (VDI) using Python 3. This is the version of the code that applies aLMI to an input `Xarray` dataset and stores the aLMI results into a new `Xarray` dataset.


&nbsp;
## Background

This Python code was originally cloned from a [CSIRO BitBucket repository](https://bitbucket.csiro.au/projects/CMAR_RS/repos/ereefs-lmi-python/), which was itself translated from an IDL version of the aLMI implementation (available from another [CSIRO BitBucket repo](https://bitbucket.csiro.au/projects/CMAR_RS/repos/ereefs-lmi-idl/)).

The original Python version of the aLMI code was subsequently converted from Python 2 to Python 3, debugged, tested on the NCI's Virtual Desktop Interface (VDI), and then ported and implemented as a batch job on NCI's Gadi infrastructure in March 2020. This version of the code is available on [this GitHub page](https://github.com/GeoscienceAustralia/DEA-Water-Quality/tree/master/Python-aLMI).

Finally, the code was updated to work with `Xarray`s instead of loading the data from a `.nc`/`.hdf` dataset and saving the results as a new dataset to disk. This update allowed for a significant simplification of the code base; in comparison with the earlier Python code, the current version has thus been updated to remove any part of the code related to unnecessary operations that are no longer in use in the latest implementation. 

This latest version of the Python code is provided in the current GitHub repository. The Python notebook labelled `Aquatic_ARD_aLMI` provides a demonstration of how to apply the Python3 aLMI code to an `Xarray` dataset of remote-sensing data. 

The aLMI algorithm implemented by this code is described in the following paper: https://doi.org/10.1364/AO.51.002808.



&nbsp;
## Preliminaries

The previous version of the Python aLMI code required the Python package `pyhdf` to be installed manually by the user. Given that the present version of the code works with `Xarray`s to implement data input and output, the code has been updated to remove the functionality of loading and saving data to `.hdf` files. Consequently, the `pyhdf` package is not required any longer and the users do not have to install this package and make it available on their Python paths.

In terms of dealing with `.nc` files (used to store the SIOP datasets), the `netCDF4` Python package is already available on the NCI / VDI as part of the Python version loaded up by the DEA module. A separate install of this package is thus not required by the user either, and `netCDF4` will be simply loaded when executing the usual DEA "module-load" commands:

```bash
module use /g/data/v10/public/modules/modulefiles
module load dea
```



&nbsp;
## Repository structure

All the Python files necessary for the execution of the aLMI code on Gadi or the VDI are available from the current folder in this repository. A number of ancillary files can also be found in the following sub-directories.

### Subfolder 'ini_files'

This directory provides some examples of configuration files (`.ini` files), which represent an input to the aLMI code. The `.ini` files contained in this subfolder were used during the initial testing phase of the code on the VDI.

### Subfolder 'data'

This folder contains some examples of SIOP sets data files (`.nc` format), used during the testing of aLMI with input data from various sensors (MODIS and Sentinel-2).



&nbsp;
## Running the aLMI code

The Python notebook `Aquatic_ARD_aLMI` provides a demonstration of how to apply the Python3 aLMI code to an `Xarray` dataset of remote-sensing data. In essence, the aLMI function simply requires as input: 1) an `Xarray` dataset of RS data, and 2) the path to a configuration `.ini` file (see further below):

```python
from aLMI_main import aLIM_main
CONFIG_FILE = "./path/to/aLMI_config.ini"
output_XarrayDS = aLMI_main(input_XarrayDS, CONFIG_FILE)
```

where `input_XarrayDS` is the `Xarray` Dataset of input data. The code is currently implemented to only deal with a single time slice of input data. If `input_XarrayDS` contains more than one time slice, the aLMI execution will exit with an error message. A further condition on `input_XarrayDS` is that it must contain all the bands specified by the user via the configuration variable `inputSpectrumVarNames` within the input `.ini` file (see next section).



&nbsp;
## Configuration (.ini) file structure

One of the main inputs to `aLMI_main.py` is the configuration `.ini` file of aLMI input parameters. This section provides a comprehensive description of the various parameters that can be used and set up in this type of file. Additionally, the example `.ini` files located in the `ini_files` directory can be investigated to find some typical values for these parameters.

A configuration `.ini` file for aLMI typically contains a number of sections, each containing a number of fields, as follows.

* Section `[inputParameters]`:
	- `SIOP_SETS_FILE`: path to the file of SIOP sets (`.nc` file)  
	_Example:_ SIOP_SETS_FILE = /g/data/r78/aLMI/data/siops_MODIS_all_CLT4.nc
	- `inputSpectrumVarNames`: list of strings representing the names of the reflectance bands to use by aLMI.  
	**Note**: these must match the band names in the input dataset to aLMI.  
	_Example:_ inputSpectrumVarNames = \["Rrs_412" , "Rrs_443" , "Rrs_488"\]
	- `useWavelengths`: vector of wavelengths (in nm) for which to extract the SIOPs from the SIOP sets file (likely corresponding wavelengths to the reflectance bands in  `inputSpectrumVarNames`).  
	**Note**: this vector must have the same number of elements as `inputSpectrumVarNames`.  
	_Example:_ useWavelengths = \[411.5, 441.5, 486.5\]
	- `tolerance`: tolerance (in nm) to use around the stated wavelength values in `useWavelengths` when extracting the SIOP sets (i.e., if found, the extracted SIOP sets' wavelengths will be within +/- `tolerance` of the desired value)  
	_Example:_ tolerance = 5.0
	- `components`: (list of strings) constituent concentrations that will be calculated by aLMI  
	_Example:_ components = \["CHL", "CDOM", "NAP"\]
	- `chunkProcessConfig`: Python dictionary-like object of configuration parameters, containing namely:
		+ `costType`: the selected aLMI cost function type (one of `RMSE`, `RMSRE`, or `RMSE_LOG`)
		+ `minValidConc`: minimum valid concentration, for all constituents (pixels whose concentrations are less than `minValidConc` are deemed to be invalid)
		+ `outputFillValue`: defines the 'no-data' fill value (used for most output arrays)
		+ `costThreshhold`: upper cost limit for pixels to be accepted during the aLMI procedure (pixels must have a cost < `costThreshhold`); this is equivalent to the `D_R_THRESHOLD` variable in the IDL code  
	_Example:_   
	chunkProcessConfig = {'costType':'RMSE', 'minValidConc':0.00001, 'outputFillValue':-999., 'costThreshhold':100.}
	- `inputType`: type of data provided in the input dataset (one of `reflAboveSurf`, `reflBelowSurf`, `uIopRatio`)  
	_Example:_ inputType = reflAboveSurf
	- `SolarZenithBandName`: name of the band in the input dataset representing the solar zenith angle values  
	_Example:_ SolarZenithBandName = oa_solar_zenith  
	- `numberOfLinesPerChunk`: processing of the input data occurs in chunks to lower the memory requirements; this parameter corresponds to the number of lines of input data to process in each chunk  
	_Example:_ numberOfLinesPerChunk = 500  
	- `verbosity`: level of verbosity of the aLMI messages printed during execution of the code; relevant values are 0, 1 and 2, representing increasing levels of verbosity  
	_Example:_ verbosity = 1  

* Section `[outputParameters]`:
	- `outWavelengthLabels`: vector of labels (strings) to be used in the labelling of the output variables.  
	**Note**: this vector must have the same number of elements as `inputSpectrumVarNames`, and the values must be within `tolerance` of `useWavelengths`  (corresponding array values).  
	_Example:_ outWavelengthLabels =   \[ "412", "441", "488" \]  
	- `outputType`: type of data to be saved into the output dataset (one of `reflAboveSurf`, `reflBelowSurf`, `uIopRatio` or `""`; with the latter option, `outputType` will be set to the same type as `inputType`)  
	_Example:_ outputType = ""  

* Section `[optionalParameters]`:
	- `SIOP_SETS_10nm_FILE`: path to a file of SIOP sets (`.nc` file) with 10nm-spaced wavelengths to use if any of `Kd_par_MIM`, `Kd_490_MIM`, or `SD_MIM` is specified (see below).  
	**Note:** calculation of Kd and Secchi depth estimates (i.e. using any of these three options) also requires the input dataset of reflectances to contain the solar zenith angle in a variable labelled as per the `SolarZenithBandName` parameter.  
	_Example:_ SIOP_SETS_10nm_FILE = /g/data/r78/aLMI/data/siops_10nm_all_CLT4.nc
	- `Kd_WavelengthsRange`: start and end of the range of wavelengths (in nm) to extract from the 10nm SIOP dataset (`SIOP_SETS_10nm_FILE`), to be used for the calculation of Kd and Secchi depth estimates  
	_Example:_ Kd_WavelengthsRange = \[420, 750\]
	**Note:** if the `Kd_490_MIM` option is used (see below), `Kd_WavelengthsRange` must contain the value 490 within it (increments of 10nm from the start of the range).  
	- `Kd_par_MIM`: flag to enable the calculation of Kd_par (will be saved in the aLMI output dataset). Simply omit this parameter to disable this feature.
	- `Kd_490_MIM`: flag to enable the calculation of Kd_490 (will be saved in the aLMI output dataset). Simply omit this parameter to disable this feature.
	- `SD_MIM`: flag to enable the calculation of Secchi depth (will be saved in the aLMI output dataset). Simply omit this parameter to disable this feature.
	- `a_wavelength`: selected wavelength (in nm) to use for the calculation of absorption coefficients (see below)  
	_Example:_ a_wavelength = 441
	- `a_phy_MIM`: (this parameter requires `a_wavelength` to be defined) flag to enable the calculation of a_phy (will be saved in the aLMI output dataset). Simply omit this parameter to disable this feature.
	- `a_CDOM_MIM`: (this parameter requires `a_wavelength` to be defined) flag to enable the calculation of a_CDOM (will be saved in the aLMI output dataset). Simply omit this parameter to disable this feature.
	- `a_NAP_MIM`: (this parameter requires `a_wavelength` to be defined) flag to enable the calculation of a_NAP (will be saved in the aLMI output dataset). Simply omit this parameter to disable this feature.
	- `a_P_MIM`: (this parameter requires `a_wavelength` to be defined) flag to enable the calculation of a_P (will be saved in the aLMI output dataset). Simply omit this parameter to disable this feature.
	- `a_CDM_MIM`: (this parameter requires `a_wavelength` to be defined) flag to enable the calculation of a_CDM (will be saved in the aLMI output dataset). Simply omit this parameter to disable this feature.
	- `a_tot_MIM`: (this parameter requires `a_wavelength` to be defined) flag to enable the calculation of a_tot (will be saved in the aLMI output dataset). Simply omit this parameter to disable this feature.

	- `bb_wavelength`: selected wavelength (in nm) to use for the calculation of backscatter coefficients (see below)  
	_Example:_ bb_wavelength = 551
	- `bb_phy_MIM`: (this parameter requires `bb_wavelength` to be defined) flag to enable the calculation of bb_phy (will be saved in the aLMI output dataset). Simply omit this parameter to disable this feature.
	- `bb_NAP_MIM`: (this parameter requires `bb_wavelength` to be defined) flag to enable the calculation of bb_NAP (will be saved in the aLMI output dataset). Simply omit this parameter to disable this feature.
	- `bb_P_MIM`: (this parameter requires `bb_wavelength` to be defined) flag to enable the calculation of bb_P (will be saved in the aLMI output dataset). Simply omit this parameter to disable this feature.

* Section `[ggParameters]`:
	- `g0`: definition of the absorption/backscatter model constant g<sub>0</sub>  
	_Example:_ g0 = 0.084
	- `g1`: definition of the absorption/backscatter model constant g<sub>1</sub>  
	_Example:_ g1 = 0.17



&nbsp; 
## 

**Author:** Eric A. Lehmann, CSIRO Data61  
**Date**: June 2021.
