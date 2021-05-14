# Adaptive Linear Matrix Inversion in Python

The code base provided in this repository has been written to be able to run the adaptive linear matrix inversion (aLMI) method on the National Computational Infrastructure (NCI) using Python 3.

This Python code was originally cloned from a [CSIRO BitBucket repository](https://bitbucket.csiro.au/projects/CMAR_RS/repos/ereefs-lmi-python/), which was itself translated from an IDL version of the aLMI implementation (available from another [CSIRO BitBucket repo](https://bitbucket.csiro.au/projects/CMAR_RS/repos/ereefs-lmi-idl/)).

The original Python version of the aLMI code was subsequently converted from Python 2 to Python 3, debugged, tested on the NCI's Virtual Desktop Interface (VDI), and finally ported and implemented as a batch job on NCI's Gadi infrastructure in March 2020.

The aLMI algorithm implemented by this code is described in the following paper: https://doi.org/10.1364/AO.51.002808.



&nbsp;
## Repository structure

All the Python files necessary for the execution of the aLMI code on Gadi are available from the current folder in this repository. 

A number of ancillary files can also be found in the following sub-directories.

### Subfolder 'testing'

This directory contains a number of Python files and shell scripts that were used during the initial testing of the Python code on the VDI. It also contains a Python notebook (`code_testing.ipynb`) demonstrating the results from these VDI tests.

### Subfolder 'ini_files'

This directory provides some examples of configuration files (`.ini` files), which represent one of the inputs to the aLMI code. The `.ini` files contained in this subfolder were used during the initial testing phase on the VDI.



&nbsp;
## Preliminaries

The aLMI code provided here requires the Python package `pyhdf` and `netCDF4`, which are not installed by default on the NCI (Gadi). These packages thus require a manual install by the user, which can be performed as follows. 

### Python package 'pyhdf'

In a new terminal on Gadi, type in the following commands:

```bash
module load python3/3.7.4
module load hdf4/4.2.14
pip3 install -v --no-binary :all: --prefix=~/.local pyhdf    # use the relevant prefix path here!
```

This will install the `pyhdf` package in the user's home directory (`~/.local`), though a different path can also be used if desired.

### Python package 'netCDF4'

In a new terminal on Gadi (or re-using the previous one), type in the following commands:

```bash
module load python3/3.7.4
module load netcdf/4.7.3
module load hdf5/1.10.5
pip3 install -v --no-binary :all: --prefix=~/.local netCDF4    # use the relevant prefix path here!
```

### After the new packages are installed

Once the `pyhdf` and `netCDF4` packages are installed, Python needs to be able to locate the relevant package files during execution of the aLMI code. This can be achieved by executing the following command prior to running the aLMI code:

```bash
export PYTHONPATH=~/.local/lib/python3.7/site-packages/:$PYTHONPATH    # use the relevant path here!
```

Here again, the user should double-check that the path in the above command is indeed the correct path to use on their respective system. The above `PYTHONPATH` command is already implemented within the example NCI job scripts provided in this repository (see next section).



&nbsp;
## Running the aLMI code on Gadi

The main Python aLMI function is `aLMI_main.py`, which calls upon a series of subroutines provided in the various other `.py` files. The main aLMI routine expects three main input parameters:
1. the input file of remote sensing (RS) reflectance data, which can be a NetCDF (`.nc`) or HDF (`.hdf`) file
1. a configuration (`.ini`) file of input parameters for the aLMI run, which, among others, contains the path to the SIOP sets file (`.nc` file) to use (see next section); and 
1. the path and name of the output file (`.nc` or `.hdf` file) where the aLMI results will be saved.

The shell scripts `aLMI_proc_gadi_N.sh` provide examples of how to execute the aLMI processing on Gadi as a batch job. The first shell script (`aLMI_proc_gadi_1.sh`) provides an example of how to process a HDF file of input data, while the second shell script (`aLMI_proc_gadi_2.sh`) can be used to process NetCDF files.

To execute such aLMI batch jobs, the user can first edit the `.sh` scripts and enter / update the following information:

* PBS directives for the current job:
	- `#PBS -P ...`: NCI project to use
	- `#PBS -l walltime=HH:MM:SS`: walltime required to process the data
	- `#PBS -l mem=...GB`: memory required for this job
	- `#PBS -l ncpus=...`: number of CPUs (parallel threads) required for this job
	- `#PBS -l storage=...`: file systems that the job will need to access during execution
* Section of USER INPUTS:
	- `USER_PYTHON_PATH`: path to the manually installed Python packages (`pyhdf`, `netCDF4`); see above notes under 'Preliminaries'
	- `CONFIG_FILE`: path to the configuration file for this job (`.ini` file)
	- `INPUT_FILE`: path to the file of input data (`.nc` or `.hdf` file)
	- `OUTPUT_DIR`: output directory where the aLMI data will be saved
	- `OUTPUT_FORMAT`: output file format for the saved dataset (`.nc` or `.hdf` file).

Once the job's `.sh` script has been edited as desired, the job can then be submitted to the PBS queue on Gadi, as follows:

```bash
qsub aLMI_proc_gadi_1.sh
```

The processing will then occur, writing the aLMI processing messages to the job's log file, and saving the aLMI output data to the selected output directory (in the specified file format). In addition, the shell script will also generate a `.dump` file of the aLMI file's contents in the same output directory.


&nbsp;
## Configuration file structure

One of the main inputs to `aLMI_main.py` is the configuration `.ini` file of aLMI input parameters. This section provides a comprehensive description of the various parameters that can be used and set up in this type of file. Additionally, the example `.ini` files located in the `ini_files` directory can be investigated to find some typical values for these parameters.

A configuration `.ini` file typically contains a number of sections, each containing a number of fields, as follows.

* Section `[inputParameters]`:
	- `SIOP_SETS_FILE`: path to the file of SIOP sets (`.nc` file)  
	_Example:_ SIOP_SETS_FILE = /g/data/r78/aLMI/data/siops_MODIS_all_CLT4.nc
	- `inputSpectrumVarNames`: list of strings representing the names of the reflectance bands to use by aLMI.  
	**Note**: these must match the band names in the input data file to aLMI.  
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
	- `inputType`: type of data provided in the input file (one of `reflAboveSurf`, `reflBelowSurf`, `uIopRatio`)  
	_Example:_ inputType = reflAboveSurf
	- `numberOfLinesPerChunk`: processing of the input data occurs in chunks to lower the memory requirements; this parameter corresponds to the number of lines of input data to process in each chunk  
	_Example:_ numberOfLinesPerChunk = 500  

* Section `[outputParameters]`:
	- `outWavelengthLabels`: vector of labels (strings) to be used in the labelling of the output variables.  
	**Note**: this vector must have the same number of elements as `inputSpectrumVarNames`, and the values must be within `tolerance` of `useWavelengths`.  
	_Example:_ outWavelengthLabels =   \[ "412", "441", "488" \]  
	- `outputType`: type of data to be written to the output file (one of `reflAboveSurf`, `reflBelowSurf`, `uIopRatio` or `""`; with the latter option, `outputType` will be set to the same type as `inputType`)  
	_Example:_ outputType = ""  

* Section `[optionalParameters]`:
	- `test_data`: (only used for testing) path to some test dataset (can be used in conjunction with the next two parameters)  
	_Example:_ test_data = /g/data/r78/aLMI/data
	- `test_ANN_file`: (only used for testing) path to a file (`.nc` or `.hdf`) of RS reflectance data that can be used as input to aLMI  
	_Example:_  
	test_ANN_file = %(test_data)s/A20120403_0410.20130805213444.L2.ANN_P134_V20140704.hdf
	- `test_IDL_LMI_file`: (only used for testing) path to a file of IDL-based aLMI outputs, to be used in a comparison with the Python-based aLMI outputs (after processing the data in `test_ANN_file`)  
	_Example:_  
	test_IDL_LMI_file = %(test_data)s/A20120403_0410.20130805213444.L2.ANN_P134_V20140704.MIM_CLT4_gLee_412_748.hdf
	- `copyInputSpectrum`: flag to indicate that the input data spectrum is to be copied from the input reflectance file into the output aLMI file. Simply omit this parameter to disable this feature.
	- `SIOP_SETS_10nm_FILE`: path to a file of SIOP sets (`.nc` file) with 10nm-spaced wavelengths to use if any of `Kd_par_MIM`, `Kd_490_MIM`, or `SD_MIM` is specified (see below).  
	**Note:** calculation of Kd and Secchi depth estimates (i.e. using any of these three options) also requires the input file of reflectances to contain the solar zenith angle in a variable labelled `zen`.  
	_Example:_ SIOP_SETS_10nm_FILE = /g/data/r78/aLMI/data/siops_10nm_all_CLT4.nc
	- `Kd_WavelengthsRange`: start and end of the range of wavelengths (in nm) to extract from the 10nm SIOP dataset (`SIOP_SETS_10nm_FILE`), to be used for the calculation of Kd and Secchi depth estimates  
	_Example:_ Kd_WavelengthsRange = \[420, 750\]
	- `Kd_par_MIM`: flag to enable the calculation of Kd_par (will be saved in the aLMI output file). Simply omit this parameter to disable this feature.
	- `Kd_490_MIM`: flag to enable the calculation of Kd_490 (will be saved in the aLMI output file). Simply omit this parameter to disable this feature.
	- `SD_MIM`: flag to enable the calculation of Secchi depth (will be saved in the aLMI output file). Simply omit this parameter to disable this feature.
	- `a_wavelength`: selected wavelength (in nm) to use for the calculation of absorption coefficients (see below)  
	_Example:_ a_wavelength = 441
	- `a_phy_MIM`: (this parameter requires `a_wavelength` to be defined) flag to enable the calculation of a_phy (will be saved in the aLMI output file). Simply omit this parameter to disable this feature.
	- `a_CDOM_MIM`: (this parameter requires `a_wavelength` to be defined) flag to enable the calculation of a_CDOM (will be saved in the aLMI output file). Simply omit this parameter to disable this feature.
	- `a_NAP_MIM`: (this parameter requires `a_wavelength` to be defined) flag to enable the calculation of a_NAP (will be saved in the aLMI output file). Simply omit this parameter to disable this feature.
	- `a_P_MIM`: (this parameter requires `a_wavelength` to be defined) flag to enable the calculation of a_P (will be saved in the aLMI output file). Simply omit this parameter to disable this feature.
	- `a_CDM_MIM`: (this parameter requires `a_wavelength` to be defined) flag to enable the calculation of a_CDM (will be saved in the aLMI output file). Simply omit this parameter to disable this feature.
	- `a_tot_MIM`: (this parameter requires `a_wavelength` to be defined) flag to enable the calculation of a_tot (will be saved in the aLMI output file). Simply omit this parameter to disable this feature.

	- `bb_wavelength`: selected wavelength (in nm) to use for the calculation of backscatter coefficients (see below)  
	_Example:_ bb_wavelength = 551
	- `bb_phy_MIM`: (this parameter requires `bb_wavelength` to be defined) flag to enable the calculation of bb_phy (will be saved in the aLMI output file). Simply omit this parameter to disable this feature.
	- `bb_NAP_MIM`: (this parameter requires `bb_wavelength` to be defined) flag to enable the calculation of bb_NAP (will be saved in the aLMI output file). Simply omit this parameter to disable this feature.
	- `bb_P_MIM`: (this parameter requires `bb_wavelength` to be defined) flag to enable the calculation of bb_P (will be saved in the aLMI output file). Simply omit this parameter to disable this feature.

* Section `[ggParameters]`:
	- `g0`: definition of the absorption/backscatter model constant g<sub>0</sub>  
	_Example:_ g0 = 0.084
	- `g1`: definition of the absorption/backscatter model constant g<sub>1</sub>  
	_Example:_ g1 = 0.17


&nbsp;
## A note on selecting resources for Gadi jobs

The selection of walltime and memory resources for PBS jobs to be executed on Gadi (see PBS directives `PBS -l mem=...` and `PBS -l walltime=...` above, under 'Running the aLMI code on Gadi') obviously depends on the characteristics and size of the input dataset to process (total number of pixels, number of 'no-data' values, etc.). Some trial and error is typically needed at first in order to select optimal walltime and memory values for a given input dataset.

One aspect of Gadi, however, is that the cost (to the selected NCI project, in Service Units) of a batch job execution depends the largest contribution from either: a) the amount of CPU time **required** by the job (i.e. number of CPUs multiplied by walltime), or b) the **requested** amount of memory (proportionally to the total amount of memory available on each node). The previous NCI system (Raijin) only accounted for the CPU time in the cost calculations.

This means that on Gadi, it is important to avoid grossly over-estimating the MEM requirements for a given job. Similarly, it is most economical to keep the CPU and MEM requirements consistent and balanced. 

For instance, given that the above aLMI code is not currently parallelised, it is only worth submitting a job with `PBS -l ncpus=1`. Also, each node on Gadi has 48 CPUs and a total of 192GB of RAM available. Thus, selecting `PBS -l mem=4GB` for some aLMI job will be on a par with the CPU requirements, as the CPU equivalent of this MEM requirement is 48 x 4GB / 192GB = 1 CPU. Selecting any MEM value larger than 4GB (for a single-threaded job) will thus increase the execution cost to the selected project. For instance, 8GB of MEM would achieve the same cost as if 2 CPUs were selected for this job (though the above code would be unable to actually take advantage of having more than one CPU). 

For illustration, another (hypothetical) example is as follows. If a given (parallelised) NCI job is submitted with a request for 10 CPUs, the user could also request up to 40GB of RAM for the job without incurring any extra cost, since 48 x 40GB / 192GB = 10.

Of course, the user should ultimately select a MEM amount that is large enough to process a given input dataset (or face the inevitable prospect of the code crashing into a heap of cryptic error messages!). Keep in mind, however, that the MEM requirements can be altered by means of the `numberOfLinesPerChunk` parameter above: a smaller `numberOfLinesPerChunk` leads to a lower MEM requirement.


&nbsp; 
## 

**Author:** Eric A. Lehmann, CSIRO Data61  
**Date**: March 2020.
