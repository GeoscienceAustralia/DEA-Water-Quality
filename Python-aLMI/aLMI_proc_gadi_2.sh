#!/bin/bash
#PBS -P r78
#PBS -l walltime=15:00,mem=4GB,ncpus=1
#PBS -l wd
#PBS -l storage=gdata/r78+gdata/v10

######################################################################
#  PBS shell script to be executed on NCI facilities.                #
#  This script is used to run the aLMI Python code on some file of   #
#  reflectance data (.nc or .hdf), based on parameters defined       #
#  in a configuration (.ini) file.                                   #
######################################################################
# Notes:                                                             #
#  . the command 'module load dea' automatically loads up Python     #
#    3.6.7 on Gadi                                                   #
#  . the aLMI code requires the Python package 'pyhdf' which needs   #
#    to be manually installed on Gadi; the 'PYTHONPATH' line below   #
#    is then required to let Python know where the 'pyhdf' packge    #
#    was installed                                                   #
#  . submit this job to the queue on Gadi with:                      #
#      $> qsub aLMI_proc_gadi.sh                                     #
######################################################################


### USER INPUTS ######################################################
#-- Path to manually installed Python packages (e.g. pyhdf, netCDF4):
USER_PYTHON_PATH=~/.local/lib/python3.7/site-packages/

#-- Config file for this NCI job:
CONFIG_FILE=./ini_files/aLMI_config_VDItest_nc.ini

#-- Input file for this job: either define literally or get from the config file
# INPUT_DIR=/g/data/r78/aLMI/data/
# INPUT_FILE=${INPUT_DIR}A20120403_0410.20150928173107.L2OC_BASE.ANN_P134_V20140704.nc
INPUT_FILE=`./configUtils.py $CONFIG_FILE optionalParameters test_ANN_file`

#-- Output directory, where the aLMI data will be saved:
OUTPUT_DIR=/g/data/r78/aLMI/gadi_proc_out/

#-- Output file format (hdf or nc)
OUTPUT_FORMAT=nc
######################################################################



#### System definitions:
module use /g/data/v10/public/modules/modulefiles
module load dea
module load python3/3.7.4

expanded_dir=`cd $USER_PYTHON_PATH; pwd`
export PYTHONPATH=$expanded_dir:$PYTHONPATH


### Output file for this job (based on input file name):
input_ext=${INPUT_FILE##*.}
input_basename=$(basename "${INPUT_FILE}" ".${input_ext}")
OUTPUT_FILE=${OUTPUT_DIR}${input_basename}.PyALMI.$OUTPUT_FORMAT


### Execute aLMI code with input parameters:
python3.7 aLMI_main.py $INPUT_FILE $CONFIG_FILE $OUTPUT_FILE -v

### Var dump for output file:
if [ "$input_ext" == "hdf" ]; then
	hdp dumpsds -h $OUTPUT_FILE > ${OUTPUT_DIR}${input_basename}.PyALMI.dump
elif [ "$input_ext" == "nc" ]; then
	ncdump -h $OUTPUT_FILE > ${OUTPUT_DIR}${input_basename}.PyALMI.dump
fi

