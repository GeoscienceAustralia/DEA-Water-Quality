#!/usr/bin/env bash
######################################################################
## DATE:         Jan 2015 -- updated by EL, March 2020
## AUTHOR:       CSIRO & BOM
## SCRIPT:       main_test.sh
## LANGUAGE:     bash
##
## USAGE: main_test.sh INPUT_FILE CONFIG_FILE OUTPUT_FILE [-s]
##        NOTE: At the moment -s is the only optional argument. It only 
##              runs the "unit tests" and skips testing of the aLMI_main 
##              code (faster, less compehensive test).
##
## DEPENDENCIES: modules environment set.
##
## PURPOSE:     To test each function of the aLMI software by running
##              a set of standard tests.
##
######################################################################
#set -e

# Test a module; may be a Python file or a shell script
RunTest()
{
	MODULE=`expr "$1" : "\(.*\)\..*"`
	SUFFIX=`expr "$1" : ".*\(\..*\)"`
	if [ "$SUFFIX" = ".sh" ]; then
		USE_FILE=$TEST_CODE/$1
	else
		USE_FILE=$SRC/$1
	fi
	shift 1			# the rest are args for the test
	
	$USE_FILE $* >& $LOG_FILE_DIR/$MODULE.log; rc=$?
	
	if [[ $rc != 0 ]]; then
	    echo "---| FAILED $MODULE"
	    N_FAILED=`expr $N_FAILED + 1`		# count them
	else
	    echo "+++| PASSED $MODULE"
	    N_PASSED=`expr $N_PASSED + 1`		# count them
	fi
}


if [ $# -lt 4 ]; then
	echo "$0 INPUT_FILE CONFIG_FILE OUTPUT_FILE VAR_PREFIX [-s]" >&2
	exit 1
fi
INPUT_FILE=$1
CONFIG_FILE=$2
OUTPUT_FILE=$3
VAR_PREFIX=$4
OPTION=$5

# Checking for any commandline arguments given
if [[ $OPTION == "-s" ]]; then
	echo "***"
	echo "*** Running the 'unit tests' only."
	echo "***"
else
	echo "***"
	echo "*** Running all tests, including aLMI_main."
	echo "***"
fi

echo "*** Setting up $0"

#== source /etc/profile # Useful for making sure that the module env is set but might not be needed?

#### Setting up and configuring the test environment.
TITLE="Testing the aLMI python software on $HOSTNAME"
TIME_STAMP="Tests ran at $(date +"%x %r %Z") by $USER"
HERE=`pwd`
TEST_CODE=`dirname $0`      # where test scripts are
SRC=$TEST_CODE/..       # where the aLMI code is
LOG_FILE_DIR=$HERE/log_files
[ -d $LOG_FILE_DIR ] || mkdir -p $LOG_FILE_DIR		# ensure directory exists

#== PYTHONPATH=${PYTHONPATH}:$SRC:$SRC/io_layer
PYTHONPATH=${PYTHONPATH}:$SRC
export PYTHONPATH

N_PASSED=0
N_FAILED=0

### A pre-amble for the main log file describing who what when and where the test was run.
echo $TITLE
echo $TIME_STAMP
echo "*** SETUP ***"
echo "Input file was: $INPUT_FILE"
echo "Config file was: $CONFIG_FILE"
echo "Output file was: $OUTPUT_FILE"
echo "*** Ready to go. Starting tests. ***"

### Starting to test the python modules.
RunTest RrsAboveToBelow.py
RunTest RrsBelowToAbove.py
RunTest calcCost.py
RunTest calcReflForward.py
RunTest calcUBackward.py
RunTest calcY.py
RunTest calc_mu_d.py
RunTest svd_LMI.py
RunTest var_dump.py

### Now test the modules that require command-line input(s):
RunTest configUtils.py $CONFIG_FILE
RunTest calc_kd_SD.py $CONFIG_FILE
RunTest calcUForward.py $CONFIG_FILE
RunTest SIOP_sets_load.py $CONFIG_FILE get_attrs
RunTest wavelengthsToVarNames.py $INPUT_FILE $VAR_PREFIX 412 685 550
RunTest chunkProcessLMI.py $CONFIG_FILE $VAR_PREFIX

### Now do the almi_main if the user didn't request the short run.
#== if [[ $SHORT_TEST==0 ]]; then
if [[ $OPTION != "-s" ]]; then
    echo "*** Testing the aLMI_main script - this will take a few minutes. ***"
    RunTest aLMI_main.py $INPUT_FILE $CONFIG_FILE $OUTPUT_FILE -v		# takes a few minutes
    #== RunTest aLMI_main.sh $INPUT_FILE $CONFIG_FILE $OUTPUT_FILE -v		# takes a few minutes; does profiling, and compares with IDL output
fi

### Fin.
echo "***"
echo "*** Tests complete; $N_PASSED passed, $N_FAILED failed ***"
echo "***"
echo "*** For individual log files see $LOG_FILE_DIR"
echo "***"



