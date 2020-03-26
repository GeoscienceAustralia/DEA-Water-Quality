#! /bin/sh
# Test aLMI_main.py, with all 9 inputType/outputType combinations

TEST_CODE=`dirname $0`		# where test scripts are
SRC=$TEST_CODE/..		# where the aLMI code is

if [ $# -eq 0 ]; then
	echo "$0 INPUT_FILE CONFIG_FILE_TEMPLATE OUTPUT_DIR [OPTIONS]" >&2
	exit 1
fi

INPUT_FILE=$1
CONFIG_FILE_TEMPLATE=$2
OUTPUT_DIR=$3
shift 3
OPTIONS="$*"			# usually "-v", so log files show progress

LOG_FILE_DIR=./log_files_9_tests
[ -d $LOG_FILE_DIR ] || mkdir -p $LOG_FILE_DIR		# ensure directory exists

N_PASSED=0
N_FAILED=0

TITLE="Testing the aLMI python software (9_tests) on $HOSTNAME"
TIME_STAMP="Tests ran at $(date +"%x %r %Z") by $USER"
echo $TITLE
echo $TIME_STAMP
echo "*** SETUP ***"
echo "Input file was: $INPUT_FILE"
echo "Config file was: $CONFIG_FILE_TEMPLATE"
echo "Output dir was: $OUTPUT_DIR"
echo "*** Ready to go. Starting tests. ***"

for inputType in reflAboveSurf reflBelowSurf uIopRatio ; do
	for outputType in reflAboveSurf reflBelowSurf uIopRatio ; do
		TAG=${inputType}In_${outputType}Out
		CONFIG_FILE=$LOG_FILE_DIR/$TAG.ini
		cat $CONFIG_FILE_TEMPLATE | sed "s/^inputType *[:=] *\(.*\)/inputType = $inputType/" | sed "s/^outputType *[:=] *\(.*\)/outputType = $outputType/" > $CONFIG_FILE
		OUTPUT_FILE=$OUTPUT_DIR/$TAG.PyLMI.hdf

		$SRC/aLMI_main.py $INPUT_FILE $CONFIG_FILE $OUTPUT_FILE $OPTIONS > $LOG_FILE_DIR/$TAG.log
		
		if [[ $? != 0 ]]; then
			echo "---| FAILED $TAG"
			N_FAILED=`expr $N_FAILED + 1`		# count them
		else
			echo "+++| PASSED $TAG"
			N_PASSED=`expr $N_PASSED + 1`		# count them
		fi
		
		sleep 1
	done
done

### Fin.
echo "***"
echo "*** 9_tests complete; $N_PASSED passed, $N_FAILED failed ***"
echo "***"
echo "*** For individual log files see $LOG_FILE_DIR"
echo "***"

