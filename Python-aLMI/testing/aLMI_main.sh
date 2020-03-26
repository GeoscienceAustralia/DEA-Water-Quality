#! /bin/sh
# Test aLMI_main.py #== + does profiling, and compares with IDL output; use aLMI_main.py directly for testing otherwise.
set -e
echo "aLMI_main.sh v1.0"

TEST_CODE=`dirname $0`		# where test scripts are
SRC=$TEST_CODE/..		# where the aLMI code is

#== PYTHONPATH=${PYTHONPATH}:$SRC:$SRC/io_layer
PYTHONPATH=${PYTHONPATH}:$SRC
export PYTHONPATH

if [ $# -lt 3 ]; then
	echo "$0 INPUT_FILE CONFIG_FILE OUTPUT_FILE [OPTIONS]" >&2
	exit 1
fi
INPUT_FILE=$1
CONFIG_FILE=$2
OUTPUT_FILE=$3		# suffix determines output file format
shift 3
OPTIONS="$*"

rm -f $OUTPUT_FILE					# make sure it doesn't exist
touch $OUTPUT_FILE.start			# to give start time
t0=`date '+%s'`

# aLMI_main.py $OPTIONS $INPUT_FILE $CONFIG_FILE $OUTPUT_FILE
#== time python -m cProfile -o $OUTPUT_FILE.prof $SRC/aLMI_main.py $OPTIONS $INPUT_FILE $CONFIG_FILE $OUTPUT_FILE
time python3 -m cProfile -o $OUTPUT_FILE.prof $SRC/aLMI_main.py $OPTIONS $INPUT_FILE $CONFIG_FILE $OUTPUT_FILE

STATUS=$? # always 0? nope... Rob got it to exit 1 by stuffing up the path to the siops in the config file.
# if [[ $STATUS != 0 ]]; then exit 1; fi
if [[ $STATUS != 0 ]]; then echo 'aLMI failed...'; exit 1; fi

t1=`date '+%s'`
echo "t0 = $t0, t1 = $t1, elapsed = `expr $t1 - $t0` seconds"

# Summarise the most time-consuming functions:
#== python $TEST_CODE/profile_stats.py $OUTPUT_FILE.prof
python3 $TEST_CODE/profile_stats.py $OUTPUT_FILE.prof

if [ -s $OUTPUT_FILE ]; then
	OUTPUT_TYPE=`expr "$OUTPUT_FILE" : ".*\.\(.*\)"`
	if [ "$OUTPUT_TYPE" = "hdf" ]; then
		hdp dumpsds -h $OUTPUT_FILE > $OUTPUT_FILE.dump
	elif [ "$OUTPUT_TYPE" = "nc" ]; then
		ncdump -h $OUTPUT_FILE > $OUTPUT_FILE.dump
	else
		echo "$0:  unexpected output type:  $OUTPUT_TYPE"
	fi
	echo

	#== OUT_DIR=`dirname $OUTPUT_FILE`
	#== STEM=`basename $INPUT_FILE .hdf`
	IDL_FILE=`$SRC/configUtils.py $CONFIG_FILE optionalParameters test_IDL_LMI_file`
	OUT_VAR_TYPE=`$SRC/configUtils.py $CONFIG_FILE outputParameters outputType`
	echo "Compare IDL and Python results (OUT_VAR_TYPE = '$OUT_VAR_TYPE'):"
	$TEST_CODE/compare_LMIs.py $IDL_FILE $OUTPUT_FILE $OUT_VAR_TYPE
else
	echo "$0:  aLMI failed; no dump or comparison"
    exit 1
fi

echo "$0:  done"
