#!/usr/bin/env bash

ERR_COLOR='\033[0;31m'
MSG_COLOR='\033[0;32m'
END='\033[0m'

function check_diff {
   DIFF=$(diff $1 $2)
   if [ "$DIFF" != "" ]
       then
           printf "${ERR_COLOR}[output from mead train]: ${OUTPUT_TRAIN}, [output from model load]: ${OUTPUT_LOAD} does not match \n${END}"
           exit 1
   fi
}

# run mead-train on a classify config. load the model file, run test on the same data, compare if they are equal
# need to make sure that clean=false in preproc section of classify config
printf "${MSG_COLOR}training model \n${END}"
BASELINE_DIR=${HOME}/dev/work/baseline
CONFIG_FILE=${BASELINE_DIR}/python/mead/config/sst2-clean-false.json
TEST_DATA=${BASELINE_DIR}/python/tests/test_data/stsa.binary.test.data
MODEL_PREFIX=sst2
OUTPUT_TRAIN=sst2.csv
OUTPUT_LOAD=sst2-load.csv
DRIVER=${BASELINE_DIR}/api-examples/classify-text.py

mead-clean
mead-train --config ${CONFIG_FILE} &
PID=$!
wait ${PID}
TRAINED_MODEL=${MODEL_PREFIX}-${PID}.zip
printf "${MSG_COLOR}classifying with trained model \n${END}"
python ${DRIVER} --model ${TRAINED_MODEL} --text ${TEST_DATA} > ${OUTPUT_LOAD}
check_diff ${OUTPUT_TRAIN} ${OUTPUT_LOAD}
printf "${MSG_COLOR}[output from mead train]: ${OUTPUT_TRAIN}, [output from model load]: ${OUTPUT_LOAD} match \n${END}"
mead-clean
rm ${TRAINED_MODEL}
rm ${OUTPUT_TRAIN}
rm ${OUTPUT_LOAD}
exit 0