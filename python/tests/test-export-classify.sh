#!/usr/bin/env bash
RED='\033[0;31m'
GREEN='\033[0;32m'
END='\033[0m'
MSG_COLOR=${GREEN}
## check tf version

TF_VERSION_TEST=`python -c 'import tensorflow; from distutils.version import LooseVersion; import sys; i = "fail" if LooseVersion(tensorflow.__version__) < LooseVersion("1.12.0") else "pass"; print(i)'`

if [ "$TF_VERSION_TEST" == "fail" ]
    then
        printf "${RED}models were trained with tf version 1.12, you have a lower version. please upgrade.\n${END}"
        exit 1
    fi


BASELINE_DIR=`git rev-parse --show-toplevel`
CLASSIFY_MODEL=sst2-9494.zip
TEST_FILE=stsa.binary.test
CONFIG_FILE=sst2-comb.json
TEST_LOAD=${TEST_FILE}.load
TEST_SERVE=${TEST_FILE}.serve
TEST_SERVE_PREPROC=${TEST_FILE}.serve_preproc
EXPORT_DIR=${PWD}/models
EXPORT_DIR_PREPROC=${PWD}/models-preproc
MODEL_NAME=sst2
NUM_FEATURES=3
SLEEP=5

docker stop tfserving > /dev/null 2>&1
docker rm tfserving > /dev/null 2>&1
[ -e ${CLASSIFY_MODEL} ] && rm ${CLASSIFY_MODEL}
[ -e ${CONFIG_FILE} ] && rm ${CONFIG_FILE}
[ -e ${TEST_FILE} ] && rm ${TEST_FILE}
[ -e ${TEST_LOAD} ] && rm ${TEST_LOAD}
[ -e ${TEST_SERVE} ] && rm ${TEST_SERVE}
[ -e ${TEST_SERVE_PREPROC} ] && rm ${TEST_SERVE_PREPROC}
[ -e ${EXPORT_DIR} ] && rm -rf ${EXPORT_DIR}
[ -e ${EXPORT_DIR_PREPROC} ] && rm -rf ${EXPORT_DIR_PREPROC}

printf "${MSG_COLOR}running test for classify\n${END}"
printf "${MSG_COLOR}------------------------\n${END}"

printf "${MSG_COLOR} downloading trained classifier model, config and test file\n${END}"
wget https://www.dropbox.com/s/u1fbagrbi03kfvh/${CLASSIFY_MODEL}?dl=1 -O ${CLASSIFY_MODEL}
wget https://www.dropbox.com/s/zm6y79wczkaliat/${TEST_FILE}?dl=1 -O ${TEST_FILE}
wget https://www.dropbox.com/s/77m4xjpb6rpm2fx/${CONFIG_FILE}?dl=1 -O ${CONFIG_FILE}

printf "${MSG_COLOR}classifying by loading the model\n${END}"
python ${BASELINE_DIR}/api-examples/classify-text.py --model ${CLASSIFY_MODEL} --text ${TEST_FILE}  > ${TEST_LOAD}
sleep ${SLEEP}

printf "${MSG_COLOR}exporting w/o preproc\n${END}"

mkdir -p ${EXPORT_DIR}
mead-export --config ${CONFIG_FILE} --model ${CLASSIFY_MODEL} --is_remote false --output_dir ${EXPORT_DIR}/${MODEL_NAME}
sleep ${SLEEP}

printf "${MSG_COLOR}running tf serving\n${END}"
docker stop tfserving > /dev/null 2>&1
docker rm tfserving > /dev/null 2>&1
docker run -p 8501:8501 -p 8500:8500 --name tfserving -v ${EXPORT_DIR}:/models -e MODEL_NAME=${MODEL_NAME} -t tensorflow/serving &
sleep ${SLEEP}

printf "${MSG_COLOR}classifying with served model w/o preproc\n${END}"
python ${BASELINE_DIR}/api-examples/classify-text.py --model ${EXPORT_DIR}/${MODEL_NAME}/1/ --text ${TEST_FILE} --remote localhost:8500 --name ${MODEL_NAME} > ${TEST_SERVE}
sleep ${SLEEP}

printf "${MSG_COLOR}exporting with preproc\n${END}"
mkdir -p ${EXPORT_DIR_PREPROC}
mead-export --config ${CONFIG_FILE} --model ${CLASSIFY_MODEL} --is_remote false --exporter_type preproc --modules preproc_exporters preprocessors --output_dir ${EXPORT_DIR_PREPROC}/${MODEL_NAME}
sleep ${SLEEP}

printf "${MSG_COLOR}running tf serving\n${END}"
docker stop tfserving > /dev/null 2>&1
docker rm tfserving > /dev/null 2>&1
docker run -p 8501:8501 -p 8500:8500 --name tfserving -v ${EXPORT_DIR_PREPROC}:/models -e MODEL_NAME=${MODEL_NAME} -t tensorflow/serving &
sleep ${SLEEP}

printf "${MSG_COLOR}classifying with served model w preproc\n${END}"
python ${BASELINE_DIR}/api-examples/classify-text.py --model ${EXPORT_DIR_PREPROC}/${MODEL_NAME}/1/ --text ${TEST_FILE} --remote localhost:8500 --name ${MODEL_NAME} --preproc server > ${TEST_SERVE_PREPROC}

docker stop tfserving > /dev/null 2>&1
docker rm tfserving > /dev/null 2>&1

#remove prints coming from baseline
NUM_LINES_TO_REMOVE_LOAD=`expr "$NUM_FEATURES" + 2`
NUM_LINES_TO_REMOVE_SERVE=`expr "$NUM_FEATURES" + 1`
sed -i -e 1,${NUM_LINES_TO_REMOVE_LOAD}d ${TEST_LOAD}
sed -i -e 1,${NUM_LINES_TO_REMOVE_SERVE}d ${TEST_SERVE}
sed -i -e 1,${NUM_LINES_TO_REMOVE_SERVE}d ${TEST_SERVE_PREPROC}

DIFF=$(diff ${TEST_LOAD} ${TEST_SERVE})
    if [ "$DIFF" != "" ]
    then
        printf "${RED}${TEST_LOAD} does not match with ${TEST_SERVE}, exporting failed. \n${END}"
        exit 1
    fi

DIFF=$(diff ${TEST_SERVE} ${TEST_SERVE_PREPROC})
    if [ "$DIFF" != "" ]
    then
        printf "${RED}${TEST_SERVE} does not match with ${TEST_SERVE_PREPROC}, exporting failed. \n${END}"
        exit 1
    fi

printf "${GREEN}classify exporting successful. \n${END}"

rm ${CLASSIFY_MODEL}
rm ${TEST_FILE}
rm ${CONFIG_FILE}
rm -rf ${EXPORT_DIR}
rm -rf ${EXPORT_DIR_PREPROC}
rm ${TEST_LOAD}
rm ${TEST_SERVE}
rm ${TEST_SERVE_PREPROC}

exit 0
