#!/usr/bin/env bash

function get_file {
     if [ -f $1 ];
         then
         printf "${MSG_COLOR} $1 locally found, not downloading\n${END}"
     else
         printf "${MSG_COLOR} $1 locally not found, downloading $2\n${END}"
         wget $2 -O $1
     fi
}

eval $(sed -e 's/:[^:\/\/]/="/g;s/$/"/g;s/ *=/=/g' $1)

## check tf version

TF_VERSION_TEST=`python -c 'import tensorflow; from distutils.version import LooseVersion; import sys; i = "fail" if LooseVersion(tensorflow.__version__) < LooseVersion("1.12.0") else "pass"; print(i)'`

if [ "$TF_VERSION_TEST" == "fail" ]
    then
        printf "${ERROR_COLOR}models were trained with tf version 1.12, you have a lower version. please upgrade.\n${END}"
        exit 1
    fi



docker stop tfserving > /dev/null 2>&1
docker rm tfserving > /dev/null 2>&1
[ -e ${TEST_LOAD} ] && rm ${TEST_LOAD}
[ -e ${TEST_SERVE} ] && rm ${TEST_SERVE}
[ -e ${TEST_SERVE_PREPROC} ] && rm ${TEST_SERVE_PREPROC}
[ -e ${EXPORT_DIR} ] && rm -rf ${EXPORT_DIR}
[ -e ${EXPORT_DIR_PREPROC} ] && rm -rf ${EXPORT_DIR_PREPROC}

#remove prints coming from baseline
NUM_LINES_TO_REMOVE_LOAD=`expr "$NUM_FEATURES" + 3`
NUM_LINES_TO_REMOVE_SERVE=`expr "$NUM_FEATURES" + 2`

printf "${MSG_COLOR}running test for ${TASK}\n${END}"
printf "${MSG_COLOR}------------------------\n${END}"

get_file ${MODEL_FILE} ${MODEL_FILE_LINK}
get_file ${TEST_FILE} ${TEST_FILE_LINK}
get_file ${CONFIG_FILE} ${CONFIG_FILE_LINK}

printf "${MSG_COLOR}processing by loading the model\n${END}"
python ${DRIVER} --model ${MODEL_FILE} --text ${TEST_FILE}  > ${TEST_LOAD}
sleep ${SLEEP}

printf "${MSG_COLOR}exporting model with preproc=client\n${END}"

mkdir -p ${EXPORT_DIR}
mead-export --config ${CONFIG_FILE} --model ${MODEL_FILE} --is_remote false --output_dir ${EXPORT_DIR}/${MODEL_NAME}
sleep ${SLEEP}

printf "${MSG_COLOR}running tf serving\n${END}"
docker stop tfserving > /dev/null 2>&1
docker rm tfserving > /dev/null 2>&1
docker run -p 8501:8501 -p 8500:8500 --name tfserving -v ${EXPORT_DIR}:/models -e MODEL_NAME=${MODEL_NAME} -t tensorflow/serving &
sleep ${SLEEP}

printf "${MSG_COLOR}processing with served model, preproc=client\n${END}"
python ${DRIVER} --model ${EXPORT_DIR}/${MODEL_NAME}/1/ --text ${TEST_FILE} --remote localhost:8500 --name ${MODEL_NAME} > ${TEST_SERVE}
sleep ${SLEEP}

sed -i -e 1,${NUM_LINES_TO_REMOVE_LOAD}d ${TEST_LOAD}
sed -i -e 1,${NUM_LINES_TO_REMOVE_SERVE}d ${TEST_SERVE}

DIFF=$(diff ${TEST_LOAD} ${TEST_SERVE})
    if [ "$DIFF" != "" ]
    then
        printf "${ERROR_COLOR}${TEST_LOAD} does not match with ${TEST_SERVE}, exporting failed. \n${END}"
        exit 1
    fi


printf "${MSG_COLOR}exporting model with preproc=server\n${END}"
mkdir -p ${EXPORT_DIR_PREPROC}
mead-export --config ${CONFIG_FILE} --model ${MODEL_FILE} --is_remote false --exporter_type preproc --output_dir ${EXPORT_DIR_PREPROC}/${MODEL_NAME}
sleep ${SLEEP}

printf "${MSG_COLOR}running tf serving\n${END}"
docker stop tfserving > /dev/null 2>&1
docker rm tfserving > /dev/null 2>&1
docker run -p 8501:8501 -p 8500:8500 --name tfserving -v ${EXPORT_DIR_PREPROC}:/models -e MODEL_NAME=${MODEL_NAME} -t tensorflow/serving &
sleep ${SLEEP}

printf "${MSG_COLOR}processing with served model, preproc=server\n${END}"
python ${DRIVER} --model ${EXPORT_DIR_PREPROC}/${MODEL_NAME}/1/ --text ${TEST_FILE} --remote localhost:8500 --name ${MODEL_NAME} --preproc server > ${TEST_SERVE_PREPROC}

docker stop tfserving > /dev/null 2>&1
docker rm tfserving > /dev/null 2>&1


sed -i -e 1,${NUM_LINES_TO_REMOVE_SERVE}d ${TEST_SERVE_PREPROC}

DIFF=$(diff ${TEST_SERVE} ${TEST_SERVE_PREPROC})
    if [ "$DIFF" != "" ]
    then
        printf "${ERROR_COLOR}${TEST_SERVE} does not match with ${TEST_SERVE_PREPROC}, exporting failed. \n${END}"
        exit 1
    fi

printf "${MSG_COLOR}${TASK} export successful.\n${END}"

if [ "$CLEAN_AFTER_TEST" == "true" ]
then
    mead-clean
    rm ${MODEL_FILE}
    rm ${TEST_FILE}
    rm ${CONFIG_FILE}
    rm -rf ${EXPORT_DIR}
    rm -rf ${EXPORT_DIR_PREPROC}
    rm ${TEST_LOAD}
    rm ${TEST_SERVE}
    rm ${TEST_SERVE_PREPROC}
fi
exit 0
