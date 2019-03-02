#!/usr/bin/env bash

function msg_print {
    printf "${MSG_COLOR}$1\n${END}"
}

function err_print {
    printf "${ERROR_COLOR}$1\n${END}"
}

function tf_version_test {
    TF_VERSION_TEST=`python -c 'import tensorflow; from distutils.version import LooseVersion; import sys; i = "fail" if LooseVersion(tensorflow.__version__) < LooseVersion("1.12.0") else "pass"; print(i)'`

    if [ "$TF_VERSION_TEST" == "fail" ]
        then
            err_print "models were trained with tf version 1.12, you have a lower version. please upgrade."
            exit 1
    fi
}

function clean {
    if [ "$CLEAN_AFTER_TEST" == "true" ]
    then
        mead-clean
        remove_files "${FILES_TO_REMOVE[@]}"
    fi
    exit 0
}

function docker_clear {
    docker rm -f ${SERVING_CONTAINER_NAME} > /dev/null 2>&1
}

function docker_run {
    docker run -p ${REMOTE_PORT}:${REMOTE_PORT} --name ${SERVING_CONTAINER_NAME} -v $1 -e MODEL_NAME=${MODEL_NAME} -t tensorflow/serving &
}

function get_file {
     if [ -f $1 ];
         then
         msg_print " $1 locally found, not downloading"
     else
         msg_print " $1 locally not found, downloading $2"
         wget $2 -O $1
     fi
}

function mead_export {
    mead-export --config ${CONFIG_FILE} --settings ${EXPORT_SETTINGS_MEAD} --datasets ${EXPORT_SETTINGS_DATASETS} --task ${TASK} --exporter_type ${1} --model ${MODEL_FILE} --model_version ${MODEL_VERSION} --output_dir $2 --is_remote ${IS_REMOTE} --return_labels ${3}
}

function check_diff {
    DIFF=$(diff ${1} ${2})
    if [ "$DIFF" != "" ]
    then
        err_print "${1} does not match with ${2}, exporting failed. "
        docker_clear
        exit 1
    fi
}

function remove_files {
    arr=("$@")
    for file in "${arr[@]}"
        do
            [ -e "${file}" ] && rm -rf "${file}"
        done
}

function classify_text {
    if [ -z "$2" ]
    then
        python ${DRIVER} --model $1 --text ${TEST_FILE} --name ${MODEL_NAME} > $4
    else
        python ${DRIVER} --model $1 --text ${TEST_FILE} --remote ${2} --name ${MODEL_NAME} --preproc $3 > $4
    fi
}


function tag_text {
    if [ -z "$4" ]
    then
        python ${DRIVER} --model $1 --text ${TEST_FILE} --conll $2 --features $3 --name ${MODEL_NAME} > $7
    elif [ -z "$6" ]
    then
        python ${DRIVER} --model $1 --text ${TEST_FILE} --conll $2 --features $3 --remote ${4} --name ${MODEL_NAME} --preproc $5 > $7
    else
        python ${DRIVER} --model $1 --text ${TEST_FILE} --conll $2 --features $3 --remote ${4} --name ${MODEL_NAME} --preproc $5 --exporter_field_feature_map $6 > $7
    fi
}


## get the variables defined in the config into shell
eval $(sed -e 's/:[^:\/\/]/="/;s/$/"/g;s/ *=/=/g' $1)
## check tf version
tf_version_test
docker_clear

## remove files from previous run, if any
FILES_TO_REMOVE=(${TEST_LOAD} ${TEST_SERVE} ${TEST_SERVE_PREPROC} ${EXPORT_DIR} ${EXPORT_DIR_PREPROC})
remove_files "${FILES_TO_REMOVE[@]}"

msg_print "running test for ${TASK}"
msg_print "------------------------"

## get the files
get_file ${MODEL_FILE} ${MODEL_FILE_LINK}
get_file ${TEST_FILE} ${TEST_FILE_LINK}
get_file ${CONFIG_FILE} ${CONFIG_FILE_LINK}

## load model and process data
msg_print "processing by loading the model"
case ${TASK} in
    classify)
        classify_text ${MODEL_FILE} "" client ${TEST_LOAD} # remote end point is empty, preproc is client
        ;;
    tagger)
        tag_text ${MODEL_FILE} ${CONLL} "${FEATURES}" "" client "" ${TEST_LOAD}  # remote end point is empty, preproc is client
        ;;
    *)
        err_print "Unsupported task"
        exit 1
        ;;
esac
sleep ${SLEEP}

## export with preproc=client and process data
msg_print "exporting model with preproc=client"
mkdir -p ${EXPORT_DIR}
mead_export default ${EXPORT_DIR}/${MODEL_NAME} ${RETURN_LABELS}
sleep ${SLEEP}
msg_print "running tf serving"
docker_clear
docker_run ${EXPORT_DIR}:/models
sleep ${SLEEP}
msg_print "processing with served model, preproc=client"
case ${TASK} in
    classify)
        classify_text ${EXPORT_DIR}/${MODEL_NAME}/1/ ${REMOTE_HOST}:${REMOTE_PORT} client ${TEST_SERVE} # valid remote end points, preproc is client.
        ;;
    tagger)
        tag_text ${EXPORT_DIR}/${MODEL_NAME}/1/ ${CONLL} "${FEATURES}" ${REMOTE_HOST}:${REMOTE_PORT} client "" ${TEST_SERVE}
        ;;
    *)
        err_print "Unsupported task"
        exit 1
        ;;
esac
sleep ${SLEEP}
## remove first few lines and check if the outputs match
sed -i -e 1,${NUM_LINES_TO_REMOVE_LOAD}d ${TEST_LOAD}
sed -i -e 1,${NUM_LINES_TO_REMOVE_SERVE}d ${TEST_SERVE}
check_diff ${TEST_LOAD} ${TEST_SERVE}

## export with preproc=server and process data
msg_print "exporting model with preproc=server"
mkdir -p ${EXPORT_DIR_PREPROC}
mead_export preproc ${EXPORT_DIR_PREPROC}/${MODEL_NAME} ${RETURN_LABELS}
sleep ${SLEEP}
msg_print "running tf serving"
docker_clear
docker_run ${EXPORT_DIR_PREPROC}:/models
msg_print "processing with served model, preproc=server"
case ${TASK} in
    classify)
        classify_text ${EXPORT_DIR_PREPROC}/${MODEL_NAME}/1/ ${REMOTE_HOST}:${REMOTE_PORT} server ${TEST_SERVE_PREPROC} # valid remote end points, preproc is server.
        ;;
    tagger)
        tag_text ${EXPORT_DIR_PREPROC}/${MODEL_NAME}/1/ ${CONLL} "${FEATURES}" ${REMOTE_HOST}:${REMOTE_PORT} server "${EXPORTER_FIELD_FEATURE_MAP}" ${TEST_SERVE_PREPROC}

        ;;
    *)
        err_print "Unsupported task"
        exit 1
        ;;

esac
docker_clear
## remove first few lines and check if the outputs match
sed -i -e 1,${NUM_LINES_TO_REMOVE_SERVE}d ${TEST_SERVE_PREPROC}
check_diff ${TEST_SERVE} ${TEST_SERVE_PREPROC}
msg_print "${TASK} export successful."

## if successful, clean the files
clean