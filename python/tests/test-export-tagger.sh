#!/usr/bin/env bash
BASELINE_DIR=${HOME}/dev/work/baseline
TAGGER_MODEL=tagger-24429.zip
TEST_FILE=eng.testb.small
TEST_LOAD=$TEST_FILE.load
TEST_SERVE=$TEST_FILE.serve
TEST_SERVE_PREPROC=$TEST_FILE.serve_preproc
NUM_FEATURES=3
SLEEP=5

docker stop tfserving
docker rm tfserving
[ -e $TAGGER_MODEL ] && rm $TAGGER_MODEL
[ -e $TEST_FILE ] && rm $TEST_FILE
[ -e $TEST_LOAD ] && rm $TEST_LOAD
[ -e $TEST_SERVE ] && rm $TEST_SERVE
[ -e $TEST_SERVE_PREPROC ] && rm $TEST_SERVE_PREPROC

echo "running test for tagger"
echo "------------------------"

echo "downloading trained tagger model and test file"
wget https://www.dropbox.com/s/86xzsofbgk3dzyj/tagger-24429.zip?dl=1 -O $TAGGER_MODEL
wget https://www.dropbox.com/s/pp3pcq5q5ko1df6/eng.testb.small?dl=1 -O $TEST_FILE

echo "tagging by loading the model"
python $BASELINE_DIR/api-examples/tag-text.py --model $TAGGER_MODEL --text $TEST_FILE  > $TEST_LOAD
sleep $SLEEP

echo "exporting w/o preproc"
MDIR=models/conll
rm -rf $MDIR
mkdir -p $MDIR
mead-export --config $BASELINE_DIR/python/mead/config/conll.json --model $TAGGER_MODEL --is_remote false --output_dir $MDIR
sleep $SLEEP

echo "running tf serving"
docker stop tfserving
docker rm tfserving
docker run -p 8501:8501 -p 8500:8500 --name tfserving -v ${PWD}/models:/models -e MODEL_NAME=conll -t tensorflow/serving &
sleep $SLEEP

echo "tagging with served model w/o preproc"
python $BASELINE_DIR/api-examples/tag-text.py --model ./models/conll/1/ --text $TEST_FILE --remote localhost:8500 --name conll > $TEST_SERVE

echo "exporting with preproc"
MDIR=models-preproc/conll
rm -rf $MDIR
mkdir -p $MDIR
mead-export --config $BASELINE_DIR/python/mead/config/conll.json --model $TAGGER_MODEL --is_remote false --output_dir $MDIR --modules preproc-exporters preprocessors --exporter_type preproc
sleep $SLEEP

echo "running tf serving"
docker stop tfserving
docker rm tfserving
docker run -p 8501:8501 -p 8500:8500 --name tfserving -v ${PWD}/models-preproc:/models -e MODEL_NAME=conll -t tensorflow/serving &
sleep $SLEEP

echo "tagging with served model w preproc"
python $BASELINE_DIR/api-examples/tag-text.py --model ./models-preproc/conll/1/ --text $TEST_FILE --remote localhost:8500 --name conll --preproc true > $TEST_SERVE_PREPROC

docker stop tfserving
docker rm tfserving

#remove prints coming from baseline
NUM_LINES_TO_REMOVE_LOAD=`expr "$NUM_FEATURES" + 2`
NUM_LINES_TO_REMOVE_SERVE=`expr "$NUM_FEATURES" + 1`
sed -i -e 1,${NUM_LINES_TO_REMOVE_LOAD}d $TEST_LOAD
sed -i -e 1,${NUM_LINES_TO_REMOVE_SERVE}d $TEST_SERVE
sed -i -e 1,${NUM_LINES_TO_REMOVE_SERVE}d $TEST_SERVE_PREPROC

DIFF=$(diff ${TEST_LOAD} ${TEST_SERVE})
    if [ "$DIFF" != "" ]
    then
        echo $TEST_LOAD does not match with $TEST_SERVE, exporting failed.
        exit 1
    fi

DIFF=$(diff ${TEST_SERVE} ${TEST_SERVE_PREPROC})
    if [ "$DIFF" != "" ]
    then
        echo $TEST_SERVE does not match with $TEST_SERVE_PREPROC, exporting failed.
        exit 1
    fi

echo "tagger exporting successful"
exit 0
