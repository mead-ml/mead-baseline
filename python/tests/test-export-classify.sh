BASELINE_DIR=${HOME}/dev/work/baseline
CLASSIFY_MODEL=sst2-25195.zip
TEST_FILE=stsa.binary.test
TEST_LOAD=$TEST_FILE.load
TEST_SERVE=$TEST_FILE.serve
TEST_SERVE_PREPROC=$TEST_FILE.serve_preproc
SLEEP=5

docker stop tfserving
docker rm tfserving
[ -e $CLASSIFY_MODEL ] && rm $CLASSIFY_MODEL
[ -e $TEST_FILE ] && rm $TEST_FILE
[ -e $TEST_LOAD ] && rm $TEST_LOAD
[ -e $TEST_SERVE ] && rm $TEST_SERVE
[ -e $TEST_SERVE_PREPROC ] && rm $TEST_SERVE_PREPROC

echo "running test for classify"
echo "------------------------"

echo "downloading trained classifier model and test file"
wget https://www.dropbox.com/s/orcrqkla4z0t5tk/sst2-25195.zip?dl=1 -O $CLASSIFY_MODEL
wget https://www.dropbox.com/s/zm6y79wczkaliat/stsa.binary.test?dl=1 -O $TEST_FILE

echo "classifying by loading the model"
python $BASELINE_DIR/api-examples/classify-text.py --model $CLASSIFY_MODEL --text $TEST_FILE  > $TEST_LOAD
sleep $SLEEP

echo "exporting w/o preproc"
MDIR=models/sst2
rm -rf $MDIR
mkdir -p $MDIR
echo $CLASSIFY_MODEL
mead-export --config $BASELINE_DIR/python/mead/config/sst2.json --model $CLASSIFY_MODEL --is_remote false --output_dir $MDIR
sleep $SLEEP

echo "running tf serving"
docker stop tfserving
docker rm tfserving
docker run -p 8501:8501 -p 8500:8500 --name tfserving -v ${PWD}/models:/models -e MODEL_NAME=sst2 -t tensorflow/serving &
sleep $SLEEP

echo "classifying with served model w/o preproc"
python $BASELINE_DIR/api-examples/classify-text.py --model ./models/sst2/1/ --text $TEST_FILE --remote localhost:8500 --name sst2 > $TEST_SERVE

echo "exporting with preproc"
MDIR=models-preproc/sst2
rm -rf $MDIR
mkdir -p $MDIR
mead-export --config $BASELINE_DIR/python/mead/config/sst2.json --model $CLASSIFY_MODEL --is_remote false --output_dir $MDIR
sleep $SLEEP

echo "running tf serving"
docker stop tfserving
docker rm tfserving
docker run -p 8501:8501 -p 8500:8500 --name tfserving -v ${PWD}/models-preproc:/models -e MODEL_NAME=sst2 -t tensorflow/serving &
sleep $SLEEP

echo "classifying with served model w preproc"
python $BASELINE_DIR/api-examples/classify-text.py --model ./models-preproc/sst2/1/ --text $TEST_FILE --remote localhost:8500 --name sst2 > $TEST_SERVE_PREPROC

docker stop tfserving
docker rm tfserving

#remove prints coming from baseline
sed -i -e 1,4d $TEST_LOAD 
sed -i -e 1,3d $TEST_SERVE 
sed -i -e 1,3d $TEST_SERVE_PREPROC

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

echo "classify exporting successful"
exit 0

