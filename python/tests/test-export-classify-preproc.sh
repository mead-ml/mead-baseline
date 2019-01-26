BASELINE_DIR=${HOME}/dev/work/baseline
CLASSIFY_MODEL=sst2-25195.zip

[ -e $CLASSIFY_MODEL ] && rm $CLASSIFY_MODEL

echo "downloading trained classifier model and test file"
wget https://www.dropbox.com/s/orcrqkla4z0t5tk/sst2-25195.zip?dl=1 -O $CLASSIFY_MODEL

echo "exporting with preproc"
MDIR=models-preproc/sst2
rm -rf $MDIR
mkdir -p $MDIR
mead-export --config $BASELINE_DIR/python/mead/config/sst2.json --model $CLASSIFY_MODEL --is_remote false --exporter_type preproc --modules preproc-exporters preprocessors --output_dir $MDIR

