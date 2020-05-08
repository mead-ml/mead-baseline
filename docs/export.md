DNNs can be deployed within a model serving framework such as __TensorFlow Serving__, a high-performance and highly scalable deployment architecture. All of the baseline implementations have exporters, though some customization may be required for user-defined models, for which we provide interfaces. The exporter transforms the model to include pre-processing and service-consumable output.

Currently we provide code to export and serve TensorFlow models when the backend is TensorFlow, and code to export ONNX models from PyTorch. 

## Exporting a model

To export a model, use [export.py](../python/mead/export.py). The following are the important arguments for the exporter:

- `--config`: configuration file for the trained model (see [mead](mead.md))
- `--task`: classify,taggger,seq2seq,lm (for which task the model was trained, see [baseline](baseline.md))
- `--model`: location of the saved model checkpoint.
- `--output_dir`: location of the directory where the exported model is saved.
- `--backend`: If you override this param from the config in `mead-train`, pass the same argument here
- `--is_remote`: If you set this to false, the model's vectorizers and meta-data files will be packaged in the same place as the exported model
Here is an example of running for a saved model from training.  Adjust for your model name:

```
mead-export --config config/sst2.json --model sst2-29858.zip

```

The last command will store the model at `./models/1` (note the version number). Please see `export.py` for more customization. 

## Serving a TensorFlow Model with TensorFlow Serving

TODO: Update this section, its quite old!

To serve the model you must run [Tensorflow Serving](https://github.com/tensorflow/serving).  

### TensorFlow Serving using bazel:

1. Clone serving:

```
git clone --recursive https://github.com/tensorflow/serving

```
2. Install bazel:

```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

sudo apt-get update && sudo apt-get install bazel
```
3. Serve model using bazel (CPU version):

build:

```
cd serving
bazel build //tensorflow_serving/model_servers:tensorflow_model_server
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8500 --model_name=test --model_config_file=/data/model-store/models.config
```
4. Serve model using bazel (GPU version):

Add this to `.bashrc`

```
export "TF_NEED_CUDA=1"
```
build:

```
bazel build -c opt --config=cuda --spawn_strategy=standalone //tensorflow_serving/model_servers:tensorflow_model_server
```

5. Start serving models:

```
bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8500 --model_config_file=models.config

```

A sample models.config file looks like this:

```
model_config_list: {
  config: {
    name: "age",
    base_path: "/data/model-store/age",
    model_platform: "tensorflow"
  }
}
```
### The Docker way

TODO: This is way out of date

You can alternately test your module by running a docker image pulled from [epigram ai](https://github.com/tensorflow/serving): `docker pull epigramai/model-server:light-1.5`. 


Once you have an exported model and the docker image pulled, you can load it in a docker container as follows:

```
docker run -it --name tf_classify -p 9000:9000 -v ./models:/models/ epigramai/model-server:light --port=9000 --model_name=tf_classify --model_base_path=/models/

```
## Exporting an ONNX model from PyTorch

To export a model to ONNX for a PyTorch model, use `mead-export` as usual.  If you want to run the model locally with the api-examples, use the `is_remote` parameter.

```
mead-export --config config/conll.json --model tagger-31405.zip --name conll-iobes --use_version false --zip true --is_remote False
```

The options above make tell `mead-export` to ignore versioning and generate a fully-contained `onnx` bundle in zip format.

### Testing out your ONNX model with api-examples

The API examples `classify-text` and `text-text` use the `baseline.Service` classes to load and run models that are either remote or local.
They can run `onnx` models if you pass `--backend onnx` to them.  To do this, you need the `onnxruntime` package installed.

You can install it manually like this:

```
pip install onnxruntime
```

Or you can install it as a dependency of mead like this
```
pip install mead-baseline[onnx]
```

To run the checkpoint we just exported:

```
(base) dpressel@dpressel-CORSAIR-ONE:~/dev/work/baseline/api-examples$ python tag-text.py --model ../mead/models/conll-iobes-11335.zip --text "Mr. Jones flew to New York ." --backend onnx
/tmp/43cd89440a7111ce9180faee69b1e75aac71c661/conll-iobes/tagger-model-31405.onnx
/tmp/43cd89440a7111ce9180faee69b1e75aac71c661/conll-iobes/vocabs-word-31405.json
/tmp/43cd89440a7111ce9180faee69b1e75aac71c661/conll-iobes/vocabs-char-31405.json
/tmp/43cd89440a7111ce9180faee69b1e75aac71c661/conll-iobes/vocabs-senna-31405.json
2020-05-08 13:45:41.131100225 [W:onnxruntime:, graph.cc:2413 CleanUnusedInitializers] Removing initializer 'embeddings.embeddings.2.proj.bias'. It is not used by any node and should be removed from the model.
2020-05-08 13:45:41.131122836 [W:onnxruntime:, graph.cc:2413 CleanUnusedInitializers] Removing initializer 'embeddings.embeddings.2.proj.weight'. It is not used by any node and should be removed from the model.
Mr. O
Jones S-PER
flew O
to O
New B-LOC
York E-LOC
. O

```

### MEAD Baseline PyTorch model ONNX Capabilities

The following MEAD Baseline models have been tested with ONNX

----------------------------------------------------------------
| Task       | PyTorch | Encoder        | Decoder | ORT Version |
|------------|---------|----------------|---------|-------------|
| Tagger     | 1.4     |  BiLSTM        | Greedy  |       1.2.0 |
| Tagger     | 1.4     |  BiLSTM        |    CRF  |       1.2.0 |
| Classifier | 1.4     |     CNN        |    N/A  |       1.2.0 |
| Classifier | 1.4     |    LSTM        |    N/A  |       1.2.0 |
| Classifier | 1.4     | Fine-Tune BERT |    N/A  |       1.2.0 |
----------------------------------------------------------------

## Paths

A few commandline arguments to `mead-export` can be used to control what the output paths look like.

The first is `--output_dir` This controls the root of the exported model. The argument `--project` creates a subdirectory for the model and `--name` can be used to create a second one. `--is_remote` is used to split the export into a `client` and `server` trees. Here are a few examples. When neither a `--project` nor as `--name` are given then a directory with the same name is the basename of the `--output_dir` is created. `--model_version` is the last directory.

 * `mead-export ... --output_dir path/to/models --is_remote true --model_version 3` -> `path/to/models/client/models/3` and `path/to/models/server/models/3`
 * `mead-export ... --output_dir path/to/models --is_remote false --model_version 19` -> `path/to/models/19`
 * `mead-export ... --output_dir path/to/models --project proj --model_version 2` -> `path/to/models/client/proj/2` and `path/to/models/server/proj/2`
 * `mead-export ... --output_dir path/to/models --project example --name bob --model_version 1 --is_remote false` -> `path/to/models/example/bob/1`

`--model_version` can be set to `None` and it will search in the export dir and use a model number one larger than the largest one currently in there.
