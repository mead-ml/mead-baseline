DNNs can be deployed within a model serving framework such as __TensorFlow Serving__, a high-performance and highly scalable deployment architecture. All of the baseline implementations have exporters, though some customization may be required for user-defined models, for which we provide interfaces. The exporter transforms the model to include pre-processing and service-consumable output.

Currently we provide code to export and serve TensorFlow models when the backend is TensorFlow, and code to export ONNX models from PyTorch. 

## Exporting a model

To export a model, use [export.py](../python/mead/export.py). The following are the important arguments for the exporter:

- `--config`: configuration file for the trained model (see [mead](mead.md))
- `--task`: classify,taggger,seq2seq,lm (for which task the model was trained, see [baseline](baseline.md))
- `--model`: location of the saved model checkpoint.
- `--output_dir`: location of the directory where the exported model is saved.
- `--backend`: If you override this param from the config in `mead-train`, pass the same argument here

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
mead-export --config config/conll.json --model tagger-29891.zip --is_remote false
```

### Testing out your ONNX model with api-examples

There are API examples that mirror their `*-text.py` counterparts, and are named `*-text-onnx.py*` for trying out your ONNX-exported models with `onnxruntime`.  To make this work, you need to first install that:

```
pip install onnxruntime
```

To run the checkpoint we just exported:

```
python tag-text-onnx.py --model ../mead/models/1/ --text "Mr. Jones went to Last Vegas ."
../mead/models/1/tagger-model-29891.labels
../mead/models/1/vocabs-char-29891.json
../mead/models/1/vocabs-senna-29891.json
../mead/models/1/vocabs-word-29891.json
2020-04-29 21:01:58.184187301 [W:onnxruntime:, graph.cc:2413 CleanUnusedInitializers] Removing initializer 'embeddings.embeddings.2.proj.bias'. It is not used by any node and should be removed from the model.
2020-04-29 21:01:58.184211421 [W:onnxruntime:, graph.cc:2413 CleanUnusedInitializers] Removing initializer 'embeddings.embeddings.2.proj.weight'. It is not used by any node and should be removed from the model.
Mr. B-PER
Jones E-PER
went O
to O
Last B-LOC
Vegas E-LOC
. O

```

## Paths

A few commandline arguments to `mead-export` can be used to control what the output paths look like.

The first is `--output_dir` This controls the root of the exported model. The argument `--project` creates a subdirectory for the model and `--name` can be used to create a second one. `--is_remote` is used to split the export into a `client` and `server` trees. Here are a few examples. When neither a `--project` nor as `--name` are given then a directory with the same name is the basename of the `--output_dir` is created. `--model_version` is the last directory.

 * `mead-export ... --output_dir path/to/models --is_remote true --model_version 3` -> `path/to/models/client/models/3` and `path/to/models/server/models/3`
 * `mead-export ... --output_dir path/to/models --is_remote false --model_version 19` -> `path/to/models/19`
 * `mead-export ... --output_dir path/to/models --project proj --model_version 2` -> `path/to/models/client/proj/2` and `path/to/models/server/proj/2`
 * `mead-export ... --output_dir path/to/models --project example --name bob --model_version 1 --is_remote false` -> `path/to/models/example/bob/1`

`--model_version` can be set to `None` and it will search in the export dir and use a model number one larger than the largest one currently in there.
