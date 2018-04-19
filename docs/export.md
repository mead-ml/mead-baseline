DNNs can be deployed within a model serving framework such as \textbf{TensorFlow Serving}, a high-performance and highly scalable deployment architecture. All of the baseline implementations have exporters, though some customization may be required for user-defined models, for which we provide interfaces. The exporter transforms the model to include pre-processing and service-consumable output.

Currently we provide code to export and serve TensorFlow models. The saved model is a typical TensorFlow checkpoint: model graph and data (values for the variables).

## Exporting a model

To export a model, use [export.py](../python/mead/export.py). The following are the important arguments for the exporter:

- `--config`: configuration file for the trained model (see [mead](mead.md))
- `--task`: classify,taggger,seq2seq,lm (for which task the model was trained, see [baseline](baseline.md))
- `--model`: location of the saved model checkpoint.
- `--output_dir`: location of the directory where the exported model is saved.

Here is an example of running for a saved model with pid 27015.  Adjust for your model name:

```
python export.py --config config/sst2.json --model classify-model-tf-27015

```
The last command will store the model at `./models/1` (note the version number). Please see `export.py` for more customization. 

## Serving a model

To serve the model you must run [Tensorflow Serving](https://github.com/tensorflow/serving).  

### Serving the model using bazel:

1. Clone serving:

```
git clone --recursive https://github.com/tensorflow/serving

```
2. Install bazel:

```
sudo tee /etc/apt/sources.list.d/bazel.list
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
### The docker way

You can alternately test your module by running a docker image pulled from [epigram ai](https://github.com/tensorflow/serving): `docker pull epigramai/model-server:light-1.5`. 


Once you have an exported model and the docker image pulled, you can load it in a docker container as follows:

```
docker run -it --name tf_classify -p 9000:9000 -v ./models:/models/ epigramai/model-server:light --port=9000 --model_name=tf_classify --model_base_path=/models/

```  
