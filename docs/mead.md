## mead - Model Exploration And Development

This is a simple configuration-driven training/test system that facilitates exploration and development of model architectures for common tasks in NLP. It is built on [Baseline](baseline.md) and makes it easy to explore hyper-parameters and model architectures using Baseline`s extension capabilities. While its always possible to use Baseline as a library or call it from the task-specific drivers, its sometimes easier to work with a set of simple configuration files against a single driver that makes it easier to make variations without any code changes.  Additionally, when extension or new model architectures are required, its nice to be able to build them easily, plug then into an existing infrastructure that takes care of the boilerplate setup, and track them against a set of standard deep baselines.

We define a problem as a `<task, dataset>` tuple. To use mead, we simply set up a JSON configuration file that tells the driver what models, readers and trainers to run, with what parameters and HPs. For any task, the configuration file should contain 1. The dataset name, 2. Embedding type, 3. Reader type, 4. Model type, 5. Model hyper-parameters, 6. Training parameters (number of epochs, optimizers, optimizer specific parameters, patience for early stopping), 7. Pre-processing information. Reasonable default values are provided where possible. Thus, the whole experiment including hyper-parameters is uniquely identified by the sha1 hash of the configuration file. An experiment produces comprehensive logs including step-wise loss on the training data and task-specific metrics on the development and test sets. The reporting hooks support popular visualization frameworks including Tensorboard logging and Visdom. The model is persisted after each epoch, or, when early-stopping is enabled, whenever the model improves on the target development metric. The persisted model can be used to restart the training process or perform inference.

Here is a simple example for configuring the `default` model for SST2, with a TensorFlow backend for 2 epochs:


```
{
    "task": "classify",
    "batchsz": 50,
    "preproc": {
	"mxlen": 100,
	"rev": false,
	"clean": true
    },
    "backend": "tensorflow",
    "dataset": "SST2",
    "loader": {
	"reader_type": "default"
    },
    "unif": 0.25,
    "model": {
	"model_type": "default",
	"filtsz": [3,4,5],
	"cmotsz": 100,
	"dropout": 0.5,
	"finetune": true
    },
    "word_embeddings": {
	"label": "w2v-gn"
    },
    "train": {
	"epochs": 2,
	"optim": "adadelta",
	"eta": 1.0,
	"model_base": "./models/sst2",
	"early_stopping_metric": "acc"
    }
}

```

Here is a simple example of configuring `mead` to run a BLSTM-CRF in `pytorch` against the CONLL dataset:

```
{
    "task": "tagger",
    "batchsz": 10,
    "conll_output": "conllresults.conll",
    "charsz": 30,
    "unif": 0.1,
    "preproc": {
        "mxlen": -1,
        "mxwlen": -1,
        "lower": true
    },
    "backend": "pytorch",
    "dataset": "conll",
    "loader": {
        "reader_type": "default"
    },
    "model": {
        "model_type": "default",
        "cfiltsz": [3],
        "hsz": 200,
        "wsz": 30,
        "dropout": 0.5,
        "rnntype": "blstm",
        "layers": 2,
	"crf": 1
    },

    "word_embeddings": {
        "label": "glove-6B-100"
    },
    "train": {
        "epochs": 100,
        "optim": "sgd",
        "decay": 0,
        "eta": 0.015,
        "mom": 0.9,
        "patience": 40,
        "early_stopping_metric": "f1",
        "clip": 5.0
    }
}

```

### Training 

To train a model use the [trainer.py](../python/mead/trainer.py) utility in the directory:

```
python trainer.py --config config/conll.json
```

See more running options in [trainer.py](../python/mead/trainer.py).


### Dataset and Embeddings
You can provide your own dataset and embedding files in `mead` by changing the `datasets.json` or `embeddings.json`. We provide some standard ones, see [this doc](dataset-embedding.md) for details.

### Adding new models

Adding new models in mead is easy: under `addons/` add 1. model architecture, 2. trainer (if necessary) 3. data reader (if necessary). The model files should start with `task_`, eg, if you are adding a `tagger` model, it should be called `tagger_xxx.py`. Similarly, the reader files should start with `reader_`.  In the configuration file, change the `model_type` and/or `reader_type` to `xxx` from `default`. Add this directory to your `PYTHONPATH` and run the code as before i.e., `python trainer.py --config config/xxx.json` and Baseline will pick up the new model automatically.  

### Support for features

For `tagger`, Baseline supports multuple features in the CoNLL IOB file. For eg., you can include the POS chunks as:

```
Barrack B-NP B-PER
Obama B-NP I-PER
```

If this feature is called `pos`, your configuration file would change slightly:

```
"loader": {
        "reader_type": "default",
        "extended_features": {
            "pos": 1
        }
    },
```

`"pos":1` indicates that `pos` is the first feature in the CoNLL file. You can include multiple features. The last column in the ConLL file is considered the `IOB label`. For each feature, you *CAN* create an embedding file (in `Word2Vec` or `Glove` format). The embedding file is the mapping from the string feature to real values, eg: 

```
B-NP 010001
I-NP 012221
```
shows an embedding file `pos.txt` in `Word2Vec` format. Again, to use this, you would need to change the config a little:

```
    "word_embeddings": {"label": "glove-42B"},
    "extended_embed_info": {
        "pos": {"embedding":"/data/embeddings/pos.txt"}
    },

```

Creating an embedding file is not necessary, if you do not provide one we will initialize these features with random values. However, you must provide a dimension size in that case. Your config file will change to:

```
    "word_embeddings": {"label": "glove-42B"},
    "extended_embed_info": {
        "pos": {"dsz":10}
    },

```

An example of this is provided in gazetter model (one that includes gazetter information in NER tagging): [the code](../python/addons/tagger_gazetteer.py) and the [JSON file](../python/mead/config/wnut-gazetteer.json). Including gazetteers does improve the result on wnut17 dataset.
