## mead - Model Exploration an Development

This is a simple configuration-driven training system that facilitates exploration and development of model architectures for common tasks in NLP. It is built on `baseline` and makes it easy to explore hyper-parameters and model architectures using `baseline`s extension capabilities. While its always possible to use `baseline` as a library or call it from the task-specific drivers, its sometimes easier to work with a set of simple configuration files against a single driver that makes it easier to make variations without any code changes.  Additionally, when extension or new model architectures are required, its nice to be able to build them easily, plug then into an existing infrastructure that takes care of the boilerplate setup, and track them against a set of standard deep baselines.

To use mead, we simply set up a JSON configuration file that tells the driver what models, readers and trainers to run, with what parameters and HPs.  Here is a simple example for configuring the `default` model for SST2, with a TensorFlow backend for 2 epochs:

```
{
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
    "batchsz": 10,
    "conll_output": "conllresults.conll",
    "test_thresh": 10,
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

