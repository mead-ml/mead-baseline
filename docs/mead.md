## mead - Model Exploration And Development

This is a simple configuration-driven training/test system that facilitates exploration and development of model architectures for common tasks in NLP. It is built on [Baseline](baseline.md) and makes it easy to explore hyper-parameters and model architectures using Baseline`s extension capabilities.
While its always possible to use Baseline as a library or call it from the task-specific drivers, its sometimes easier to work with a set of simple configuration files against a single driver that makes it easier to make variations without any code changes.
Additionally, when extension or new model architectures are required, its nice to be able to build them easily, plug then into an existing infrastructure that takes care of the boilerplate setup, and track them against a set of standard deep baselines.

We define a problem as a `<task, dataset>` tuple.
To use mead, we simply set up a JSON configuration file that tells the driver what models, readers and trainers to run, with what parameters and HPs. For any task, the configuration file should contain 1. The dataset name, 2. Embedding type, 3. Reader type, 4. Model type, 5. Model hyper-parameters, 6. Training parameters (number of epochs, optimizers, optimizer specific parameters, patience for early stopping), 7. Pre-processing information. Reasonable default values are provided where possible. Thus, the whole experiment including hyper-parameters is uniquely identified by the sha1 hash of the configuration file. An experiment produces comprehensive logs including step-wise loss on the training data and task-specific metrics on the development and test sets. The reporting hooks [support popular visualization](reporting.md) frameworks including Tensorboard logging and [visdom](https://github.com/facebookresearch/visdom). The model is persisted after each epoch, or, when early-stopping is enabled, whenever the model improves on the target development metric. The persisted model can be used to restart the training process or perform inference.

Here is a simple example for configuring the `default` model for SST2, with a TensorFlow backend for 2 epochs:


```

  "version": 2,
  "task": "classify",
  "basedir": "./sst2",
  "batchsz": 50,
  "features": [
    {
      "name": "word",
      "vectorizer": {
        "type": "token1d",
	"transform": "baseline.lowercase"
      },
      "embeddings": {
        "label": "w2v-gn"
      }
    }
  ],
  "preproc": {
    "rev": false,
    "clean": true
  },
  "backend": "tensorflow",
  "dataset": "SST2",
  "reader": {
    "type": "default"
  },
  "unif": 0.25,
  "model": {
    "type": "default",
    "filtsz": [
      3,
      4,
      5
    ],
    "cmotsz": 100,
    "dropout": 0.5,
    "finetune": true
  },
  "train": {
    "epochs": 2,
    "optim": "adadelta",
    "eta": 1.0,
    "model_zip": true,
    "early_stopping_metric": "acc",
    "verbose": {
      "console": true,
      "file": "sst2-cm.csv"
    }
  }
}

```

Here is an example of configuring `mead` to run a BLSTM-CRF in `pytorch` against the CONLL dataset:

```
{
 
  "task": "tagger",
  "conll_output": "conllresults.conll",
  "unif": 0.1,
  "features": [
    {
      "name": "word",
      "vectorizer": {
        "type": "dict1d",
        "fields": "text",
        "transform": "baseline.lowercase"
      },
      "embeddings": {
        "label": "glove-6B-100"
      }
    },
    {
      "name": "senna",
      "vectorizer": {
        "type": "dict1d",
        "fields": "text",
        "transform": "baseline.lowercase"
      },
      "embeddings": {
        "label": "senna"
      }
    },
    {
      "name": "char",
      "vectorizer": {
        "type": "dict2d"
      },
      "embeddings": { "dsz": 30, "wsz": 30, "type": "char-conv" }
    }
  ],
  "backend": "pytorch",
  "dataset": "conll-iobes",
  "reader": {
    "type": "default",
    "named_fields": {
      "0": "text",
      "-1": "y"
    }
  },
  "model": {
    "type": "default",
    "cfiltsz": [
      3
    ],
    "hsz": 400,
    "dropout": 0.5,
    "dropin": {"word": 0.1,"senna": 0.1},
    "rnntype": "blstm",
    "layers": 1,
    "constrain_decode": true,
    "crf": 1
  },
  "train": {
    "batchsz": 10,
    "epochs": 100,
    "optim": "sgd",
    "eta": 0.015,
    "mom": 0.9,
    "patience": 40,
    "early_stopping_metric": "f1",
    "clip": 5.0,
    "span_type": "iobes"
  }
}

```

### Training 

To train a model use the [mead-train](../mead/trainer.py) utility in the directory:

```
mead-train --config config/conll.json
```

To change the backend without having to change the configuration, we can pass the `--backend` argument (assuming that backend is installed)

```
mead-train --config config/sst2.json --backend pytorch
mead-train --config config/conll.json --backend tf
```


### Dataset and Embeddings

You can provide your own dataset and embedding files in `mead` by changing the `datasets.json` or `embeddings.json`.
For more information about how we handle datasets, embeddings and featurization, please read [this document](dataset-embedding.md).

### Support for features

In the NER example above, we saw the use of multiple features simultaneously to produce a single word representation.
All of the features used in that example were built right off the surface `text` feature in the loader.
For the `tagger` loader, Baseline supports multiple feature columns in the CONLL files.
For example, you can include the POS features from the second column:

```
Barack B-NP B-PER
Obama E-NP E-PER
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

`"pos":1` indicates that `pos` is the first feature in the CoNLL file. You can include multiple features.
The last column in the ConLL file is considered the `IOB label`.
To use this custom feature you would want to give it a name and reference some embeddings (either pretrained, or randomly initialized):

```
   {
      "name": "pos",
      "vectorizer": {
        "type": "dict1d",
        "fields": "pos"
      },
      "embeddings": {
        "dsz": 300,
        "type": "default"
      }
    },
```

### Exporting

Once you have a model trained with `mead-train`, you might want to deploy it into production.
The [mead-export](export.md) tool provides an extensible interface and a driver program to deploy your models.