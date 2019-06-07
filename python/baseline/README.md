# Overview

This is the core part of the library. The top-level module provides base classes for data loading and evaluation. The data loader reads common file formats for classification, CONLL-formatted files for sequence tagging, TSV and standard parallel corpora files for Neural Machine Translation and text files for language modeling. The data is masked and padded as necessary. It is also shuffled, sorted and batched such that data vectors in each batch have similar lengths. For sequence tagging problems, the loader supports multiple user-defined features. Also, the reader supports common formats for pre-trained embeddings. The library also supports common data cleaning procedures.

The top-level module also provides model base classes for four tasks: 
 
- [Text Classification](classify.md)
- [Tagging with RNNs](tagging.md)
- [Seq2Seq for Encoder-Decoder](seq2seq.md)
- [Language Modeling with RNNs](lm.md)

These are the most common NLP tasks and many common problems can be mapped to them ( NER, slot filling -> Sequence Tagging, translation -> encoder decoder). The lower-level modules provide at least one implementation for each task in TensorFlow, PyTorch and DyNet. These are well-known algorithms with strong results so that new algorithms can be compared against them. 

The library provides methods to calculate standard evaluation metrics including precision, recall, F1, average loss, and perplexity. It also provides high-level utility support for common architecture layers and paradigms such as attention, highway and skip connections. The default trainer supports multiple optimizers, early stopping, and various learning rate schedules.

Model architecture development is a common use-case for a researcher. The library is designed to make this process extremely easy. The user can build a model by overriding the create and load methods of a model base class and can run an experiment with the new model by passing the class name as an argument to the driver program. The data loading and training algorithm is decoupled from the model and can be overridden if necessary.

Driver programs are provided to train/test a model from the command line. In the following docs, we document how to use them, the implemented algorithms and the results in details. However, we recommend using [mead](mead.md) and [xpctl](xpctl.md) to run deep learning experiments.  


## Dependencies

The Baseline module has dependencies on:

- `tensorflow`, `pytorch` or `dynet`
- `numpy`
- `six`
- `requests`
- for visualization: `tensorboard_logger` and/or `visdom` are optional. To enable reporting with `visdom`/`tensorboard_logger`, just pass `--visdom 1`/`--tensorboard 1` in any command line program. 
- `PyYAML` is an optional dependency, which, if installed, allows [mead](mead.md) configurations to be provided with YAML instead of JSON
- When the GPU is used, the code assumes that `cudnn` is available and installed. This is critical for good performance.

## Saving the results

We provide different [reporting hooks](reporting.md) for displaying the results on console/ saving to a log file or database or visualizations. All reporting hooks can be run simultaneously.

## Running the code

The easiest way to train is using [mead](../python/mead/README.md). For running through mead, use the [trainer.py](../python/mead/trainer.py) utility in the directory:

```
python trainer.py --config config/conll.json --task tagger
```

See more running options in [trainer.py](../python/mead/trainer.py).

To run this code `baseline/python` should be available in your `PYTHONPATH` variable.

It is possible to use the Baseline models without mead as in this [example using tf estimators](../api-examples/tf-estimator.py).

## Installing Baseline as a Python Package

Baseline can be installed as a python package using the script [install_dev.sh](../python/install_dev.sh). To install, run the command: `./install.sh baseline`. Once installed, you can use the commands: `mead-train` and `mead-export` to  run the [trainer](../python/mead/trainer.py) or the [exporter](../python/mead/export.py) (with the same options as before) w/o putting baseline in PYTHONPATH. 

## Baseline as an API

The code provides a high-level Python API to access common deep-learning NLP approaches.  This should facilitate faster research in any language, as these tasks are fairly standard for NLP.  The data loaders and data feeds are all reusable, as are the basic harnesses for the APIs.  To get an understanding for how to structure a program, have a look at the [api-examples](../api-examples).

You can also think of the library itself as an abstraction layer at the "solution" or algorithm level with sub-modules built with each framework. Adding a new framework is straightforward using the methods shown in the library.

### As scaffolding for an experiment

If you have a problem where the input is the same as a `baseline` task, you can easily use the API to set up your boilerplate work for you, and focus on your model, by creating a user-defined `addon`.  This is just a normal python file with a creation and load hooks (see the [addons area](../python/addons) for examples). 

Then pass `--model_type {model}` to the driver program for that task.  The driver program will look to see if it has an implementation within the library and will not find the one in its registry.  So it will import the module and call its `create_model` function with the arguments and use the provided model.


### Training details

#### A note about losses and reporting

When tracking losses for reporting the average on the loss is undone and the total loss is tracked. At the end of the epoch this total loss is averaged over all of the examples seen. This allows for statistically correct reporting when the size of a batch is variable. NStep reporting can stride dev evaluations so it is possible for there to be a spike in nstep times if that step happens to have a dev set evaluation run during it.


#### Injecting inputs & avoiding placeholders in TensorFlow

The Baseline models are defined to support input inject, which bypasses placeholder creation.  This means that from custom user code, it is possible to directly inject the inputs. In the example below, we assume that embeddings is a dict containing the single key 'word', and we inject the input to that embedding via keyword arguments:

```
model = bl.model.create_model(embeddings, labels=params['labels'], word=word_source, y=y, **model_params)
```

This allows the user to access some of the advanced input capabilities in TensorFlow like `tf.dataset`s and `tf.Queue`s

#### TRAIN_FLAG() in TensorFlow backend

In the TensorFlow backend, we use a global method function `TRAIN_FLAG()` to determine if things like dropout should be applied.  If the user is running `mead` to train (which is the typical case), this flag is automatically defined as a `tf.placeholder` that will default `False` (meaning no dropout will be applied).

This is convenient because it is possible to use the same graph definition for training and evaluation by simply re-defining the placeholder to `True` when we are training.

However, in some cases, we may wish to define multiple graphs, in which case this default behavior may not be suitable.  A typical example of this would be if the user is running a Baseline model outside of `mead` and wants to define separate graphs depending on what mode is running.  In this case, the `TRAIN_FLAG()` may be better suited as a boolean.  To facilitate this, we provide a method which allows the user to override the value of the `TRAIN_FLAG()` on demand.  This makes it possible to do code such as:

```
bl.tf.SET_TRAIN_FLAG(True)
model = bl.model.create_model(embeddings, labels=params['labels'], word=features['word'], y=y, **model_params)
```

This is particularly helpful when using the `tf.estimator` API.  See [this example](../api-examples/tf-estimator.py) for details
