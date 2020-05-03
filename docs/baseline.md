# Overview

Baseline models are built on a small NLP-specific layer API that is unified between PyTorch and TensorFlow, called [8 mile](../layers/eight_mile).
The 8 mile API offers layers built on the underlying API's primitives.  For example, using TensorFlow, 8 mile defines a set of layers built on `tf.keras.layers.Layer`.
The same API is implemented in PyTorch, but there, its built on top of `nn.Module`.  8 mile also defines metrics for experiments and many utilitieis that make it easy to write your own Deep Learning architectures.

Baseline applies the 8 mile layers to form a set of strong baseline models for various tasks, and also provides important utilities for loading data and handling each specific task.
it brings together several core capabilities that it uses to provide implementations of several key deep learning NLP tasks.
The data loader reads common file formats for classification, CONLL-formatted files for sequence tagging, TSV and standard parallel corpora files for Neural Machine Translation and text files for language modeling. The data is masked and padded as necessary. It is also shuffled, sorted and batched such that data vectors in each batch have similar lengths. For sequence tagging problems, the loader supports multiple user-defined features. Also, the reader supports common formats for pre-trained embeddings. The library also supports common data cleaning procedures.
The top-level module also provides model base classes for four tasks: 
 
- [Text Classification](classify.md)
- [Tagging with RNNs](tagging.md)
- [Seq2Seq for Encoder-Decoder](seq2seq.md)
- [Language Modeling with RNNs](lm.md)

These are the most common NLP tasks and many common problems can be mapped to them.
For example NER and slot filling are typically implemented as Sequence Tagging tasks.
Machine Translation is typically implemented using Encoder-Decoder models.

The task modules provide at least one implementation for each task in TensorFlow, PyTorch. 
These are well-known algorithms with strong results so that new algorithms can be compared against them. 

The library provides methods to calculate standard evaluation metrics including precision, recall, F1, average loss, and perplexity.
It also provides high-level utility support for common architecture layers and paradigms such as attention, highway and skip connections.
The default trainer supports multiple optimizers, early stopping, and various learning rate schedules.

Model architecture development is a common use-case for a researcher.
The library is designed to make this process extremely easy.
The user can build a model by overriding the methods of a model base class and can run an experiment with the new model by passing the class name as an argument to the driver program.
The data loading and training algorithm is decoupled from the model and can be overridden if necessary.

## Dependencies

The Baseline module has almost no dependencies. The requirements are `six` and `numpy`, and your choice of either `tensorflow` or `pytorch` as a deep learning backend.

- for visualization: `tensorboard_logger` and/or `visdom` are optional.
To enable reporting with `visdom`/`tensorboard_logger`, just pass `--visdom 1`/`--tensorboard 1` in any command line program. 
- `PyYAML` is an optional dependency, which, if installed, allows [mead](mead.md) configurations to be provided with YAML instead of JSON
- When the GPU is used, the code assumes that `cudnn` is available and installed. This is critical for good performance.

## Saving the results

We provide different [reporting hooks](reporting.md) for displaying the results on console/ saving to a log file or database or visualizations. All reporting hooks can be run simultaneously.

## Running the code

The easiest way to train is using [mead](mead.md)

```
mead-train --config config/conll.json
```


See more running options in [trainer.py](../mead/trainer.py).

It is possible to use the Baseline models without mead as in this [example using tf estimators](../api-examples/tf-estimator.py).

## Baseline as an API

The code provides a high-level Python API to access common deep-learning NLP approaches.
This should facilitate faster research in any language, as these tasks are fairly standard for NLP.
The data loaders and data feeds are all reusable, as are the basic harnesses for the APIs.
To get an understanding for how to structure a program, have a look at the [api-examples](../api-examples).

You can also think of the library itself as an abstraction layer at the "solution" or algorithm level with sub-modules built with each framework. Adding a new framework is straightforward using the methods shown in the library.

### As scaffolding for an experiment

If you have a problem where the input is the same as a `baseline` task, you can easily use the API to set up your boilerplate work for you, and focus on your model, by creating a user-defined `addon`.  This is just a sub-class of the bae task in the framework you wish to use, with a [@register_model annotation](addons.md) around it, identifying the name by which to reference the model from mead
There are many example addons on [mead-hub](https://github.com/mead-ml/hub/tree/master/v1/addons) and the [mead/config](../mead/config) directory contains examples for using addons

In the mead config, under the `model` block, you identify the `type` of model, which is a reference to the name you gave when registering your model.

You can find details of the API and implementation for each task under the task-specific documentation.

### Training details

#### A note about losses and reporting

When tracking losses for reporting the average on the loss is undone and the total loss is tracked. At the end of the epoch this total loss is averaged over all of the examples seen. This allows for statistically correct reporting when the size of a batch is variable. NStep reporting can stride dev evaluations so it is possible for there to be a spike in nstep times if that step happens to have a dev set evaluation run during it.
