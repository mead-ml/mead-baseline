baseline
=========

Simple, Strong Deep-Learning Baselines for NLP in several frameworks

Baseline algorithms and data support implemented with multiple deep learning tools, including sentence classification, tagging, seq2seq, and language modeling.  Can be used as stand-alone command line tools or as a Python library.  The library attempts to provide a common interface for several common deep learning tasks, as well as easy-to-use file loaders to make it easy to publish standard results, compare against strong baselines without concern for mistakes and to support rapid experiments to try and beat these baselines.

# Overview

A few strong, deep baseline algorithms for several common NLP tasks, including sentence classification, tagging, sequence-to-sequence and language modeling problems.  Considerations are conceptual simplicity, efficiency and accuracy.  The included approaches are (hopefully) the first "deep learning thing" you might think of for a certain type of problem, and are currently near SoTA for many datasets.  Below you can find descriptions of the algorithms, and the status of implementation for each framework.

When the GPU is used, the code assumes that cudnn is available* and installed. This is critical for good performance.

## Supported Tasks

- [Text Classification](docs/classify.md)
- [Tagging with RNNs](docs/tagging.md)
- [Seq2Seq](docs/seq2seq.md)
- [Language Modeling with RNNs](docs/lm.md)

## Reporting with Visdom and tensorboard

### Visdom

First install it:

`pip install visdom`

To enable reporting with visdom, just pass `--visdom 1` in any command line program.

### Tensorboard

First install tensorboard logger (which is independent of tensorflow):

`pip install tensorboard_logger`

To enable reporting with tensorboard, just pass `--tensorboard 1` in any command line program.
You must have tensorboard installed to use this.

## Baseline as an API

The latest code provides a high-level Python API to access common deep-learning NLP approaches.  This should facilitate faster research in any language, as these tasks are fairly standard for NLP.  The data loaders and data feeds are all reusable, as are the basic harnesses for the APIs.  To get an understanding for how to structure a program to use baseline, have a look at the command line programs for each task.

You can also think of the library itself as an abstraction layer at the "solution" or algorithm level with sub-modules built with each framework. Adding a new framework is straightforward using the methods shown in the library.

### As scaffolding for an experiment

If you have a problem where the input is the same as a `baseline` task, you can easily use the API to set up your boilerplate work for you, and focus on your model, by creating a user-defined `addon`.  This is just a normal python file with a creation and load hooks (see the [addons area](python/addons) for examples). 

Then pass `--model_type {model}` to the driver program for that task.  The driver program will look to see if it has an implementation within the library and will not find the one in its registry.  So it will import the module and call its `create_model` function with the arguments and use the provided model.


## Running from configuration files, using `mead`

We provide a single driver to train all of the tasks from a simple JSON configuration file as part of [mead](docs/mead.md).  This makes it easy to explore model architetures, track your experiments and deploy models to production easily.  Sample configurations are provided for the tasks

