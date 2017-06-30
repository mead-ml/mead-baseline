baseline
=========

Simple, Strong Deep-Learning Baselines for NLP in several frameworks

Baseline algorithms and data support implemented with multiple deep learning tools, including CNN sentence modeling, RNN/LSTM-based tagging, seq2seq, and language modeling.  Can be used as stand-alone command line tools or as a Python library.  The library attempts to provide a common interface for several common deep learning tasks, as well as easy-to-use file loaders to make it easy to publish standard results, compare against strong baselines without concern for mistakes and to support rapid experiments to try and beat these baselines.


**Update 6/30/2017: The code has been refactored to support reusing components and use as a python module.  If you used to run code from the `<task>/python/<framework>` areas, you can now run those tasks from `baseline/python`.  See usage in each task [document](docs)**

# Overview

A few strong, deep baseline algorithms for several common NLP tasks,
including sentence classification, tagging, sequence-to-sequence and language modeling problems.  Considerations are conceptual simplicity, efficiency and accuracy.  This is intended as a simple to understand reference for building strong baselines with deep learning.

After considering other strong, shallow baselines, we have found that even incredibly simple, moderately deep models often perform better.  These models are only slightly more complex to implement than strong shallow baselines such as shingled SVMs and NBSVMs, and support multi-class output easily.  Additionally, they are (hopefully) the first "deep learning thing" you might think of for a certain type of problem.  Using these stronger baselines as a reference point hopefully yields more productive algorithms and experimentation.

Each algorithm and implementation for each DL framework can be run as a separate command line program, or as a library.  Below you can find descriptions of the algorithms, and the status of implementation for each framework.

When the GPU is used, the code *assumes that cudnn (>= R5) is available* and installed. This is critical for good performance.

## Supported Tasks

- [Text Classification](docs/cmot.md)
- [Tagging with RNNs](docs/tagging.md)
- [Seq2Seq](docs/seq2seq.md)
- [Language Modeling with RNNs](docs/lm.md)

## Reporting with Visdom

To enable reporting with visdom, just pass `--visdom 1` in any command line program.  Baseline uses visdom with all framework implementations


## Baseline as an API

The latest code provides a high-level Python API to access common deep-learning NLP approaches.  This should facilitate faster research in any language, as these tasks are fairly standard for NLP.  The data loaders and data feeds are all reusable, as are the basic harnesses for the APIs.  To get an understanding for how to structure a program to use baseline, have a look at the command line programs for each task.

You can also think of the library itself as an abstraction layer at the "solution" or algorithm level with sub-modules built with each framework. Adding a new framework is straightforward using the methods shown in the library.

