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

## Running the codes

The driver programs available in baseline are:

- **Classification**: _train_: [classify_sentence.py](../python/classify_sentence.py) 
- **Sequence Tagging**: _train_ : [tag_char_rnn.py](../python/tag_char_rnn.py), _test_: [tag.py](../python/tag.py)
- **Language Modeling**: _train_: [wchar_lm.py](../python/wchar_lm.py) 
- **Encoder Decoder**: _train_ [seq2seq.py](../python/seq2seq.py), _test_: [translate.py](../python/translate.py)  
  
However, we recommend training the models through [mead](../python/mead/README.md). For running through mead, use the [trainer.py](../python/mead/trainer.py) utility in the directory:

```
python trainer.py --config config/conll.json --task tagger
```

See more running options in [trainer.py](../python/mead/trainer.py).

To run these codes `baseline/python` should be available in your `PYTHONPATH` variable.  

### Installing Baseline as a Python Package

Baseline can be installed as a python package using the script [install_dev.sh](../python/install_dev.sh). To install, run the command: `./install.sh baseline`. Once installed, you can use the commands: `mead.train` and `mead.export` to  run the [trainer](../python/mead/trainer.py) or the [exporter](../python/mead/export.py) (with the same options as before) w/o putting baseline in PYTHONPATH. 

## Baseline as an API

The latest code provides a high-level Python API to access common deep-learning NLP approaches.  This should facilitate faster research in any language, as these tasks are fairly standard for NLP.  The data loaders and data feeds are all reusable, as are the basic harnesses for the APIs.  To get an understanding for how to structure a program to use baseline, have a look at the command line programs for each task.

You can also think of the library itself as an abstraction layer at the "solution" or algorithm level with sub-modules built with each framework. Adding a new framework is straightforward using the methods shown in the library.

### As scaffolding for an experiment

If you have a problem where the input is the same as a `baseline` task, you can easily use the API to set up your boilerplate work for you, and focus on your model, by creating a user-defined `addon`.  This is just a normal python file with a creation and load hooks (see the [addons area](../python/addons) for examples). 

Then pass `--model_type {model}` to the driver program for that task.  The driver program will look to see if it has an implementation within the library and will not find the one in its registry.  So it will import the module and call its `create_model` function with the arguments and use the provided model.

