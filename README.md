# Baseline

*Version 2.x*

Baseline is a library for reproducible deep learning research and fast model
development for NLP. The library provides easily extensible abstractions and
implementations for data loading, model development, training and export of deep
learning architectures. It also provides implementations for high-performance,
deep learning models for various NLP tasks, against which newly developed models
can be compared. Deep learning experiments are hard to reproduce, Baseline
provides functionalities to track them. The goal is to allow a researcher to
focus on model development, delegating the repetitive tasks to the library.

## Components

- [**baseline**](docs/baseline.md): An object-oriented Python library for
  rapid development of deep learning algorithms. The library provides extensible
  base classes for common components in a deep learning architecture (data
  loading, model development, training, evaluation, and export) in TensorFlow,
  PyTorch and DyNet. In addition, it provides strong, deep learning baselines
  for four fundamental NLP tasks -- [Classification](./docs/classify.md),
  [Sequence Tagging](./docs/tagging.md), [Seq-to-Seq Encoder-Decoders](./docs/seq2seq.md)
  and [Language Modeling](./docs/lm.md). Many NLP problems can be seen as
  variants of these tasks. For example, Part of Speech (POS) Tagging, Named
  Entity Recognition (NER) and Slot-filling are all Sequence Tagging tasks,
  Neural Machine Translation (NMT) is typically modeled as an Encoder-Decoder
  task. An end-user can easily implement a new model and delegate the rest to
  the library.

- [**mead**](docs/mead.md): Software for fast modeling, experimentation
  and development built on top of [baseline](docs/baseline.md) core modules. It contains driver programs to run experiments from JSON or YAML
  configuration files to completely control the reader, trainer, model, and
  hyper-parameters. 

- [**xpctl**](docs/xpctl.md): A command-line interface to track experimental
  results and provide access to a global leaderboard. After running an
  experiment through mead, the results and the logs are committed to a database.
  Several commands are provided to show the best experimental results under
  various constraints.

- [**hpctl**](docs/hpctl.md): A library for sampling configurations and training
  models to help find good hyper parameters.

## Workflow

The workflow for developing a deep learning model using baseline is as follows:

1. Map the problem to one of the existing tasks using a `<$task, dataset$>`
   tuple, eg., NER on CoNLL 2003 dataset is a `<tagger task, conll>`.
2. Use the existing implementations in `Baseline` or extend the base model class
   to create a new architecture.
3. Define a configuration file in `mead` and run an experiment.
4. Use `xpctl` to compare the result with the previous experiments, commit the
   results to the leaderboard database and the model files to a persistent
   storage if desired.

Additionally, the base models provided by the library can be
[exported from saved checkpoints](docs/export.md) directly into
[TensorFlow Serving](https://www.tensorflow.org/serving/) for deployment in a
production environment. [The framework can be run within a Docker container](docs/docker.md)
to reduce the installation complexity and to isolate experiment configurations
and variants. It is actively maintained by a team of core developers and accepts
public contributions.

## Installation

Baseline can be installed as a Python package:

    ./install_dev.sh baseline

Currently, xpctl depends on baseline but baseline is not available on PyPI so
you need to install baseline before you install xpctl:

    ./install_dev.sh xpctl

## A Note About Versions

Deep Learning Frameworks are evolving quickly, and changes are not always
backwards compatible. We recommend recent versions of each framework. Baseline
is known to work on most versions of TensorFlow, and is currently being run on
versions between 1.5 and 1.13. The PyTorch backend requires at least version 1.0.

## Citing

If you use the library, please cite the following paper:

```
@InProceedings{W18-2506,
  author =    "Pressel, Daniel
               and Ray Choudhury, Sagnik
               and Lester, Brian
               and Zhao, Yanjie
               and Barta, Matt",
  title =     "Baseline: A Library for Rapid Modeling, Experimentation and
               Development of Deep Learning Algorithms targeting NLP",
  booktitle = "Proceedings of Workshop for NLP Open Source Software (NLP-OSS)",
  year =      "2018",
  publisher = "Association for Computational Linguistics",
  pages =     "34--40",
  location =  "Melbourne, Australia",
  url =       "http://aclweb.org/anthology/W18-2506"
}
```
