## Workflow

The workflow for developing a deep learning model using baseline is as follows:

1. Map the problem to one of the existing tasks using a `<task, dataset>`
   tuple, eg., NER on CoNLL 2003 dataset is a `<tagger, conll-iobes>`.
2. Use the existing implementations in `Baseline` or extend the base model class
   to create a new architecture.
3. Define a configuration file in `mead` and run an experiment.
4. Use `xpctl` to compare the result with the previous experiments, commit the
   results to the leaderboard database and the model files to a persistent
   storage if desired.

Additionally, the base models provided by the library can be
[exported from saved checkpoints](export.md) directly into
[TensorFlow Serving](https://www.tensorflow.org/serving/) or [ONNX]() for deployment in a
production environment.

## Components

- [**8 mile**]: An lightweight implementation of NLP layers and embeddings built on Keras and PyTorch.  This is used
  by `mead-baseline` to create reusable models that are nearly identical in PyTorch and TensorFlow.  It also provides
  core utilities and metrics APIs used by the library
  
- [**baseline**](baseline.md): An object-oriented Python library for
  rapid development of deep learning algorithms. The library provides extensible
  base classes for common components in a deep learning architecture (data
  loading, model development, training, evaluation, and export) in TensorFlow and
  PyTorch. In addition, it provides strong, deep learning baselines
  for four fundamental NLP tasks -- [Classification](classify.md),
  [Sequence Tagging](tagging.md), [Seq-to-Seq Encoder-Decoders](seq2seq.md)
  and [Language Modeling](lm.md). Many NLP problems can be seen as
  variants of these tasks. For example, Part of Speech (POS) Tagging, Named
  Entity Recognition (NER) and Slot-filling are all Sequence Tagging tasks,
  Neural Machine Translation (NMT) is typically modeled as an Encoder-Decoder
  task. An end-user can easily implement a new model and delegate the rest to
  the library.

- [**mead**](mead.md): Software for fast modeling, experimentation
  and development built on top of [baseline](docs/baseline.md) core modules. It contains driver programs to run experiments from JSON or YAML
  configuration files to completely control the reader, trainer, model, and
  hyper-parameters. 

- [**xpctl**](xpctl.md): A command-line interface to track experimental
  results and provide access to a global leaderboard. After running an
  experiment through mead, the results and the logs are committed to a database.
  Several commands are provided to show the best experimental results under
  various constraints.

- [**hpctl**](hpctl.md): A library for sampling configurations and training
  models to help find good hyper parameters.

## Design

- [Fine-tuning pre-trained models](fine-tuning.md)
- [Deep-learning framework implementation details](fw.md)
- [How addons are implemented](addons.md)
- [Extensible reporting in MEAD](reporting.md)
- [Version 2 release notes](v2.md)
