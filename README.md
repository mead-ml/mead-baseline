Baseline
=========

Baseline is a library for reproducible deep learning research and fast model development for NLP. The library provides easily extensible abstractions and implementations for data loading, model development, training and export of deep learning architectures. It also provides implementations for simple, high-performance, deep learning models for various NLP tasks, against which newly developed models can be compared. Deep learning experiments are hard to reproduce, Baseline provides functionalities to track them. The goal is to allow a researcher to focus on model development, delegating the repetitive tasks to the library.



It has three components: 

- [**baseline-core**](docs/baseline.md): An object-oriented Python library for rapid development of deep learning algorithms. The library provides extensible base classes for common components in a deep learning architecture (data loading, model development, training, evaluation, and export) in TensorFlow and PyTorch. In addition, it provides strong, deep learning baselines for four fundamental NLP tasks -- Classification, Sequence Tagging, Sequence-to-Sequence Encoder-Decoders and Language Modeling. Many NLP problems can be seen as variants of these tasks. For example, Part of Speech (POS) Tagging, Named Entity Recognition (NER) and Slot-filling are all Sequence Tagging tasks, Neural Machine Translation (NMT) is typically modeled as an Encoder-Decoder task. An end-user can easily implement a new model and delegate the rest to the library.

- [**mead**](docs/mead.md): A library built on  for fast _M_odeling, _E_xperimentation _A_nd _D_evelopment. It contains driver programs to run experiments from JSON configuration files to completely control the reader, trainer, model, and hyper-parameters. 
  
- [**xpctl**](docs/xpctl.md): A command-line interface to track experimental results and provide access to a global leaderboard. After running an experiment through mead, the results and the logs are committed to a database. Several commands are provided to show the best experimental results under various constraints. 

The workflow for developing a deep learning model using baseline is simple: 

1. Map the problem to one of the existing tasks using a `<$task, dataset$>` tuple, eg., NER on CoNLL 2003 dataset is a `<tagger task, conll>`.

2. Use the existing implementations in `Baseline` or extend the base model class to create a new architecture. 

3. Define a configuration file in `mead` and run an experiment. 

4. Use `xpctl` to compare the result with the previous experiments, commit the results to the leaderboard database and the model files to a persistent storage if desired.

Additionally, the base models provided by the library can be [exported from saved checkpoints](docs/export.md) directly into [TensorFlow Serving](https://www.tensorflow.org/serving/) for deployment in a production environment. [The framework can be run within a Docker container](docker/README.md) to reduce the installation complexity and to isolate experiment configurations and variants. It is actively maintained by a team of core developers and accepts public contributions.
