baseline addons
===============

These are `addon` models that can be run in `baseline` using `mead`.  To run these models, make sure this directory is in your `$PYTHONPATH`

# Overview

These models are coded in a backend that is supported by baseline (currently either TensorFlow or PyTorch)

- [Text Classification using ELMo](classifier_elmo.py)
  - State-of-the-art classification model, uses [ELMo embeddings](https://export.arxiv.org/pdf/1802.05365) concatenated with word embeddings. It achieves *90.6* accuracy on SST2 after a single epoch
  - Model is based on the baseline convolutional model
  - Run using [this config](../mead/config/sst2-elmo.json)
  - This model requires `tensorflow_hub`, which requires TensorFlow 1.7.   You can get `hub` by running
    - `pip install tensorflow-hub`
    - https://www.tensorflow.org/hub/installation

- [Tagger with Gazetteer](tagger_gazetteer.py)
  - This model allows gazetteer features to be used along with normal word and character embeddings
  - Achieves *40.2537* on WNUT, a significant improvement over the baseline