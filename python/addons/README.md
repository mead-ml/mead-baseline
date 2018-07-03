baseline addons
===============

These are `addon` models that can be run in `baseline` using `mead`.  To run these models, make sure this directory is in your `$PYTHONPATH`

# Overview

These models are coded in a backend that is supported by baseline (currently either TensorFlow or PyTorch)

- [Text Classification using ELMo](classify_elmo.py)
  - Uses [ELMo embeddings](https://export.arxiv.org/pdf/1802.05365) concatenated with word embeddings. It achieves *90.6* accuracy on SST2 after a single epoch
  - Model is based on the baseline convolutional model
  - This model requires `tensorflow_hub`, which requires TensorFlow 1.7.   You can get `hub` by running
    - `pip install tensorflow-hub`
    - https://www.tensorflow.org/hub/installation
  - Run using [this config](../mead/config/sst2-elmo.json)
    - `mead-train --config config/sst2-elmo.json`

- [Tagger with Gazetteer](tagger_gazetteer.py)
  - This model allows gazetteer features to be used along with normal word and character embeddings
  - Achieves *40.2537* on WNUT17, an improvement over the baseline

- [Tagger with ELMo embeddings](tagger_elmo.py)
  - Uses ELMo embeddings (see info above) concatenated with word embeddings using 1 or 2 bidirectional LSTM layers.  With a single BLSTM layer, it achieves *44* F1 on WNUT17, and up to *92* F1 on CONLL2003
  - This model requires `tensorflow_hub` (see info above)
  - Run using [this config](../mead/config/wnut-elmo.json)
    - `mead-train --config config/wnut-elmo.json`

- [Encoder Decoder with Transformer](seq2seq_transformer.py)
  - This model implements https://arxiv.org/abs/1706.03762
    - Based on this excellent reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html
