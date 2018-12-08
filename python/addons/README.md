baseline addons
===============

These are `addon` models that can be run in `baseline` using `mead`.  To run these models, make sure this directory is in your `$PYTHONPATH`

# Overview

These models are coded in a backend that is supported by baseline (currently either TensorFlow or PyTorch)


- [Tagger with Gazetteer](tagger_gazetteer.py)
  - This model allows gazetteer features to be used along with normal word and character embeddings

- [ELMo pre-trained embeddings](embed_elmo.py)
   - Requires `tensorflow_hub`
     - `pip install tensorflow-hub`
     - https://www.tensorflow.org/hub/installation

  - This provides ELMo embeddings to use with any model
  - The peformance of this model is listed under the [tagger section](../docs/tagger.md)

  - For example, you can run this with CONLL2003 as follows:

```
mead-train --config config/conll-elmo.yml
```

