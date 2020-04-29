"""Train a tagger with TensorFlow

This module supports 2 different ways of training a model

1. feed_dict
2. datasets (`default` if not eager)
3. eager mode (`default` if eager)
4. distributed eager mode (`distributed`)
"""
from baseline.tf.tagger.training import *


