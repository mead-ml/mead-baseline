"""Train a classifier with TensorFlow

This module supports several different ways of training a model

1. feed_dict
2. datasets (`default` default for non-eager)
3. eager mode (`default` for eager)
4. distributed eager mode (`distributed`)
"""
from baseline.tf.classify.training import *
