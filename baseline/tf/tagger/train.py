"""Train a tagger with TensorFlow

This module supports 2 different ways of training a model

1. feed_dict
2. datasets

It doesnt currently support estimators and multi-GPU as there is currently no good way to early stop with
`train_and_evaluate`, nor is there an easy way to stop on the metrics we define here
"""
from baseline.tf.tagger.training import *


