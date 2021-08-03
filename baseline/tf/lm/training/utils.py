import os
import time
import numpy as np
import tensorflow as tf
from baseline.tf.tfy import TRAIN_FLAG, SET_TRAIN_FLAG
from baseline.train import Trainer, register_trainer
from baseline.model import create_model_for
from collections import OrderedDict


# Number of batches to prefetch if using tf.datasets
NUM_PREFETCH = 2
# The shuffle buffer
SHUF_BUF_SZ = 5000


def to_tensors(ts):
    """Convert a data feed into a tuple of `features` (`dict`) and `y` values

    This method is required to produce `tf.dataset`s from the input data feed.
    Any fields ending with `_lengths` are ignored, unless they match the
    `src_lengths_key` or `tgt_lengths_key`, in which case, they are converted to `src_len` and `tgt_len`

    :param ts: The data feed to convert
    :param lengths_key: This is a field passed from the model params specifying source of truth of the temporal lengths
    :return: A `tuple` of `features` and `y` (labels)
    """
    keys = ts[0].keys()
    # This is kind of a hack
    keys = [k for k in keys if k != 'ids']

    features = dict((k, []) for k in keys)

    for sample in ts:
        for k in features.keys():
            for s in sample[k]:
                features[k].append(s)

    features = dict((k, np.stack(v).astype(np.int32)) for k, v in features.items())
    tgt = features.pop('y')
    return features, tgt

