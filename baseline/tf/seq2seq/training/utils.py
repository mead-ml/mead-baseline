import os
import time
import numpy as np
import tensorflow as tf
from eight_mile.progress import create_progress_bar
from eight_mile.bleu import bleu

from baseline.utils import (
    convert_seq2seq_golds,
    convert_seq2seq_preds,
)

from baseline.train import Trainer, register_trainer
from baseline.tf.tfy import TRAIN_FLAG, SET_TRAIN_FLAG

from baseline.model import create_model_for

# Number of batches to prefetch if using tf.datasets
NUM_PREFETCH = 2
# The shuffle buffer
SHUF_BUF_SZ = 5000


def to_tensors(ts, src_lengths_key, dst=False):
    """Convert a data feed into a tuple of `features` (`dict`) and `y` values

    This method is required to produce `tf.dataset`s from the input data feed.
    Any fields ending with `_lengths` are ignored, unless they match the
    `src_lengths_key` or `tgt_lengths_key`, in which case, they are converted to `src_len` and `tgt_len`

    :param ts: The data feed to convert
    :param lengths_key: This is a field passed from the model params specifying source of truth of the temporal lengths
    :param dst: `bool` that says if we should prepare a `dst` tensor.  This is needed in distributed mode
    :return: A `tuple` of `features` and `y` (labels)
    """
    keys = ts[0].keys()
    # This is kind of a hack
    keys = [k for k in keys if '_lengths' not in k and k != 'ids'] + [src_lengths_key, "tgt_lengths"]

    features = dict((k, []) for k in keys)
    for sample in ts:
        for k in keys:
            for s in sample[k]:
                features[k].append(s)
    features['src_len'] = features[src_lengths_key]
    del features[src_lengths_key]
    features['tgt_len'] = features['tgt_lengths']
    del features['tgt_lengths']
    features = dict((k, np.stack(v).astype(np.int32)) for k, v in features.items())
    if dst:
        features['dst'] = features['tgt'][:, :-1]
    tgt = features.pop('tgt')

    return features, tgt

