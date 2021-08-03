import six
import os
import time
import logging
import tensorflow as tf

from eight_mile.confusion import ConfusionMatrix
from eight_mile.utils import listify
from eight_mile.tf.layers import reload_checkpoint
from eight_mile.tf.optz import optimizer

from eight_mile.progress import create_progress_bar
from baseline.utils import get_model_file, get_metric_cmp
from baseline.tf.tfy import _add_ema, TRAIN_FLAG, SET_TRAIN_FLAG
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.utils import verbose_output
from baseline.model import create_model_for

import numpy as np

# Number of batches to prefetch if using tf.datasets
NUM_PREFETCH = 2
# The shuffle buffer
SHUF_BUF_SZ = 5000

log = logging.getLogger('baseline.timing')


def to_tensors(ts, lengths_key):
    """Convert a data feed into a tuple of `features` (`dict`) and `y` values

    This method is required to produce `tf.dataset`s from the input data feed

    :param ts: The data feed to convert
    :return: A `tuple` of `features` and `y` (labels)
    """
    keys = ts[0].keys()
    features = dict((k, []) for k in keys)
    for sample in ts:
        for k in features.keys():
            # add each sample
            for s in sample[k]:
                features[k].append(s)

    features = dict((k, np.stack(v)) for k, v in features.items())
    if lengths_key and lengths_key in features:
        features['lengths'] = features[lengths_key]
        del features[lengths_key]
    y = features.pop('y')
    return features, y


def _report(step, metrics, start, phase, tt, reporting_fns, steps=1):
    """Make a report (both metric and timing).

    :param step: `int` The step number of this report (epoch or nstep number).
    :param metrics: `dict` The metrics to report.
    :param start: `int` The starting time of this segment.
    :param phase: `str` The phase type. {'Train', 'Valid', 'Test'}
    :param tt: `str` The tick type. {'STEP', 'EPOCH'}
    :param reporting_fns: `List[Callable]` The list of reporting functions to call.
    :param steps: `int` The number of steps in this segment, used to normalize the time.
    """
    elapsed = time.perf_counter() - start
    for reporting in reporting_fns:
        reporting(metrics, step, phase, tt)
    log.debug({
        'tick_type': tt, 'tick': step, 'phase': phase,
        'time': elapsed / float(steps),
        'step/sec': steps / float(elapsed)
    })

