import os
import time
import numpy as np
import tensorflow as tf
from eight_mile.tf.layers import reload_checkpoint
from eight_mile.tf.optz import optimizer
from baseline.tf.tfy import TRAIN_FLAG, SET_TRAIN_FLAG
from baseline.train import Trainer, register_trainer
from baseline.model import create_model_for
from collections import OrderedDict


# Number of batches to prefetch if using tf.datasets
NUM_PREFETCH = 2
# The shuffle buffer
SHUF_BUF_SZ = 5000

_EVENT_FILE_GLOB_PATTERN = 'events.out.tfevents.*'


def _summaries(eval_dir):
    """Yields `tensorflow.Event` protos from event files in the eval dir.
    Args:
      eval_dir: Directory containing summary files with eval metrics.
    Yields:
      `tensorflow.Event` object read from the event files.
    """
    if tf.gfile.Exists(eval_dir):
        for event_file in tf.gfile.Glob(os.path.join(eval_dir, _EVENT_FILE_GLOB_PATTERN)):
            for event in tf.train.summary_iterator(event_file):
                yield event


def read_eval_metrics(eval_dir):
    """Helper to read eval metrics from eval summary files.
    Args:
      eval_dir: Directory containing summary files with eval metrics.
    Returns:
      A `dict` with global steps mapping to `dict` of metric names and values.
    """
    eval_metrics_dict = {}
    for event in _summaries(eval_dir):
        if not event.HasField('summary'):
            continue
        metrics = {}
        for value in event.summary.value:
            if value.HasField('simple_value'):
                metrics[value.tag] = value.simple_value
        if metrics:
            eval_metrics_dict[event.step] = metrics
    return OrderedDict(sorted(eval_metrics_dict.items(), key=lambda t: t[0]))


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


@register_trainer(task='lm', name='default')
class LanguageModelTrainerTf(Trainer):
    """A Trainer to use if not using eager mode

    The trainer can run in 2 modes: `dataset` and `feed_dict`.  When the former, the graph is assumed to
    be connected by features attached to the input so the `feed_dict` will only be used to pass dropout information.

    When the latter, we will use the baseline DataFeed to read the object into the `feed_dict`
    """
    def __init__(self, model_params, **kwargs):
        super().__init__()
        if type(model_params) is dict:
            self.model = create_model_for('lm', **model_params)
        else:
            self.model = model_params
        self.sess = self.model.sess
        self.loss = self.model.create_loss()
        self.test_loss = self.model.create_test_loss()
        self.global_step, self.train_op = optimizer(self.loss, colocate_gradients_with_ops=True, variables=self.model.trainable_variables, **kwargs)
        self.nsteps = kwargs.get('nsteps', 500)
        init = tf.compat.v1.global_variables_initializer()
        self.model.sess.run(init)
        saver = tf.compat.v1.train.Saver()
        self.model.set_saver(saver)
        checkpoint = kwargs.get('checkpoint')
        if checkpoint is not None:
            skip_blocks = kwargs.get('blocks_to_skip', ['OptimizeLoss'])
            reload_checkpoint(self.model.sess, checkpoint, skip_blocks)

    def checkpoint(self):
        """This method saves a checkpoint

        :return: None
        """
        checkpoint_dir = '{}-{}'.format("./tf-lm", os.getpid())
        self.model.saver.save(self.sess,
                              os.path.join(checkpoint_dir, 'lm'),
                              global_step=self.global_step,
                              write_meta_graph=False)

    def recover_last_checkpoint(self):
        """Recover the last saved checkpoint

        :return: None
        """
        checkpoint_dir = '{}-{}'.format("./tf-lm", os.getpid())
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        self.model.saver.restore(self.model.sess, latest)

    @staticmethod
    def _num_toks(batch):
        return np.prod(batch['y'].shape)

    def train(self, ts, reporting_fns, dataset=True):
        """Train by looping over the steps

        For a `tf.dataset`-backed `fit_func`, we are using the previously wired `dataset`s
        in the model (and `dataset` is `True`).  For `feed_dict`, we convert the ts samples
        to `feed_dict`s and hand them in one-by-one

        :param ts: The training set
        :param reporting_fns: A list of reporting hooks
        :param dataset: (`bool`) Are we using `tf.dataset`s
        :return: Metrics
        """
        epoch_loss = 0.0
        epoch_toks = 0

        if self.model.requires_state:
            state = self.model.sess.run(self.model.initial_state, self.model.make_input(ts[0], True))

        fetches = {
            "loss": self.loss,
            "train_op": self.train_op,
            "global_step": self.global_step
        }

        if self.model.requires_state:
            fetches["final_state"] = self.model.final_state

        start = time.time()
        self.nstep_start = start
        for batch_dict in ts:

            if dataset:
                feed_dict = {TRAIN_FLAG(): 1}
            else:
                feed_dict = self.model.make_input(batch_dict, True)
                _, global_step, lossv = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)

            # In Keras LSTM, the order is h first, c second, its the opposite in TF 1, however I dont think it
            # ends up mattering here
            if self.model.requires_state:
                for i, (s1, s2) in enumerate(self.model.initial_state):
                    feed_dict[s1] = state[i][0]  #.c  # 0
                    feed_dict[s2] = state[i][1]  #.h  # 1

            vals = self.model.sess.run(fetches, feed_dict)
            loss = vals["loss"]

            if self.model.requires_state:
                state = vals["final_state"]
            global_step = vals["global_step"]
            toks = self._num_toks(batch_dict)
            report_loss = loss * toks
            epoch_loss += report_loss
            epoch_toks += toks
            self.nstep_agg += report_loss
            self.nstep_div += toks

            if (global_step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    global_step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()

        metrics = self.calc_metrics(epoch_loss, epoch_toks)
        self.train_epochs += 1
        self.report(
            self.train_epochs, metrics, start,
            'Train', 'EPOCH', reporting_fns
        )
        return metrics

    def calc_metrics(self, agg, norm):
        metrics = super().calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics

    def test(self, vs, reporting_fns, phase, dataset=True):
        """Run an epoch of testing over the dataset

        If we are using a `tf.dataset`-based `fit_func`, we will just
        cycle the number of steps and let the `dataset` yield new batches.

        If we are using `feed_dict`s, we convert each batch from the `DataFeed`
        and pass that into TF as the `feed_dict`

        :param vs: A validation set
        :param reporting_fns: Reporting hooks
        :param phase: The phase of evaluation (`Test`, `Valid`)
        :param dataset: (`bool`) Are we using `tf.dataset`s
        :return: Metrics
        """
        total_loss = 0.0
        total_toks = 0
        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs

        if self.model.requires_state:
            state = self.model.sess.run(self.model.initial_state, self.model.make_input(vs[0], False))

        fetches = {
            "loss": self.test_loss,
        }

        if self.model.requires_state:
            fetches["final_state"] = self.model.final_state

        start = time.time()

        for batch_dict in vs:
            feed_dict = {}
            if not dataset:
                feed_dict = self.model.make_input(batch_dict, False)
            # In Keras LSTM, the order is h first, c second, its the opposite in TF 1, however I dont think it
            # ends up mattering here
            if self.model.requires_state:

                for i, (s1, s2) in enumerate(self.model.initial_state):
                    feed_dict[s1] = state[i][0]  # .c  # 0
                    feed_dict[s2] = state[i][1]  # .h  # 1

            vals = self.model.sess.run(fetches, feed_dict)
            loss = vals["loss"]
            toks = self._num_toks(batch_dict)
            if self.model.requires_state:
                state = vals["final_state"]
            total_loss += loss * toks
            total_toks += toks

        metrics = self.calc_metrics(total_loss, total_toks)
        self.report(
            epochs, metrics, start,
            phase, 'EPOCH', reporting_fns
        )
        return metrics
