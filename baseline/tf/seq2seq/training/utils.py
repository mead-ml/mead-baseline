import os
import time
import numpy as np
import tensorflow as tf
from eight_mile.tf.layers import create_session, reload_checkpoint
from eight_mile.tf.optz import optimizer
from baseline.progress import create_progress_bar
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


@register_trainer(task='seq2seq', name='default')
class Seq2SeqTrainerTf(Trainer):
    """A non-eager mode Trainer for seq2seq

    The trainer can run in 2 modes: `dataset` and `feed_dict`.  When the former, the graph is assumed to
    be connected by features attached to the input so the `feed_dict` will only be used to pass dropout information.

    When the latter, we will use the baseline DataFeed to read the object into the `feed_dict`
    """
    def __init__(self, model_params, **kwargs):
        """Create a Trainer, and give it the parameters needed to instantiate the model

        :param model_params: The model parameters
        :param kwargs: See below

        :Keyword Arguments:

          * *nsteps* (`int`) -- If we should report every n-steps, this should be passed
          * *ema_decay* (`float`) -- If we are doing an exponential moving average, what decay to us4e
          * *clip* (`int`) -- If we are doing gradient clipping, what value to use
          * *optim* (`str`) -- The name of the optimizer we are using
          * *lr* (`float`) -- The learning rate we are using
          * *mom* (`float`) -- If we are using SGD, what value to use for momentum
          * *beta1* (`float`) -- Adam-specific hyper-param, defaults to `0.9`
          * *beta2* (`float`) -- Adam-specific hyper-param, defaults to `0.999`
          * *epsilon* (`float`) -- Adam-specific hyper-param, defaults to `1e-8
          * *tgt_rlut* (`dict`) -- This is a dictionary that converts from ints back to strings, used for predictions
          * *beam* (`int`) -- The beam size to use at prediction time, defaults to `10`

        """
        super().__init__()
        if type(model_params) is dict:
            self.model = create_model_for('seq2seq', **model_params)
        else:
            self.model = model_params
        self.sess = self.model.sess
        self.loss = self.model.create_loss()
        self.test_loss = self.model.create_test_loss()
        self.tgt_rlut = kwargs['tgt_rlut']
        self.base_dir = kwargs['basedir']
        self.global_step, self.train_op = optimizer(self.loss, colocate_gradients_with_ops=True, variables=self.model.trainable_variables, **kwargs)
        self.nsteps = kwargs.get('nsteps', 500)
        self.beam = kwargs.get('beam', 10)
        tables = tf.compat.v1.tables_initializer()
        self.model.sess.run(tables)
        self.model.sess.run(tf.compat.v1.global_variables_initializer())
        self.model.set_saver(tf.compat.v1.train.Saver())
        self.bleu_n_grams = int(kwargs.get("bleu_n_grams", 4))

        init = tf.compat.v1.global_variables_initializer()
        self.model.sess.run(init)
        checkpoint = kwargs.get('checkpoint')
        if checkpoint is not None:
            skip_blocks = kwargs.get('blocks_to_skip', ['OptimizeLoss'])
            reload_checkpoint(self.model.sess, checkpoint, skip_blocks)

    def checkpoint(self):
        """This method saves a checkpoint

        :return: None
        """
        checkpoint_dir = '{}-{}'.format("./tf-seq2seq", os.getpid())
        self.model.saver.save(self.sess,
                              os.path.join(checkpoint_dir, 'seq2seq'),
                              global_step=self.global_step,
                              write_meta_graph=False)

    def recover_last_checkpoint(self):
        """Recover the last saved checkpoint

        :return: None
        """
        latest = os.path.join(self.base_dir, 'seq2seq-model-tf-%d' % os.getpid())
        # logger.info('Reloading %s', latest)
        g = tf.Graph()
        with g.as_default():
            SET_TRAIN_FLAG(None)
            sess = create_session()
            self.model = self.model.load(latest, predict=True, beam=self.beam, session=sess)

    def _num_toks(self, lens):
        return np.sum(lens)

    def calc_metrics(self, agg, norm):
        """Calculate metrics

        :param agg: The aggregated loss
        :param norm: The number of steps to average over
        :return: The metrics
        """
        metrics = super().calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics

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
        epoch_loss = 0
        epoch_toks = 0

        start = time.time()
        self.nstep_start = start
        for batch_dict in ts:
            if dataset:
                _, global_step, lossv = self.sess.run([self.train_op, self.global_step, self.loss],
                                                      feed_dict={TRAIN_FLAG(): 1})
            else:
                feed_dict = self.model.make_input(batch_dict, True)
                _, global_step, lossv = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)

            # ?? How to get this cleaner?
            toks = self._num_toks(batch_dict['tgt_lengths'])
            report_loss = lossv * toks

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

    def _evaluate(self, es, reporting_fns):
        """Run the model with beam search and report Bleu.

        :param es: `DataFeed` of input
        :param reporting_fns: Input hooks
        """
        pg = create_progress_bar(len(es))
        preds = []
        golds = []
        start = time.time()
        for batch_dict in pg(es):
            tgt = batch_dict.pop('tgt')
            tgt_lens = batch_dict.pop('tgt_lengths')
            pred = [p[0] for p in self.model.predict(batch_dict)[0]]
            preds.extend(convert_seq2seq_preds(pred, self.tgt_rlut))
            golds.extend(convert_seq2seq_golds(tgt, tgt_lens, self.tgt_rlut))
        metrics = {'bleu': bleu(preds, golds, self.bleu_n_grams)[0]}
        self.report(
            0, metrics, start, 'Test', 'EPOCH', reporting_fns
        )
        return metrics

    def test(self, vs, reporting_fns, phase='Valid', dataset=True):
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
        if phase == 'Test' and not dataset:
            return self._evaluate(vs, reporting_fns)
        self.valid_epochs += 1

        total_loss = 0
        total_toks = 0
        preds = []
        golds = []

        start = time.time()
        pg = create_progress_bar(len(vs))
        for batch_dict in pg(vs):

            if dataset:
                lossv, top_preds = self.model.sess.run([self.test_loss, self.model.decoder.best])
            else:
                feed_dict = self.model.make_input(batch_dict)
                lossv, top_preds = self.model.sess.run([self.test_loss, self.model.decoder.best], feed_dict=feed_dict)
            toks = self._num_toks(batch_dict['tgt_lengths'])
            total_loss += lossv * toks
            total_toks += toks

            preds.extend(convert_seq2seq_preds(top_preds.T, self.tgt_rlut))
            golds.extend(convert_seq2seq_golds(batch_dict['tgt'], batch_dict['tgt_lengths'], self.tgt_rlut))

        metrics = self.calc_metrics(total_loss, total_toks)
        metrics['bleu'] = bleu(preds, golds, self.bleu_n_grams)[0]
        self.report(
            self.valid_epochs, metrics, start,
            phase, 'EPOCH', reporting_fns
        )
        return metrics
