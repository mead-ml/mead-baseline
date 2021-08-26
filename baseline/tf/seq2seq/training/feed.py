import os
import time
import logging
import numpy as np
import tensorflow as tf
from eight_mile.bleu import bleu
from eight_mile.tf.optz import optimizer
from eight_mile.progress import create_progress_bar
from baseline.utils import convert_seq2seq_golds, convert_seq2seq_preds
from eight_mile.tf.layers import reload_checkpoint, create_session
from baseline.tf.tfy import TRAIN_FLAG, SET_TRAIN_FLAG
from eight_mile.utils import listify
from baseline.utils import get_model_file, get_metric_cmp
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func
from baseline.model import create_model_for
from collections import OrderedDict



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

    def train(self, ts, reporting_fns):
        """Train by looping over the steps

        :param ts: The training set
        :param reporting_fns: A list of reporting hooks
        :return: Metrics
        """
        epoch_loss = 0
        epoch_toks = 0

        start = time.perf_counter()
        self.nstep_start = start
        for batch_dict in ts:

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
        start = time.perf_counter()
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

    def test(self, vs, reporting_fns, phase='Valid'):
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
        if phase == 'Test':
            return self._evaluate(vs, reporting_fns)
        self.valid_epochs += 1

        total_loss = 0
        total_toks = 0
        preds = []
        golds = []

        start = time.perf_counter()
        pg = create_progress_bar(len(vs))
        for batch_dict in pg(vs):

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


@register_training_func('seq2seq')
def fit(model_params, ts, vs, es=None, **kwargs):
    """
    Train an encoder-decoder network using TensorFlow with a `feed_dict`.

    :param model_params: The model (or parameters to create the model) to train
    :param ts: A training data set
    :param vs: A validation data set
    :param es: A test data set, can be None
    :param kwargs:
        See below

    :Keyword Arguments:
        * *do_early_stopping* (``bool``) --
          Stop after evaluation data is no longer improving.  Defaults to True
        * *verbose* (`dict`) A dictionary containing `console` boolean and `file` name if on
        * *epochs* (``int``) -- how many epochs.  Default to 5
        * *outfile* -- Model output file
        * *patience* --
           How many epochs where evaluation is no longer improving before we give up
        * *reporting* --
           Callbacks which may be used on reporting updates
        * *nsteps* (`int`) -- If we should report every n-steps, this should be passed
        * *ema_decay* (`float`) -- If we are doing an exponential moving average, what decay to us4e
        * *clip* (`int`) -- If we are doing gradient clipping, what value to use
        * *optim* (`str`) -- The name of the optimizer we are using
        * *lr* (`float`) -- The learning rate we are using
        * *mom* (`float`) -- If we are using SGD, what value to use for momentum
        * *beta1* (`float`) -- Adam-specific hyper-param, defaults to `0.9`
        * *beta2* (`float`) -- Adam-specific hyper-param, defaults to `0.999`
        * *epsilon* (`float`) -- Adam-specific hyper-param, defaults to `1e-8
        * *after_train_fn* (`func`) -- A callback to fire after ever epoch of training

    :return: None
    """
    epochs = int(kwargs.get('epochs', 5))
    patience = int(kwargs.get('patience', epochs))
    model_file = get_model_file('seq2seq', 'tf', kwargs.get('basedir'))

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'perplexity')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = create_trainer(model_params, **kwargs)

    last_improved = 0

    for epoch in range(epochs):
        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(trainer.model)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            print('New best %.3f' % best_metric)
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on %s: %.3f at epoch %d' % (early_stopping_metric, best_metric, last_improved))
    if es is not None:
        trainer.recover_last_checkpoint()
        trainer.test(es, reporting_fns, phase='Test')
