import os
import time
import logging
import numpy as np
import tensorflow as tf
from eight_mile.tf.optz import optimizer
from eight_mile.tf.layers import reload_checkpoint
from baseline.tf.tfy import TRAIN_FLAG, SET_TRAIN_FLAG
from eight_mile.utils import listify
from baseline.utils import get_model_file, get_metric_cmp
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func
from baseline.model import create_model_for
from collections import OrderedDict


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

    def train(self, ts, reporting_fns):
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

        start = time.perf_counter()
        self.nstep_start = start
        for batch_dict in ts:

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

    def test(self, vs, reporting_fns, phase):
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

        start = time.perf_counter()

        for batch_dict in vs:
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


@register_training_func('lm')
def fit(model_params, ts, vs, es=None, **kwargs):
    """
    Train an language model using TensorFlow with a `feed_dict`.

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
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs

    model_file = get_model_file('lm', 'tf', kwargs.get('basedir'))
    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = create_trainer(model_params, **kwargs)

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    best_metric = 1000
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'avg_loss')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

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

