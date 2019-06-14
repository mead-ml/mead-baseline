import os
import time
import logging
import numpy as np
import tensorflow as tf
from baseline.tf.optz import optimizer
from baseline.utils import listify, get_model_file, get_metric_cmp
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func


logger = logging.getLogger('baseline')


@register_trainer(task='lm', name='default')
class LanguageModelTrainerTf(Trainer):

    def __init__(self, model, **kwargs):
        super(LanguageModelTrainerTf, self).__init__()
        self.model = model
        self.loss = model.create_loss()
        self.test_loss = model.create_test_loss()
        if kwargs.get('eval_mode', False):
            # When reloaded a model creating the training op will break things.
            self.train_op = tf.no_op()
        else:
            self.global_step, self.train_op = optimizer(self.loss, **kwargs)
        self.nsteps = kwargs.get('nsteps', 500)

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-lm-%d/lm" % os.getpid(), global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-lm-%d" % os.getpid())
        logger.info("Reloading %s", latest)
        self.model.saver.restore(self.model.sess, latest)

    @staticmethod
    def _num_toks(batch):
        return np.prod(batch['y'].shape)

    def train(self, ts, reporting_fns):
        epoch_loss = 0.0
        epoch_toks = 0

        xfer_state = hasattr(self.model, 'initial_state')
        if xfer_state:
            state = self.model.sess.run(self.model.initial_state, self.model.make_input(ts[0], True))

        fetches = {
            "loss": self.loss,
            "train_op": self.train_op,
            "global_step": self.global_step
        }

        if xfer_state:
            fetches["final_state"] = self.model.final_state

        start = time.time()
        self.nstep_start = start
        for batch_dict in ts:

            feed_dict = self.model.make_input(batch_dict, True)
            if xfer_state:
                for i, (c, h) in enumerate(self.model.initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

            vals = self.model.sess.run(fetches, feed_dict)
            loss = vals["loss"]

            if xfer_state:
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
        metrics = super(LanguageModelTrainerTf, self).calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics

    def test(self, ts, reporting_fns, phase, **kwargs):
        total_loss = 0.0
        total_toks = 0
        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs
        xfer_state = hasattr(self.model, 'initial_state')

        if xfer_state:
            state = self.model.sess.run(self.model.initial_state, self.model.make_input(ts[0], False))

        fetches = {
            "loss": self.test_loss,
        }

        if xfer_state:
            fetches["final_state"] = self.model.final_state

        start = time.time()

        for batch_dict in ts:
            feed_dict = self.model.make_input(batch_dict, False)
            if xfer_state:
                for i, (c, h) in enumerate(self.model.initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

            vals = self.model.sess.run(fetches, feed_dict)
            loss = vals["loss"]
            toks = self._num_toks(batch_dict)
            if xfer_state:
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
def fit(model, ts, vs, es=None, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs

    model_file = get_model_file('lm', 'tf', kwargs.get('basedir'))
    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = create_trainer(model, **kwargs)
    init = tf.global_variables_initializer()
    m = model.replicas[0] if hasattr(model, 'replicas') else model
    feed_dict = {k: v for e in m.embeddings.values() for k, v in e.get_feed_dict().items()}
    model.sess.run(init, feed_dict)
    saver = tf.train.Saver()
    model.set_saver(saver)
    checkpoint = kwargs.get('checkpoint')
    if checkpoint is not None:
        latest = tf.train.latest_checkpoint(checkpoint)
        print('Reloading ' + latest)
        model.saver.restore(model.sess, latest)

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    best_metric = 1000
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'avg_loss')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)

    last_improved = 0

    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(model)

        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            logger.info('New best %.3f', best_metric)
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif (epoch - last_improved) > patience:
            logger.info('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        logger.info('Best performance on %s: %.3f at epoch %d', early_stopping_metric, best_metric, last_improved)
    if es is not None:
        trainer.recover_last_checkpoint()
        test_metrics = trainer.test(es, reporting_fns, phase='Test')
    return test_metrics
