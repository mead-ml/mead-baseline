import os
import time
import logging
import numpy as np
import tensorflow as tf
from baseline.tf.optz import optimizer
from baseline.utils import listify, get_model_file
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func


@register_trainer(task='seq2seq', name='default')
class Seq2SeqTrainerTf(Trainer):

    def __init__(self, model, **kwargs):
        super(Seq2SeqTrainerTf, self).__init__()
        self.sess = model.sess
        self.loss = model.create_loss()
        self.test_loss = model.create_test_loss()
        self.model = model
        self.global_step, self.train_op = optimizer(self.loss, colocate_gradients_with_ops=True, **kwargs)
        self.nsteps = kwargs.get('nsteps', 500)

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-seq2seq-%d/seq2seq" % os.getpid(), global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-seq2seq-%d" % os.getpid())
        print('Reloading ' + latest)
        self.model.saver.restore(self.model.sess, latest)

    def prepare(self, saver):
        self.model.set_saver(saver)

    def _num_toks(self, lens):
        return np.sum(lens)

    def calc_metrics(self, agg, norm):
        metrics = super(Seq2SeqTrainerTf, self).calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics

    def train(self, ts, reporting_fns):
        epoch_loss = 0
        epoch_toks = 0

        start = time.time()
        self.nstep_start = start
        for batch_dict in ts:
            feed_dict = self.model.make_input(batch_dict, True)
            _, global_step, lossv = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)

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
                    'Train', 'STEP', reporting_fns
                )
                self.reset_nstep()

        metrics = self.calc_metrics(epoch_loss, epoch_toks)
        self.train_epochs += 1
        self.report(
            self.train_epochs, metrics, start,
            'Train', 'EPOCH', reporting_fns
        )
        return metrics

    def test(self, vs, reporting_fns, phase='Valid'):
        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs

        total_loss = 0
        total_toks = 0
        metrics = {}

        start = time.time()
        for batch_dict in vs:

            feed_dict = self.model.make_input(batch_dict)
            lossv = self.model.sess.run(self.test_loss, feed_dict=feed_dict)
            toks = self._num_toks(batch_dict['tgt_lengths'])
            total_loss += lossv * toks
            total_toks += toks

        metrics = self.calc_metrics(total_loss, total_toks)
        self.report(
            epochs, metrics, start,
            phase, 'EPOCH', reporting_fns
        )


@register_training_func('seq2seq')
def fit(model, ts, vs, es=None, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs

    model_file = get_model_file('seq2seq', 'tf', kwargs.get('basedir'))
    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = create_trainer(model, **kwargs)
    init = tf.global_variables_initializer()
    model.sess.run(init)
    saver = tf.train.Saver()
    trainer.prepare(saver)

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'avg_loss')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    min_metric = 10000
    last_improved = 0

    for epoch in range(epochs):

        #if after_train_fn is not None:
        #    after_train_fn(model)

        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(model)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif test_metrics[early_stopping_metric] < min_metric:
            last_improved = epoch
            min_metric = test_metrics[early_stopping_metric]
            print('New min %.3f' % min_metric)
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on min_metric %.3f at epoch %d' % (min_metric, last_improved))
    if es is not None:

        trainer.recover_last_checkpoint()
        trainer.test(es, reporting_fns, phase='Test')
