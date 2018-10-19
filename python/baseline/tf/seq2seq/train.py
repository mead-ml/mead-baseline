import os
import time
import logging
import numpy as np
import tensorflow as tf
from baseline.utils import zip_model
from baseline.tf.tfy import optimizer
from baseline.utils import listify, get_model_file
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func
import time
import os


@register_trainer(task='seq2seq', name='default')
class Seq2SeqTrainerTf(Trainer):

    def __init__(self, model, **kwargs):
        super(Seq2SeqTrainerTf, self).__init__()
        self.sess = model.sess
        self.loss = model.create_loss()
        self.test_loss = model.create_test_loss()
        self.model = model
        self.global_step, self.train_op = optimizer(self.loss, **kwargs)
        self.log = logging.getLogger('baseline.timing')

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-seq2seq-%d/seq2seq" % os.getpid(), global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-seq2seq-%d" % os.getpid())
        print('Reloading ' + latest)
        self.model.saver.restore(self.model.sess, latest)

    def prepare(self, saver):
        self.model.set_saver(saver)

    def train(self, ts, reporting_fns):
        total_loss = 0
        steps = 0
        metrics = {}
        duration = 0

        fetches = {
            "loss": self.loss,
            "train_op": self.train_op,
            "global_step": self.global_step}

        start = time.time()
        for batch_dict in ts:
            start_time = time.time()
            steps += 1
            feed_dict = self.model.make_input(batch_dict, do_dropout=True)
            vals = self.model.sess.run(fetches, feed_dict=feed_dict)
            global_step = vals["global_step"]
            lossv = vals["loss"]

            total_loss += lossv
            duration += time.time() - start_time

            if steps % 500 == 0:
                print('Step time (%.3f sec)' % (duration / 500.))
                duration = 0
                metrics['avg_loss'] = total_loss / steps
                metrics['perplexity'] = np.exp(total_loss / steps)
                for reporting in reporting_fns:
                    reporting(metrics, global_step.item(), 'Train')

        assert(steps == len(ts))

        self.log.debug({'phase': 'Train', 'time': time.time() - start})
        metrics['avg_loss'] = total_loss / steps
        metrics['perplexity'] = np.exp(total_loss / steps)
        for reporting in reporting_fns:
            reporting(metrics, global_step.item(), 'Train')
        return metrics

    def test(self, vs, reporting_fns, phase='Valid'):
        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs

        fetches = {
            "loss": self.test_loss,
        }

        total_loss = 0
        steps = len(vs)
        metrics = {}

        start = time.time()
        for batch_dict in vs:

            feed_dict = self.model.make_input(batch_dict)
            vals = self.model.sess.run(fetches, feed_dict)
            lossv = vals["loss"]
            total_loss += lossv

        self.log.debug({'phase': phase, 'time': time.time() - start})
        avg_loss = total_loss/steps
        metrics['avg_loss'] = avg_loss
        metrics['perplexity'] = np.exp(avg_loss)
        for reporting in reporting_fns:
            reporting(metrics, epochs, phase)
        return metrics


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

