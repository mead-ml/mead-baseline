import tensorflow as tf
import numpy as np
from baseline.utils import listify, get_model_file
from baseline.reporting import basic_reporting
from baseline.tf.tfy import optimizer
from baseline.train import Trainer
import time
import os


class Seq2SeqTrainerTf(Trainer):

    def __init__(self, model, **kwargs):
        super(Seq2SeqTrainerTf, self).__init__()
        self.sess = model.sess
        self.loss = model.create_loss()
        self.model = model
        self.global_step, self.train_op = optimizer(self.loss, **kwargs)

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-seq2seq-%d/seq2seq" % os.getpid(), global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-seq2seq-%d" % os.getpid())
        print('Reloading ' + latest)
        self.model.saver.restore(self.model.sess, latest)

    def prepare(self, saver):
        self.model.saver = saver

    def train(self, ts, reporting_fns):
        total_loss = 0
        steps = 0
        metrics = {}
        duration = 0
        for src, tgt, src_len, tgt_len in ts:

            start_time = time.time()
            steps += 1
            feed_dict = self.model.make_feed_dict(src, src_len, tgt, tgt_len)
            _, global_step, lossv = self.model.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)
            total_loss += lossv
            duration += time.time() - start_time

            if steps % 500 == 0:
                print('Step time (%.3f sec)' % (duration / 500.))
                duration = 0
                metrics['avg_loss'] = total_loss / steps
                metrics['perplexity'] = np.exp(total_loss / steps)
                for reporting in reporting_fns:
                    reporting(metrics, global_step, 'Train')
            
        assert(steps == len(ts))

        metrics['avg_loss'] = total_loss / steps
        metrics['perplexity'] = np.exp(total_loss / steps)
        for reporting in reporting_fns:
            reporting(metrics, global_step, 'Train')
        return metrics

    def test(self, vs, reporting_fns, phase='Valid'):
        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs

        total_loss = 0
        steps = len(vs)
        metrics = {}
        for src,tgt,src_len,tgt_len in vs:
            feed_dict = self.model.make_feed_dict(src, src_len, tgt, tgt_len)
            lossv = self.model.sess.run(self.loss, feed_dict=feed_dict)
            total_loss += lossv

        avg_loss = total_loss/steps
        metrics['avg_loss'] = avg_loss
        metrics['perplexity'] = np.exp(avg_loss)
        for reporting in reporting_fns:
            reporting(metrics, epochs, phase)
        return metrics


def fit(seq2seq, ts, vs, es=None, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs

    model_file = get_model_file(kwargs, 'seq2seq', 'tf')
    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = Seq2SeqTrainerTf(seq2seq, **kwargs)
    init = tf.global_variables_initializer()
    seq2seq.sess.run(init)
    saver = tf.train.Saver()
    trainer.prepare(saver)

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'avg_loss')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', basic_reporting))
    print('reporting', reporting_fns)

    min_metric = 10000
    last_improved = 0

    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(seq2seq)
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

