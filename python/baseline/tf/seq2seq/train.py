import tensorflow as tf
import numpy as np
import time

from baseline.utils import listify
from baseline.reporting import basic_reporting
from baseline.progress import ProgressBar


class Seq2SeqTrainerTf:

    def __init__(self, model, **kwargs):

        eta = kwargs.get('eta', kwargs.get('lr', 0.01))
        print('using eta [%.3f]' % eta)
        mom = kwargs.get('mom', 0.9)
        optim = kwargs.get('optim', 'sgd')
        print('using optim [%s]' % optim)
        clip = float(kwargs['clip']) if 'clip' in kwargs else 5

        self.sess = model.sess
        self.loss = model.create_loss()
        self.model = model
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if optim == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(eta, 0.95, 1e-6)
        elif optim == 'adam':
            self.optimizer = tf.train.AdamOptimizer(eta)
        elif mom > 0:
            self.optimizer = tf.train.MomentumOptimizer(eta, mom)
            print('using mom [%.3f]' % mom)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(eta)

        gvs = self.optimizer.compute_gradients(self.loss)
        self.train_op = self.optimizer.apply_gradients(gvs, global_step=self.global_step)

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-checkpoints/seq2seq", global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-checkpoints")
        print('Reloading ' + latest)
        self.model.saver.restore(self.model.sess, latest)

    def prepare(self, saver):
        self.model.saver = saver

    def train(self, ts):
        total_loss = 0
        steps = len(ts)
        metrics = {}
        pg = ProgressBar(steps)
        for src, tgt, src_len, tgt_len in ts:
            # TODO: there is a bug that occurs if mx_tgt_len == mxlen

            feed_dict = self.model.make_feed_dict(src, src_len, tgt, tgt_len)
            _, step, lossv = self.model.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)
            total_loss += lossv
            pg.update()

        pg.done()
        avg_loss = total_loss/steps
        metrics['avg_loss'] = avg_loss
        metrics['perplexity'] = np.exp(avg_loss)
        return metrics

    def test(self, ts):

        total_loss = 0
        steps = len(ts)
        metrics = {}
        for src,tgt,src_len,tgt_len in ts:
            feed_dict = self.model.make_feed_dict(src, src_len, tgt, tgt_len)
            lossv = self.model.sess.run(self.loss, feed_dict=feed_dict)
            total_loss += lossv

        avg_loss = total_loss/steps
        metrics['avg_loss'] = avg_loss
        metrics['perplexity'] = np.exp(avg_loss)
        return metrics


def fit(seq2seq, ts, vs, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs

    model_file = kwargs['outfile'] if 'outfile' in kwargs and kwargs['outfile'] is not None else './seq2seq-model-tf'
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

    for epoch in range(epochs):

        start_time = time.time()
        train_metrics = trainer.train(ts)
        train_duration = time.time() - start_time
        print('Training time (%.3f sec)' % train_duration)

        if after_train_fn is not None:
            after_train_fn(seq2seq)

        start_time = time.time()
        test_metrics = trainer.test(vs)
        test_duration = time.time() - start_time
        print('Validation time (%.3f sec)' % test_duration)

        for reporting in reporting_fns:
            reporting(train_metrics, epoch, 'Train')
            reporting(test_metrics, epoch, 'Valid')

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
