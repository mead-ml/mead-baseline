import six
import os
import time
import logging
import tensorflow as tf

from eight_mile.confusion import ConfusionMatrix
from eight_mile.utils import listify
from eight_mile.tf.layers import reload_checkpoint
from eight_mile.tf.optz import optimizer

from baseline.progress import create_progress_bar
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
    elapsed = time.time() - start
    for reporting in reporting_fns:
        reporting(metrics, step, phase, tt)
    log.debug({
        'tick_type': tt, 'tick': step, 'phase': phase,
        'time': elapsed / float(steps),
        'step/sec': steps / float(elapsed)
    })


@register_trainer(task='classify', name='default')
class ClassifyTrainerTf(EpochReportingTrainer):
    """A non-eager mode Trainer for classification

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

        """
        super().__init__()
        if type(model_params) is dict:
            self.model = create_model_for('classify', **model_params)
        else:
            self.model = model_params
        self.sess = self.model.sess
        self.loss = self.model.create_loss()
        self.test_loss = self.model.create_test_loss()
        self.global_step, train_op = optimizer(self.loss, colocate_gradients_with_ops=True, variables=self.model.trainable_variables, **kwargs)
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)
        decay = kwargs.get('ema_decay', None)
        if decay is not None:
            self.ema = True
            ema_op, self.ema_load, self.ema_restore = _add_ema(self.model, float(decay))
            with tf.compat.v1.control_dependencies([ema_op]):
                self.train_op = tf.identity(train_op)
        else:
            self.ema = False
            self.train_op = train_op

        tables = tf.compat.v1.tables_initializer()
        self.model.sess.run(tables)
        self.model.sess.run(tf.compat.v1.global_variables_initializer())
        self.model.set_saver(tf.compat.v1.train.Saver())
        checkpoint = kwargs.get('checkpoint')
        if checkpoint is not None:
            skip_blocks = kwargs.get('blocks_to_skip', ['OptimizeLoss'])
            reload_checkpoint(self.model.sess, checkpoint, skip_blocks)

    @staticmethod
    def _get_batchsz(batch_dict):
        return len(batch_dict['y'])

    def _train(self, loader, dataset=True, **kwargs):
        """Train an epoch of data using either the input loader or using `tf.dataset`

        In non-`tf.dataset` mode, we cycle the loader data feed, and pull a batch and feed it to the feed dict
        When we use `tf.dataset`s under the hood, this function simply uses the loader to know how many steps
        to train.  We do use a `feed_dict` for passing the `TRAIN_FLAG` in either case

        :param loader: A data feed
        :param kwargs: See below

        :Keyword Arguments:
         * *dataset* (`bool`) Set to `True` if using `tf.dataset`s, defaults to `True`
         * *reporting_fns* (`list`) A list of reporting hooks to use

        :return: Metrics
        """
        if self.ema:
            self.sess.run(self.ema_restore)

        reporting_fns = kwargs.get('reporting_fns', [])
        epoch_loss = 0
        epoch_div = 0
        steps = len(loader)
        pg = create_progress_bar(steps)
        for batch_dict in pg(loader):
            if dataset:
                _, step, lossv = self.sess.run([self.train_op, self.global_step, self.loss],
                                               feed_dict={TRAIN_FLAG(): 1})
            else:
                feed_dict = self.model.make_input(batch_dict, True)
                _, step, lossv = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)

            batchsz = self._get_batchsz(batch_dict)
            report_lossv = lossv * batchsz
            epoch_loss += report_lossv
            epoch_div += batchsz
            self.nstep_agg += report_lossv
            self.nstep_div += batchsz

            if (step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()

        metrics = self.calc_metrics(epoch_loss, epoch_div)
        return metrics

    def _test(self, loader, **kwargs):
        """Test an epoch of data using either the input loader or using `tf.dataset`

        In non-`tf.dataset` mode, we cycle the loader data feed, and pull a batch and feed it to the feed dict
        When we use `tf.dataset`s under the hood, this function simply uses the loader to know how many steps
        to train.

        :param loader: A data feed
        :param kwargs: See below

        :Keyword Arguments:
          * *dataset* (`bool`) Set to `True` if using `tf.dataset`s, defaults to `True`
          * *reporting_fns* (`list`) A list of reporting hooks to use
          * *verbose* (`dict`) A dictionary containing `console` boolean and `file` name if on

        :return: Metrics
        """
        if self.ema:
            self.sess.run(self.ema_load)

        use_dataset = kwargs.get('dataset', True)

        cm = ConfusionMatrix(self.model.labels)
        steps = len(loader)
        total_loss = 0
        total_norm = 0
        verbose = kwargs.get("verbose", None)

        pg = create_progress_bar(steps)
        for i, batch_dict in enumerate(pg(loader)):
            y = batch_dict['y']
            if use_dataset:
                guess, lossv = self.sess.run([self.model.best, self.test_loss])
            else:
                feed_dict = self.model.make_input(batch_dict, False)
                guess, lossv = self.sess.run([self.model.best, self.test_loss], feed_dict=feed_dict)

            batchsz = len(guess)
            total_loss += lossv * batchsz
            total_norm += batchsz
            cm.add_batch(y, guess)

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss / float(total_norm)
        verbose_output(verbose, cm)

        return metrics

    def checkpoint(self):
        """This method saves a checkpoint

        :return: None
        """
        checkpoint_dir = '{}-{}'.format("./tf-classify", os.getpid())
        self.model.saver.save(self.sess,
                              os.path.join(checkpoint_dir, 'classify'),
                              global_step=self.global_step,
                              write_meta_graph=False)

    def recover_last_checkpoint(self):
        """Recover the last saved checkpoint

        :return: None
        """
        checkpoint_dir = '{}-{}'.format("./tf-classify", os.getpid())
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('Reloading ' + latest)
        self.model.saver.restore(self.model.sess, latest)

