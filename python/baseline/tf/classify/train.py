import six
import os
import time
from baseline.utils import fill_y
import tensorflow as tf
from baseline.confusion import ConfusionMatrix
from baseline.progress import create_progress_bar
from baseline.utils import listify, get_model_file, get_metric_cmp
from baseline.tf.tfy import _add_ema, TRAIN_FLAG
from baseline.tf.optz import optimizer
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.utils import verbose_output
from baseline.model import create_model_for
import numpy as np

NUM_PREFETCH = 2
SHUF_BUF_SZ = 5000


@register_trainer(task='classify', name='default')
class ClassifyTrainerTf(EpochReportingTrainer):

    def __init__(self, model_params, **kwargs):
        super(ClassifyTrainerTf, self).__init__()

        if type(model_params) is dict:
            self.model = create_model_for('classify', **model_params)
        else:
            self.model = model_params

        self.sess = self.model.sess
        self.loss = self.model.create_loss()
        self.test_loss = self.model.create_test_loss()
        self.global_step, train_op = optimizer(self.loss, colocate_gradients_with_ops=True, **kwargs)
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)
        decay = kwargs.get('ema_decay', None)
        if decay is not None:
            self.ema = True
            ema_op, self.ema_load, self.ema_restore = _add_ema(self.model, float(decay))
            with tf.control_dependencies([ema_op]):
                self.train_op = tf.identity(train_op)
        else:
            self.ema = False
            self.train_op = train_op

        tables = tf.tables_initializer()
        self.model.sess.run(tables)
        self.model.sess.run(tf.global_variables_initializer())
        self.model.set_saver(tf.train.Saver())

    @staticmethod
    def _get_batchsz(batch_dict):
        return len(batch_dict['y'])

    def _train(self, loader, **kwargs):

        if self.ema:
            self.sess.run(self.ema_restore)

        reporting_fns = kwargs.get('reporting_fns', [])
        epoch_loss = 0
        epoch_div = 0
        steps = len(loader)
        pg = create_progress_bar(steps)
        for batch_dict in pg(loader):
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

        if self.ema:
            self.sess.run(self.ema_load)

        cm = ConfusionMatrix(self.model.labels)
        steps = len(loader)
        total_loss = 0
        total_norm = 0
        verbose = kwargs.get("verbose", None)

        pg = create_progress_bar(steps)
        for batch_dict in pg(loader):
            y = batch_dict['y']
            feed_dict = self.model.make_input(batch_dict)
            guess, lossv = self.sess.run([self.model.best, self.test_loss], feed_dict=feed_dict)
            batchsz = self._get_batchsz(batch_dict)
            total_loss += lossv * batchsz
            total_norm += batchsz
            cm.add_batch(y, guess)

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss / float(total_norm)
        verbose_output(verbose, cm)

        return metrics

    def checkpoint(self):
        self.model.saver.save(self.sess, "./tf-classify-%d/classify" % os.getpid(), global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-classify-%d" % os.getpid())
        print('Reloading ' + latest)
        self.model.saver.restore(self.model.sess, latest)


def to_tensors(model_params, ts):

    keys = ts[0].keys()

    d = dict((k, []) for k in keys)## if k != 'y')
    for sample in ts:
        # How do I know NC?
        sample['y'] = fill_y(2, sample['y'])
        for k in d.keys():
            # add each sample
            for s in sample[k]:
                d[k].append(s)
    d = dict((k, np.stack(v)) for k, v in d.items())
    return d


@register_training_func('classify')
def fit(model_params, ts, vs, es=None, **kwargs):
    """
    Train a classifier using TensorFlow

    :param model: The model to train
    :param ts: A training data set
    :param vs: A validation data set
    :param es: A test data set, can be None
    :param kwargs:
        See below

    :Keyword Arguments:
        * *do_early_stopping* (``bool``) --
          Stop after evaluation data is no longer improving.  Defaults to True

        * *epochs* (``int``) -- how many epochs.  Default to 20
        * *outfile* -- Model output file, defaults to classifier-model.pyth
        * *patience* --
           How many epochs where evaluation is no longer improving before we give up
        * *reporting* --
           Callbacks which may be used on reporting updates
        * Additional arguments are supported, see :func:`baseline.tf.optimize` for full list
    :return:
    """
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    verbose = kwargs.get('verbose', {'console': kwargs.get('verbose_console', False), 'file': kwargs.get('verbose_file', None)})
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file('classify', 'tf', kwargs.get('basedir'))

    batchsz = kwargs['batchsz']
    ## First, make tf.datasets for ts, vs and es
    d = to_tensors(model_params, ts)
    train_dataset = tf.data.Dataset.from_tensor_slices(d)
    train_dataset = train_dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
    # effective_batch_sz = args.batchsz*args.gpus
    train_dataset = train_dataset.batch(batchsz // kwargs.get('gpus', 1))
    train_dataset = train_dataset.repeat(epochs)
    train_dataset = train_dataset.prefetch(NUM_PREFETCH)
    #train_dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))

    valid_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(model_params, vs))
    valid_dataset = valid_dataset.batch(batchsz // kwargs.get('gpus', 1))
    #valid_dataset = valid_dataset.map(lambda *args: (dict((k, v) for k, v in zip(keys, args[:-1])), args[-1]))
    valid_dataset = valid_dataset.repeat(epochs)
    valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)
    #valid_dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))

    iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

    features = iter.get_next()
    model_params.update(features)
    #model_params.update({'y': y})
    # create the initialisation operations
    train_init_op = iter.make_initializer(train_dataset)
    valid_init_op = iter.make_initializer(valid_dataset)

    ema = True if kwargs.get('ema_decay') is not None else False

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    TRAIN_FLAG()
    trainer = create_trainer(model_params, **kwargs)


    last_improved = 0

    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
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
        print('Reloading best checkpoint')
        trainer.recover_last_checkpoint()
        trainer.test(es, reporting_fns, phase='Test', verbose=verbose)
