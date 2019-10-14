"""Train a classifier with TensorFlow

This module supports 3 different ways of training a model

1. feed_dict
2. datasets
3. datasets + estimators

"""
import six
import os
import time
import logging
import tensorflow as tf

from eight_mile.confusion import ConfusionMatrix
from eight_mile.progress import create_progress_bar
from eight_mile.utils import listify
from eight_mile.tf.optz import optimizer

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


def model_creator(model_params):
    """Create a model function which itself, if called yields a new model

    :param model_params: The parameters from MEAD for a model
    :return: A model function
    """
    def model_fn(features, labels, mode, params):
        """This function is used by estimators to create a model

        :param features: (`dict`) A dictionary of feature names mapped to tensors (iterators)
        :param labels: (`int`) These are the raw labels from file
        :param mode: (`str`): A TF mode including ModeKeys.PREDICT|TRAIN|EVAL
        :param params: A set of user-defined hyper-parameters passed by the estimator
        :return: A new model
        """
        model_params.update(features)
        model_params['sess'] = None
        if labels is not None:
            model_params['y'] = tf.one_hot(tf.reshape(labels, [-1, 1]), len(params['labels']))

        if mode == tf.estimator.ModeKeys.PREDICT:
            SET_TRAIN_FLAG(False)
            model = create_model_for('classify', **model_params)
            predictions = {
                'classes': model.best,
                'probabilities': model.probs,
                'logits': model.logits,
            }
            outputs = tf.estimator.export.PredictOutput(predictions['classes'])
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs={'classes': outputs})

        elif mode == tf.estimator.ModeKeys.EVAL:
            SET_TRAIN_FLAG(False)
            model = create_model_for('classify', **model_params)
            loss = model.create_loss()
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(
                    labels=labels, predictions=model.best)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=model.logits, loss=loss, eval_metric_ops=eval_metric_ops)

        SET_TRAIN_FLAG(True)
        model = create_model_for('classify', **model_params)
        loss = model.create_loss()
        colocate = True if params['gpus'] > 1 else False
        global_step, train_op = optimizer(loss,
                                          optim=params['optim'],
                                          eta=params.get('lr', params.get('eta')),
                                          colocate_gradients_with_ops=colocate)

        return tf.estimator.EstimatorSpec(mode=mode, predictions=model.logits,
                                          loss=loss,
                                          train_op=train_op)
    return model_fn


@register_trainer(task='classify', name='default')
class ClassifyTrainerTf(EpochReportingTrainer):
    """A Trainer to use if not using tf Estimators

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

            batchsz = self._get_batchsz(batch_dict)
            assert len(guess) == batchsz
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
        self.model.saver.save(self.sess, os.path.join(checkpoint_dir, 'classify'), global_step=self.global_step)

    def recover_last_checkpoint(self):
        """Recover the last saved checkpoint

        :return: None
        """
        checkpoint_dir = '{}-{}'.format("./tf-classify", os.getpid())
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('Reloading ' + latest)
        self.model.saver.restore(self.model.sess, latest)


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
    features['lengths'] = features[lengths_key]
    del features[lengths_key]
    y = features.pop('y')
    return features, y


def create_train_input_fn(ts, batchsz=1, gpus=1, **kwargs):
    """Creator function for an estimator to get a train dataset

    We use a closure to encapsulate the outer parameters

    :param ts: The data feed
    :param batchsz: The batchsz to use
    :param gpus: The number of GPUs to use
    :param kwargs: Keyword args
    :return: Return an input function that is suitable for an estimator
    """
    # Precompute this
    lengths_key = kwargs.get('lengths_key')
    tensors = to_tensors(ts, lengths_key)

    def train_input_fn():
        epochs = None
        train_dataset = tf.data.Dataset.from_tensor_slices(tensors)
        train_dataset = train_dataset.shuffle(buffer_size=SHUF_BUF_SZ)
        train_dataset = train_dataset.batch(batchsz // gpus, drop_remainder=False)
        train_dataset = train_dataset.repeat(epochs)
        train_dataset = train_dataset.prefetch(NUM_PREFETCH)
        _ = train_dataset.make_one_shot_iterator()
        return train_dataset
    return train_input_fn


def create_valid_input_fn(vs, batchsz=1, **kwargs):
    """Creator function for an estimator to get a valid dataset

    We use a closure to encapsulate the outer parameters

    :param vs: The data feed
    :param batchsz: The batchsz to use
    :param gpus: The number of GPUs to use
    :param epochs: The number of epochs to train
    :param kwargs: Keyword args
    :return: Return an input function that is suitable for an estimator
    """
    # Precompute this
    lengths_key = kwargs.get('lengths_key')
    tensors = to_tensors(vs, lengths_key)

    def eval_input_fn():
        valid_dataset = tf.data.Dataset.from_tensor_slices(tensors)
        valid_dataset = valid_dataset.batch(batchsz, drop_remainder=False)
        valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)
        _ = valid_dataset.make_one_shot_iterator()
        return valid_dataset

    return eval_input_fn


def create_eval_input_fn(es, test_batchsz=1, **kwargs):
    """Creator function for an estimator to get a test dataset

    We use a closure to encapsulate the outer parameters

    :param es: The data feed
    :param batchsz: The batchsz to use
    :param gpus: The number of GPUs to use
    :param epochs: The number of epochs to train
    :param kwargs: Keyword args
    :return: Return an input function that is suitable for an estimator
    """
    # Precompute this
    lengths_key = kwargs.get('lengths_key')
    tensors = to_tensors(es, lengths_key)

    def predict_input_fn():
        test_dataset = tf.data.Dataset.from_tensor_slices(tensors)
        test_dataset = test_dataset.batch(test_batchsz, drop_remainder=False)
        test_dataset = test_dataset.prefetch(NUM_PREFETCH)
        _ = test_dataset.make_one_shot_iterator()
        return test_dataset

    return predict_input_fn


@register_training_func('classify')
def fit_datasets(model_params, ts, vs, es=None, **kwargs):
    """
    Train a classifier using TensorFlow with `tf.dataset`.  This
    is the default behavior for training.

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
        * *epochs* (``int``) -- how many epochs.  Default to 20
        * *outfile* -- Model output file, defaults to classifier-model.pyth
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

    :return: None
    """
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    verbose = kwargs.get('verbose', {'console': kwargs.get('verbose_console', False), 'file': kwargs.get('verbose_file', None)})
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file('classify', 'tf', kwargs.get('basedir'))

    batchsz = kwargs['batchsz']
    lengths_key = model_params.get('lengths_key')

    ## First, make tf.datasets for ts, vs and es
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
    # effective_batch_sz = args.batchsz*args.gpus
    test_batchsz = kwargs.get('test_batchsz', batchsz)
    train_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(ts, lengths_key))
    train_dataset = train_dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    train_dataset = train_dataset.batch(batchsz, drop_remainder=False)
    train_dataset = train_dataset.repeat(epochs + 1)
    train_dataset = train_dataset.prefetch(NUM_PREFETCH)

    valid_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(vs, lengths_key))
    valid_dataset = valid_dataset.batch(batchsz, drop_remainder=False)
    valid_dataset = valid_dataset.repeat(epochs + 1)
    valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)

    test_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(es, lengths_key))
    test_dataset = test_dataset.batch(test_batchsz, drop_remainder=False)
    test_dataset = test_dataset.repeat(epochs + 1)
    test_dataset = test_dataset.prefetch(NUM_PREFETCH)

    iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

    features, y = iter.get_next()
    # Add features to the model params
    model_params.update(features)
    model_params['y'] = tf.one_hot(tf.reshape(y, [-1, 1]), len(model_params['labels']))
    # create the initialisation operations
    train_init_op = iter.make_initializer(train_dataset)
    valid_init_op = iter.make_initializer(valid_dataset)
    test_init_op = iter.make_initializer(test_dataset)

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
        trainer.sess.run(train_init_op)
        trainer.train(ts, reporting_fns)
        trainer.sess.run(valid_init_op)
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
        trainer.sess.run(test_init_op)
        trainer.test(es, reporting_fns, phase='Test', verbose=verbose)


@register_training_func('classify', 'feed_dict')
def fit(model_params, ts, vs, es=None, **kwargs):
    """
    Train a classifier using TensorFlow with a `feed_dict`.  This
    is the previous default behavior for training.  To use this, you need to pass
    `fit_func: feed_dict` in your MEAD config

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
        * *epochs* (``int``) -- how many epochs.  Default to 20
        * *outfile* -- Model output file, defaults to classifier-model.pyth
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

    :return: None
    """
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    verbose = kwargs.get('verbose', {'console': kwargs.get('verbose_console', False), 'file': kwargs.get('verbose_file', None)})
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file('classify', 'tf', kwargs.get('basedir'))
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

        trainer.train(ts, reporting_fns, dataset=False)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid', dataset=False)

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
        trainer.test(es, reporting_fns, phase='Test', verbose=verbose, dataset=False)


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


@register_training_func('classify', 'estimator')
def fit_estimator(model_params, ts, vs, es=None, epochs=20, gpus=1, **kwargs):
    """Train the model with an `tf.estimator`

    To use this, you pass `fit_func: estimator` in your MEAD config

    This flavor of training utilizes both `tf.dataset`s and `tf.estimator`s to train.
    It is the preferred method for distributed training.

    FIXME: currently this method doesnt support early stopping.

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
        * *epochs* (``int``) -- how many epochs.  Default to 20
        * *outfile* -- Model output file, defaults to classifier-model.pyth
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

    :return: None
    """
    model_fn = model_creator(model_params)
    labels = model_params['labels']
    lengths_key = model_params.get('lengths_key')

    params = {
        'labels': labels,
        'optim': kwargs['optim'],
        'lr': kwargs.get('lr', kwargs.get('eta')),
        'epochs': epochs,
        'gpus': gpus,
        'batchsz': kwargs['batchsz'],
        'lengths_key': lengths_key,
        'test_batchsz': kwargs.get('test_batchsz', kwargs.get('batchsz'))
    }

    checkpoint_dir = '{}-{}'.format("./tf-classify", os.getpid())
    # We are only distributing the train function for now
    # https://stackoverflow.com/questions/52097928/does-tf-estimator-estimator-evaluate-always-run-on-one-gpu
    config = tf.estimator.RunConfig(model_dir=checkpoint_dir,
                                    train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=gpus))
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=params)

    train_input_fn = create_train_input_fn(ts, **params)
    valid_input_fn = create_valid_input_fn(vs, **params)
    predict_input_fn = create_eval_input_fn(es, **params)

    # This is going to be None because train_and_evaluate controls the max steps so repeat doesnt matter
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=epochs * len(ts))
    # This is going to be None because the evaluation will run for 1 pass over the data that way
    eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, steps=None)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    y_test = [sample['y'] for sample in es]
    start = time.time()
    predictions = np.array([p['classes'] for p in estimator.predict(input_fn=predict_input_fn)])

    cm = ConfusionMatrix(labels)
    for truth, guess in zip(y_test, predictions):
        cm.add(truth, guess)

    metrics = cm.get_all_metrics()

    reporting_fns = listify(kwargs.get('reporting', []))
    _report(0, metrics, start, 'Test', "EPOCH", reporting_fns)

    verbose = kwargs.get('verbose')
    verbose_output(verbose, cm)

    return metrics
