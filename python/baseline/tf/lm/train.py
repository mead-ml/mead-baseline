import os
import time
import logging
import numpy as np
import tensorflow as tf
from eight_mile.tf.optz import optimizer
from baseline.tf.tfy import TRAIN_FLAG, SET_TRAIN_FLAG
from eight_mile.utils import listify
from baseline.utils import get_model_file, get_metric_cmp
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func
from baseline.model import create_model_for
from collections import OrderedDict

# Number of batches to prefetch if using tf.datasets
NUM_PREFETCH = 2
# The shuffle buffer
SHUF_BUF_SZ = 5000

_EVENT_FILE_GLOB_PATTERN = 'events.out.tfevents.*'


def _summaries(eval_dir):
    """Yields `tensorflow.Event` protos from event files in the eval dir.
    Args:
      eval_dir: Directory containing summary files with eval metrics.
    Yields:
      `tensorflow.Event` object read from the event files.
    """
    if tf.gfile.Exists(eval_dir):
        for event_file in tf.gfile.Glob(os.path.join(eval_dir, _EVENT_FILE_GLOB_PATTERN)):
            for event in tf.train.summary_iterator(event_file):
                yield event


def read_eval_metrics(eval_dir):
    """Helper to read eval metrics from eval summary files.
    Args:
      eval_dir: Directory containing summary files with eval metrics.
    Returns:
      A `dict` with global steps mapping to `dict` of metric names and values.
    """
    eval_metrics_dict = {}
    for event in _summaries(eval_dir):
        if not event.HasField('summary'):
            continue
        metrics = {}
        for value in event.summary.value:
            if value.HasField('simple_value'):
                metrics[value.tag] = value.simple_value
        if metrics:
            eval_metrics_dict[event.step] = metrics
    return OrderedDict(sorted(eval_metrics_dict.items(), key=lambda t: t[0]))


class EvalMetricsHook(tf.train.SessionRunHook):
    def __init__(self, eval_dir, phase):
        super(EvalMetricsHook, self).__init__()

        self.phase = phase
        self.eval_dir = eval_dir
        self.record = {}

    def after_run(self, run_context, run_values):
        metrics = read_eval_metrics(self.eval_dir)
        if metrics:
            for k, v in metrics.items():
                if k not in self.record:
                    print({'STEP': k, 'loss': v['loss'], 'perplexity': np.exp(v['loss']), 'phase': self.phase})
                    print({'STEP': k, 'loss': v['loss'], 'perplexity': np.exp(v['loss']), 'phase': self.phase})
                    self.record[k] = v

        #    perplexity = np.exp(loss)
        #    print({"loss": loss, "perplexity": perplexity})


def to_tensors(ts):
    """Convert a data feed into a tuple of `features` (`dict`) and `y` values

    This method is required to produce `tf.dataset`s from the input data feed.
    Any fields ending with `_lengths` are ignored, unless they match the
    `src_lengths_key` or `tgt_lengths_key`, in which case, they are converted to `src_len` and `tgt_len`

    :param ts: The data feed to convert
    :param lengths_key: This is a field passed from the model params specifying source of truth of the temporal lengths
    :return: A `tuple` of `features` and `y` (labels)
    """
    keys = ts[0].keys()
    # This is kind of a hack
    keys = [k for k in keys if k != 'ids']

    features = dict((k, []) for k in keys)

    for sample in ts:
        for k in features.keys():
            for s in sample[k]:
                features[k].append(s)

    features = dict((k, np.stack(v).astype(np.int32)) for k, v in features.items())
    tgt = features.pop('y')
    return features, tgt


@register_trainer(task='lm', name='default')
class LanguageModelTrainerTf(Trainer):
    """A Trainer to use if not using tf Estimators

    The trainer can run in 2 modes: `dataset` and `feed_dict`.  When the former, the graph is assumed to
    be connected by features attached to the input so the `feed_dict` will only be used to pass dropout information.

    When the latter, we will use the baseline DataFeed to read the object into the `feed_dict`
    """
    def __init__(self, model_params, **kwargs):
        super(LanguageModelTrainerTf, self).__init__()
        if type(model_params) is dict:
            self.model = create_model_for('lm', **model_params)
        else:
            self.model = model_params
        self.sess = self.model.sess
        self.loss = self.model.create_loss()
        self.test_loss = self.model.create_test_loss()
        self.global_step, self.train_op = optimizer(self.loss, **kwargs)
        self.nsteps = kwargs.get('nsteps', 500)
        init = tf.global_variables_initializer()
        self.model.sess.run(init)
        saver = tf.train.Saver()
        self.model.set_saver(saver)

    def checkpoint(self):
        """This method saves a checkpoint

        :return: None
        """
        checkpoint_dir = '{}-{}'.format("./tf-lm", os.getpid())
        self.model.saver.save(self.sess, os.path.join(checkpoint_dir, 'lm'), global_step=self.global_step)

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

    def train(self, ts, reporting_fns, dataset=True):
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

            if dataset:
                feed_dict = {TRAIN_FLAG(): 1}
            else:
                feed_dict = self.model.make_input(batch_dict, True)
                _, global_step, lossv = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)

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

    def test(self, vs, reporting_fns, phase, dataset=True):
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
        xfer_state = hasattr(self.model, 'initial_state')

        if xfer_state:
            state = self.model.sess.run(self.model.initial_state, self.model.make_input(vs[0], False))

        fetches = {
            "loss": self.test_loss,
        }

        if xfer_state:
            fetches["final_state"] = self.model.final_state

        start = time.time()

        for batch_dict in vs:
            feed_dict = {}
            if not dataset:
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


@register_training_func('lm', 'feed_dict')
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

        trainer.train(ts, reporting_fns, dataset=False)
        if after_train_fn is not None:
            after_train_fn(trainer.model)

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
        trainer.recover_last_checkpoint()
        trainer.test(es, reporting_fns, phase='Test', dataset=False)


@register_training_func('lm')
def fit_datasets(model_params, ts, vs, es=None, **kwargs):
    """
    Train an language model using TensorFlow with `tf.dataset`.  This
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

    epochs = int(kwargs.get('epochs', 5))
    patience = int(kwargs.get('patience', epochs))

    model_file = get_model_file('lm', 'tf', kwargs.get('basedir'))

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'avg_loss')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    batchsz = kwargs['batchsz']
    test_batchsz = kwargs.get('test_batchsz', batchsz)
    tgt_key = model_params.get('tgt_key')

    train_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(ts))
    train_dataset = train_dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    train_dataset = train_dataset.batch(batchsz, drop_remainder=False)
    train_dataset = train_dataset.repeat(epochs + 1)
    train_dataset = train_dataset.prefetch(NUM_PREFETCH)

    valid_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(vs))
    valid_dataset = valid_dataset.batch(batchsz, drop_remainder=False)
    valid_dataset = valid_dataset.repeat(epochs + 1)
    valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)

    test_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(es))
    test_dataset = test_dataset.batch(test_batchsz, drop_remainder=False)
    test_dataset = test_dataset.repeat(epochs + 1)
    test_dataset = test_dataset.prefetch(NUM_PREFETCH)

    iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

    features, tgt = iter.get_next()
    # Add features to the model params
    model_params.update(features)
    model_params.update({'y': tgt})

    # create the initialization operations
    train_init_op = iter.make_initializer(train_dataset)
    valid_init_op = iter.make_initializer(valid_dataset)
    test_init_op = iter.make_initializer(test_dataset)

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
        trainer.test(es, reporting_fns, phase='Test')


def create_train_input_fn(ts, batchsz=1, gpus=1, **kwargs):
    """Creator function for an estimator to get a train dataset

    We use a closure to encapsulate the outer parameters

    :param ts: The data feed
    :param src_lengths_key: The key identifying the data feed field corresponding to length
    :param batchsz: The batchsz to use
    :param gpus: The number of GPUs to use
    :param kwargs: Keyword args
    :return: Return an input function that is suitable for an estimator
    """
    # Precompute this
    tensors = to_tensors(ts)

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
    :param src_lengths_key: The key identifying the data feed field corresponding to length
    :param batchsz: The batchsz to use
    :param gpus: The number of GPUs to use
    :param epochs: The number of epochs to train
    :param kwargs: Keyword args
    :return: Return an input function that is suitable for an estimator
    """
    # Precompute this
    tensors = to_tensors(vs)

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
    tensors = to_tensors(es)

    def predict_input_fn():
        test_dataset = tf.data.Dataset.from_tensor_slices(tensors)
        test_dataset = test_dataset.batch(test_batchsz, drop_remainder=False)
        test_dataset = test_dataset.prefetch(NUM_PREFETCH)
        _ = test_dataset.make_one_shot_iterator()
        return test_dataset

    return predict_input_fn


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
        model_params.update({'y': labels})
        model_params['sess'] = None

        if mode == tf.estimator.ModeKeys.EVAL:
            SET_TRAIN_FLAG(False)
            model = create_model_for('lm', **model_params)
            loss = model.create_loss()
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

        SET_TRAIN_FLAG(True)
        model = create_model_for('lm', **model_params)
        loss = model.create_loss()
        colocate = True if params['gpus'] > 1 else False
        global_step, train_op = optimizer(loss,
                                          optim=params['optim'],
                                          eta=params.get('lr', params.get('eta')),
                                          colocate_gradients_with_ops=colocate)

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
    return model_fn


@register_training_func('lm', 'estimator')
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
    params = {
        'optim': kwargs['optim'],
        'lr': kwargs.get('lr', kwargs.get('eta')),
        'epochs': epochs,
        'gpus': gpus,
        'batchsz': kwargs['batchsz'],
        'test_batchsz': kwargs.get('test_batchsz', kwargs.get('batchsz'))
    }
    print(params)
    checkpoint_dir = '{}-{}'.format("./tf-lm", os.getpid())
    # We are only distributing the train function for now
    # https://stackoverflow.com/questions/52097928/does-tf-estimator-estimator-evaluate-always-run-on-one-gpu
    config = tf.estimator.RunConfig(model_dir=checkpoint_dir,
                                    train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=gpus),
                                    log_step_count_steps=500)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=params)
    train_input_fn = create_train_input_fn(ts, **params)
    valid_input_fn = create_valid_input_fn(vs, **params)
    eval_input_fn = create_eval_input_fn(es, **params)

    #valid_metrics = EvalMetricsHook(estimator.eval_dir(), 'Valid')
    # This is going to be None because train_and_evaluate controls the max steps so repeat doesnt matter
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=epochs * len(ts))
    # This is going to be None because the evaluation will run for 1 pass over the data that way
    eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, steps=None)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    test_metrics = EvalMetricsHook(estimator.eval_dir(), 'Test')
    estimator.evaluate(input_fn=eval_input_fn, hooks=[test_metrics])
