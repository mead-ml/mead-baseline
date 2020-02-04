import six
import os
import time
import logging
import tensorflow as tf

from eight_mile.confusion import ConfusionMatrix
from eight_mile.utils import listify
from eight_mile.tf.optz import optimizer

from baseline.utils import get_model_file, get_metric_cmp
from baseline.tf.tfy import _add_ema, TRAIN_FLAG, SET_TRAIN_FLAG
from baseline.tf.classify.training.utils import to_tensors, _report
from baseline.train import register_training_func
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
