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

