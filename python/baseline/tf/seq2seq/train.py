import os
import time
import numpy as np
import tensorflow as tf
from eight_mile.tf.optz import optimizer
from eight_mile.progress import create_progress_bar
from eight_mile.utils import listify
from eight_mile.bleu import bleu

from baseline.utils import (
    get_model_file,
    get_metric_cmp,
    convert_seq2seq_golds,
    convert_seq2seq_preds,
)

from baseline.train import Trainer, create_trainer, register_trainer, register_training_func
from baseline.tf.tfy import TRAIN_FLAG, SET_TRAIN_FLAG

from baseline.model import create_model_for


# Number of batches to prefetch if using tf.datasets
NUM_PREFETCH = 2
# The shuffle buffer
SHUF_BUF_SZ = 5000


def to_tensors(ts, src_lengths_key):
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
    keys = [k for k in keys if '_lengths' not in k and k != 'ids'] + [src_lengths_key, "tgt_lengths"]

    features = dict((k, []) for k in keys)

    for sample in ts:
        for k in features.keys():
            for s in sample[k]:
                features[k].append(s)

    features['src_len'] = features[src_lengths_key]
    del features[src_lengths_key]
    features['tgt_len'] = features['tgt_lengths']
    del features['tgt_lengths']
    features = dict((k, np.stack(v).astype(np.int32)) for k, v in features.items())
    tgt = features.pop('tgt')
    return features, tgt


@register_trainer(task='seq2seq', name='default')
class Seq2SeqTrainerTf(Trainer):
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
          * *tgt_rlut* (`dict`) -- This is a dictionary that converts from ints back to strings, used for predictions
          * *beam* (`int`) -- The beam size to use at prediction time, defaults to `10`

        """
        super(Seq2SeqTrainerTf, self).__init__()
        if type(model_params) is dict:
            self.model = create_model_for('seq2seq', **model_params)
        else:
            self.model = model_params
        self.sess = self.model.sess
        self.loss = self.model.create_loss()
        self.test_loss = self.model.create_test_loss()
        self.tgt_rlut = kwargs['tgt_rlut']
        self.base_dir = kwargs['basedir']
        self.global_step, self.train_op = optimizer(self.loss, colocate_gradients_with_ops=True, **kwargs)
        self.nsteps = kwargs.get('nsteps', 500)
        self.beam = kwargs.get('beam', 10)
        tables = tf.tables_initializer()
        self.model.sess.run(tables)
        self.model.sess.run(tf.global_variables_initializer())
        self.model.set_saver(tf.train.Saver())

        init = tf.global_variables_initializer()
        self.model.sess.run(init)

    def checkpoint(self):
        """This method saves a checkpoint

        :return: None
        """
        checkpoint_dir = '{}-{}'.format("./tf-seq2seq", os.getpid())
        self.model.saver.save(self.sess, os.path.join(checkpoint_dir, 'seq2seq'), global_step=self.global_step)

    def recover_last_checkpoint(self):
        """Recover the last saved checkpoint

        :return: None
        """
        checkpoint_dir = '{}-{}'.format("./tf-seq2seq", os.getpid())
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        self.model.saver.restore(self.model.sess, latest)
        #print('Reloading ' + latest)
        #g = tf.Graph()
        #with g.as_default():
        #    SET_TRAIN_FLAG(None)
        #    sess = tf.Session()
        #    self.model = self.model.load(latest, predict=True, beam=self.beam, session=sess)

    def _num_toks(self, lens):
        return np.sum(lens)

    def calc_metrics(self, agg, norm):
        """Calculate metrics

        :param agg: The aggregated loss
        :param norm: The number of steps to average over
        :return: The metrics
        """
        metrics = super(Seq2SeqTrainerTf, self).calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics

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
        epoch_loss = 0
        epoch_toks = 0

        start = time.time()
        self.nstep_start = start
        for batch_dict in ts:
            if dataset:
                _, global_step, lossv = self.sess.run([self.train_op, self.global_step, self.loss],
                                                      feed_dict={TRAIN_FLAG(): 1})
            else:
                feed_dict = self.model.make_input(batch_dict, True)
                _, global_step, lossv = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)

            # ?? How to get this cleaner?
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

    def _evaluate(self, es, reporting_fns):
        """Run the model with beam search and report Bleu.

        :param es: `DataFeed` of input
        :param reporting_fns: Input hooks
        """
        pg = create_progress_bar(len(es))
        preds = []
        golds = []
        start = time.time()
        for batch_dict in pg(es):
            tgt = batch_dict.pop('tgt')
            tgt_lens = batch_dict.pop('tgt_lengths')
            pred = [p[0] for p in self.model.predict(batch_dict)]
            preds.extend(convert_seq2seq_preds(pred, self.tgt_rlut))
            golds.extend(convert_seq2seq_golds(tgt, tgt_lens, self.tgt_rlut))
        metrics = {'bleu': bleu(preds, golds)[0]}
        self.report(
            0, metrics, start, 'Test', 'EPOCH', reporting_fns
        )
        return metrics

    def test(self, vs, reporting_fns, phase='Valid', dataset=True):
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
        if phase == 'Test' and not dataset:
            return self._evaluate(vs, reporting_fns)
        self.valid_epochs += 1

        total_loss = 0
        total_toks = 0
        preds = []
        golds = []

        start = time.time()
        pg = create_progress_bar(len(vs))
        for batch_dict in pg(vs):

            if dataset:
                lossv, top_preds = self.model.sess.run([self.test_loss, self.model.decoder.best])
            else:
                feed_dict = self.model.make_input(batch_dict)
                lossv, top_preds = self.model.sess.run([self.test_loss, self.model.decoder.best], feed_dict=feed_dict)
            toks = self._num_toks(batch_dict['tgt_lengths'])
            total_loss += lossv * toks
            total_toks += toks

            preds.extend(convert_seq2seq_preds(top_preds.T, self.tgt_rlut))
            golds.extend(convert_seq2seq_golds(batch_dict['tgt'], batch_dict['tgt_lengths'], self.tgt_rlut))

        metrics = self.calc_metrics(total_loss, total_toks)
        metrics['bleu'] = bleu(preds, golds)[0]
        self.report(
            self.valid_epochs, metrics, start,
            phase, 'EPOCH', reporting_fns
        )
        return metrics


@register_training_func('seq2seq', 'feed_dict')
def fit(model_params, ts, vs, es=None, **kwargs):
    """
    Train an encoder-decoder network using TensorFlow with a `feed_dict`.

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
    epochs = int(kwargs.get('epochs', 5))
    patience = int(kwargs.get('patience', epochs))
    model_file = get_model_file('seq2seq', 'tf', kwargs.get('basedir'))

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'bleu')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = create_trainer(model_params, **kwargs)

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


@register_training_func('seq2seq', 'dataset')
def fit_datasets(model_params, ts, vs, es=None, **kwargs):
    """
    Train an encoder-decoder network using TensorFlow with `tf.dataset`.  This
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
    model_file = get_model_file('seq2seq', 'tf', kwargs.get('basedir'))
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'bleu')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    batchsz = kwargs['batchsz']
    ## First, make tf.datasets for ts, vs and es
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
    # effective_batch_sz = args.batchsz*args.gpus
    test_batchsz = kwargs.get('test_batchsz', batchsz)
    src_lengths_key = model_params.get('src_lengths_key')
    train_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(ts, src_lengths_key))
    train_dataset = train_dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    train_dataset = train_dataset.batch(batchsz, drop_remainder=False)
    train_dataset = train_dataset.repeat(epochs + 1)
    train_dataset = train_dataset.prefetch(NUM_PREFETCH)

    valid_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(vs, src_lengths_key))
    valid_dataset = valid_dataset.batch(batchsz, drop_remainder=False)
    valid_dataset = valid_dataset.repeat(epochs + 1)
    valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)

    test_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(es, src_lengths_key))
    test_dataset = test_dataset.batch(test_batchsz, drop_remainder=False)
    test_dataset = test_dataset.repeat(epochs + 1)
    test_dataset = test_dataset.prefetch(NUM_PREFETCH)

    iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

    features, tgt = iter.get_next()
    # Add features to the model params
    model_params.update(features)
    # This is kind of crazy, but seems to work, hardwire a graph op for `mx_tgt_len`
    model_params.update({'tgt': tgt, 'mx_tgt_len': tf.reduce_max(features['tgt_len'])})

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


def create_train_input_fn(ts, src_lengths_key, batchsz=1, gpus=1, **kwargs):
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
    tensors = to_tensors(ts, src_lengths_key)

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


def create_valid_input_fn(vs, src_lengths_key, batchsz=1, **kwargs):
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
    tensors = to_tensors(vs, src_lengths_key)

    def eval_input_fn():
        valid_dataset = tf.data.Dataset.from_tensor_slices(tensors)
        valid_dataset = valid_dataset.batch(batchsz, drop_remainder=False)
        valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)
        _ = valid_dataset.make_one_shot_iterator()
        return valid_dataset

    return eval_input_fn


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

        # This is kind of crazy, but seems to work, hardwire a graph op for `mx_tgt_len`
        model_params.update({'tgt': labels, 'mx_tgt_len': tf.reduce_max(features['tgt_len'])})

        if mode == tf.estimator.ModeKeys.EVAL:
            SET_TRAIN_FLAG(False)
            model = create_model_for('seq2seq', **model_params)
            loss = model.create_loss()
            return tf.estimator.EstimatorSpec(mode=mode, predictions=model.decoder.best, loss=loss)

        SET_TRAIN_FLAG(True)
        model = create_model_for('seq2seq', **model_params)
        loss = model.create_loss()
        colocate = True if params['gpus'] > 1 else False
        global_step, train_op = optimizer(loss,
                                          optim=params['optim'],
                                          eta=params.get('lr', params.get('eta')),
                                          colocate_gradients_with_ops=colocate)

        return tf.estimator.EstimatorSpec(mode=mode, predictions=model.decoder.best,
                                          loss=loss,
                                          train_op=train_op)
    return model_fn


@register_training_func('seq2seq') ##, 'estimator')
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
    #labels = model_params['labels']
    params = {
        #'labels': labels,
        'optim': kwargs['optim'],
        'lr': kwargs.get('lr', kwargs.get('eta')),
        'epochs': epochs,
        'gpus': gpus,
        'batchsz': kwargs['batchsz'],
        'test_batchsz': kwargs.get('test_batchsz', kwargs.get('batchsz'))
    }
    print(params)
    checkpoint_dir = '{}-{}'.format("./tf-seq2seq", os.getpid())
    # We are only distributing the train function for now
    # https://stackoverflow.com/questions/52097928/does-tf-estimator-estimator-evaluate-always-run-on-one-gpu
    config = tf.estimator.RunConfig(model_dir=checkpoint_dir,
                                    train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=gpus))
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=params)
    src_lengths_key = model_params.get('src_lengths_key')
    train_input_fn = create_train_input_fn(ts, src_lengths_key, **params)
    valid_input_fn = create_valid_input_fn(vs, src_lengths_key, **params)
    ##predict_input_fn = create_eval_input_fn(es, **params)

    # This is going to be None because train_and_evaluate controls the max steps so repeat doesnt matter
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=epochs * len(ts))
    # This is going to be None because the evaluation will run for 1 pass over the data that way
    eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, steps=None)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
