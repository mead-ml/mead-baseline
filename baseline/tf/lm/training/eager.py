import os
import numpy as np
import time
import tensorflow as tf
from eight_mile.utils import listify, get_version
from eight_mile.tf.layers import SET_TRAIN_FLAG, get_shape_as_list
from eight_mile.tf.optz import EagerOptimizer
from baseline.utils import get_model_file, get_metric_cmp
from baseline.model import create_model_for
from baseline.train import register_training_func, Trainer
from baseline.tf.lm.training.utils import to_tensors, SHUF_BUF_SZ, NUM_PREFETCH


def loss_with_state(model, h, x, y):
    logits, h_out = model(x, h)
    vsz = model.vsz
    targets = tf.reshape(y, [-1])
    bt_x_v = tf.nn.log_softmax(tf.reshape(logits, [-1, vsz]), axis=-1)
    one_hots = tf.one_hot(targets, vsz)
    example_loss = -tf.reduce_sum(one_hots * bt_x_v, axis=-1)
    loss = tf.reduce_mean(example_loss)
    return loss, h_out


def loss_without_state(model, x, y):
    # Model will produce a null hidden state
    logits = model(x, None)[0]
    vsz = model.vsz
    targets = tf.reshape(y, [-1])
    bt_x_v = tf.nn.log_softmax(tf.reshape(logits, [-1, vsz]), axis=-1)
    one_hots = tf.one_hot(targets, vsz)
    example_loss = -tf.reduce_sum(one_hots * bt_x_v, axis=-1)
    loss = tf.reduce_mean(example_loss)
    return loss


class LanguageModelTrainerEagerTf(Trainer):
    """A Trainer to use for eager mode

    """
    def __init__(self, model_params, **kwargs):
        super().__init__()

        if type(model_params) is dict:
            self.model = create_model_for('lm', **model_params)
        else:
            self.model = model_params

        loss_fn = loss_with_state if self.model.requires_state else loss_without_state
        self.optimizer = EagerOptimizer(loss_fn, **kwargs)
        self.nsteps = kwargs.get('nsteps', 500)
        self._checkpoint = tf.train.Checkpoint(optimizer=self.optimizer.optimizer, model=self.model)
        checkpoint_dir = '{}-{}'.format("./tf-lm", os.getpid())

        self.checkpoint_manager = tf.train.CheckpointManager(self._checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=5)

    def checkpoint(self):
        """This method saves a checkpoint

        :return: None
        """
        self.checkpoint_manager.save()

    def recover_last_checkpoint(self):
        """Recover the last saved checkpoint

        :return: None
        """
        print(self._checkpoint.restore(self.checkpoint_manager.latest_checkpoint))

    @staticmethod
    def _num_toks(y):
        return tf.reduce_prod(get_shape_as_list(y))

    def train(self, ts, reporting_fns):
        """Train by looping over the steps

        For a `tf.dataset`-backed `fit_func`, we are using the previously wired `dataset`s
        in the model (and `dataset` is `True`).  For `feed_dict`, we convert the ts samples
        to `feed_dict`s and hand them in one-by-one

        :param ts: The training set
        :param reporting_fns: A list of reporting hooks
        :param dataset: (`bool`) Are we using `tf.dataset`s
        :return: Metrics
        """

        SET_TRAIN_FLAG(True)
        epoch_loss = tf.Variable(0.0)
        epoch_div = tf.Variable(0, dtype=tf.int32)
        nstep_loss = tf.Variable(0.0)
        nstep_div = tf.Variable(0, dtype=tf.int32)
        self.nstep_start = time.perf_counter()
        start = time.perf_counter()

        def _train_step_no_state(inputs):
            """Replicated training step."""

            features, y = inputs
            loss = self.optimizer.update(self.model, features, y)
            toks = self._num_toks(y)
            report_loss = loss * tf.cast(toks, tf.float32)
            return report_loss, toks

        def _train_step_with_state(inputs, hidden):
            """Replicated training step."""

            features, y = inputs
            loss, hidden = self.optimizer.update_with_hidden(self.model, hidden, features, y)
            toks = self._num_toks(y)
            report_loss = loss * tf.cast(toks, tf.float32)
            return hidden, report_loss, toks

        if get_version(tf) >= 2:
            _train_step_with_state = tf.function(_train_step_with_state)
            _train_step_no_state = tf.function(_train_step_no_state)

        h = None
        for inputs in ts:
            if self.model.requires_state:
                h, step_report_loss, step_toks = _train_step_with_state(inputs, h)
            else:
                step_report_loss, step_toks = _train_step_no_state(inputs)

            epoch_loss.assign_add(step_report_loss)
            nstep_loss.assign_add(step_report_loss)
            epoch_div.assign_add(step_toks)
            nstep_div.assign_add(step_toks)

            step = self.optimizer.global_step.numpy() + 1
            if step % self.nsteps == 0:
                metrics = self.calc_metrics(nstep_loss.numpy(), nstep_div.numpy())
                self.report(
                    step, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                nstep_loss.assign(0.0)
                nstep_div.assign(0)
                self.nstep_start = time.perf_counter()

        epoch_loss = epoch_loss.numpy()
        epoch_div = epoch_div.numpy()
        metrics = self.calc_metrics(epoch_loss, epoch_div)
        self.train_epochs += 1
        self.report(
            self.train_epochs, metrics, start,
            'Train', 'EPOCH', reporting_fns
        )
        return metrics

    def calc_metrics(self, agg, norm):
        metrics = super().calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics

    def test(self, vs, reporting_fns, phase):
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
        SET_TRAIN_FLAG(False)

        start = time.perf_counter()
        h = None
        for features, y in vs:
            if self.model.requires_state:
                loss_value, h = loss_with_state(self.model, h, features, y)
            else:
                loss_value = loss_without_state(self.model, features, y)
            loss_value = loss_value.numpy()
            toks = self._num_toks(y)
            total_loss += loss_value * tf.cast(toks, tf.float32).numpy()
            total_toks += toks.numpy()

        metrics = self.calc_metrics(total_loss, total_toks)
        self.report(
            epochs, metrics, start,
            phase, 'EPOCH', reporting_fns
        )
        return metrics


@register_training_func('lm')
def fit_eager(model_params, ts, vs, es=None, **kwargs):
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
    train_dataset = train_dataset.prefetch(NUM_PREFETCH)

    valid_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(vs))
    valid_dataset = valid_dataset.batch(batchsz, drop_remainder=False)
    valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)

    trainer = LanguageModelTrainerEagerTf(model_params, **kwargs)
    last_improved = 0
    SET_TRAIN_FLAG(True)

    for epoch in range(epochs):

        trainer.train(train_dataset, reporting_fns)
        test_metrics = trainer.test(valid_dataset, reporting_fns, phase='Valid')

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
        test_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(es))
        test_dataset = test_dataset.batch(test_batchsz, drop_remainder=False)
        test_dataset = test_dataset.prefetch(NUM_PREFETCH)
        trainer.test(test_dataset, reporting_fns, phase='Test')

