import six
import os
import time
import logging
import tensorflow as tf

from eight_mile.progress import create_progress_bar
from eight_mile.utils import listify, get_version, Offsets
from eight_mile.metrics import LCM, UCM, LAS, UAS
from eight_mile.tf.layers import SET_TRAIN_FLAG, get_shape_as_list, autograph_options, masked_fill
from eight_mile.tf.optz import *
from baseline.utils import get_model_file, get_metric_cmp
from baseline.train import EpochReportingTrainer, register_trainer, register_training_func
from baseline.model import create_model_for
import numpy as np

# Number of batches to prefetch if using tf.datasets
NUM_PREFETCH = 2
# The shuffle buffer
SHUF_BUF_SZ = 5000

log = logging.getLogger('baseline.timing')

TF_VERSION = get_version(tf)
if TF_VERSION < 2:
    tf.enable_eager_execution()


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
    heads = features.pop('heads')
    labels = features.pop('labels')
    return features, (heads, labels)


def arc_label_loss(model, x, y):

    arcs_pred, labels_pred = model(x)
    arcs_gold, labels_gold = y
    # First, trim gold labels to length of the input
    B, T = get_shape_as_list(labels_pred)[:2]
    flat_labels_pred = tf.reshape(labels_pred, (B*T, T, -1))
    flat_arcs_pred = tf.reshape(arcs_pred, (B*T, T))
    flat_labels_gold = tf.reshape(labels_gold[:, :T], [-1])
    flat_arcs_gold = tf.reshape(arcs_gold[:, :T], [-1])

    idx = tf.cast(tf.range(get_shape_as_list(flat_arcs_gold)[0]), dtype=tf.int64)
    indices = tf.stack([idx, flat_arcs_gold], 1)
    flat_labels_gather = tf.gather_nd(flat_labels_pred, indices)

    arc_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(logits=flat_arcs_pred, labels=flat_arcs_gold)
    rel_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(logits=flat_labels_gather, labels=flat_labels_gold)

    return arc_loss + rel_loss


@register_trainer(task='deps')
class DependencyParserTrainerEagerTf(EpochReportingTrainer):
    """A Trainer to use if using TF2.0
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
            self.model = create_model_for('deps', **model_params)
        else:
            self.model = model_params

        self.optimizer = EagerOptimizer(arc_label_loss, **kwargs)
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)
        self.punct_eval = kwargs.get('punct_eval', False)
        self._checkpoint = tf.train.Checkpoint(optimizer=self.optimizer.optimizer, model=self.model)
        checkpoint_dir = '{}-{}'.format("./tf-deps", os.getpid())

        self.checkpoint_manager = tf.train.CheckpointManager(self._checkpoint,
                                                             directory=checkpoint_dir,
                                                             max_to_keep=5)

    def _train(self, loader, steps=0, **kwargs):
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

        SET_TRAIN_FLAG(True)
        reporting_fns = kwargs.get('reporting_fns', [])
        pg = create_progress_bar(steps)
        epoch_loss = tf.Variable(0.0)
        epoch_div = tf.Variable(0, dtype=tf.int32)
        nstep_loss = tf.Variable(0.0)
        nstep_div = tf.Variable(0, dtype=tf.int32)
        self.nstep_start = time.perf_counter()

        @tf.function
        def _train_step(inputs):
            """Replicated training step."""
            features, y = inputs
            loss = self.optimizer.update(self.model, features, y)
            batchsz = len(y[0])
            report_loss = loss * batchsz
            return report_loss, batchsz

        #with autograph_options({"function_optimization": False, "layout_optimizer": False}):

        for inputs in pg(loader):
            step_report_loss, step_batchsz = _train_step(inputs)
            epoch_loss.assign_add(step_report_loss)
            nstep_loss.assign_add(step_report_loss)
            epoch_div.assign_add(step_batchsz)
            nstep_div.assign_add(step_batchsz)
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
        return metrics

    def _test(self, loader, steps=0, **kwargs):
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

        metrics = [LAS(), UAS(), LCM(), UCM()]

        pg = create_progress_bar(steps)

        SET_TRAIN_FLAG(False)
        for features, y in pg(loader):
            heads_gold, labels_gold = y
            greedy_heads_pred, greedy_labels_pred = self.model.decode(features)
            B, T = get_shape_as_list(greedy_labels_pred)[:2]
            labels_gold_trimmed = labels_gold[:, :T].numpy()
            heads_gold_trimmed = heads_gold[:, :T].numpy()

            for i in range(B):
                for m in metrics:
                    if self.punct_eval is False:
                        labels_gold_trimmed[i] = masked_fill(labels_gold_trimmed[i],
                                                             labels_gold_trimmed[i] == self.model.punct, Offsets.PAD)
                    m.add(greedy_heads_pred[i], heads_gold_trimmed[i], greedy_labels_pred[i], labels_gold_trimmed[i])

        metrics = {m.name: m.score for m in metrics}
        return metrics

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


@register_training_func('deps', name='default')
def fit_eager(model_params, ts, vs, es=None, **kwargs):

    """
    Train a dependency parser using TensorFlow with `tf.dataset`.  This
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

    test_batchsz = kwargs.get('test_batchsz', batchsz)
    train_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(ts, lengths_key))
    train_dataset = train_dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    train_dataset = train_dataset.batch(batchsz, drop_remainder=False)
    train_dataset = train_dataset.prefetch(NUM_PREFETCH)

    valid_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(vs, lengths_key))
    valid_dataset = valid_dataset.batch(batchsz, drop_remainder=False)
    valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)
    SET_TRAIN_FLAG(True)
    trainer = DependencyParserTrainerEagerTf(model_params, **kwargs)
    last_improved = 0

    for epoch in range(epochs):

        trainer.train(train_dataset, reporting_fns, steps=len(ts))
        test_metrics = trainer.test(valid_dataset, reporting_fns, phase='Valid', steps=len(vs))

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
        test_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(es, lengths_key))
        test_dataset = test_dataset.batch(test_batchsz, drop_remainder=False)
        test_dataset = test_dataset.prefetch(NUM_PREFETCH)
        trainer.test(test_dataset, reporting_fns, phase='Test', verbose=verbose, steps=len(es))
