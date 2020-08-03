import six
import os
import time
from itertools import zip_longest
import numpy as np
import tensorflow as tf
import logging
from eight_mile.utils import listify, revlut, to_spans, write_sentence_conll, per_entity_f1, span_f1, conlleval_output, Offsets
from eight_mile.tf.layers import TRAIN_FLAG, SET_TRAIN_FLAG, reload_checkpoint, get_shape_as_list, autograph_options
from eight_mile.tf.optz import EagerOptimizer
from baseline.progress import create_progress_bar
from baseline.model import create_model_for
from baseline.train import register_training_func, EpochReportingTrainer
from baseline.utils import get_model_file, get_metric_cmp
from baseline.tf.tagger.training.utils import to_tensors
# Number of batches to prefetch if using tf.datasets
NUM_PREFETCH = 2
# The shuffle buffer
SHUF_BUF_SZ = 5000

logger = logging.getLogger('baseline')


def loss(model, x, y):
    unary = model.transduce(x)
    return model.decoder.neg_log_loss(unary, y, x['lengths'])


class TaggerEvaluatorEagerTf:
    """Performs evaluation on tagger output
    """
    def __init__(self, model, span_type, verbose):
        """Construct from an existing model

        :param model: A model
        :param span_type: (`str`) The span type
        :param verbose: (`bool`) Be verbose?
        """
        self.model = model
        self.idx2label = revlut(model.labels)
        self.span_type = span_type
        if verbose:
            print('Setting span type {}'.format(self.span_type))
        self.verbose = verbose

    def process_batch(self, batch, truth, handle=None, txts=None, ids=None):
        guess = self.model(batch)
        sentence_lengths = batch['lengths']

        correct_labels = 0
        total_labels = 0

        # For fscore
        gold_chunks = []
        pred_chunks = []

        # For each sentence
        for b in range(len(guess)):
            length = sentence_lengths[b]
            sentence = guess[b][:length].numpy()
            # truth[b] is padded, cutting at :length gives us back true length
            gold = truth[b][:length].numpy()

            valid_guess = sentence[gold != Offsets.PAD]
            valid_gold = gold[gold != Offsets.PAD]
            valid_sentence_length = np.sum(gold != Offsets.PAD)

            correct_labels += np.sum(np.equal(valid_guess, valid_gold))
            total_labels += valid_sentence_length

            gold_chunks.append(set(to_spans(valid_gold, self.idx2label, self.span_type, self.verbose)))
            pred_chunks.append(set(to_spans(valid_guess, self.idx2label, self.span_type, self.verbose)))

            if not (handle is None or txts is None):
                example_id = ids[b]
                example_txt = txts[example_id]
                write_sentence_conll(handle, valid_guess, valid_gold, example_txt, self.idx2label)

        return correct_labels, total_labels, gold_chunks, pred_chunks

    def test(self, ts, steps=0, **kwargs):
        """Method that evaluates on some data.  There are 2 modes this can run in, `feed_dict` and `dataset`

        In `feed_dict` mode, the model cycles the test data batch-wise and feeds each batch in with a `feed_dict`.
        In `dataset` mode, the data is still passed in to this method, but it is not passed in a `feed_dict` and is
        mostly superfluous since the features are grafted right onto the graph.  However, we do use it for supplying
        the ground truth, ids and text, so it is essential that the caller does not shuffle the data
        :param ts: The test set
        :param conll_output: (`str`) An optional file output
        :param txts: A list of text data associated with the encoded batch
        :param dataset: (`bool`) Is this using `tf.dataset`s
        :return: The metrics
        """
        total_correct = total_sum = 0
        gold_spans = []
        pred_spans = []

        handle = None
        if kwargs.get("conll_output") is not None and kwargs.get('txts') is not None:
            handle = open(kwargs.get("conll_output"), "w")

        try:
            pg = create_progress_bar(steps)
            metrics = {}
            for (features, y), batch in pg(zip_longest(ts, kwargs.get('batches', []), fillvalue={})):
                correct, count, golds, guesses = self.process_batch(features, y, handle=handle, txts=kwargs.get("txts"), ids=batch.get("ids"))
                total_correct += correct
                total_sum += count
                gold_spans.extend(golds)
                pred_spans.extend(guesses)

            total_acc = total_correct / float(total_sum)
            # Only show the fscore if requested
            metrics['f1'] = span_f1(gold_spans, pred_spans)
            metrics['acc'] = total_acc
            if self.verbose:
                conll_metrics = per_entity_f1(gold_spans, pred_spans)
                conll_metrics['acc'] = total_acc * 100
                conll_metrics['tokens'] = total_sum
                logger.info(conlleval_output(conll_metrics))
        finally:
            if handle is not None:
                handle.close()

        return metrics


class TaggerTrainerEagerTf(EpochReportingTrainer):
    """A Trainer to use for eager mode training
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
            self.model = create_model_for('tagger', **model_params)
        else:
            self.model = model_params
        span_type = kwargs.get('span_type', 'iob')
        verbose = kwargs.get('verbose', False)
        self.evaluator = TaggerEvaluatorEagerTf(self.model, span_type, verbose)
        self.optimizer = EagerOptimizer(loss, **kwargs)
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)
        self._checkpoint = tf.train.Checkpoint(optimizer=self.optimizer.optimizer, model=self.model)
        checkpoint_dir = '{}-{}'.format("./tf-tagger", os.getpid())

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
    def _get_batchsz(batch_dict):
        return batch_dict['y'].shape[0]

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
        self.nstep_start = time.time()

        @tf.function
        def _train_step(inputs):
            features, y = inputs
            loss = self.optimizer.update(self.model, features, y)
            batchsz = get_shape_as_list(y)[0]
            report_loss = loss * batchsz
            return report_loss, batchsz

        with autograph_options({"function_optimization": False, "layout_optimizer": False}):
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
                    self.nstep_start = time.time()

        epoch_loss = epoch_loss.numpy()
        epoch_div = epoch_div.numpy()
        metrics = self.calc_metrics(epoch_loss, epoch_div)
        return metrics

    def _test(self, ts, steps=0, **kwargs):
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
        return self.evaluator.test(ts, steps, **kwargs)


@register_training_func('tagger')
def fit_eager(model_params, ts, vs, es=None, **kwargs):
    """
    Train a tagger using TensorFlow with `tf.dataset`.  This
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
    conll_output = kwargs.get('conll_output', None)
    span_type = kwargs.get('span_type', 'iob')
    txts = kwargs.get('txts', None)
    model_file = get_model_file('tagger', 'tf', kwargs.get('basedir'))

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    verbose = kwargs.get('verbose', {'console': kwargs.get('verbose_console', False), 'file': kwargs.get('verbose_file', None)})
    epochs = int(kwargs.get('epochs', 20))

    batchsz = kwargs['batchsz']
    test_batchsz = kwargs.get('test_batchsz', batchsz)
    lengths_key = model_params.get('lengths_key')

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

    trainer = TaggerTrainerEagerTf(model_params, **kwargs)

    last_improved = 0

    SET_TRAIN_FLAG(True)

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
        evaluator = TaggerEvaluatorEagerTf(trainer.model, span_type, verbose)
        start = time.time()
        test_metrics = evaluator.test(test_dataset, conll_output=conll_output, txts=txts, batches=es, steps=len(es))
        duration = time.time() - start
        for reporting in reporting_fns:
            reporting(test_metrics, 0, 'Test')
        trainer.log.debug({'phase': 'Test', 'time': duration})
