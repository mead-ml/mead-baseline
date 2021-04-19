import os
import time
import sys
import logging
import numpy as np
import tensorflow as tf
from eight_mile.tf.optz import optimizer
from eight_mile.progress import create_progress_bar
from eight_mile.utils import listify, Timer, to_spans, Offsets, revlut, span_f1, per_entity_f1, conlleval_output, write_sentence_conll
from baseline.train import create_trainer, register_training_func, register_trainer, EpochReportingTrainer
from baseline.model import create_model_for
from baseline.tf.tfy import TRAIN_FLAG, reload_checkpoint
from baseline.utils import get_model_file, get_metric_cmp

logger = logging.getLogger('baseline')


class TaggerEvaluatorTf:
    """Performs evaluation on tagger output
    """
    def __init__(self, model, span_type, verbose):
        """Construct from an existing model

        :param model: A model
        :param span_type: (`str`) The span type
        :param verbose: (`bool`) Be verbose?
        """
        self.model = model
        self.sess = self.model.sess
        self.idx2label = revlut(model.labels)
        self.span_type = span_type
        if verbose:
            print('Setting span type {}'.format(self.span_type))
        self.verbose = verbose

    def process_batch(self, batch_dict, handle, txts):
        feed_dict = self.model.make_input(batch_dict)
        guess = self.sess.run(self.model.best, feed_dict=feed_dict)
        sentence_lengths = batch_dict[self.model.lengths_key]

        ids = batch_dict['ids']
        truth = batch_dict['y']
        correct_labels = 0
        total_labels = 0

        # For fscore
        gold_chunks = []
        pred_chunks = []

        # For each sentence
        for b in range(len(guess)):
            length = sentence_lengths[b]
            sentence = guess[b][:length]
            # truth[b] is padded, cutting at :length gives us back true length
            gold = truth[b][:length]

            valid_guess = sentence[gold != Offsets.PAD]
            valid_gold = gold[gold != Offsets.PAD]
            valid_sentence_length = np.sum(gold != Offsets.PAD)
            correct_labels += np.sum(np.equal(valid_guess, valid_gold))
            total_labels += valid_sentence_length

            gold_chunks.append(set(to_spans(valid_gold, self.idx2label, self.span_type, self.verbose)))
            pred_chunks.append(set(to_spans(valid_guess, self.idx2label, self.span_type, self.verbose)))

            # Should we write a file out?  If so, we have to have txts
            if handle is not None:
                id = ids[b]
                txt = txts[id]
                write_sentence_conll(handle, valid_guess, valid_gold, txt, self.idx2label)

        return correct_labels, total_labels, gold_chunks, pred_chunks

    def test(self, ts, conll_output=None, txts=None):
        """Method that evaluates on some data.

        :param ts: The test set
        :param conll_output: (`str`) An optional file output
        :param txts: A list of text data associated with the encoded batch
        :return: The metrics
        """
        total_correct = total_sum = 0
        gold_spans = []
        pred_spans = []

        steps = len(ts)
        pg = create_progress_bar(steps)
        metrics = {}
        # Only if they provide a file and the raw txts, we can write CONLL file
        handle = None
        if conll_output is not None and txts is not None:
            handle = open(conll_output, "w")

        try:
            for batch_dict in pg(ts):
                correct, count, golds, guesses = self.process_batch(batch_dict, handle, txts)
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


@register_trainer(task='tagger', name='default')
class TaggerTrainerTf(EpochReportingTrainer):
    """A Trainer to use if not using eager mode

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
            self.model = create_model_for('tagger', **model_params)
        else:
            self.model = model_params
        self.sess = self.model.sess
        self.loss = self.model.create_loss()
        span_type = kwargs.get('span_type', 'iob')
        verbose = kwargs.get('verbose', False)
        self.evaluator = TaggerEvaluatorTf(self.model, span_type, verbose)
        self.global_step, self.train_op = optimizer(self.loss, colocate_gradients_with_ops=True, variables=self.model.trainable_variables, **kwargs)
        self.nsteps = kwargs.get('nsteps', sys.maxsize)
        tables = tf.compat.v1.tables_initializer()
        self.model.sess.run(tables)
        init = tf.compat.v1.global_variables_initializer()
        self.model.sess.run(init)
        saver = tf.compat.v1.train.Saver()
        self.model.save_using(saver)
        checkpoint = kwargs.get('checkpoint')
        if checkpoint is not None:
            skip_blocks = kwargs.get('blocks_to_skip', ['OptimizeLoss'])
            reload_checkpoint(self.model.sess, checkpoint, skip_blocks)

    def checkpoint(self):
        """This method saves a checkpoint

        :return: None
        """
        checkpoint_dir = '{}-{}'.format("./tf-tagger", os.getpid())
        self.model.saver.save(self.sess, os.path.join(checkpoint_dir, 'tagger'),
                              global_step=self.global_step,
                              write_meta_graph=False)

    def recover_last_checkpoint(self):
        """Recover the last saved checkpoint

        :return: None
        """
        checkpoint_dir = '{}-{}'.format("./tf-tagger", os.getpid())
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('Reloading ' + latest)
        self.model.saver.restore(self.model.sess, latest)

    @staticmethod
    def _get_batchsz(batch_dict):
        return batch_dict['y'].shape[0]

    def _train(self, ts, **kwargs):
        """Train an epoch of data using either the input loader or using `tf.dataset`

        In non-`tf.dataset` mode, we cycle the loader data feed, and pull a batch and feed it to the feed dict
        When we use `tf.dataset`s under the hood, this function simply uses the loader to know how many steps
        to train.  We do use a `feed_dict` for passing the `TRAIN_FLAG` in either case

        :param ts: A data feed
        :param kwargs: See below

        :Keyword Arguments:
         * *dataset* (`bool`) Set to `True` if using `tf.dataset`s, defaults to `True`
         * *reporting_fns* (`list`) A list of reporting hooks to use

        :return: Metrics
        """
        reporting_fns = kwargs.get('reporting_fns', [])
        epoch_loss = 0
        epoch_div = 0
        steps = len(ts)
        pg = create_progress_bar(steps)
        for batch_dict in pg(ts):
            feed_dict = self.model.make_input(batch_dict, True)
            _, step, lossv = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)

            batchsz = self._get_batchsz(batch_dict)
            report_loss = lossv * batchsz
            epoch_loss += report_loss
            epoch_div += batchsz
            self.nstep_agg += report_loss
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

    def _test(self, ts):
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
        return self.evaluator.test(ts)


@register_training_func('tagger')
def fit(model_params, ts, vs, es, **kwargs):
    """
    Train a classifier using TensorFlow with a `feed_dict`.  This
    is the previous default behavior for training.  To use this, you need to pass
    `fit_func: feed_dict` in your MEAD config

    :param model_params: The model to train
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
    conll_output = kwargs.get('conll_output', None)
    span_type = kwargs.get('span_type', 'iob')
    txts = kwargs.get('txts', None)
    model_file = get_model_file('tagger', 'tf', kwargs.get('basedir'))
    TRAIN_FLAG()

    trainer = create_trainer(model_params, **kwargs)

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    verbose = bool(kwargs.get('verbose', False))

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

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

        trainer.recover_last_checkpoint()
        # What to do about overloading this??
        evaluator = TaggerEvaluatorTf(trainer.model, span_type, verbose)
        timer = Timer()
        test_metrics = evaluator.test(es, conll_output=conll_output, txts=txts)
        duration = timer.elapsed()
        for reporting in reporting_fns:
            reporting(test_metrics, 0, 'Test')
        trainer.log.debug({'phase': 'Test', 'time': duration})
