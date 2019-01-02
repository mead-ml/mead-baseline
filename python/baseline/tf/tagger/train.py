"""Train a tagger with TensorFlow

This module supports 2 different ways of training a model

1. feed_dict
2. datasets

It doesnt currently support estimators and multi-GPU as there is currently no good way to early stop with
`train_and_evaluate`, nor is there an easy way to stop on the metrics we define here
"""
import six
import os
import time
import numpy as np
import tensorflow as tf
from baseline.tf.optz import optimizer
from baseline.progress import create_progress_bar
from baseline.utils import to_spans, f_score, listify, revlut, get_model_file, write_sentence_conll, get_metric_cmp
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.tf.tfy import TRAIN_FLAG
from baseline.model import create_model_for

# Number of batches to prefetch if using tf.datasets
NUM_PREFETCH = 2
# The shuffle buffer
SHUF_BUF_SZ = 5000



def to_tensors(ts, lengths_key):
    """Convert a data feed into a tuple of `features` (`dict`) and `y` values

    This method is required to produce `tf.dataset`s from the input data feed.
    Any fields ending with `_lengths` are ignored, unless they match the
    `lengths_key` name (as are `ids`)

    :param ts: The data feed to convert
    :param lengths_key: This is a field passed from the model params specifying source of truth of the temporal lengths
    :return: A `tuple` of `features` and `y` (labels)
    """
    keys = ts[0].keys()
    # This is kind of a hack
    keys = [k for k in keys if '_lengths' not in k and k != 'ids'] + [lengths_key]

    features = dict((k, []) for k in keys)
    for sample in ts:
        for k in features.keys():
            # add each sample
            for s in sample[k]:
                features[k].append(s)

    features['lengths'] = features[lengths_key]
    del features[lengths_key]
    features = dict((k, np.stack(v)) for k, v in features.items())
    y = features.pop('y')
    return features, y


class TaggerEvaluatorTf(object):
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

    def _process_batch(self, guess, sentence_lengths, ids, truth, handle=None, txts=None):

        correct_labels = 0
        total_labels = 0

        # For fscore
        gold_count = 0
        guess_count = 0
        overlap_count = 0

        # For each sentence
        for b in range(len(guess)):
            length = sentence_lengths[b]
            assert(length == len(guess[b]))
            sentence = guess[b]
            # truth[b] is padded, cutting at :length gives us back true length
            gold = truth[b][:length]
            correct_labels += np.sum(np.equal(sentence, gold))
            total_labels += length

            gold_chunks = to_spans(gold, self.idx2label, self.span_type, self.verbose)
            gold_count += len(gold_chunks)

            guess_chunks = to_spans(sentence, self.idx2label, self.span_type, self.verbose)
            guess_count += len(guess_chunks)

            overlap_chunks = gold_chunks & guess_chunks
            overlap_count += len(overlap_chunks)

            # Should we write a file out?  If so, we have to have txts
            if handle is not None:
                id = ids[b]
                txt = txts[id]
                write_sentence_conll(handle, sentence, gold, txt, self.idx2label)

        return correct_labels, total_labels, overlap_count, gold_count, guess_count

    def test(self, ts, conll_output=None, txts=None, dataset=True):
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
        total_gold_count = total_guess_count = total_overlap_count = 0
        steps = len(ts)
        pg = create_progress_bar(steps)
        metrics = {}
        # Only if they provide a file and the raw txts, we can write CONLL file
        handle = None
        if conll_output is not None and txts is not None:
            handle = open(conll_output, "w")

        try:
            for batch_dict in pg(ts):
                if dataset:
                    bestv = self.sess.run(self.model.best)
                else:
                    feed_dict = self.model.make_input(batch_dict)
                    bestv = self.sess.run(self.model.best, feed_dict=feed_dict)

                lengths = batch_dict[self.model.lengths_key]
                guess = [pij[:sl] for pij, sl in zip(bestv, lengths)]
                sentence_lengths = batch_dict[self.model.lengths_key]
                ids = batch_dict['ids']
                truth = batch_dict['y']
                assert len(batch_dict['y']) == len(guess)
                correct, count, overlaps, golds, guesses = self._process_batch(guess,
                                                                               sentence_lengths,
                                                                               ids,
                                                                               truth,
                                                                               handle,
                                                                               txts)

                total_correct += correct
                total_sum += count
                total_gold_count += golds
                total_guess_count += guesses
                total_overlap_count += overlaps

            total_acc = total_correct / float(total_sum)
            # Only show the fscore if requested
            metrics['f1'] = f_score(total_overlap_count, total_gold_count, total_guess_count)
            metrics['acc'] = total_acc
        finally:
            if handle is not None:
                handle.close()

        return metrics


@register_trainer(task='tagger', name='default')
class TaggerTrainerTf(EpochReportingTrainer):
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
        super(TaggerTrainerTf, self).__init__()
        if type(model_params) is dict:
            self.model = create_model_for('tagger', **model_params)
        else:
            self.model = model_params
        self.sess = self.model.sess
        self.loss = self.model.create_loss()
        span_type = kwargs.get('span_type', 'iob')
        verbose = kwargs.get('verbose', False)
        self.evaluator = TaggerEvaluatorTf(self.model, span_type, verbose)
        self.global_step, self.train_op = optimizer(self.loss, **kwargs)
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)
        tables = tf.tables_initializer()
        self.model.sess.run(tables)
        init = tf.global_variables_initializer()
        self.model.sess.run(init)
        saver = tf.train.Saver()
        self.model.save_using(saver)

    def checkpoint(self):
        """This method saves a checkpoint

        :return: None
        """
        checkpoint_dir = '{}-{}'.format("./tf-tagger", os.getpid())
        self.model.saver.save(self.sess, os.path.join(checkpoint_dir, 'tagger'), global_step=self.global_step)

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
        use_dataset = kwargs.get('dataset', True)
        reporting_fns = kwargs.get('reporting_fns', [])
        epoch_loss = 0
        epoch_div = 0
        steps = len(ts)
        pg = create_progress_bar(steps)
        for batch_dict in pg(ts):
            if use_dataset:
                _, step, lossv = self.sess.run([self.train_op, self.global_step, self.loss],
                                               feed_dict={TRAIN_FLAG(): 1})
            else:
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

    def _test(self, ts, dataset=True):
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
        return self.evaluator.test(ts, dataset=dataset)


@register_training_func('tagger')
def fit_datasets(model_params, ts, vs, es=None, **kwargs):
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
    ## First, make tf.datasets for ts, vs and es
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
    # effective_batch_sz = args.batchsz*args.gpus
    test_batchsz = kwargs.get('test_batchsz', batchsz)
    # This is a little awkward:
    lengths_key = model_params.get('lengths_key')

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
    model_params['y'] = y
    # create the initialisation operations
    train_init_op = iter.make_initializer(train_dataset)
    valid_init_op = iter.make_initializer(valid_dataset)
    test_init_op = iter.make_initializer(test_dataset)

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
        # What to do about overloading this??
        evaluator = TaggerEvaluatorTf(trainer.model, span_type, verbose)
        start = time.time()
        test_metrics = evaluator.test(es, conll_output=conll_output, txts=txts)
        duration = time.time() - start
        for reporting in reporting_fns:
            reporting(test_metrics, 0, 'Test')
        trainer.log.debug({'phase': 'Test', 'time': duration})


@register_training_func('tagger', 'feed_dict')
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

        trainer.recover_last_checkpoint()
        # What to do about overloading this??
        evaluator = TaggerEvaluatorTf(trainer.model, span_type, verbose)
        start = time.time()
        test_metrics = evaluator.test(es, conll_output=conll_output, txts=txts, dataset=False)
        duration = time.time() - start
        for reporting in reporting_fns:
            reporting(test_metrics, 0, 'Test')
        trainer.log.debug({'phase': 'Test', 'time': duration})
