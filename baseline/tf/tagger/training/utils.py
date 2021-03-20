import six
import os
import time
import numpy as np
import tensorflow as tf
import logging
from eight_mile.tf.optz import optimizer
from eight_mile.utils import to_spans, Offsets, revlut, span_f1, per_entity_f1, conlleval_output, write_sentence_conll

from eight_mile.progress import create_progress_bar
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.tf.tfy import TRAIN_FLAG
from baseline.model import create_model_for
from baseline.utils import get_model_file, get_metric_cmp
from eight_mile.tf.layers import reload_checkpoint

logger = logging.getLogger('baseline')


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
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)
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
