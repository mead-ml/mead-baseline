import six
import os
import time
import numpy as np
import tensorflow as tf
import logging

from eight_mile.utils import to_spans, Offsets, revlut, span_f1, per_entity_f1, conlleval_output, write_sentence_conll

from eight_mile.progress import create_progress_bar
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.tf.tfy import TRAIN_FLAG
from baseline.model import create_model_for
from baseline.utils import get_model_file, get_metric_cmp


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

