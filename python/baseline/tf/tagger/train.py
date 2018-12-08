import six
import os
import time
import numpy as np
import tensorflow as tf
from baseline.tf.optz import optimizer
from baseline.progress import create_progress_bar
from baseline.utils import to_spans, f_score, listify, revlut, get_model_file, write_sentence_conll, get_metric_cmp
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func


class TaggerEvaluatorTf(object):

    def __init__(self, model, span_type, verbose):
        self.model = model
        self.idx2label = revlut(model.labels)
        self.span_type = span_type
        if verbose:
            print('Setting span type {}'.format(self.span_type))
        self.verbose = verbose

    def process_batch(self, batch_dict, handle=None, txts=None):

        guess = self.model.predict(batch_dict)
        sentence_lengths = batch_dict[self.model.lengths_key]
        ids = batch_dict['ids']
        truth = batch_dict['y']
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

    def test(self, ts, conll_output=None, txts=None):

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
                correct, count, overlaps, golds, guesses = self.process_batch(batch_dict, handle, txts)
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

    def __init__(self, model, **kwargs):
        super(TaggerTrainerTf, self).__init__()
        self.loss = model.create_loss()
        self.model = model
        span_type = kwargs.get('span_type', 'iob')
        verbose = kwargs.get('verbose', False)
        self.evaluator = TaggerEvaluatorTf(model, span_type, verbose)
        self.global_step, self.train_op = optimizer(self.loss, **kwargs)
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-tagger-%d/tagger" % os.getpid(), global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-tagger-%d" % os.getpid())
        print("Reloading " + latest)
        self.model.saver.restore(self.model.sess, latest)

    @staticmethod
    def _get_batchsz(batch_dict):
        return batch_dict['y'].shape[0]

    def _train(self, ts, **kwargs):
        reporting_fns = kwargs.get('reporting_fns', [])
        epoch_loss = 0
        epoch_norm = 0
        steps = len(ts)
        pg = create_progress_bar(steps)
        for batch_dict in pg(ts):
            feed_dict = self.model.make_input(batch_dict, True)
            _, step, lossv = self.model.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)
            bsz = self._get_batchsz(batch_dict)
            report_loss = lossv * bsz
            epoch_loss += report_loss
            epoch_norm += bsz
            self.nstep_agg += report_loss
            self.nstep_div += bsz
            if (step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()

        metrics = self.calc_metrics(epoch_loss, epoch_norm)
        return metrics

    def _test(self, ts):
        return self.evaluator.test(ts)


@register_training_func('tagger')
def fit(model, ts, vs, es, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs
    conll_output = kwargs.get('conll_output', None)
    span_type = kwargs.get('span_type', 'iob')
    txts = kwargs.get('txts', None)
    model_file = get_model_file('tagger', 'tf', kwargs.get('basedir'))
    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = create_trainer(model, **kwargs)
    tables = tf.tables_initializer()
    model.sess.run(tables)
    init = tf.global_variables_initializer()
    model.sess.run(init)
    saver = tf.train.Saver()
    model.save_using(saver)
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
        if after_train_fn is not None:
            after_train_fn(model)

        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.checkpoint()
            model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            print('New best %.3f' % best_metric)
            trainer.checkpoint()
            model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on %s: %.3f at epoch %d' % (early_stopping_metric, best_metric, last_improved))
    if es is not None:

        trainer.recover_last_checkpoint()
        # What to do about overloading this??
        evaluator = TaggerEvaluatorTf(model, span_type, verbose)
        start = time.time()
        test_metrics = evaluator.test(es, conll_output=conll_output, txts=txts)
        duration = time.time() - start
        for reporting in reporting_fns:
            reporting(test_metrics, 0, 'Test')
        trainer.log.debug({'phase': 'Test', 'time': duration})

