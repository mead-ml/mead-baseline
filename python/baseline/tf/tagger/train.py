import tensorflow as tf
import numpy as np
from baseline.utils import to_spans, f_score, listify, revlut
from baseline.progress import ProgressBar
from baseline.reporting import basic_reporting
from baseline.tf.tfy import optimizer
from baseline.train import EpochReportingTrainer


class TaggerEvaluatorTf:

    def __init__(self, model):
        self.model = model
        self.idx2label = revlut(model.labels)

    def _write_sentence_conll(self, handle, sentence, gold, txt):

        if len(txt) != len(sentence):
            txt = txt[:len(sentence)]

        try:
            for word, truth, guess in zip(txt, gold, sentence):
                handle.write('%s %s %s\n' % (word, self.idx2label[truth], self.idx2label[guess]))
            handle.write('\n')
        except:
            print('ERROR: Failed to write lines... closing file')
            handle.close()
            handle = None

    def process_batch(self, x, xch, truth, sentence_lengths, ids, handle=None, txts=None):

        guess = self.model.predict(x, xch, sentence_lengths)
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

            gold_chunks = to_spans(gold, self.idx2label)
            gold_count += len(gold_chunks)

            guess_chunks = to_spans(sentence, self.idx2label)
            guess_count += len(guess_chunks)

            overlap_chunks = gold_chunks & guess_chunks
            overlap_count += len(overlap_chunks)

            # Should we write a file out?  If so, we have to have txts
            if handle is not None:
                id = ids[b]
                txt = txts[id]
                self._write_sentence_conll(handle, sentence, gold, txt)

        return correct_labels, total_labels, overlap_count, gold_count, guess_count

    def test(self, ts, conll_file=None, txts=None):

        total_correct = total_sum = 0
        total_gold_count = total_guess_count = total_overlap_count = 0

        steps = len(ts)
        pg = ProgressBar(steps)
        metrics = {}
        # Only if they provide a file and the raw txts, we can write CONLL file
        handle = None
        if conll_file is not None and txts is not None:
            handle = open(conll_file, "w")

        for x, xch, y, lengths, id in ts:
            correct, count, overlaps, golds, guesses = self.process_batch(x, xch, y, lengths, id, handle, txts)
            total_correct += correct
            total_sum += count
            total_gold_count += golds
            total_guess_count += guesses
            total_overlap_count += overlaps
            pg.update()
        pg.done()

        total_acc = total_correct / float(total_sum)
        # Only show the fscore if requested
        metrics['f1'] = f_score(total_overlap_count, total_gold_count, total_guess_count)
        metrics['acc'] = total_acc

        if handle is not None:
            handle.close()

        return metrics


class TaggerTrainerTf(EpochReportingTrainer):

    def __init__(self, model, **kwargs):
        super(TaggerTrainerTf, self).__init__()
        self.loss = model.create_loss()
        self.model = model
        self.evaluator = TaggerEvaluatorTf(model)
        self.global_step, self.train_op = optimizer(self.loss, **kwargs)

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-checkpoints/tagger", global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-checkpoints")
        print("Reloading " + latest)
        self.model.saver.restore(self.model.sess, latest)

    def _train(self, ts):
        total_loss = 0
        steps = len(ts)
        metrics = {}
        pg = ProgressBar(steps)
        for x, xch, y, lengths, id in ts:
            feed_dict = self.model.make_feed_dict(x, xch, y, do_dropout=True)
            _, step, lossv = self.model.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)
            total_loss += lossv
            pg.update()
        pg.done()
        metrics['avg_loss'] = float(total_loss)/steps
        return metrics

    def _test(self, ts, conll_file=None, txts=None):
        return self.evaluator.test(ts, conll_file, txts)

def fit(model, ts, vs, es, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs
    conll_file = kwargs.get('conll_file', None)
    txts = kwargs.get('txts', None)
    model_file = kwargs['outfile'] if 'outfile' in kwargs and kwargs['outfile'] is not None else './seq2seq-model-tf'
    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = TaggerTrainerTf(model, **kwargs)
    init = tf.global_variables_initializer()
    model.sess.run(init)
    saver = tf.train.Saver()
    model.save_using(saver)
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', basic_reporting))
    print('reporting', reporting_fns)

    max_metric = 0
    last_improved = 0
    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(model)

        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            model.save(model_file)

        elif test_metrics[early_stopping_metric] > max_metric:
            last_improved = epoch
            max_metric = test_metrics[early_stopping_metric]
            print('New max %.3f' % max_metric)
            model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on max_metric %.3f at epoch %d' % (max_metric, last_improved))
    if es is not None:

        model.restore(tf.Session(), model_file)
        evaluator = TaggerEvaluatorTf(model)
        test_metrics = evaluator.test(es, conll_file=conll_file, txts=txts)
        for reporting in reporting_fns:
            reporting(test_metrics, 0, 'Test')
