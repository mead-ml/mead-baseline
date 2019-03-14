import six
import time
import logging
import dynet as dy
import numpy as np
from baseline.dy.optz import *
from baseline.progress import create_progress_bar
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.utils import listify, get_model_file, revlut, to_spans, f_score, write_sentence_conll, get_metric_cmp, span_f1, per_entity_f1, conlleval_output

logger = logging.getLogger('baseline')


@register_trainer(task='tagger', name='default')
class TaggerTrainerDyNet(EpochReportingTrainer):

    def __init__(self, model, **kwargs):

        super(TaggerTrainerDyNet, self).__init__()

        self.span_type = kwargs.get('span_type', 'iob')
        logger.info('Setting span type %s', self.span_type)
        self.gpu = not bool(kwargs.get('nogpu', False))
        self.model = model
        self.idx2label = revlut(self.model.labels)
        self.autobatchsz = kwargs.get('autobatchsz')
        self.labels = model.labels
        self.optimizer = OptimizerManager(model, **kwargs)
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)
        self.verbose = kwargs.get('verbose', False)

    # Guess is a list over time
    def process_output(self, guess, truth, sentence_lengths, ids, handle=None, txts=None):

        correct_labels = 0
        total_labels = 0
        truth_n = truth
        # For fscore
        gold_chunks = []
        pred_chunks = []
        # For each sentence
        for b in range(len(guess)):

            sentence_length = sentence_lengths[b]
            gold = truth_n[b, :sentence_length]
            sentence = guess[b]
            sentence = sentence[:sentence_length]
            correct_labels += np.sum(np.equal(sentence, gold))
            total_labels += sentence_length
            gold_chunks.append(set(to_spans(gold, self.idx2label, self.span_type)))
            pred_chunks.append(set(to_spans(sentence, self.idx2label, self.span_type)))
            # Should we write a file out?  If so, we have to have txts
            if handle is not None:
                id = ids[b]
                txt = txts[id]
                write_sentence_conll(handle, sentence, gold, txt, self.idx2label)

        return correct_labels, total_labels, gold_chunks, pred_chunks


    def _test(self, ts, **kwargs):

        self.model.train = False
        total_correct = 0
        total_sum = 0

        gold_spans = []
        pred_spans = []

        metrics = {}
        steps = len(ts)
        conll_output = kwargs.get('conll_output', None)
        txts = kwargs.get('txts', None)
        handle = None
        if conll_output is not None and txts is not None:
            handle = open(conll_output, "w")
        pg = create_progress_bar(steps)
        for batch_dict in pg(ts):

            lengths = batch_dict[self.model.lengths_key]
            ids = batch_dict['ids']
            y = batch_dict['y']
            pred = self.model.predict(batch_dict)
            correct, count, golds, guesses = self.process_output(pred, y, lengths, ids, handle, txts)
            total_correct += correct
            total_sum += count
            gold_spans.extend(golds)
            pred_spans.extend(guesses)

        total_acc = total_correct / float(total_sum)
        metrics['acc'] = total_acc
        # Only show the fscore if requested
        metrics['f1'] = span_f1(gold_spans, pred_spans)
        if self.verbose:
            conll_metrics = per_entity_f1(gold_spans, pred_spans)
            conll_metrics['acc'] = total_acc * 100
            conll_metrics['tokens'] = total_sum
            logger.info(conlleval_output(conll_metrics))
        return metrics

    @staticmethod
    def _get_batchsz(y):
        # Because we only support autobatch this is just 1
        return 1

    def _train(self, ts, **kwargs):
        self.model.train = True
        reporting_fns = kwargs.get('reporting_fns', [])
        epoch_loss = 0
        epoch_norm = 0
        auto_norm = 0
        metrics = {}
        steps = len(ts)
        last = steps
        losses = []
        i = 1
        pg = create_progress_bar(steps)
        dy.renew_cg()
        for batch_dict in pg(ts):

            inputs = self.model.make_input(batch_dict)
            y = inputs.pop('y')
            pred = self.model.compute_unaries(inputs)
            bsz = self._get_batchsz(y)
            if self.autobatchsz is None:
                losses = self.model.loss(pred, y)
                loss = dy.mean_batches(losses)
                lossv = loss.npvalue().item()
                report_loss = lossv * bsz
                epoch_loss += report_loss
                epoch_norm += bsz
                self.nstep_agg += report_loss
                self.nstep_div += bsz
                loss.backward()
                self.optimizer.update()
                dy.renew_cg()
                # TODO: Abstract this somewhat, or else once we have a batched tagger have 2 trainers
                if (self.optimizer.global_step + 1) % self.nsteps == 0:
                    metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                    self.report(
                        self.optimizer.global_step + 1, metrics, self.nstep_start,
                        'Train', 'STEP', reporting_fns, self.nsteps
                    )
                    self.reset_nstep()
            else:
                loss = self.model.loss(pred, y)
                losses.append(loss)
                self.nstep_div += bsz
                epoch_norm += bsz
                auto_norm += bsz

                if i % self.autobatchsz == 0 or i == last:
                    loss = dy.average(losses)
                    lossv = loss.npvalue().item()
                    loss.backward()
                    self.optimizer.update()
                    report_loss = lossv * auto_norm
                    epoch_loss += report_loss
                    self.nstep_agg += report_loss
                    losses = []
                    dy.renew_cg()
                    if (self.optimizer.global_step + 1) % self.nsteps == 0:
                        metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                        self.report(
                            self.optimizer.global_step + 1, metrics, self.nstep_start,
                            'Train', 'STEP', reporting_fnsa, self.nsteps
                        )
                        self.reset_nstep()
                    auto_norm = 0
            i += 1

        metrics = self.calc_metrics(epoch_loss, epoch_norm)
        return metrics


@register_training_func('tagger')
def fit(model, ts, vs, es, **kwargs):

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file('tagger', 'dynet', kwargs.get('basedir'))
    conll_output = kwargs.get('conll_output', None)
    txts = kwargs.get('txts', None)

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_metric'))
        patience = kwargs.get('patience', epochs)
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)

    #validation_improvement_fn = kwargs.get('validation_improvement', None)

    after_train_fn = kwargs.get('after_train_fn', None)
    trainer = create_trainer(model, **kwargs)

    last_improved = 0
    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(model)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            logger.info('New max %.3f', best_metric)
            model.save(model_file)


        elif (epoch - last_improved) > patience:
            logger.info('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        logger.info('Best performance on %s: %.3f at epoch %d', early_stopping_metric, best_metric, last_improved)

    if es is not None:
        logger.info('Reloading best checkpoint')
        model = model.load(model_file)
        trainer = create_trainer(model, **kwargs)
        test_metircs = trainer.test(es, reporting_fns, conll_output=conll_output, txts=txts, phase='Test')
    return test_metrics
