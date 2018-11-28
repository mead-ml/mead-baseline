import six
import time
from baseline.progress import create_progress_bar
from baseline.utils import listify, to_spans, f_score, revlut, get_model_file, write_sentence_conll
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.pytorch.torchy import *
from baseline.pytorch.optz import OptimizerManager


@register_trainer(task='tagger', name='default')
class TaggerTrainerPyTorch(EpochReportingTrainer):

    def __init__(self, model, **kwargs):
        super(TaggerTrainerPyTorch, self).__init__()
        self.gpu = not bool(kwargs.get('nogpu', False))
        # By default support IOB1/IOB2
        self.span_type = kwargs.get('span_type', 'iob')
        self.verbose = kwargs.get('verbose', False)

        if self.verbose:
            print('Setting span type {}'.format(self.span_type))
        self.model = model
        self.idx2label = revlut(self.model.labels)
        self.clip = float(kwargs.get('clip', 5))
        self.optimizer = OptimizerManager(self.model, **kwargs)
        if self.gpu:
            self.model = model.to_gpu()
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)

    @staticmethod
    def _get_batchsz(batch_dict):
        return batch_dict['y'].shape[0]

    def process_output(self, guess, truth, sentence_lengths, ids, handle=None, txts=None):

        correct_labels = 0
        total_labels = 0
        truth_n = truth.cpu().numpy()
        # For fscore
        gold_count = 0
        guess_count = 0
        overlap_count = 0

        # For each sentence
        for b in range(len(guess)):

            sentence = guess[b].cpu().numpy()

            sentence_length = sentence_lengths[b]
            gold = truth_n[b, :sentence_length]
            correct_labels += np.sum(np.equal(sentence, gold))
            total_labels += sentence_length
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

    def _test(self, ts, **kwargs):

        self.model.eval()
        total_correct = 0
        total_sum = 0
        total_gold_count = 0
        total_guess_count = 0
        total_overlap_count = 0
        metrics = {}
        steps = len(ts)
        conll_output = kwargs.get('conll_output', None)
        txts = kwargs.get('txts', None)
        handle = None
        if conll_output is not None and txts is not None:
            handle = open(conll_output, "w")
        pg = create_progress_bar(steps)
        for batch_dict in pg(ts):

            inputs = self.model.make_input(batch_dict)
            y = inputs.pop('y')
            lengths = inputs['lengths']
            ids = inputs['ids']
            pred = self.model(inputs)
            correct, count, overlaps, golds, guesses = self.process_output(pred, y.data, lengths, ids, handle, txts)
            total_correct += correct
            total_sum += count
            total_gold_count += golds
            total_guess_count += guesses
            total_overlap_count += overlaps

        total_acc = total_correct / float(total_sum)
        # Only show the fscore if requested
        metrics['f1'] = f_score(total_overlap_count, total_gold_count, total_guess_count)
        metrics['acc'] = total_acc
        return metrics

    def _train(self, ts, **kwargs):
        self.model.train()
        reporting_fns = kwargs.get('reporting_fns', [])
        epoch_loss = 0
        epoch_norm = 0
        steps = len(ts)
        pg = create_progress_bar(steps)
        for batch_dict in pg(ts):
            inputs = self.model.make_input(batch_dict)
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            bsz = self._get_batchsz(batch_dict)
            report_loss = loss.item() * bsz
            epoch_loss += report_loss
            epoch_norm += bsz
            self.nstep_agg += report_loss
            self.nstep_div += bsz
            if (self.optimizer.global_step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    self.optimizer.global_step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()

        metrics = self.calc_metrics(epoch_loss, epoch_norm)
        return metrics


@register_training_func('tagger')
def fit(model, ts, vs, es, **kwargs):

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file('tagger', 'pytorch', kwargs.get('basedir'))
    conll_output = kwargs.get('conll_output', None)
    txts = kwargs.get('txts', None)

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    #validation_improvement_fn = kwargs.get('validation_improvement', None)

    after_train_fn = kwargs.get('after_train_fn', None)
    trainer = create_trainer(model, **kwargs)

    last_improved = 0
    max_metric = 0
    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(model)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            model.save(model_file)

        elif test_metrics[early_stopping_metric] > max_metric:
            #if validation_improvement_fn is not None:
            #    validation_improvement_fn(early_stopping_metric, test_metrics, epoch, max_metric, last_improved)
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
        print('Reloading best checkpoint')
        model = torch.load(model_file)
        trainer = create_trainer(model, **kwargs)
        trainer.test(es, reporting_fns, conll_output=conll_output, txts=txts, phase='Test')
