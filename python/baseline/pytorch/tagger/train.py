from baseline.pytorch.torchy import *
from baseline.utils import listify, to_spans, f_score, revlut
from baseline.reporting import basic_reporting
from baseline.progress import ProgressBar
from baseline.train import EpochReportingTrainer


class TaggerTrainerPyTorch(EpochReportingTrainer):

    def __init__(self, model, **kwargs):
        super(TaggerTrainerPyTorch, self).__init__()
        self.gpu = not bool(kwargs.get('nogpu', False))
        optim = kwargs.get('optim', 'adam')
        eta = float(kwargs.get('eta', 0.01))
        mom = float(kwargs.get('mom', 0.9))
        self.clip = float(kwargs.get('clip', 5))
        self.model = model
        self.idx2label = revlut(self.model.labels)
        if optim == 'adadelta':
            self.optimizer = torch.optim.Adadelta(model.parameters(), lr=eta)
        elif optim == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=eta)
        elif optim == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=eta)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=mom)

        self.crit = model.create_loss()
        if self.gpu:
            self.model = torch.nn.DataParallel(model).cuda()
            self.crit.cuda()

    def _wrap(self, x, xch, y):

        if self.gpu:
            x = x.cuda()
            xch = xch.cuda()
            y = y.cuda()

        return torch.autograd.Variable(x), torch.autograd.Variable(xch),\
               torch.autograd.Variable(y.contiguous())

    # Ok, accuracy only for now, B x T x C
    def _right(self, pred, y):
        _, best = pred.max(2)
        best = best.data.long().squeeze()
        yt = y.data.long()
        mask = yt.ne(0)
        return torch.sum(mask * (yt == best))

    def _total(self, y):
        yt = y.data.long()
        return torch.sum(yt.ne(0))

    def process_output(self, guess, truth, sentence_lengths, ids, handle=None, txts=None):

        correct_labels = 0
        total_labels = 0
        guess_n = guess.data.cpu().numpy()
        truth_n = truth.data.cpu().numpy()
        # For fscore
        gold_count = 0
        guess_count = 0
        overlap_count = 0

        # For each sentence
        for b in range(len(guess_n)):
            sentence = np.argmax(guess_n[b], -1)
            sentence_length = sentence_lengths[b]
            gold = truth_n[b,:sentence_length]
            sentence = sentence[:sentence_length]
            correct_labels += np.sum(np.equal(sentence, gold))
            total_labels += sentence_length
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

    def _test(self, ts):

        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_sum = 0
        total_gold_count = 0
        total_guess_count = 0
        total_overlap_count = 0
        metrics = {}
        steps = len(ts)
        pg = ProgressBar(steps)
        for x, xch, y, lengths, ids in ts:
            x, xch, y = self._wrap(x, xch, y)
            pred = self.model((x, xch))
            loss = self.crit(pred, y)
            total_loss += loss.data[0]
            correct, count, overlaps, golds, guesses = self.process_output(pred, y, lengths, ids, None, None)
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
        metrics['avg_loss'] = float(total_loss)/total_sum
        return metrics

    def _train(self, ts):
        total_loss = total = 0
        metrics = {}
        steps = len(ts)
        pg = ProgressBar(steps)
        for x, xch, y, lengths, ids in ts:
            x, xch, y = self._wrap(x, xch, y)
            self.optimizer.zero_grad()
            pred = self.model((x, xch))
            loss = self.crit(pred, y)
            total_loss += loss.data[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
            total += self._total(y)
            self.optimizer.step()
            pg.update()

        pg.done()
        metrics['avg_loss'] = float(total_loss)/total
        return metrics


def fit(model, ts, vs, es, **kwargs):

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    epochs = int(kwargs.get('epochs', 20))
    model_file = kwargs.get('outfile', './tagger-model.pyth')

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', basic_reporting))
    print('reporting', reporting_fns)

    after_train_fn = kwargs.get('after_train_fn', None)
    trainer = TaggerTrainerPyTorch(model, **kwargs)

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
        trainer = TaggerTrainerPyTorch(model, **kwargs)
        trainer.test(es, reporting_fns, phase='Test')
