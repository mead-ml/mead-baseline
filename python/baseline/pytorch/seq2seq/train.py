import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import numpy as np
from baseline.progress import ProgressBar
from baseline.reporting import basic_reporting
from baseline.utils import listify


class Seq2SeqTrainerPyTorch:

    def __init__(self, model, **kwargs):

        self.gpu = bool(kwargs.get('gpu', True))
        optim = kwargs.get('optim', 'adam')
        eta = float(kwargs.get('eta', 0.01))
        mom = float(kwargs.get('mom', 0.9))
        self.clip = float(kwargs.get('clip', 5))

        if optim == 'adadelta':
            self.optimizer = torch.optim.Adadelta(model.parameters(), lr=eta)
        elif optim == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=eta)
        elif optim == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=eta)
        else:
            self.optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=mom)
        self.model = model
        self.crit = model.create_loss()
        if self.gpu:
            self.model = torch.nn.DataParallel(model).cuda()
            self.crit.cuda()
    
    def _wrap(self, src, tgt):
        dst = tgt[:,:-1]
        tgt = tgt[:,1:]
        if self.gpu:
            src = src.cuda()
            dst = dst.cuda()
            tgt = tgt.cuda()
        return Variable(src), Variable(dst), Variable(tgt)
    
    def _total(self, tgt):
        tgtt = tgt.data.long()
        return torch.sum(tgtt.ne(0))

    def test(self, ts):
        self.model.eval()
        metrics = {}
        total_loss = total = 0
        steps = len(ts)
        pg = ProgressBar(steps)
        for src, tgt, src_len, tgt_len in ts:
            src, dst, tgt = self._wrap(src, tgt)
            pred = self.model((src, dst))
            loss = self.crit(pred, tgt)
            total_loss += loss.data[0]
            total += self._total(tgt)
            pg.update()
        pg.done()

        avg_loss = float(total_loss)/total
        metrics['avg_loss'] = avg_loss
        metrics['perplexity'] = np.exp(avg_loss)
        return metrics

    def train(self, ts):
        self.model.train()

        metrics = {}
        steps = len(ts)
        total_loss = total = 0
        pg = ProgressBar(steps)
        for src, tgt, src_len, tgt_len in ts:
            self.optimizer.zero_grad()
            src, dst, tgt = self._wrap(src, tgt)
            pred = self.model((src, dst))
            loss = self.crit(pred, tgt)
            total_loss += loss.data[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
            total += self._total(tgt)
            self.optimizer.step()
            pg.update()
        pg.done()

        avg_loss = float(total_loss)/total
        metrics['avg_loss'] = avg_loss
        metrics['perplexity'] = np.exp(avg_loss)
        return metrics


def fit(model, ts, vs, es=None, **kwargs):

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    epochs = int(kwargs.get('epochs', 20))
    model_file = kwargs.get('outfile', './seq2seq-model.pyth')

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'avg_loss')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', basic_reporting))
    print('reporting', reporting_fns)

    after_train_fn = kwargs.get('after_train_fn', None)
    trainer = Seq2SeqTrainerPyTorch(model, **kwargs)

    min_metric = 10000
    last_improved = 0
    for epoch in range(epochs):

        start_time = time.time()
        train_metrics = trainer.train(ts)
        train_duration = time.time() - start_time
        print('Training time (%.3f sec)' % train_duration)

        if after_train_fn is not None:
            after_train_fn(model)

        start_time = time.time()
        test_metrics = trainer.test(vs)
        test_duration = time.time() - start_time
        print('Validation time (%.3f sec)' % test_duration)

        for reporting in reporting_fns:
            reporting(train_metrics, epoch, 'Train')
            reporting(test_metrics, epoch, 'Valid')

        if do_early_stopping is False:
            model.save(model_file)

        elif test_metrics[early_stopping_metric] < min_metric:
            last_improved = epoch
            min_metric = test_metrics[early_stopping_metric]
            print('New min %.3f' % min_metric)
            model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on min_metric %.3f at epoch %d' % (min_metric, last_improved))

    if es is not None:
        model.load(model_file)
        trainer = Seq2SeqTrainerPyTorch(model, **kwargs)
        start_time = time.time()
        test_metrics = trainer.test(es)
        test_duration = time.time() - start_time
        print('Test time (%.3f sec)' % test_duration)

        for reporting in reporting_fns:
            reporting(test_metrics, 0, 'Test')