import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from baseline.progress import create_progress_bar
from baseline.utils import listify, get_model_file
from baseline.train import Trainer, create_trainer
from baseline.pytorch.torchy import pytorch_prepare_optimizer


class Seq2SeqTrainerPyTorch(Trainer):

    def __init__(self, model, **kwargs):
        super(Seq2SeqTrainerPyTorch, self).__init__()
        self.steps = 0
        self.gpu = bool(kwargs.get('gpu', True))
        self.clip = float(kwargs.get('clip', 5))
        self.model = model
        self.optimizer, self.scheduler = pytorch_prepare_optimizer(self.model, **kwargs)
        self._input = model.make_input
        self.crit = model.create_loss()
        if self.gpu:
            self.model = torch.nn.DataParallel(model).cuda()
            self.crit.cuda()
        self.log = logging.getLogger('baseline.timing')

    def _total(self, tgt):
        tgtt = tgt.data.long()
        return torch.sum(tgtt.ne(0)).item()

    def test(self, vs, reporting_fns, phase):
        self.model.eval()
        metrics = {}
        total_loss = total = 0
        steps = len(vs)
        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs

        start = time.time()
        pg = create_progress_bar(steps)
        for batch_dict in vs:
            input = self._input(batch_dict)
            tgt = input['tgt']
            pred = self.model(input)
            loss = self.crit(pred, tgt)
            total_loss += loss.item()
            total += self._total(tgt)
            pg.update()
        pg.done()

        self.log.debug({'phase': phase, 'time': time.time() - start})
        avg_loss = float(total_loss)/total
        metrics['avg_loss'] = avg_loss
        metrics['perplexity'] = np.exp(avg_loss)
        for reporting in reporting_fns:
            reporting(metrics, epochs, phase)
        return metrics

    def train(self, ts, reporting_fns):
        self.model.train()

        metrics = {}

        total_loss = total = 0
        duration = 0
        if self.scheduler is not None:
            self.scheduler.step()

        start = time.time()
        for batch_dict in ts:

            start_time = time.time()
            self.steps += 1
            self.optimizer.zero_grad()
            input = self._input(batch_dict)
            tgt = input['tgt']
            pred = self.model(input)
            loss = self.crit(pred, tgt)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            total += self._total(tgt)
            self.optimizer.step()
            duration += time.time() - start_time

            if self.steps % 500 == 0:
                print('Step time (%.3f sec)' % (duration / 500.))
                duration = 0
                avg_loss = float(total_loss)/total
                metrics['avg_loss'] = avg_loss
                metrics['perplexity'] = np.exp(avg_loss)
                for reporting in reporting_fns:
                    reporting(metrics, self.steps, 'Train')

        self.log.debug({'phase': 'Train', 'time': time.time() - start})
        self.train_epochs += 1
        avg_loss = float(total_loss)/total
        metrics['avg_loss'] = avg_loss
        metrics['perplexity'] = np.exp(avg_loss)
        for reporting in reporting_fns:
            reporting(metrics, self.steps, 'Train')

        return metrics


def fit(model, ts, vs, es=None, **kwargs):

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file('seq2seq', 'pytorch', kwargs.get('basedir'))

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'avg_loss')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    after_train_fn = kwargs.get('after_train_fn', None)
    trainer = create_trainer(Seq2SeqTrainerPyTorch, model, **kwargs)

    min_metric = 10000
    last_improved = 0
    for epoch in range(epochs):


        #if after_train_fn is not None:
        #    after_train_fn(model)

        trainer.train(ts, reporting_fns)

        if after_train_fn is not None:
            after_train_fn(model)

        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

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
        trainer.test(es, reporting_fns, phase='Test')
