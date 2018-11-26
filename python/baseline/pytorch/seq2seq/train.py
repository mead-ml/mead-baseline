import time
import logging
import torch
import numpy as np
from baseline.progress import create_progress_bar
from baseline.utils import listify, get_model_file
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func
from baseline.pytorch.optz import OptimizerManager


@register_trainer(task='seq2seq', name='default')
class Seq2SeqTrainerPyTorch(Trainer):

    def __init__(self, model, **kwargs):
        super(Seq2SeqTrainerPyTorch, self).__init__()
        self.gpu = bool(kwargs.get('gpu', True))
        self.clip = float(kwargs.get('clip', 5))
        self.model = model
        self.optimizer = OptimizerManager(self.model, **kwargs)
        self._input = model.make_input
        self.crit = model.create_loss()
        if self.gpu:
            self.model = torch.nn.DataParallel(model).cuda()
            self.crit.cuda()
        self.nsteps = kwargs.get('nsteps', 500)

    @staticmethod
    def _num_toks(tgt_lens):
        return np.sum(tgt_lens)

    def calc_metrics(self, agg, norm):
        metrics = super(Seq2SeqTrainerPyTorch, self).calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics

    def test(self, vs, reporting_fns, phase):
        self.model.eval()
        total_loss = total_toks = 0
        steps = len(vs)
        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs

        start = time.time()
        pg = create_progress_bar(steps)
        for batch_dict in pg(vs):
            input_ = self._input(batch_dict)
            tgt = input_['tgt']
            tgt_lens = batch_dict['tgt_lengths']
            pred = self.model(input_)
            loss = self.crit(pred, tgt)
            toks = self._num_toks(tgt_lens)
            total_loss += loss.item() * toks
            total_toks += toks

        metrics = self.calc_metrics(total_loss, total_toks)
        self.report(
            epochs, metrics, start,
            phase, 'EPOCH', reporting_fns
        )
        return metrics

    def train(self, ts, reporting_fns):
        self.model.train()

        epoch_loss = 0
        epoch_toks = 0

        start = time.time()
        self.nstep_start = start
        for batch_dict in ts:

            start_time = time.time()
            self.optimizer.zero_grad()
            input_ = self._input(batch_dict)
            tgt = input_['tgt']
            pred = self.model(input_)
            loss = self.crit(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            tgt_lens = batch_dict['tgt_lengths']
            tok_count = self._num_toks(tgt_lens)
            reporting_loss = loss.item() * tok_count
            epoch_loss += reporting_loss
            epoch_toks += tok_count
            self.nstep_agg += reporting_loss
            self.nstep_div += tok_count

            if (self.optimizer.global_step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    self.optimizer.global_step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns
                )
                self.reset_nstep()

        metrics = self.calc_metrics(epoch_loss, epoch_toks)
        self.train_epochs += 1
        self.report(
            self.train_epochs, metrics, start,
            'Train', 'EPOCH', reporting_fns
        )
        return metrics


@register_training_func('seq2seq')
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
    trainer = create_trainer(model, **kwargs)

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
