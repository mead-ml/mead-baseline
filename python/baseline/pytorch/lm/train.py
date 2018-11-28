import time
import logging
from baseline.pytorch.torchy import *
from baseline.utils import listify, revlut, get_model_file
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func
from baseline.pytorch.optz import OptimizerManager


@register_trainer(task='lm', name='default')
class LanguageModelTrainerPyTorch(Trainer):

    def __init__(self, model, **kwargs):
        super(LanguageModelTrainerPyTorch, self).__init__()
        self.model = model
        self.clip = float(kwargs.get('clip', 5))
        self.gpu = not bool(kwargs.get('nogpu', False))
        self.crit = model.create_loss()

        if self.gpu:
            self.model = self.model.cuda()
            self.crit.cuda()
        self.nsteps = kwargs.get('nsteps', 500)

        self.optimizer = OptimizerManager(self.model, **kwargs)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    @staticmethod
    def _get_dims(batch_dict):
        return batch_dict['y'].shape

    @staticmethod
    def _num_toks(batch_dict):
        return np.prod(LanguageModelTrainerPyTorch._get_dims(batch_dict))

    def calc_metrics(self, agg, norm):
        metrics = super(LanguageModelTrainerPyTorch, self).calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics

    def test(self, vs, reporting_fns, phase='Valid'):
        epoch = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epoch = self.valid_epochs
        start = time.time()
        self.model.eval()
        total_loss = 0
        total_toks = 0
        metrics = {}
        batchsz, nctx = self._get_dims(vs[0])

        hidden = self.model.init_hidden(batchsz)

        for batch_dict in vs:
            inputs = self.model.make_input(batch_dict)
            y = inputs.pop('y')
            output, hidden = self.model(inputs, hidden)
            toks = self._num_toks(batch_dict)
            total_loss += self.crit(output, y).item() * toks
            total_toks += toks
            if hidden is not None:
                hidden = self.repackage_hidden(hidden)
        metrics = self.calc_metrics(total_loss, total_toks)
        self.report(
            epoch, metrics, start,
            phase, 'EPOCH', reporting_fns
        )
        return metrics

    def train(self, ts, reporting_fns):
        start = time.time()
        self.nstep_start = start
        self.model.train()
        epoch_loss = 0
        epoch_toks = 0
        batchsz, nctx = self._get_dims(ts[0])
        hidden = self.model.init_hidden(batchsz)

        for batch_dict in ts:
            if hidden is not None:
                hidden = self.repackage_hidden(hidden)
            inputs = self.model.make_input(batch_dict)
            y = inputs.pop('y')
            self.optimizer.zero_grad()
            output, hidden = self.model(inputs, hidden)
            loss = self.crit(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            toks = self._num_toks(batch_dict)
            report_loss = loss.item() * toks
            epoch_loss += report_loss
            epoch_toks += toks
            self.nstep_agg += report_loss
            self.nstep_div += toks
            if (self.optimizer.global_step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    self.optimizer.global_step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()

        metrics = self.calc_metrics(epoch_loss, epoch_toks)
        self.train_epochs += 1
        self.report(
            self.train_epochs, metrics, start,
            'Train', 'EPOCH', reporting_fns
        )
        return metrics


@register_training_func('lm')
def fit(model, ts, vs, es, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    model_file = get_model_file('lm', 'pytorch', kwargs.get('basedir'))

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
        print('Best performance on max_metric %.3f at epoch %d' % (min_metric, last_improved))

    if es is not None:
        print('Reloading best checkpoint')
        model = torch.load(model_file)
        trainer = create_trainer(model, **kwargs)
        trainer.test(es, reporting_fns, phase='Test')
