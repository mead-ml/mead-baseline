import time
import logging
from baseline.pytorch.torchy import *
from eight_mile.utils import listify, revlut
from eight_mile.pytorch.optz import OptimizerManager
from baseline.utils import get_model_file, get_metric_cmp
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func
from baseline.model import create_model_for
from torch.utils.data import DataLoader
logger = logging.getLogger('baseline')


@register_trainer(task='lm', name='default')
class LanguageModelTrainerPyTorch(Trainer):

    def __init__(self, model, **kwargs):
        super().__init__()
        if type(model) is dict:
            checkpoint = kwargs.get('checkpoint')
            if checkpoint:
                model['checkpoint'] = checkpoint
            model = create_model_for('lm', **model)
        self.model = model
        self.clip = float(kwargs.get('clip', 5))
        self.gpus = kwargs.get('gpus', 1)
        if self.gpus > 0:
            self.crit = model.create_loss().cuda()
            if self.gpus > 1:
                self.model = torch.nn.DataParallel(model).cuda()
            else:
                self.model.cuda()
        else:
            logger.warning("Requested training on CPU.  This will be slow.")
            self.crit = model.create_loss()

        self.nsteps = kwargs.get('nsteps', 500)
        self.optimizer = OptimizerManager(self.model, **kwargs)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def save(self, model_file):
        self._get_pytorch_model().save(model_file)

    def _get_pytorch_model(self):
        return self.model.module if self.gpus > 1 else self.model

    @staticmethod
    def _get_dims(loader):
        batch_dict = loader.dataset[0]
        return batch_dict['y'].shape

    @staticmethod
    def _num_toks(batch_dict):
        return np.prod(batch_dict['y'].shape)

    def calc_metrics(self, agg, norm):
        metrics = super().calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics

    def test(self, vs, reporting_fns, phase='Valid', **kwargs):
        epoch = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epoch = self.valid_epochs
        start = time.perf_counter()
        self.model.eval()
        total_loss = 0
        total_toks = 0
        batchsz, nctx = self._get_dims(vs)
        hidden = self._get_pytorch_model().zero_state(batchsz)

        for batch_dict in vs:
            inputs = self._get_pytorch_model().make_input(batch_dict)
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
        start = time.perf_counter()
        self.nstep_start = start
        self.model.train()
        epoch_loss = 0
        epoch_toks = 0
        batchsz, nctx = self._get_dims(ts)
        hidden = self._get_pytorch_model().zero_state(batchsz)

        for batch_dict in ts:
            if hidden is not None:
                hidden = self.repackage_hidden(hidden)
            inputs = self._get_pytorch_model().make_input(batch_dict)
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
                metrics['lr'] = self.optimizer.current_lr

                self.report(
                    self.optimizer.global_step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()

        metrics = self.calc_metrics(epoch_loss, epoch_toks)
        metrics['lr'] = self.optimizer.current_lr

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

    num_loader_workers = int(kwargs.get('num_loader_workers', 0))
    pin_memory = bool(kwargs.get('pin_memory', True))
    if not isinstance(ts, DataLoader):
        ts = DataLoader(ts, num_workers=num_loader_workers, batch_size=None, pin_memory=pin_memory)
    if not isinstance(vs, DataLoader):
        vs = DataLoader(vs, batch_size=None, pin_memory=pin_memory)
    if es and not isinstance(es, DataLoader):
        es = DataLoader(es, batch_size=None, pin_memory=pin_memory)
    best_metric = 10000
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'avg_loss')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)

    after_train_fn = kwargs.get('after_train_fn', None)
    trainer = create_trainer(model, **kwargs)
    last_improved = 0
    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(model)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            logger.info('New best %.3f', best_metric)
            trainer.save(model_file)

        elif (epoch - last_improved) > patience:
            logger.info('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        logger.info('Best performance on %s: %.3f at epoch %d', early_stopping_metric, best_metric, last_improved)

    if es is not None:
        logger.info('Reloading best checkpoint')
        model = torch.load(model_file)
        trainer = create_trainer(model, **kwargs)
        test_metrics = trainer.test(es, reporting_fns, phase='Test')
    return test_metrics
