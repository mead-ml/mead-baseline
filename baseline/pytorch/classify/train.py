import six
import logging
import torch
import torch.autograd
import os

from eight_mile.confusion import ConfusionMatrix
from eight_mile.utils import listify
from eight_mile.pytorch.optz import OptimizerManager

from baseline.progress import create_progress_bar
from baseline.utils import verbose_output, get_model_file, get_metric_cmp
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.model import create_model_for
from torch.utils.data import DataLoader
logger = logging.getLogger('baseline')


def _add_to_cm(cm, y, pred):
    _, best = pred.max(1)
    yt = y.cpu().int()
    yp = best.cpu().int()
    cm.add_batch(yt.data.numpy(), yp.data.numpy())


@register_trainer(task='classify', name='default')
class ClassifyTrainerPyTorch(EpochReportingTrainer):

    def __init__(self, model, **kwargs):

        if type(model) is dict:
            model = create_model_for('classify', **model)
        super(ClassifyTrainerPyTorch, self).__init__()
        if type(model) is dict:
            model = create_model_for('classify', **model)
        self.clip = float(kwargs.get('clip', 5))
        self.labels = model.labels
        self.gpus = int(kwargs.get('gpus', 1))
        if self.gpus == -1:
            self.gpus = len(os.getenv('CUDA_VISIBLE_DEVICES', os.getenv('NV_GPU', '0')).split(','))

        self.optimizer = OptimizerManager(model, **kwargs)
        self.model = model
        if self.gpus > 0 and self.model.gpu:
            self.crit = model.create_loss().cuda()
            if self.gpus > 1:
                self.model = torch.nn.DataParallel(model).cuda()
            else:
                self.model.cuda()
        else:
            logger.warning("Requested training on CPU.  This will be slow.")
            self.crit = model.create_loss()
            self.model = model
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)

    def _get_pytorch_model(self):
        return self.model.module if self.gpus > 1 else self.model

    def save(self, model_file):
        self._get_pytorch_model().save(model_file)

    def _make_input(self, batch_dict, **kwargs):
        return self._get_pytorch_model().make_input(batch_dict, **kwargs)

    @staticmethod
    def _get_batchsz(batch_dict):
        return len(batch_dict['y'])

    def _test(self, loader, **kwargs):
        self.model.eval()
        total_loss = 0
        total_norm = 0
        steps = len(loader)
        pg = create_progress_bar(steps)
        cm = ConfusionMatrix(self.labels)
        verbose = kwargs.get("verbose", None)
        output = kwargs.get('output')
        txts = kwargs.get('txts')
        handle = None
        line_number = 0
        if output is not None and txts is not None:
            handle = open(output, "w")

        for batch_dict in pg(loader):
            example = self._make_input(batch_dict)
            ys = example.pop('y')
            pred = self.model(example)
            loss = self.crit(pred, ys)
            if handle is not None:
                for p, y in zip(pred, ys):
                    handle.write('{}\t{}\t{}\n'.format(" ".join(txts[line_number]), self.model.labels[p], self.model.labels[y]))
                    line_number += 1
            batchsz = self._get_batchsz(batch_dict)
            total_loss += loss.item() * batchsz
            total_norm += batchsz
            _add_to_cm(cm, ys, pred)

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss / float(total_norm)
        verbose_output(verbose, cm)
        if handle is not None:
            handle.close()

        return metrics

    def _train(self, loader, **kwargs):
        self.model.train()
        reporting_fns = kwargs.get('reporting_fns', [])
        steps = len(loader)
        pg = create_progress_bar(steps)
        cm = ConfusionMatrix(self.labels)
        epoch_loss = 0
        epoch_div = 0
        for batch_dict in pg(loader):
            self.optimizer.zero_grad()
            example = self._make_input(batch_dict)
            y = example.pop('y')
            pred = self.model(example)
            loss = self.crit(pred, y)
            batchsz = self._get_batchsz(batch_dict)
            report_loss = loss.item() * batchsz
            epoch_loss += report_loss
            epoch_div += batchsz
            self.nstep_agg += report_loss
            self.nstep_div += batchsz
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            _add_to_cm(cm, y, pred)
            self.optimizer.step()

            if (self.optimizer.global_step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    self.optimizer.global_step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = epoch_loss / float(epoch_div)
        return metrics


@register_training_func('classify')
def fit(model_params, ts, vs, es, **kwargs):
    """
    Train a classifier using PyTorch
    :param model_params: The model to train
    :param ts: A training data set
    :param vs: A validation data set
    :param es: A test data set, can be None
    :param kwargs: See below

    :Keyword Arguments:
        * *do_early_stopping* (``bool``) -- Stop after eval data is not improving. Default to True
        * *epochs* (``int``) -- how many epochs.  Default to 20
        * *outfile* -- Model output file, defaults to classifier-model.pyth
        * *patience* --
           How many epochs where evaluation is no longer improving before we give up
        * *reporting* --
           Callbacks which may be used on reporting updates
        * *optim* --
           Optimizer to use, defaults to `sgd`
        * *eta, lr* (``float``) --
           Learning rate, defaults to 0.01
        * *mom* (``float``) --
           Momentum (SGD only), defaults to 0.9 if optim is `sgd`
    :return:
    """
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    verbose = kwargs.get('verbose', {'console': kwargs.get('verbose_console', False), 'file': kwargs.get('verbose_file', None)})
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file('classify', 'pytorch', kwargs.get('basedir'))
    output = kwargs.get('output')
    txts = kwargs.get('txts')

    num_loader_workers = int(kwargs.get('num_loader_workers', 0))
    pin_memory = bool(kwargs.get('pin_memory', True))
    ts = DataLoader(ts, num_workers=num_loader_workers, batch_size=None, pin_memory=pin_memory)
    vs = DataLoader(vs, batch_size=None, pin_memory=pin_memory)
    es = DataLoader(es, batch_size=None, pin_memory=pin_memory) if es is not None else None

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)
    trainer = create_trainer(model_params, **kwargs)

    last_improved = 0

    for epoch in range(epochs):
        trainer.train(ts, reporting_fns)
        test_metrics = trainer.test(vs, reporting_fns)

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
        test_metrics = trainer.test(es, reporting_fns, phase='Test', verbose=verbose, output=output, txts=txts)
    return test_metrics
