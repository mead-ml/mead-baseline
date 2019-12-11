import six
import logging
import torch
import torch.autograd
import os

from eight_mile.confusion import ConfusionMatrix
from eight_mile.progress import create_progress_bar
from eight_mile.utils import listify
from eight_mile.pytorch.optz import OptimizerManager

from baseline.utils import verbose_output, get_model_file, get_metric_cmp
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.model import create_model_for

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
        if self.gpus > 0:
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

    def _make_input(self, batch_dict):
        return self._get_pytorch_model().make_input(batch_dict)

    @staticmethod
    def _get_batchsz(batch_dict):
        return len(batch_dict['y'])

    def _test(self, loader, **kwargs):
        self.model.eval()
        total_loss = 0
        total_norm = 0
        steps = len(loader)
        pg = create_progress_bar(steps)
        if self.model.is_multilabel:
            tps = torch.zeros(len(self.labels))
            tp_fns = torch.zeros(len(self.labels))
            tp_fps = torch.zeros(len(self.labels))
            total_tps = 0.
            total_tp_fps = 0.
            total_tp_fns = 0.
        else:
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

            if self.model.is_multilabel:
                ons = torch.sigmoid(pred)
                ons[pred > 0.5] = 1
                ons = ons.int()
                yt = ys.int()
                tps += (yt & ons).sum(0).cpu().float()
                tp_fns += yt.sum(0).cpu().float()
                tp_fps += ons.sum(0).cpu().float()

                total_tps += tps.sum().item()
                total_tp_fns += tp_fns.sum().item()
                total_tp_fps += tp_fps.sum().item()
            else:
                _add_to_cm(cm, ys, pred)

        if self.model.is_multilabel:
            precision = tps / tp_fps
            recall = tps / tp_fns
            f1 = 2 * precision * recall / (precision + recall)

            total_precision = total_tps / total_tp_fps
            total_recall = total_tps / total_tp_fns
            total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)

            metrics = {}
            metrics['precision'] = total_precision
            metrics['recall'] = total_recall
            metrics['f1'] = total_f1
            metrics['class_precision'] = {k: v.item() for k, v in zip(self.model.labels, precision) if v.item() == v.item()}
            metrics['class_recall'] = {k: v.item() for k, v in zip(self.model.labels, recall) if v.item() == v.item()}
            metrics['class_f1'] = {k: v.item() for k, v in zip(self.model.labels, f1) if v.item() == v.item()}


        else:
            metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss / float(total_norm)
        if not self.model.is_multilabel:
            verbose_output(verbose, cm)
        if handle is not None:
            handle.close()

        return metrics

    def _train(self, loader, **kwargs):
        self.model.train()
        reporting_fns = kwargs.get('reporting_fns', [])
        steps = len(loader)
        pg = create_progress_bar(steps)
        if self.model.is_multilabel:
            tps = torch.zeros(len(self.labels))
            tp_fns = torch.zeros(len(self.labels))
            tp_fps = torch.zeros(len(self.labels))
            total_tps = 0.
            total_tp_fps = 0.
            total_tp_fns = 0.
        else:
            cm = ConfusionMatrix(self.labels)
        epoch_loss = 0
        epoch_div = 0
        for batch_dict in pg(loader):
            self.optimizer.zero_grad()
            example = self._make_input(batch_dict)
            ys = example.pop('y')
            pred = self.model(example)
            loss = self.crit(pred, ys)
            batchsz = self._get_batchsz(batch_dict)
            report_loss = loss.item() * batchsz
            epoch_loss += report_loss
            epoch_div += batchsz
            self.nstep_agg += report_loss
            self.nstep_div += batchsz
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

            if self.model.is_multilabel:
                ons = torch.sigmoid(pred)
                ons[pred > 0.5] = 1
                ons = ons.int()
                yt = ys.int()
                tps += (yt & ons).sum(0).cpu().float()
                tp_fns += yt.sum(0).cpu().float()
                tp_fps += ons.sum(0).cpu().float()

                total_tps += tps.sum().item()
                total_tp_fns += tp_fns.sum().item()
                total_tp_fps += tp_fps.sum().item()
            else:
                _add_to_cm(cm, ys, pred)
            self.optimizer.step()

            if (self.optimizer.global_step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    self.optimizer.global_step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()

        if self.model.is_multilabel:
            precision = tps / tp_fps
            recall = tps / tp_fns
            f1 = 2 * precision * recall / (precision + recall)

            total_precision = total_tps / total_tp_fps
            total_recall = total_tps / total_tp_fns
            total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)

            metrics = {}
            metrics['precision'] = total_precision
            metrics['recall'] = total_recall
            metrics['f1'] = total_f1
            metrics['class_precision'] = {k: v.item() for k, v in zip(self.model.labels, precision) if v.item() == v.item()}
            metrics['class_recall'] = {k: v.item() for k, v in zip(self.model.labels, recall) if v.item() == v.item()}
            metrics['class_f1'] = {k: v.item() for k, v in zip(self.model.labels, f1) if v.item() == v.item()}


        else:
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
    
    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('eatly_stopping_cmp'))
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
