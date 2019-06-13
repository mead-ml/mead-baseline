import six
import time
import logging
import dynet as dy
import numpy as np
from baseline.utils import listify, get_model_file, get_metric_cmp
from baseline.progress import create_progress_bar
from baseline.confusion import ConfusionMatrix
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.dy.dynety import *
from baseline.dy.optz import *
from baseline.utils import verbose_output

logger = logging.getLogger('baseline')


def _add_to_cm(cm, y, preds, axis=0):
    best = np.argmax(preds, axis=axis)
    best = np.reshape(best, y.shape)
    cm.add_batch(y, best)


@register_trainer(task='classify', name='default')
class ClassifyTrainerDynet(EpochReportingTrainer):

    def __init__(self, model, **kwargs):
        super(ClassifyTrainerDynet, self).__init__()
        self.model = model
        self.labels = model.labels
        self.optimizer = OptimizerManager(model, **kwargs)
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)

    def _update(self, loss):
        loss.backward()
        self.optimizer.update()

    def _log(self, steps, loss, norm, reporting_fns):
        self.nstep_agg += loss
        self.nstep_div += norm
        if (steps + 1) % self.nsteps == 0:
            metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
            self.report(
                steps + 1, metrics, self.nstep_start,
                'Train', 'STEP', reporting_fns, self.nsteps
            )
            self.reset_nstep()

    def _dummy_log(self, *args):
        """This is a no op that validation calls to avoid adding to the nstep totals."""
        pass

    @staticmethod
    def _get_batchsz(batch_dict):
        return len(batch_dict['y'])

    def _step(self, loader, update, log, reporting_fns, verbose=None, output=None, txts=None):
        steps = len(loader)
        pg = create_progress_bar(steps)
        cm = ConfusionMatrix(self.labels)
        epoch_loss = 0
        epoch_div = 0
        handle = None
        line_number = 0
        if output is not None and txts is not None:
            handle = open(output, "w")

        for batch_dict in pg(loader):
            dy.renew_cg()
            inputs = self.model.make_input(batch_dict)
            ys = inputs.pop('y')
            preds = self.model.forward(inputs)
            losses = self.model.loss(preds, ys)
            loss = dy.mean_batches(losses)
            batchsz = self._get_batchsz(batch_dict)
            lossv = loss.npvalue().item() * batchsz
            if handle is not None:
                for p, y in zip(preds, ys):
                    handle.write('{}\t{}\t{}\n'.format(" ".join(txts[line_number]), self.model.labels[p], self.model.labels[y]))
                    line_number += 1
            epoch_loss += lossv
            epoch_div += batchsz
            _add_to_cm(cm, ys, preds.npvalue())
            update(loss)
            log(self.optimizer.global_step, lossv, batchsz, reporting_fns)

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = epoch_loss / float(epoch_div)
        verbose_output(verbose, cm)
        if handle is not None:
            handle.close()
        return metrics

    def _test(self, loader, **kwargs):
        self.model.train = False
        return self._step(loader, lambda x: None, self._dummy_log, [], kwargs.get("verbose", None), kwargs.get('output'),
                          kwargs.get('txts'))

    def _train(self, loader, **kwargs):
        self.model.train = True
        return self._step(loader, self._update, self._log, kwargs.get('reporting_fns', []))


@register_trainer(task='classify', name='autobatch')
class ClassifyTrainerAutobatch(ClassifyTrainerDynet):
    def __init__(self, model, autobatchsz=1, **kwargs):
        self.autobatchsz = autobatchsz
        super(ClassifyTrainerAutobatch, self).__init__(model, **kwargs)

    def _step(self, loader, update, log, reporting_fns, verbose=None, output=None, txts=None):
        steps = len(loader)
        pg = create_progress_bar(steps)
        cm = ConfusionMatrix(self.labels)
        epoch_loss = 0
        epoch_div = 0
        preds, losses, ys = [], [], []
        dy.renew_cg()
        for i, batch_dict in enumerate(pg(loader), 1):
            inputs = self.model.make_input(batch_dict)
            y = inputs.pop('y')
            pred = self.model.forward(inputs)
            preds.append(pred)
            loss = self.model.loss(pred, y)
            losses.append(loss)
            ys.append(y)
            if i % self.autobatchsz == 0:
                loss = dy.average(losses)
                preds = dy.concatenate_cols(preds)
                batchsz = len(losses)
                lossv = loss.npvalue().item() * batchsz
                epoch_loss += lossv
                epoch_div += batchsz
                _add_to_cm(cm, np.array(ys), preds.npvalue())
                update(loss)
                log(self.optimizer.global_step, lossv, batchsz, reporting_fns)
                preds, losses, ys = [], [], []
                dy.renew_cg()
        loss = dy.average(losses)
        preds = dy.concatenate_cols(preds)
        batchsz = len(losses)
        epoch_loss += loss.npvalue().item() * batchsz
        epoch_div += batchsz
        _add_to_cm(cm, np.array(ys), preds.npvalue())
        update(loss)

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = epoch_loss / float(epoch_div)
        verbose_output(verbose, cm)
        return metrics


@register_training_func(task='classify')
def fit(model, ts, vs, es, epochs=20, do_early_stopping=True, early_stopping_metric='acc', **kwargs):
    autobatchsz = kwargs.get('autobatchsz', 1)
    verbose = kwargs.get('verbose', {'print': kwargs.get('verbose_print', False), 'file': kwargs.get('verbose_file', None)})
    model_file = get_model_file('classify', 'dynet', kwargs.get('basedir'))
    output = kwargs.get('output')
    txts = kwargs.get('txts')
    best_metric = 0
    if do_early_stopping:
        patience = kwargs.get('patience', epochs)
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)

    trainer = create_trainer(model, **kwargs)

    last_improved = 0

    for epoch in range(epochs):
        trainer.train(ts, reporting_fns)
        test_metrics = trainer.test(vs, reporting_fns)

        if do_early_stopping is False:
            model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            logger.info('New best %.3f', best_metric)
            model.save(model_file)

        elif (epoch - last_improved) > patience:
            logger.info('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        logger.info('Best performance on %s: %.3f at epoch %d', early_stopping_metric, best_metric, last_improved)
    if es is not None:
        logger.info('Reloading best checkpoint')
        model = model.load(model_file)
        test_metrics = trainer.test(es, reporting_fns, phase='Test', verbose=verbose, output=output, txts=txts)
    return test_metrics
