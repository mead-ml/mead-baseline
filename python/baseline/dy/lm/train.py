import time
import logging
import dynet as dy
import numpy as np
from baseline.utils import listify, get_model_file, get_metric_cmp
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func
from baseline.dy.optz import OptimizerManager
from baseline.dy.dynety import *

logger = logging.getLogger('baseline')


@register_trainer(task='lm', name='default')
class LanguageModelTrainerDynet(Trainer):

    def __init__(self, model, **kwargs):
        super(LanguageModelTrainerDynet, self).__init__()
        self.model = model
        self.optimizer = OptimizerManager(model, **kwargs)
        self.nsteps = kwargs.get('nsteps', 500)

    @staticmethod
    def _loss(outputs, labels):
        losses = [dy.pickneglogsoftmax_batch(out, label) for out, label in zip(outputs, labels)]
        loss = dy.mean_batches(dy.average(losses))
        return loss

    @staticmethod
    def _num_toks(batch_dict):
        return np.prod(batch_dict['y'].shape)

    def calc_metrics(self, agg, norm):
        metrics = super(LanguageModelTrainerDynet, self).calc_metrics(agg, norm)
        metrics['perplexity'] = np.exp(metrics['avg_loss'])
        return metrics

    def train(self, loader, reporting_fns, **kwargs):
        metrics = {}
        epoch_loss = 0.0
        epoch_toks = 0
        initial_state = None
        start = time.time()
        self.nstep_start = start
        for batch_dict in loader:
            dy.renew_cg()
            inputs = self.model.make_input(batch_dict)
            y = inputs.pop('y')
            output, initial_state = self.model.forward(inputs, initial_state)
            loss = self._loss(output, y)
            toks = self._num_toks(batch_dict)
            loss_val = loss.npvalue().item() * toks
            epoch_loss += loss_val
            epoch_toks += toks
            self.nstep_agg += loss_val
            self.nstep_div += toks
            if initial_state is not None:
                initial_state = [x.npvalue() for x in initial_state]
            loss.backward()
            self.optimizer.update()

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

    def test(self, loader, reporting_fns, phase, **kwargs):
        metrics = {}
        total_loss = 0.0
        total_toks = 0
        initial_state = None
        start = time.time()
        for batch_dict in loader:
            dy.renew_cg()
            inputs = self.model.make_input(batch_dict)
            y = inputs.pop('y')
            output, initial_state = self.model.forward(inputs, initial_state, train=False)
            loss = self._loss(output, y)
            toks = self._num_toks(batch_dict)
            loss_val = loss.npvalue().item() * toks
            total_loss += loss_val
            total_toks += toks
            if initial_state is not None:
                initial_state = [x.npvalue() for x in initial_state]

        epochs = 0
        if phase == 'Valid':
            self.valid_epochs += 1
            epochs = self.valid_epochs

        metrics = self.calc_metrics(total_loss, total_toks)
        self.report(
            epochs, metrics, start,
            phase, 'EPOCH', reporting_fns
        )
        return metrics


@register_training_func('lm')
def fit(model, ts, vs, es=None, epochs=5, do_early_stopping=True, early_stopping_metric='avg_loss', **kwargs):

    patience = int(kwargs.get('patience', epochs))
    after_train_fn = kwargs.get('after_train_fn', None)

    model_file = get_model_file('lm', 'dy', kwargs.get('basedir'))

    trainer = create_trainer(model, **kwargs)

    best_metric = 10000
    if do_early_stopping:
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        logger.info("Doing early stopping on [%s] with patience [%d]", early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)

    last_improved = 0

    for epoch in range(epochs):
        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(model)

        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            logger.info("New best %.3f", best_metric)
            model.save(model_file)

        elif (epoch - last_improved) > patience:
            logger.info("Stopping due to persistent failures to improve")
            break

    if do_early_stopping is True:
        logger.info('Best performance on %s: %.3f at epoch %d', early_stopping_metric, best_metric, last_improved)

    if es is not None:
        logger.info('Reloading best checkpoint')
        model = model.load(model_file)
        test_metrics = trainer.test(es, reporting_fns, phase='Test')
    return test_metrics
