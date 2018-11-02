import time
import logging
import dynet as dy
import numpy as np
from baseline.utils import listify, get_model_file
from baseline.train import Trainer, create_trainer, register_trainer, register_training_func
from baseline.dy.optz import OptimizerManager
from baseline.dy.dynety import *


@register_trainer(task='lm', name='default')
class LanguageModelTrainerDynet(Trainer):
    def __init__(self, model, **kwargs):
        super(LanguageModelTrainerDynet, self).__init__()
        self.model = model
        self.optimizer = OptimizerManager(model, **kwargs)
        self.valid_epochs = 0
        self.log = logging.getLogger('baseline.timing')

    @staticmethod
    def _loss(outputs, labels):
        losses = [dy.pickneglogsoftmax_batch(out, label) for out, label in zip(outputs, labels)]
        loss = dy.mean_batches(dy.esum(losses))
        return loss

    def train(self, loader, reporting_fns, **kwargs):
        metrics = {}
        total_loss = 0.0
        iters = 0
        initial_state = None
        start = time.time()
        for batch_dict in loader:
            dy.renew_cg()
            inputs = self.model.make_input(batch_dict)
            y = inputs.pop('y')
            output, initial_state = self.model.forward(inputs, initial_state)
            loss = self._loss(output, y)
            loss_val = loss.npvalue().item()
            total_loss += loss_val
            if initial_state is not None:
                initial_state = [x.npvalue() for x in initial_state]
            loss.backward()
            self.optimizer.update()

            iters += len(y)

            if self.optimizer.global_step > 0 and self.optimizer.global_step % 500 == 0:
                print(total_loss, iters)
                metrics['avg_loss'] = total_loss / iters
                metrics['perplexity'] = np.exp(total_loss / iters)
                for reporting in reporting_fns:
                    reporting(metrics, self.optimizer.global_step, 'Train')

        self.log.debug({'phase': 'Train', 'time': time.time() - start})
        metrics['avg_loss'] = total_loss / iters
        metrics['perplexity'] = np.exp(total_loss / iters)
        for reporting in reporting_fns:
            reporting(metrics, self.optimizer.global_step, 'Train')
        return metrics

    def test(self, loader, reporting_fns, phase, **kwargs):
        metrics = {}
        total_loss = 0.0
        iters = 0
        initial_state = None
        start = time.time()
        for batch_dict in loader:
            dy.renew_cg()
            inputs = self.model.make_input(batch_dict)
            y = inputs.pop('y')
            output, initial_state = self.model.forward(inputs, initial_state, train=False)
            loss = self._loss(output, y)
            loss_val = loss.npvalue().item()
            total_loss += loss_val
            if initial_state is not None:
                initial_state = [x.npvalue() for x in initial_state]
            iters += len(y)

        if phase == 'Valid':
            self.valid_epochs += 1
            output = self.valid_epochs
        else:
            output = 0

        self.log.debug({'phase': phase, 'time': time.time() - start})
        metrics['avg_loss'] = total_loss / iters
        metrics['perplexity'] = np.exp(total_loss / iters)
        for reporting in reporting_fns:
            reporting(metrics, output, phase)
        return metrics


@register_training_func('lm')
def fit(model, ts, vs, es=None, epochs=5, do_early_stopping=True, early_stopping_metric='avg_loss', **kwargs):

    patience = int(kwargs.get('patience', epochs))
    after_train_fn = kwargs.get('after_train_fn', None)

    model_file = get_model_file('lm', 'dy', kwargs.get('basedir'))

    trainer = create_trainer(model, **kwargs)

    if do_early_stopping:
        print("Doing early stopping on [{}] with patience [{}]".format(early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

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
            print("New min {:.3f}".format(min_metric))
            model.save(model_file)

        elif (epoch - last_improved) > patience:
            print("Stopping due to persistent failures to improve")
            break

    if do_early_stopping is True:
        print('Best performance on min_metric {:.3f} at epoch {}'.format(min_metric, last_improved))

    if es is not None:
        print('Reloading best checkpoint')
        model = model.load(model_file)
        trainer.test(es, reporting_fns, phase='Test')
