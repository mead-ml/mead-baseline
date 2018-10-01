import time
import logging
from baseline.utils import listify, get_model_file
from baseline.progress import create_progress_bar
from baseline.train import Trainer, create_trainer, lr_decay
from baseline.dy.dynety import *


class Seq2SeqTrainerDynet(Trainer):
    def __init__(self, model, **kwargs):
        super(Seq2SeqTrainerDynet, self).__init__()
        self.model = model
        self.optimizer = optimizer(model, **kwargs)
        ##self.decay = lr_decay(**kwargs)
        self.global_step = 0
        self.valid_epochs = 0
        self.log = logging.getLogger('baseline.timing')

    @staticmethod
    def _loss(outputs, labels):
        losses = [dy.pickneglogsoftmax_batch(out, label) for out, label in zip(outputs, labels)]
        loss = dy.sum_batches(dy.esum(losses))
        return loss

    def _total(self, tgt):
        return (tgt != 0).sum()

    def train(self, loader, reporting_fns, **kwargs):
        self.model.train = True
        metrics = {}
        total_loss = 0.0
        step = 0
        total = 0
        start = time.time()
        for batch_dict in loader:
            dy.renew_cg()
            ##self.optimizer.learning_rate = self.decay(self.global_step)
            inputs = self.model.make_input(batch_dict)
            tgt = inputs.pop('tgt')
            output = self.model.forward(inputs)
            loss = self._loss(output, tgt)
            total += self._total(tgt)
            loss_val = loss.npvalue().item()
            total_loss += loss_val
            loss.backward()
            self.optimizer.update()

            step += 1
            self.global_step += 1

            if step % 500 == 0:
                avg_loss = total_loss / total
                metrics['avg_loss'] = avg_loss
                metrics['perplexity'] = np.exp(avg_loss)
                for reporting in reporting_fns:
                    reporting(metrics, self.global_step, 'Train')

        self.log.debug({'phase': 'Train', 'time': time.time() - start})
        avg_loss = total_loss / total
        metrics['avg_loss'] = avg_loss
        metrics['perplexity'] = np.exp(avg_loss)
        for reporting in reporting_fns:
            reporting(metrics, self.global_step, 'Train')
        return metrics

    def test(self, vs, reporting_fns, phase):
        self.model.train = False
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
            dy.renew_cg()
            inputs = self.model.make_input(batch_dict)
            tgt = inputs.pop('tgt')
            output = self.model.forward(inputs)
            loss = self._loss(output, tgt)
            total += self._total(tgt)
            loss_val = loss.npvalue().item()
            total_loss += loss_val
            pg.update()
        pg.done()

        self.log.debug({'phase': phase, 'time': time.time() - start})
        avg_loss = float(total_loss)/total
        metrics['avg_loss'] = avg_loss
        metrics['perplexity'] = np.exp(avg_loss)
        for reporting in reporting_fns:
            reporting(metrics, epochs, phase)
        return metrics


def fit(model, ts, vs, es=None, epochs=5, do_early_stopping=True,
        early_stopping_metric='avg_loss', **kwargs):

    patience = int(kwargs.get('patience', epochs))
    after_train_fn = kwargs.get('after_train_fn', None)

    model_file = get_model_file('seq2seq', 'dy', kwargs.get('basedir'))

    trainer = create_trainer(Seq2SeqTrainerDynet, model, **kwargs)

    if do_early_stopping:
        print("Doing early stopping on [{}] with patience [{}]".format(early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    min_metric = 10000
    last_improved = 0

    #if after_train_fn is not None:
    #    after_train_fn(model)
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
