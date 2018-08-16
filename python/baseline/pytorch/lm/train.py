from baseline.pytorch.torchy import *
from baseline.utils import listify, revlut, get_model_file
from baseline.train import Trainer, create_trainer
import time
import logging


class LanguageModelTrainerPyTorch(Trainer):

    def __init__(self, model, **kwargs):
        super(LanguageModelTrainerPyTorch, self).__init__()
        self.train_steps = 0
        self.valid_epochs = 0
        self.model = model
        self.clip = float(kwargs.get('clip', 5))
        self.gpu = not bool(kwargs.get('nogpu', False))
        self.crit = model.create_loss()

        if self.gpu:
            self.model = self.model.cuda()
            self.crit.cuda()
        self.log = logging.getLogger('baseline.reporting')

        self.optimizer, self.scheduler = pytorch_prepare_optimizer(self.model, **kwargs)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def _get_dims(self, ts):
        np_array = ts[0]['x']
        return np_array.shape

    def test(self, vs, reporting_fns, phase='Valid'):
        start_time = time.time()
        self.model.eval()
        total_loss = 0
        metrics = {}
        batchsz, nbptt = self._get_dims(vs)

        hidden = self.model.init_hidden(batchsz)
        iters = 0

        for batch_dict in vs:
            inputs = self.model.make_input(batch_dict)
            output, hidden = self.model(inputs[:-1], hidden)
            total_loss += self.crit(output, inputs[-1]).data
            hidden = self.repackage_hidden(hidden)
            iters += nbptt
        self.valid_epochs += 1

        avg_loss = float(total_loss) / iters / batchsz
        metrics['avg_loss'] = avg_loss
        metrics['perplexity'] = np.exp(avg_loss)

        duration = time.time() - start_time
        print('%s time (%.3f sec)' % (phase, duration))
        self.log.debug({'phase': phase, 'time': duration})

        for reporting in reporting_fns:
            reporting(metrics, self.valid_epochs, phase)
        return metrics

    def train(self, ts, reporting_fns):
        start_time = time.time()
        self.model.train()
        if self.scheduler is not None:
            self.scheduler.step()
        total_loss = 0
        metrics = {}
        batchsz, nbptt = self._get_dims(ts)
        hidden = self.model.init_hidden(batchsz)
        iters = 0

        for batch_dict in ts:
            hidden = self.repackage_hidden(hidden)
            inputs = self.model.make_input(batch_dict)
            self.optimizer.zero_grad()
            output, hidden = self.model(inputs[:-1], hidden)
            loss = self.crit(output, inputs[-1])
            loss.backward()
            total_loss += loss.data
            iters += nbptt
            self.train_steps += 1
            if self.train_steps % 500 == 0:
                avg_loss = float(total_loss) / iters / batchsz
                metrics['avg_loss'] = avg_loss
                metrics['perplexity'] = np.exp(avg_loss)
                for reporting in reporting_fns:
                    reporting(metrics, self.train_steps, 'Train')

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

        avg_loss = float(total_loss) / iters / batchsz
        metrics['avg_loss'] = avg_loss
        metrics['perplexity'] = np.exp(avg_loss)

        duration = time.time() - start_time
        print('Training time (%.3f sec)' % duration)
        self.log.debug({'phase': 'Train', 'time': duration})

        for reporting in reporting_fns:
            reporting(metrics, self.train_epochs * len(ts), 'Train')
        return metrics


def fit(model, ts, vs, es, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    model_file = get_model_file(kwargs, 'lm', 'pytorch')

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'avg_loss')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    after_train_fn = kwargs.get('after_train_fn', None)
    trainer = create_trainer(LanguageModelTrainerPyTorch, model, **kwargs)
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
            #if validation_improvement_fn is not None:
            #    validation_improvement_fn(early_stopping_metric, test_metrics, epoch, max_metric, last_improved)
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
        trainer = create_trainer(LanguageModelTrainerPyTorch, model, **kwargs)
        trainer.test(es, reporting_fns, phase='Test')
