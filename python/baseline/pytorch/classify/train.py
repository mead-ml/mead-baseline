import six
import time
import torch
import torch.autograd
from baseline.utils import verbose_output
from baseline.confusion import ConfusionMatrix
from baseline.progress import create_progress_bar
from baseline.utils import listify, get_model_file
from baseline.pytorch.optz import OptimizerManager
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func


def _add_to_cm(cm, y, pred):
    _, best = pred.max(1)
    yt = y.cpu().int()
    yp = best.cpu().int()
    cm.add_batch(yt.data.numpy(), yp.data.numpy())


@register_trainer(task='classify', name='default')
class ClassifyTrainerPyTorch(EpochReportingTrainer):

    def __init__(self, model, **kwargs):
        super(ClassifyTrainerPyTorch, self).__init__()
        self.clip = float(kwargs.get('clip', 5))
        self.labels = model.labels
        self.optimizer = OptimizerManager(model, **kwargs)
        self.crit = model.create_loss().cuda()
        self.model = torch.nn.DataParallel(model).cuda()
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)

    def _make_input(self, batch_dict):
        return self.model.module.make_input(batch_dict)

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

        for batch_dict in pg(loader):
            example = self._make_input(batch_dict)
            y = example.pop('y')
            pred = self.model(example)
            loss = self.crit(pred, y)
            batchsz = self._get_batchsz(batch_dict)
            total_loss += loss.item() * batchsz
            total_norm += batchsz
            _add_to_cm(cm, y, pred)

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss / float(total_norm)
        verbose_output(verbose, cm)

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
def fit(model, ts, vs, es, **kwargs):
    """
    Train a classifier using PyTorch
    :param model: The model to train
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
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)


    trainer = create_trainer(model, **kwargs)

    max_metric = 0
    last_improved = 0

    for epoch in range(epochs):
        trainer.train(ts, reporting_fns)
        test_metrics = trainer.test(vs, reporting_fns)

        if do_early_stopping is False:
            model.save(model_file)

        elif test_metrics[early_stopping_metric] > max_metric:
            last_improved = epoch
            max_metric = test_metrics[early_stopping_metric]
            print('New max %.3f' % max_metric)
            model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on max_metric %.3f at epoch %d' % (max_metric, last_improved))

    if es is not None:
        print('Reloading best checkpoint')
        model = torch.load(model_file)
        trainer = create_trainer(model, **kwargs)
        trainer.test(es, reporting_fns, phase='Test', verbose=verbose)
