from baseline.utils import listify, get_model_file
from baseline.progress import create_progress_bar
from baseline.confusion import ConfusionMatrix
from baseline.reporting import basic_reporting
from baseline.train import EpochReportingTrainer, create_trainer
import torch
import torch.autograd


def _add_to_cm(cm, y, pred):
    _, best = pred.max(1)
    yt = y.cpu().int()
    yp = best.cpu().int().squeeze()
    cm.add_batch(yt.data.numpy(), yp.data.numpy())


class ClassifyTrainerPyTorch(EpochReportingTrainer):

    def __init__(self, model, **kwargs):
        super(ClassifyTrainerPyTorch, self).__init__()
        eta = kwargs.get('eta', kwargs.get('lr', 0.01))
        print('using eta [%.3f]' % eta)
        optim = kwargs.get('optim', 'sgd')
        print('using optim [%s]' % optim)

        parameters = filter(lambda p: p.requires_grad, model.parameters())
        self.labels = model.labels
        if optim == 'adadelta':
            print('Using adadelta, ignoring learning rate')
            self.optimizer = torch.optim.Adadelta(parameters)
        elif optim == 'adam':
            self.optimizer = torch.optim.Adam(parameters)
        elif optim == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=eta)
        elif optim == 'asgd':
            self.optimizer = torch.optim.ASGD(model.parameters(), lr=eta)
        else:
            mom = kwargs.get('mom', 0.9)
            print('using mom [%.3f]' % mom)
            self.optimizer = torch.optim.SGD(parameters, lr=eta, momentum=mom)

        self.model = model.cuda()
        self.crit = model.create_loss().cuda()

    def _test(self, loader):
        self.model.eval()
        total_loss = 0
        steps = len(loader)
        pg = create_progress_bar(steps)
        cm = ConfusionMatrix(self.labels)

        for batch_dict in loader:
            x, y = self.model.make_input(batch_dict)
            pred = self.model(x)
            loss = self.crit(pred, y)
            total_loss += loss.data[0]
            _add_to_cm(cm, y, pred)
            pg.update()
        pg.done()

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss/float(steps)

        return metrics

    def _train(self, loader):
        self.model.train()
        steps = len(loader)
        pg = create_progress_bar(steps)
        cm = ConfusionMatrix(self.labels)
        total_loss = 0
        for batch_dict in loader:
            self.optimizer.zero_grad()
            x, y = self.model.make_input(batch_dict)
            pred = self.model(x)
            loss = self.crit(pred, y)
            total_loss += loss.data[0]
            loss.backward()
            _add_to_cm(cm, y, pred)
            self.optimizer.step()
            pg.update()
        pg.done()

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss/float(steps)
        return metrics


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
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file(kwargs, 'classify', 'pytorch')
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))    

    reporting_fns = listify(kwargs.get('reporting', basic_reporting))
    print('reporting', reporting_fns)


    trainer = create_trainer(ClassifyTrainerPyTorch, model, **kwargs)

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
        trainer = create_trainer(ClassifyTrainerPyTorch, model, **kwargs)
        trainer.test(es, reporting_fns, phase='Test')
