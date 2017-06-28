from baseline.utils import listify
from baseline.progress import ProgressBar
from baseline.confusion import ConfusionMatrix
from baseline.reporting import basic_reporting
import torch
import torch.autograd
import time
def _add_to_cm(cm, y, pred):
    _, best = pred.max(1)
    yt = y.cpu().int()
    yp = best.cpu().int().squeeze()
    cm.add_batch(yt.data.numpy(), yp.data.numpy())


class ClassifyTrainerPyTorch:

    def __init__(self, model, **kwargs):
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
        else:
            mom = kwargs.get('mom', 0.9)
            print('using mom [%.3f]' % mom)
            self.optimizer = torch.optim.SGD(parameters, lr=eta, momentum=mom)

        self.model = torch.nn.DataParallel(model).cuda()
        self.crit = model.create_loss().cuda()

    def test(self, loader):
        self.model.eval()
        total_loss = 0
        steps = len(loader)
        pg = ProgressBar(steps)
        cm = ConfusionMatrix(self.labels)

        for x, y in loader:
            if type(x) == list:
                x = [torch.autograd.Variable(item.cuda()) for item in x]
            else:
                x = torch.autograd.Variable(x.cuda())
            y = torch.autograd.Variable(y.cuda())
            pred = self.model(x)
            loss = self.crit(pred, y)
            total_loss += loss.data[0]
            _add_to_cm(cm, y, pred)
            pg.update()
        pg.done()

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss/float(steps)

        return metrics

    def train(self, loader):
        self.model.train()
        steps = len(loader)
        pg = ProgressBar(steps)
        cm = ConfusionMatrix(self.labels)
        total_loss = 0
        for x, y in loader:
            self.optimizer.zero_grad()
            if type(x) == list:
                x = [torch.autograd.Variable(item.cuda()) for item in x]
            else:
                x = torch.autograd.Variable(x.cuda())
            y = torch.autograd.Variable(y.cuda())
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
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    epochs = int(kwargs.get('epochs', 20))
    model_file = kwargs.get('outfile', './classifier-model.pyth')
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'f1')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))    

    reporting_fns = listify(kwargs.get('reporting', basic_reporting))
    print('reporting', reporting_fns)

    trainer = ClassifyTrainerPyTorch(model, **kwargs)
    
    max_metric = 0
    last_improved = 0

    for epoch in range(epochs):
        start_time = time.time()
        train_metrics = trainer.train(ts)
        train_duration = time.time() - start_time        
        print('Training time (%.3f sec)' % train_duration)

        start_time = time.time()
        test_metrics = trainer.test(vs)
        test_duration = time.time() - start_time
        print('Validation time (%.3f sec)' % test_duration)

        for reporting in reporting_fns:
            reporting(train_metrics, epoch, 'Train')
            reporting(test_metrics, epoch, 'Valid')
        
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
        trainer = ClassifyTrainerPyTorch(model, **kwargs)
        test_metrics = trainer.test(es)
        for reporting in reporting_fns:
            reporting(test_metrics, epoch, 'Test')

