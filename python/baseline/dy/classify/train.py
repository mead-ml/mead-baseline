from baseline.utils import listify, get_model_file
from baseline.progress import create_progress_bar
from baseline.confusion import ConfusionMatrix
from baseline.reporting import basic_reporting
from baseline.train import EpochReportingTrainer, create_trainer
import dynet as dy
import numpy as np

def _add_to_cm(cm, y, preds):
    best = np.argmax(preds, axis=0)
    best = np.reshape(best, y.shape)
    cm.add_batch(y, best)

class ClassifyTrainerDynet(EpochReportingTrainer):

    def __init__(
            self,
            model,
            optim='sgd', clip=5, mom=0.9,
            **kwargs
    ):
        super(ClassifyTrainerDynet, self).__init__()
        self.model = model
        eta = kwargs.get('eta', kwargs.get('lr', 0.01))
        print("Using eta [{:.4f}]".format(eta))
        print("Using optim [{}]".format(optim))
        self.labels = model.labels
        if optim == 'adadelta':
            self.optimizer = dy.AdadeltaTrainer(model.pc)
        elif optim == 'adam':
            self.optimizer = dy.AdamTrainer(model.pc)
        elif optim == 'rmsprop':
            self.optimizer = dy.RMSPropTrainer(model.pc, learning_rate=eta)
        else:
            print("using mom {:.3f}".format(mom))
            self.optimizer = dy.MomentumSGDTrainer(model.pc, learning_rate=eta, mom=mom)
        self.optimizer.set_clip_threshold(clip)

    def _test(self, loader):
        total_loss = 0
        steps = len(loader)
        pg = create_progress_bar(steps)
        cm = ConfusionMatrix(self.labels)

        for batch_dict in pg(loader):
            dy.renew_cg()
            xs, ys = self.model.make_input(batch_dict)
            preds = self.model.forward(xs)
            losses = self.model.loss(preds, ys)
            loss = dy.sum_batches(losses)
            total_loss += loss.npvalue().item()
            _add_to_cm(cm, ys, preds.npvalue())

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss / float(steps)
        return metrics


    def _train(self, loader):
        steps = len(loader)
        pg = create_progress_bar(steps)
        cm = ConfusionMatrix(self.labels)
        total_loss = 0
        for batch_dict in pg(loader):
            dy.renew_cg()
            xs, ys = self.model.make_input(batch_dict)
            preds = self.model.forward(xs)
            losses = self.model.loss(preds, ys)
            loss = dy.sum_batches(losses)
            total_loss += loss.npvalue().item()
            _add_to_cm(cm, ys, preds.npvalue())
            loss.backward()
            self.optimizer.update()

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss / float(steps)
        return metrics

def fit(
        model,
        ts, vs, es,
        epochs=20,
        do_early_stopping=True, early_stopping_metric='acc',
        reporting=basic_reporting,
        **kwargs
):
    model_file = get_model_file(kwargs, 'classify', 'dynet')
    if do_early_stopping:
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [{}] with patience [{}]'.format(early_stopping_metric, patience))

    reporting_fns = listify(reporting)
    print('reporting', reporting_fns)

    trainer = create_trainer(ClassifyTrainerDynet, model, **kwargs)

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
            print('New max {:.3f}'.format(max_metric))
            model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on max_metric {:.3f} at epoch {}'.format(max_metric, last_improved))

    if es is not None:
        print('Reloading best checkpoint')
        model = model.load(model_file)
        trainer = create_trainer(ClassifyTrainerDynet, model, **kwargs)
        trainer.test(es, reporting_fns, phase='Test')
