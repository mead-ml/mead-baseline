from keras.utils import np_utils
from baseline.utils import listify
from baseline.reporting import basic_reporting
from baseline.progress import ProgressBar
from baseline.train import *


class ClassifyTrainerKeras(EpochReportingTrainer):

    METRIC_REMAP = {'fmeasure': 'f1'}

    def __init__(self, model, **kwargs):
        super(ClassifyTrainerKeras, self).__init__()
        self.model = model
        optim = kwargs.get('optim', 'adam')
        self.model.impl.compile(optim, 'categorical_crossentropy', metrics=['accuracy', 'fmeasure'])

    def _train(self, loader):

        train_metrics = {}
        steps = len(loader)
        pg = ProgressBar(steps)
        for x, y in loader:
            y = np_utils.to_categorical(y,  len(self.model.labels))
            metrics = self.model.impl.train_on_batch(x, y)
            for i in range(len(self.model.impl.metrics_names)):
                name = self.model.impl.metrics_names[i]
                name = ClassifyTrainerKeras.METRIC_REMAP.get(name, name)
                train_metrics[name] = train_metrics.get(name, 0) + metrics[i]
            pg.update()

        for k, v in train_metrics.items():
            train_metrics[k] /= steps

        pg.done()
        return train_metrics

    def _test(self, loader):
        test_metrics = {}
        steps = len(loader)
        pg = ProgressBar(steps)
        for x, y in loader:
            y = np_utils.to_categorical(y, len(self.model.labels))
            metrics = self.model.impl.test_on_batch(x, y)
            for i in range(len(self.model.impl.metrics_names)):
                name = self.model.impl.metrics_names[i]
                name = ClassifyTrainerKeras.METRIC_REMAP.get(name, name)
                test_metrics[name] = test_metrics.get(name, 0) + metrics[i]
            pg.update()
        pg.done()

        for k, v in test_metrics.items():
            test_metrics[k] /= steps
        return test_metrics


def fit(model, ts, vs, es=None, **kwargs):
    """
    Train a classifier using Keras
    
    :param model: The model to train
    :param ts: A training data set
    :param vs: A validation data set
    :param es: A test data set, can be None
    :param kwargs: 
        See below
    
    :Keyword Arguments:
        * *do_early_stopping* (``bool``) --
          Stop after evaluation data is no longer improving.  Defaults to True
        
        * *epochs* (``int``) -- how many epochs.  Default to 20
        * *outfile* -- Model output file, defaults to classifier-model-keras
        * *patience* (``int``) -- 
           How many epochs where evaluation is no longer improving before we give up
        * *reporting* --
           Callbacks which may be used on reporting updates
        * *optim* --
           What optimizer to use.  Defaults to `adam`
    :return: 
    """
    trainer = ClassifyTrainerKeras(model, **kwargs)
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    epochs = int(kwargs.get('epochs', 20))
    model_file = kwargs.get('outfile', './classifier-model-keras')

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', basic_reporting))
    print('reporting', reporting_fns)

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
        model.load(model_file)
        trainer = ClassifyTrainerKeras(model, **kwargs)
        trainer.test(es, reporting_fns, phase='Test')
