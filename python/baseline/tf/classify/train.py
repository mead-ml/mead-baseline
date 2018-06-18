import tensorflow as tf
from baseline.confusion import ConfusionMatrix
from baseline.progress import create_progress_bar
from baseline.reporting import basic_reporting
from baseline.utils import listify, get_model_file
from baseline.tf.tfy import optimizer
from baseline.train import EpochReportingTrainer, create_trainer
import os


class ClassifyTrainerTf(EpochReportingTrainer):

    def __init__(self, model, **kwargs):
        super(ClassifyTrainerTf, self).__init__()
        self.sess = model.sess
        self.loss = model.create_loss()
        self.model = model
        self.global_step, self.train_op = optimizer(self.loss, **kwargs)

    def _train(self, loader):

        cm = ConfusionMatrix(self.model.labels)
        total_loss = 0
        steps = len(loader)
        pg = create_progress_bar(steps)
        for batch_dict in loader:
            y = batch_dict['y']
            feed_dict = self.model.make_input(batch_dict, do_dropout=True)
            _, step, lossv, guess = self.sess.run([self.train_op, self.global_step, self.loss, self.model.best], feed_dict=feed_dict)
            cm.add_batch(y, guess)
            total_loss += lossv
            pg.update()

        pg.done()
        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss/float(steps)
        return metrics

    def _test(self, loader, **kwargs):

        total_loss = 0
        cm = ConfusionMatrix(self.model.labels)
        steps = len(loader)
        print_cm = kwargs.get("print_cm", False)

        pg = create_progress_bar(steps)
        for batch_dict in loader:
            y = batch_dict['y']
            feed_dict = self.model.make_input(batch_dict)
            lossv, guess = self.sess.run([self.loss, self.model.best], feed_dict=feed_dict)
            cm.add_batch(y, guess)
            total_loss += lossv
            pg.update()

        pg.done()
        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss/float(steps)
        if print_cm:
            print("confusion matrix\n{}\n".format("".join(["="]*40)))
            print(cm)
            print("\n{}\n".format("".join(["="]*40)))

        return metrics

    def checkpoint(self):
        self.model.saver.save(self.sess, "./tf-classify-%d/classify" % os.getpid(), global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-classify-%d" % os.getpid())
        print('Reloading ' + latest)
        self.model.saver.restore(self.model.sess, latest)


def fit(model, ts, vs, es=None, **kwargs):
    """
    Train a classifier using TensorFlow
    
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
        * *outfile* -- Model output file, defaults to classifier-model.pyth
        * *patience* -- 
           How many epochs where evaluation is no longer improving before we give up
        * *reporting* --
           Callbacks which may be used on reporting updates
        * Additional arguments are supported, see :func:`baseline.tf.optimize` for full list
    :return: 
    """
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    print_cm = bool(kwargs.get('print_cm', False))
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file(kwargs, 'classify', 'tf')

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', basic_reporting))
    print('reporting', reporting_fns)
    
    trainer = create_trainer(ClassifyTrainerTf, model, **kwargs)
    tables = tf.tables_initializer()
    model.sess.run(tables)
    model.sess.run(tf.global_variables_initializer())
    model.saver = tf.train.Saver()

    max_metric = 0
    last_improved = 0

    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif test_metrics[early_stopping_metric] > max_metric:
            last_improved = epoch
            max_metric = test_metrics[early_stopping_metric]
            print('New max %.3f' % max_metric)
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on max_metric %.3f at epoch %d' % (max_metric, last_improved))

    if es is not None:
        print('Reloading best checkpoint')
        trainer.recover_last_checkpoint()
        trainer.test(es, reporting_fns, phase='Test', print_cm=print_cm)
