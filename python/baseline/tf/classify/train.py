import tensorflow as tf
from baseline.confusion import ConfusionMatrix
from baseline.progress import ProgressBar
from baseline.reporting import basic_reporting
from baseline.utils import listify
from baseline.tf.tfy import optimizer
import time

class ClassifyTrainerTf:

    def __init__(self, model, **kwargs):
        self.sess = model.sess
        self.loss = model.create_loss()
        self.model = model
        self.global_step, self.train_op = optimizer(self.loss, **kwargs)

    def train(self, loader):

        cm = ConfusionMatrix(self.model.labels)
        total_loss = 0
        steps = len(loader)
        pg = ProgressBar(steps)
        for x, y in loader:
            feed_dict = self.model.ex2dict(x, y, do_dropout=True)
            _, step, lossv, guess = self.sess.run([self.train_op, self.global_step, self.loss, self.model.best], feed_dict=feed_dict)
            cm.add_batch(y, guess)
            total_loss += lossv
            pg.update()

        pg.done()
        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss/float(steps)
        return metrics

    def test(self, loader):

        total_loss = 0
        cm = ConfusionMatrix(self.model.labels)
        steps = len(loader)
        pg = ProgressBar(steps)
        for x, y in loader:
            feed_dict = self.model.ex2dict(x, y)
            lossv, guess = self.sess.run([self.loss, self.model.best], feed_dict=feed_dict)
            cm.add_batch(y, guess)
            total_loss += lossv
            pg.update()

        pg.done()
        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss/float(steps)

        return metrics

    def checkpoint(self):
        self.model.saver.save(self.sess, "./tf-checkpoints/classify", global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-checkpoints")
        print('Reloading ' + latest)
        self.model.saver.restore(self.model.sess, latest)

def fit(model, ts, vs, es=None, **kwargs):

    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    epochs = int(kwargs.get('epochs', 20))
    model_file = kwargs.get('outfile', './classifier-model-tf')

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'f1')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', basic_reporting))
    print('reporting', reporting_fns)
    
    trainer = ClassifyTrainerTf(model, **kwargs)
    model.sess.run(tf.global_variables_initializer())
    model.saver = tf.train.Saver()

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

        trainer = ClassifyTrainerTf(model, **kwargs)
        test_metrics = trainer.test(es)
            
        for reporting in reporting_fns:
            reporting(test_metrics, 0, 'Test')
