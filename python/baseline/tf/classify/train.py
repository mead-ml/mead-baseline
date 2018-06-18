import os
import tensorflow as tf
from baseline.confusion import ConfusionMatrix
from baseline.progress import create_progress_bar
from baseline.reporting import basic_reporting
from baseline.utils import listify, get_model_file
from baseline.tf.tfy import optimizer, _add_ema
from baseline.train import EpochReportingTrainer, create_trainer

class ClassifyTrainerTf(EpochReportingTrainer):

    def __init__(self, model, **kwargs):
        super(ClassifyTrainerTf, self).__init__()
        self.sess = model.sess
        self.loss = model.create_loss()
        self.model = model
        self.global_step, train_op = optimizer(self.loss, **kwargs)
        decay = kwargs.get('ema_decay', None)
        if decay is not None:
            model.ema, model.ema_op, model.eval_saver = _add_ema(model, decay)
            with tf.control_dependencies([model.ema_op]):
                self.train_op = tf.identity(train_op)
        else:
            self.train_op = train_op

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

    def _test(self, loader):

        total_loss = 0
        cm = ConfusionMatrix(self.model.labels)
        steps = len(loader)
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

        return metrics

    def checkpoint(self, train=False):
        """When train is true it saves checkpoint to a subfolder for ema loading.

        When train is False it saves normally. This is the checkpoint used for
        early stopping.
        """
        file_name = './tf-classify-{}'.format(os.getpid())
        if train:
            file_name = os.path.join(file_name, "train")
            try:
                os.makedirs(file_name)
            except OSError:
                pass
        file_name = os.path.join(file_name, 'classify')
        self.model.saver.save(self.sess, file_name, global_step=self.global_step)

    def eval_restore(self, train=False):
        file_name = './tf-classify-{}'.format(os.getpid())
        if train:
            file_name = os.path.join(file_name, "train")
        latest = tf.train.latest_checkpoint(file_name)
        print('Eval Loading ' + latest)
        try:
            self.model.eval_saver.restore(self.model.sess, latest)
        except AttributeError:
            self.model.saver.restore(self.model.sess, latest)

    def recover_last_checkpoint(self, train=False):
        file_name = './tf-classify-{}'.format(os.getpid())
        if train:
            file_name = os.path.join(file_name, "train")
        latest = tf.train.latest_checkpoint(file_name)
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
        trainer.checkpoint(train=True)
        trainer.eval_restore(train=True)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')
        trainer.recover_last_checkpoint(train=True)

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
        trainer.eval_restore()
        trainer.test(es, reporting_fns, phase='Test')
