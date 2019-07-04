import six
import os
import logging
import tensorflow as tf
from baseline.confusion import ConfusionMatrix
from baseline.progress import create_progress_bar
from baseline.utils import listify, get_model_file, get_metric_cmp
from baseline.tf.tfy import _add_ema
from baseline.tf.optz import optimizer
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.utils import verbose_output, unzip_model


logger = logging.getLogger('baseline')


@register_trainer(task='classify', name='default')
class ClassifyTrainerTf(EpochReportingTrainer):

    def __init__(self, model, **kwargs):
        super(ClassifyTrainerTf, self).__init__()
        self.sess = model.sess
        self.loss = model.create_loss()
        self.test_loss = model.create_test_loss()
        self.model = model
        if kwargs.get('eval_mode', False):
            # When using a reloaded model creating the training op will break things.
            train_op = tf.no_op()
        else:
            self.global_step, train_op = optimizer(self.loss, colocate_gradients_with_ops=True, **kwargs)
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)
        decay = kwargs.get('ema_decay', None)
        if decay is not None:
            self.ema = True
            ema_op, self.ema_load, self.ema_restore = _add_ema(model, float(decay))
            with tf.control_dependencies([ema_op]):
                self.train_op = tf.identity(train_op)
        else:
            self.ema = False
            self.train_op = train_op

    @staticmethod
    def _get_batchsz(batch_dict):
        return len(batch_dict['y'])

    def _train(self, loader, **kwargs):

        if self.ema:
            self.sess.run(self.ema_restore)

        reporting_fns = kwargs.get('reporting_fns', [])
        epoch_loss = 0
        epoch_div = 0
        steps = len(loader)
        pg = create_progress_bar(steps)
        for batch_dict in pg(loader):
            feed_dict = self.model.make_input(batch_dict, True)
            _, step, lossv = self.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)
            batchsz = self._get_batchsz(batch_dict)
            report_lossv = lossv * batchsz
            epoch_loss += report_lossv
            epoch_div += batchsz
            self.nstep_agg += report_lossv
            self.nstep_div += batchsz

            if (step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                self.report(
                    step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()

        metrics = self.calc_metrics(epoch_loss, epoch_div)
        return metrics

    def _test(self, loader, **kwargs):

        if self.ema:
            self.sess.run(self.ema_load)

        cm = ConfusionMatrix(self.model.labels)
        steps = len(loader)
        total_loss = 0
        total_norm = 0
        verbose = kwargs.get("verbose", None)
        output = kwargs.get('output')
        txts = kwargs.get('txts')
        handle = None
        line_number = 0
        if output is not None and txts is not None:
            handle = open(output, "w")

        pg = create_progress_bar(steps)
        for batch_dict in pg(loader):
            y = batch_dict['y']
            feed_dict = self.model.make_input(batch_dict)
            guess, lossv = self.sess.run([self.model.best, self.test_loss], feed_dict=feed_dict)
            batchsz = self._get_batchsz(batch_dict)
            if handle is not None:
                for predicted, gold in zip(guess, y):
                    handle.write('{}\t{}\t{}\n'.format(" ".join(txts[line_number]), self.model.labels[predicted], self.model.labels[gold]))
                    line_number += 1
            total_loss += lossv * batchsz
            total_norm += batchsz
            cm.add_batch(y, guess)

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss / float(total_norm)
        verbose_output(verbose, cm)

        if handle is not None:
            handle.close()
        return metrics

    def checkpoint(self):
        self.model.saver.save(self.sess, "./tf-classify-%d/classify" % os.getpid(), global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-classify-%d" % os.getpid())
        logger.info('Reloading %s', latest)
        self.model.saver.restore(self.model.sess, latest)


@register_training_func('classify')
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
    verbose = kwargs.get('verbose', {'console': kwargs.get('verbose_console', False), 'file': kwargs.get('verbose_file', None)})
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file('classify', 'tf', kwargs.get('basedir'))
    ema = True if kwargs.get('ema_decay') is not None else False

    output = kwargs.get('output')
    txts = kwargs.get('txts')

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)

    trainer = create_trainer(model, **kwargs)
    tables = tf.tables_initializer()
    model.sess.run(tables)
    m = model.replicas[0] if hasattr(model, 'replicas') else model
    feed_dict = {k: v for e in m.embeddings.values() for k, v in e.get_feed_dict().items()}
    model.sess.run(tf.global_variables_initializer(), feed_dict)
    model.set_saver(tf.train.Saver())
    checkpoint = kwargs.get('checkpoint')
    if checkpoint is not None:
        checkpoint = unzip_model(checkpoint)
        model.saver.restore(model.sess, checkpoint)

    last_improved = 0

    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            logger.info('New best %.3f', best_metric)
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif (epoch - last_improved) > patience:
            logger.info('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        logger.info('Best performance on %s: %.3f at epoch %d', early_stopping_metric, best_metric, last_improved)

    if es is not None:
        logger.info('Reloading best checkpoint')
        trainer.recover_last_checkpoint()
        test_metrics = trainer.test(es, reporting_fns, phase='Test', verbose=verbose, output=output, txts=txts)
    return test_metrics
