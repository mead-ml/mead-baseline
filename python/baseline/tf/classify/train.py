import six
import os
import time
from baseline.utils import fill_y
import tensorflow as tf
from baseline.confusion import ConfusionMatrix
from baseline.progress import create_progress_bar
from baseline.utils import listify, get_model_file, get_metric_cmp
from baseline.tf.tfy import _add_ema, TRAIN_FLAG, SET_TRAIN_FLAG
from baseline.tf.optz import optimizer
from baseline.train import EpochReportingTrainer, create_trainer, register_trainer, register_training_func
from baseline.utils import verbose_output
from baseline.model import create_model_for
import copy
import numpy as np

NUM_PREFETCH = 2
SHUF_BUF_SZ = 5000


def model_creator(model_params):

    def model_fn(features, labels, mode, params):
        model_params.update(features)
        model_params['sess'] = None
        if labels is not None:
           #model_params.update({'y': labels})
           model_params['y'] = tf.one_hot(tf.reshape(labels, [-1, 1]), len(params['labels']))

        if mode == tf.estimator.ModeKeys.PREDICT:
            SET_TRAIN_FLAG(False)
            model = create_model_for('classify', **model_params)
            predictions = {
                'classes': model.best,
                'probabilities': model.probs,
                'logits': model.logits,
            }
            outputs = tf.estimator.export.PredictOutput(predictions['classes'])
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs={'classes': outputs})

        elif mode == tf.estimator.ModeKeys.EVAL:
            SET_TRAIN_FLAG(False)
            model = create_model_for('classify', **model_params)
            loss = model.create_loss()
            eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(
                    labels=labels, predictions=model.best)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=model.logits, loss=loss, eval_metric_ops=eval_metric_ops)

        SET_TRAIN_FLAG(True)
        model = create_model_for('classify', **model_params)
        loss = model.create_loss()
        colocate = True if params['gpus'] > 1 else False
        global_step, train_op = optimizer(loss,
                                          optim=params['optim'],
                                          eta=params.get('lr', params.get('eta')),
                                          colocate_gradients_with_ops=colocate)

        return tf.estimator.EstimatorSpec(mode=mode, predictions=model.logits,
                                          loss=loss,
                                          train_op=train_op)
    return model_fn


@register_trainer(task='classify', name='default')
class ClassifyTrainerTf(EpochReportingTrainer):

    def __init__(self, model_params, **kwargs):
        super(ClassifyTrainerTf, self).__init__()

        if type(model_params) is dict:
            self.model = create_model_for('classify', **model_params)
        else:
            self.model = model_params

        self.sess = self.model.sess
        self.loss = self.model.create_loss()
        self.test_loss = self.model.create_test_loss()
        self.global_step, train_op = optimizer(self.loss, colocate_gradients_with_ops=True, **kwargs)
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)
        decay = kwargs.get('ema_decay', None)
        if decay is not None:
            self.ema = True
            ema_op, self.ema_load, self.ema_restore = _add_ema(self.model, float(decay))
            with tf.control_dependencies([ema_op]):
                self.train_op = tf.identity(train_op)
        else:
            self.ema = False
            self.train_op = train_op

        tables = tf.tables_initializer()
        self.model.sess.run(tables)
        self.model.sess.run(tf.global_variables_initializer())
        self.model.set_saver(tf.train.Saver())

    @staticmethod
    def _get_batchsz(batch_dict):
        return len(batch_dict['y'])

    def _train(self, loader, **kwargs):

        if self.ema:
            self.sess.run(self.ema_restore)

        use_dataset = kwargs.get('dataset', True)
        reporting_fns = kwargs.get('reporting_fns', [])
        epoch_loss = 0
        epoch_div = 0
        steps = len(loader)
        pg = create_progress_bar(steps)
        for batch_dict in pg(loader):
            if use_dataset:
                _, step, lossv = self.sess.run([self.train_op, self.global_step, self.loss],
                                               feed_dict={TRAIN_FLAG(): 1})
            else:
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

        use_dataset = kwargs.get('dataset', True)

        cm = ConfusionMatrix(self.model.labels)
        steps = len(loader)
        total_loss = 0
        total_norm = 0
        verbose = kwargs.get("verbose", None)

        pg = create_progress_bar(steps)
        for i, batch_dict in enumerate(pg(loader)):
            y = batch_dict['y']
            if use_dataset:
                guess, lossv = self.sess.run([self.model.best, self.test_loss])
            else:
                feed_dict = self.model.make_input(batch_dict, False)
                guess, lossv = self.sess.run([self.model.best, self.test_loss], feed_dict=feed_dict)

            batchsz = self._get_batchsz(batch_dict)
            assert len(guess) == batchsz
            total_loss += lossv * batchsz
            total_norm += batchsz
            cm.add_batch(y, guess)

        metrics = cm.get_all_metrics()
        metrics['avg_loss'] = total_loss / float(total_norm)
        verbose_output(verbose, cm)

        return metrics

    def checkpoint(self):
        checkpoint_dir = '{}-{}'.format("./tf-classify", os.getpid())
        self.model.saver.save(self.sess, os.path.join(checkpoint_dir, 'classify'), global_step=self.global_step)

    def recover_last_checkpoint(self):
        checkpoint_dir = '{}-{}'.format("./tf-classify", os.getpid())
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('Reloading ' + latest)
        self.model.saver.restore(self.model.sess, latest)


def to_tensors(model_params, ts):  # "TRAIN_FLAG"
    keys = ts[0].keys()
    nc = len(model_params['labels'])
    d = dict((k, []) for k in keys)
    for sample in ts:
        #sample['y'] = fill_y(nc, sample['y'])
        for k in d.keys():
            # add each sample
            for s in sample[k]:
                d[k].append(s)

    d = dict((k, np.stack(v)) for k, v in d.items())
    y = d.pop('y')
    return d, y


@register_training_func('classify')
def fit_estimator(model_params, ts, vs, es=None, epochs=20, gpus=1, **kwargs):
    model_fn = model_creator(model_params)
    labels = model_params['labels']
    params = {
        'labels': labels,
        'optim': kwargs['optim'],
        'lr': kwargs.get('lr', kwargs.get('eta')),
        'epochs': epochs,
        'gpus': gpus,
        'batchsz': kwargs['batchsz'],
        'test_batchsz': kwargs.get('test_batchsz', kwargs.get('batchsz'))
    }

    checkpoint_dir = '{}-{}'.format("./tf-classify", os.getpid())
    config = tf.estimator.RunConfig(model_dir=checkpoint_dir,
                                    train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=gpus))
    # config = tf.estimator.RunConfig(model_dir=checkpoint_dir)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config, params=params)

    train_input_fn = create_train_input_fn(model_params, ts, **params)
    valid_input_fn = create_valid_input_fn(model_params, vs, **params)
    predict_input_fn = create_eval_input_fn(model_params, es, **params)
    for i in range(epochs):
        estimator.train(input_fn=train_input_fn, steps=len(ts))
        eval_results = estimator.evaluate(input_fn=valid_input_fn, steps=len(vs))
        print(eval_results)

    y_test = [sample['y'] for sample in es]
    predictions = np.array([p['classes'] for p in estimator.predict(input_fn=predict_input_fn)])

    cm = ConfusionMatrix(labels)
    for truth, guess in zip(y_test, predictions):
        cm.add(truth, guess)

    print(cm.get_all_metrics())


def create_train_input_fn(model_params, ts, batchsz=1, gpus=1, epochs=1, **kwargs):

    def train_input_fn():
        train_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(model_params, ts))
        train_dataset = train_dataset.shuffle(buffer_size=SHUF_BUF_SZ)
        train_dataset = train_dataset.batch(batchsz // gpus, drop_remainder=False)
        train_dataset = train_dataset.repeat(epochs)
        train_dataset = train_dataset.prefetch(NUM_PREFETCH)
        _ = train_dataset.make_one_shot_iterator()
        return train_dataset
    return train_input_fn


def create_valid_input_fn(model_params, vs, batchsz=1, gpus=1, epochs=1, **kwargs):
    def eval_input_fn():
        valid_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(model_params, vs))
        valid_dataset = valid_dataset.batch(batchsz // gpus, drop_remainder=False)
        valid_dataset = valid_dataset.repeat(epochs)
        valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)
        _ = valid_dataset.make_one_shot_iterator()
        return valid_dataset

    return eval_input_fn


def create_eval_input_fn(model_params, es, test_batchsz=1, gpus=1, epochs=1, **kwargs):
    def predict_input_fn():
        test_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(model_params, es))
        test_dataset = test_dataset.batch(test_batchsz // gpus, drop_remainder=False)
        test_dataset = test_dataset.prefetch(NUM_PREFETCH)
        _ = test_dataset.make_one_shot_iterator()
        return test_dataset

    return predict_input_fn


@register_training_func('classify-datasets')
def fit_datasets(model_params, ts, vs, es=None, **kwargs):
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

    batchsz = kwargs['batchsz']
    ## First, make tf.datasets for ts, vs and es
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/distribute/README.md
    # effective_batch_sz = args.batchsz*args.gpus
    test_batchsz = kwargs.get('test_batchsz', batchsz)
    train_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(model_params, ts))
    train_dataset = train_dataset.shuffle(buffer_size=SHUF_BUF_SZ)
    train_dataset = train_dataset.batch(batchsz // kwargs.get('gpus', 1), drop_remainder=False)
    train_dataset = train_dataset.repeat(epochs + 1)
    train_dataset = train_dataset.prefetch(NUM_PREFETCH)

    valid_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(model_params, vs))
    valid_dataset = valid_dataset.batch(batchsz // kwargs.get('gpus', 1), drop_remainder=False)
    valid_dataset = valid_dataset.repeat(epochs + 1)
    valid_dataset = valid_dataset.prefetch(NUM_PREFETCH)

    test_dataset = tf.data.Dataset.from_tensor_slices(to_tensors(model_params, es))
    test_dataset = test_dataset.batch(test_batchsz // kwargs.get('gpus', 1), drop_remainder=False)
    test_dataset = test_dataset.repeat(epochs + 1)
    test_dataset = test_dataset.prefetch(NUM_PREFETCH)

    iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)

    features, y = iter.get_next()
    # Add features to the model params
    model_params.update(features)
    model_params.update({'y': y})
    # create the initialisation operations
    train_init_op = iter.make_initializer(train_dataset)
    valid_init_op = iter.make_initializer(valid_dataset)
    test_init_op = iter.make_initializer(test_dataset)

    ema = True if kwargs.get('ema_decay') is not None else False

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    TRAIN_FLAG()
    trainer = create_trainer(model_params, **kwargs)

    last_improved = 0

    for epoch in range(epochs):
        trainer.sess.run(train_init_op)
        trainer.train(ts, reporting_fns)
        trainer.sess.run(valid_init_op)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            print('New best %.3f' % best_metric)
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on %s: %.3f at epoch %d' % (early_stopping_metric, best_metric, last_improved))

    if es is not None:
        print('Reloading best checkpoint')
        trainer.recover_last_checkpoint()
        trainer.sess.run(test_init_op)
        trainer.test(es, reporting_fns, phase='Test', verbose=verbose)


@register_training_func('classify-classic')
def fit(model_params, ts, vs, es=None, **kwargs):
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

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    TRAIN_FLAG()
    trainer = create_trainer(model_params, **kwargs)

    last_improved = 0

    for epoch in range(epochs):

        trainer.train(ts, reporting_fns, dataset=False)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid', dataset=False)

        if do_early_stopping is False:
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            print('New best %.3f' % best_metric)
            trainer.checkpoint()
            trainer.model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on %s: %.3f at epoch %d' % (early_stopping_metric, best_metric, last_improved))

    if es is not None:
        print('Reloading best checkpoint')
        trainer.recover_last_checkpoint()
        trainer.test(es, reporting_fns, phase='Test', verbose=verbose, dataset=False)
