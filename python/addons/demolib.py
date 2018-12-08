from baseline.train import create_trainer, register_trainer, register_training_func, Trainer
from baseline.embeddings import register_embeddings
from baseline.reporting import register_reporting, ReportingHook
from baseline.tf.embeddings import TensorFlowEmbeddings
from baseline.tf.optz import optimizer
from baseline.confusion import ConfusionMatrix
from baseline.utils import listify, get_model_file, write_json, color, Colors
from baseline.tf.tfy import embed
import tensorflow as tf
import os
import numpy as np


@register_embeddings(name='cbow')
class CharBoWEmbeddings(TensorFlowEmbeddings):
    """Bag of character embeddings, sum char embeds, so in this case `wsz == dsz`

    """
    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.int32, [None, None, None], name=name)

    def __init__(self, name, **kwargs):
        super(CharBoWEmbeddings, self).__init__()
        self.vsz = kwargs.get('vsz')
        self.dsz = kwargs.get('dsz')
        self.finetune = kwargs.get('finetune', True)
        self.name = name
        self.scope = kwargs.get('scope', '{}/CharBoWLUT'.format(self.name))
        self.weights = kwargs.get('weights')
        if self.weights is None:
            unif = kwargs.get('unif', 0.1)
            self.weights = np.random.uniform(-unif, unif, (self.vsz, self.dsz))
        self.params = kwargs

    def save_md(self, target):
        write_json({'vsz': self.get_vsz(), 'dsz': self.get_dsz()}, target)

    def encode(self, x=None):
        if x is None:
            x = CharBoWEmbeddings.create_placeholder(self.name)
        self.x = x
        return tf.reduce_sum(embed(x,
                                   self.get_vsz(),
                                   self.get_dsz(),
                                   tf.constant_initializer(self.weights, dtype=tf.float32),
                                   self.finetune,
                                   self.scope), axis=2, keepdims=False)

    def get_vsz(self):
        return self.vsz

    # Warning this function is only initialized AFTER encode
    def get_dsz(self):
        return self.dsz


@register_reporting(name='slack')
class SlackReporting(ReportingHook):

    def __init__(self, **kwargs):
        super(SlackReporting, self).__init__(**kwargs)
        self.webhook = kwargs['webhook']

    def step(self, metrics, tick, phase, tick_type=None, **kwargs):
        """Write results to `slack` (webhook)

        :param metrics: A map of metrics to scores
        :param tick: The time (resolution defined by `tick_type`)
        :param phase: The phase of training (`Train`, `Valid`, `Test`)
        :param tick_type: The resolution of tick (`STEP`, `EPOCH`)
        :return:
        """
        import requests
        chunks = ''
        if phase in ['Valid', 'Test']:
            chunks += '%s(%d) [Epoch %d] [%s]' % (os.getlogin(), os.getpid(), tick, phase)
            for k, v in metrics.items():
                if k not in ['avg_loss', 'perplexity']:
                    v *= 100.
                chunks += '\t%s=%.3f' % (k, v)
            requests.post(self.webhook, json={"text": chunks})


@register_training_func('classify', name='test_every_n_epochs')
def train(model, ts, vs, es=None, **kwargs):
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
    n = int(kwargs.get('test_epochs', 5))
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file('classify', 'tf', kwargs.get('basedir'))

    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', []))
    print('reporting', reporting_fns)

    trainer = create_trainer(model, **kwargs)
    tables = tf.tables_initializer()
    model.sess.run(tables)
    model.sess.run(tf.global_variables_initializer())
    model.set_saver(tf.train.Saver())

    max_metric = 0
    last_improved = 0

    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if epoch > 0 and epoch % n == 0 and epoch < epochs - 1:
            print(color('Running test', Colors.GREEN))
            trainer.test(es, reporting_fns, phase='Test')

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
            print(color('Stopping due to persistent failures to improve', Colors.RED))
            break

    if do_early_stopping is True:
        print('Best performance on max_metric %.3f at epoch %d' % (max_metric, last_improved))

    if es is not None:
        print(color('Reloading best checkpoint', Colors.GREEN))
        trainer.recover_last_checkpoint()
        trainer.test(es, reporting_fns, phase='Test')
