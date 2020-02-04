from baseline.embeddings import register_embeddings
from baseline.tf.embeddings import TensorFlowEmbeddingsModel, TensorFlowEmbeddings, LookupTableEmbeddings
from eight_mile.tf.layers import TRAIN_FLAG, MeanPool1D
from baseline.reporting import register_reporting, ReportingHook
from baseline.train import create_trainer, register_trainer, register_training_func, Trainer
from baseline.utils import get_metric_cmp, get_model_file, color, Colors, listify, Offsets
import tensorflow as tf
import os
import numpy as np


class CharNBoWEmbeddings(TensorFlowEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get("finetune", trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype)
        self.scope = kwargs.get("scope", "CharLUT")
        self.finetune = kwargs.get("finetune", trainable)

        self.pdrop = kwargs.get("pdrop", 0.5)
        self.x = None
        self.embed = LookupTableEmbeddings(name=f"{self.name}/CharLUT", finetune=self.finetune, **kwargs)
        #self.pool = MeanPool1D(self.get_dsz(), )

    def encode(self, x):
        self.x = x
        shape = tf.shape(x)
        B = shape[0]
        T = shape[1]
        W = shape[2]
        flat_chars = tf.reshape(x, [-1, W])
        embed_chars = self.embed(flat_chars)

        # Calculate the lengths of each word
        # You can use layers.MeanPool1D() for this as well
        # (BxT)
        #word_lengths = tf.reduce_sum(tf.cast(tf.not_equal(flat_chars, Offsets.PAD), tf.int32), axis=1)
        #result = self.pool((embed_chars, word_lengths))
        # (BxT)xD
        result = tf.reduce_mean(embed_chars, axis=1)
        return tf.reshape(result, (B, T, -1))

    def call(self, inputs):
        return self.encode(inputs)

    def get_vsz(self):
        return self.embed.get_vsz()

    def get_dsz(self):
        return self.embed.get_dsz()


@register_embeddings(name='cbow')
class CharConvEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = CharNBoWEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


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
def custom_fit_function(model_params, ts, vs, es=None, **kwargs):
    """
    Train a classifier using TensorFlow

    :param model_params: The model_params or model to train
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
        * Additional arguments are supported, see :func:`eight_mile.tf.optimize` for full list
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
    TRAIN_FLAG()

    trainer = create_trainer(model_params, **kwargs)
    max_metric = 0
    last_improved = 0

    for epoch in range(epochs):

        trainer.train(ts, reporting_fns, dataset=False)
        test_metrics = trainer.test(vs, reporting_fns, phase='Valid', dataset=False)

        if epoch > 0 and epoch % n == 0 and epoch < epochs - 1:
            print(color('Running test', Colors.GREEN))
            trainer.test(es, reporting_fns, phase='Test', dataset=False)

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
        trainer.test(es, reporting_fns, phase='Test', dataset=False)
