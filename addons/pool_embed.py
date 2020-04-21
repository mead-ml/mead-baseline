import tensorflow as tf
from eight_mile.utils import Offsets
from eight_mile.embeddings import register_embeddings
from eight_mile.tf.layers import MeanPool1D
from eight_mile.tf.embeddings import LookupTableEmbeddings
from baseline.tf.embeddings import TensorFlowEmbeddingsModel


class CBoWEmbeddings(LookupTableEmbeddings):

    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get('finetune', trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        """Create a lookup-table based embedding.

        :param name: The name of the feature/placeholder, and a key for the scope
        :param kwargs:

        :Keyword Arguments: See below
        * *vsz* -- (``int``) this is the vocabulary (input) size of the lookup table
        * *dsz* -- (``int``) the output dimension size of this embedding
        * *finetune* -- (``bool``) (default is `True`) should we allow the sub-graph to learn updated weights
        * *weights* -- (``numpy.ndarray``) Optional `vsz x dsz` weight matrix for initialization
        * *scope* -- (``str``) An optional variable scope, by default it will be `{name}/LUT`
        * *unif* -- (``float``) (defaults to `0.1`) If the weights should be created, what is the random initialization range
        """
        self.pooler = lambda x, y: tf.reduce_sum(x, axis=1)

    def encode(self, x=None):
        """Build a simple Lookup Table and set as input `x` if it exists, or `self.x` otherwise.

        :param x: An optional input sub-graph to bind to this operation or use `self.x` if `None`
        :return: The sub-graph output
        """
        if x is None:
            x = LookupTableEmbeddings.create_placeholder(self._name)
        self.x = x
        shape = tf.shape(x)
        B = shape[0]
        T = shape[1]
        W = shape[2]

        flat = tf.reshape(x, [-1, W])
        emb = super().encode(flat)

        lengths = tf.reduce_sum(tf.cast(tf.not_equal(flat, Offsets.PAD), tf.int32), axis=1)

        pooled = self.pooler(emb, lengths)

        return tf.reshape(pooled, [B, T, self.dsz])


class MaxBoWEmbeddings(CBoWEmbeddings):
    def __init__(self, trainable=True, name=None, dtype=tf.float32, **kwargs):
        trainable = kwargs.get('finetune', trainable)
        super().__init__(trainable=trainable, name=name, dtype=dtype, **kwargs)
        self.pooler = lambda x, y: tf.reduce_max(x, axis=1)


class MeanBoWEmbeddings(CBoWEmbeddings):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.pooling = MeanPool1D(None)

        def pool(x, y):
            y_ = tf.math.maximum(y, 1)
            res = self.pooling((x, y_))
            return tf.multiply(res, tf.expand_dims(tf.cast(tf.not_equal(y, 0), tf.float32), -1))

        self.pooler = pool


@register_embeddings(name='cbow')
class CBoWEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = CBoWEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='max-pool')
class MaxBoWEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = MaxBoWEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)


@register_embeddings(name='mean-pool')
class MeanBoWEmbeddingsModel(TensorFlowEmbeddingsModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name, **kwargs)
        self.embedding_layer = MeanBoWEmbeddings(name=self._name, **kwargs)

    @classmethod
    def create_placeholder(cls, name):
        return tf.compat.v1.placeholder(tf.int32, [None, None, None], name=name)

